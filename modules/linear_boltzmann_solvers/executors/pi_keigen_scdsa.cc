// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/executors/pi_keigen_scdsa.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_continuous.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/acceleration/diffusion_mip_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/acceleration/diffusion_pwlc_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/iterative_methods/ags_solver.h"
#include "framework/math/vector_ghost_communicator/vector_ghost_communicator.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/timer.h"
#include "framework/logging/log.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include <iomanip>

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, PowerIterationKEigenSCDSA);

InputParameters
PowerIterationKEigenSCDSA::GetInputParameters()
{
  InputParameters params = PowerIterationKEigen::GetInputParameters();

  params.SetGeneralDescription("Implementation of a k-eigenvalue solver using power iteration with "
                               "SCDSA acceleration.");
  params.SetDocGroup("LBSExecutors");
  params.AddOptionalParameter("accel_pi_max_its",
                              50,
                              "Maximum allowable iterations for the acceleration method's inner "
                              "power iterations");
  params.AddOptionalParameter("accel_pi_k_tol",
                              1.0e-10,
                              "k-eigenvalue tolerance for the acceleration method's inner power "
                              "iterations");
  params.AddOptionalParameter(
    "accel_pi_verbose", false, "If true, enables verbose output from the acceleration method");
  params.AddOptionalParameter("diff_accel_diffusion_l_abs_tol",
                              1.0e-10,
                              "Absolute residual tolerance to use for the diffusion accelerator");
  params.AddOptionalParameter("diff_accel_diffusion_max_iters",
                              100,
                              "Maximum allowable iterations for the diffusion accelerator");
  params.AddOptionalParameter("diff_accel_diffusion_verbose",
                              false,
                              "If true, enables verbose output of the diffusion accelerator");
  params.AddOptionalParameter("diff_accel_diffusion_petsc_options",
                              std::string("ssss"),
                              "Additional PETSc options for the diffusion accelerator");
  params.AddOptionalParameter(
    "diff_accel_sdm", "pwld", "Spatial discretization to use for the diffusion solver");
  params.ConstrainParameterRange("diff_accel_sdm", AllowableRangeList::New({"pwld", "pwlc"}));

  return params;
}

PowerIterationKEigenSCDSA::PowerIterationKEigenSCDSA(const InputParameters& params)
  : PowerIterationKEigen(params),
    accel_pi_max_its_(params.GetParamValue<int>("accel_pi_max_its")),
    accel_pi_k_tol_(params.GetParamValue<double>("accel_pi_k_tol")),
    accel_pi_verbose_(params.GetParamValue<bool>("accel_pi_verbose")),
    diffusion_solver_sdm_(params.GetParamValue<std::string>("diff_accel_sdm")),
    diff_accel_diffusion_l_abs_tol_(params.GetParamValue<double>("diff_accel_diffusion_l_abs_tol")),
    diff_accel_diffusion_max_iters_(params.GetParamValue<int>("diff_accel_diffusion_max_iters")),
    diff_accel_diffusion_verbose_(params.GetParamValue<bool>("diff_accel_diffusion_verbose")),
    diff_accel_diffusion_petsc_options_(
      params.GetParamValue<std::string>("diff_accel_diffusion_petsc_options"))
{
  if (lbs_solver_.Groupsets().size() != 1)
    throw std::logic_error("The SCDSA k-eigenvalue executor is only implemented for "
                           "problems with a single groupset.");
}

void
PowerIterationKEigenSCDSA::Initialize()
{
  PowerIterationKEigen::Initialize();

  // Make UnknownManager
  const size_t num_gs_groups = front_gs_.groups.size();
  UnknownManager uk_man;
  uk_man.AddUnknown(UnknownType::VECTOR_N, num_gs_groups);

  // Make boundary conditions
  auto bcs = TranslateBCs(lbs_solver_.SweepBoundaries(), true);

  // Make xs map
  auto matid_2_mgxs_map = PackGroupsetXS(
    lbs_solver_.GetMatID2XSMap(), front_gs_.groups.front().id, front_gs_.groups.back().id);

  // Create solver
  const auto& sdm = lbs_solver_.SpatialDiscretization();
  const auto& unit_cell_matrices = lbs_solver_.GetUnitCellMatrices();

  if (diffusion_solver_sdm_ == "pwld")
  {
    diffusion_solver_ = std::make_shared<DiffusionMIPSolver>(std::string(Name() + "_WGDSA"),
                                                             sdm,
                                                             uk_man,
                                                             bcs,
                                                             matid_2_mgxs_map,
                                                             unit_cell_matrices,
                                                             false,
                                                             true);
  }
  else
  {
    continuous_sdm_ptr_ = PieceWiseLinearContinuous::New(sdm.Grid());
    diffusion_solver_ = std::make_shared<DiffusionPWLCSolver>(std::string(Name() + "_WGDSA"),
                                                              *continuous_sdm_ptr_,
                                                              uk_man,
                                                              bcs,
                                                              matid_2_mgxs_map,
                                                              unit_cell_matrices,
                                                              false,
                                                              true);
    requires_ghosts_ = true;
    lbs_pwld_ghost_info_ =
      MakePWLDVecGhostCommInfo(lbs_solver_.SpatialDiscretization(), lbs_solver_.UnknownManager());

    const auto& cfem_sdm = *continuous_sdm_ptr_;
    const auto ghost_dof_ids = cfem_sdm.GetGhostDOFIndices(lbs_solver_.UnknownManager());
  }

  auto& ds = diffusion_solver_;
  ds->options.residual_tolerance = diff_accel_diffusion_l_abs_tol_;
  ds->options.max_iters = diff_accel_diffusion_max_iters_;
  ds->options.verbose = diff_accel_diffusion_verbose_;
  ds->options.additional_options_string = diff_accel_diffusion_petsc_options_;

  log.Log() << "Initializing diffusion solver";
  diffusion_solver_->Initialize();
  opensn::mpi_comm.barrier();
  log.Log() << "Done Initializing diffusion solver";

  log.Log() << "Assembling A and b";
  std::vector<double> dummy_rhs;
  if (diffusion_solver_sdm_ == "pwld")
    dummy_rhs.assign(sdm.GetNumLocalDOFs(uk_man), 0.0);
  else
    dummy_rhs.assign(continuous_sdm_ptr_->GetNumLocalAndGhostDOFs(uk_man), 0.0);
  diffusion_solver_->AssembleAand_b(dummy_rhs);
  log.Log() << "Done Assembling A and b";
}

void
PowerIterationKEigenSCDSA::Execute()
{
  auto phi_temp = phi_old_local_;

  /**Lambda for the creation of scattering sources but the input vector is only the zeroth moment*/
  auto SetLBSScatterSourcePhi0 = [this, &phi_temp](const std::vector<double>& input,
                                                   const bool additive,
                                                   const bool suppress_wg_scat = false)
  {
    ProjectBackPhi0(front_gs_, input, phi_temp);
    SetLBSScatterSource(phi_temp, additive, suppress_wg_scat);
  };

  k_eff_ = 1.0;
  double k_eff_prev = 1.0;
  double k_eff_change = 1.0;

  // Start power iterations
  int nit = 0;
  bool converged = false;
  while (nit < max_iters_)
  {
    // Set the fission source
    SetLBSFissionSource(phi_old_local_, false);
    Scale(q_moments_local_, 1.0 / k_eff_);
    auto Sf0_ell = CopyOnlyPhi0(front_gs_, q_moments_local_);

    // Store the last power iteration scalar flux
    auto phi0_ell = CopyOnlyPhi0(front_gs_, phi_old_local_);

    // Solve transport inners
    ags_solver_->Solve();

    // Store the latest scalar flux (lph = ell + 1/2)
    auto phi0_lph_star = CopyOnlyPhi0(front_gs_, phi_new_local_);

    // Power Iteration Acceleration
    SetLBSScatterSourcePhi0(phi0_lph_star - phi0_ell, false);
    auto Ss_res = CopyOnlyPhi0(front_gs_, q_moments_local_);

    double production_k = lbs_solver_.ComputeFissionProduction(phi_new_local_);

    std::vector<double> epsilon_k(phi0_lph_star.size(), 0.0);
    auto epsilon_kp1 = epsilon_k;

    double lambda_k = k_eff_;
    double lambda_kp1 = lambda_k;

    for (size_t k = 0; k < accel_pi_max_its_; ++k)
    {
      ProjectBackPhi0(front_gs_, epsilon_k + phi0_lph_star, phi_temp);
      SetLBSFissionSource(phi_temp, false);
      Scale(q_moments_local_, 1.0 / lambda_k);

      auto Sfaux = CopyOnlyPhi0(front_gs_, q_moments_local_);

      // Inner iterations seem extremely wasteful. Set this to 1 iteration here for further
      // investigation.
      for (int i = 0; i < 1; ++i)
      {
        SetLBSScatterSourcePhi0(epsilon_k, false, true);

        auto Ss = CopyOnlyPhi0(front_gs_, q_moments_local_);

        // Solve the diffusion system
        diffusion_solver_->Assemble_b(Ss + Sfaux + Ss_res - Sf0_ell);
        diffusion_solver_->Solve(epsilon_kp1, true);

        epsilon_k = epsilon_kp1;
      }

      ProjectBackPhi0(front_gs_, epsilon_kp1 + phi0_lph_star, phi_old_local_);

      double production_kp1 = lbs_solver_.ComputeFissionProduction(phi_old_local_);

      lambda_kp1 = production_kp1 / (production_k / lambda_k);

      const double lambda_change = std::fabs(lambda_kp1 / lambda_k - 1.0);
      if (accel_pi_verbose_ >= 1)
        log.Log() << "PISCDSA iteration " << k << " lambda " << lambda_kp1 << " lambda change "
                  << lambda_change;

      if (lambda_change < accel_pi_k_tol_)
        break;

      lambda_k = lambda_kp1;
      epsilon_k = epsilon_kp1;
      production_k = production_kp1;
    } // acceleration

    ProjectBackPhi0(front_gs_, epsilon_kp1 + phi0_lph_star, phi_new_local_);
    lbs_solver_.GSScopedCopyPrimarySTLvectors(front_gs_, phi_new_local_, phi_old_local_);

    const double production = lbs_solver_.ComputeFissionProduction(phi_old_local_);
    lbs_solver_.ScalePhiVector(PhiSTLOption::PHI_OLD, lambda_kp1 / production);

    // Recompute k-eigenvalue
    k_eff_ = lambda_kp1;

    // Check convergence, bookkeeping
    k_eff_change = std::fabs(k_eff_ / k_eff_prev - 1.0);
    k_eff_prev = k_eff_;
    nit += 1;

    if (k_eff_change < std::max(k_tolerance_, 1.0e-12))
      converged = true;

    // Print iteration summary
    if (lbs_solver_.Options().verbose_outer_iterations)
    {
      std::stringstream k_iter_info;
      k_iter_info << program_timer.GetTimeString() << " "
                  << "  Iteration " << std::setw(5) << nit << "  k_eff " << std::setw(11)
                  << std::setprecision(7) << k_eff_ << "  k_eff change " << std::setw(12)
                  << k_eff_change << "  reactivity " << std::setw(10)
                  << (k_eff_ - 1.0) / k_eff_ * 1e5;
      if (converged)
        k_iter_info << " CONVERGED\n";

      log.Log() << k_iter_info.str();
    }

    if (converged)
      break;
  } // for k iterations

  // Print summary
  log.Log() << "\n";
  log.Log() << "        Final k-eigenvalue    :        " << std::setprecision(7) << k_eff_;
  log.Log() << "        Final change          :        " << std::setprecision(6) << k_eff_change
            << " (Number of Sweeps:" << front_wgs_context_->counter_applications_of_inv_op << ")"
            << "\n";

  if (lbs_solver_.Options().use_precursors)
  {
    lbs_solver_.ComputePrecursors();
    Scale(lbs_solver_.PrecursorsNewLocal(), 1.0 / k_eff_);
  }

  lbs_solver_.UpdateFieldFunctions();

  log.Log() << "LinearBoltzmann::KEigenvalueSolver execution completed\n\n";
}

std::vector<double>
PowerIterationKEigenSCDSA::CopyOnlyPhi0(const LBSGroupset& groupset,
                                        const std::vector<double>& phi_in)
{
  const auto& lbs_sdm = lbs_solver_.SpatialDiscretization();
  const auto& diff_sdm = diffusion_solver_->SpatialDiscretization();
  const auto& diff_uk_man = diffusion_solver_->UnknownStructure();
  const auto& phi_uk_man = lbs_solver_.UnknownManager();
  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();
  const size_t diff_num_local_dofs = requires_ghosts_
                                       ? diff_sdm.GetNumLocalAndGhostDOFs(diff_uk_man)
                                       : diff_sdm.GetNumLocalDOFs(diff_uk_man);

  std::vector<double> phi_data;
  if (continuous_sdm_ptr_)
    phi_data =
      NodallyAveragedPWLDVector(phi_in, lbs_sdm, diff_sdm, phi_uk_man, lbs_pwld_ghost_info_);
  else
    phi_data = phi_in;

  std::vector<double> output_phi_local(diff_num_local_dofs, 0.0);

  for (const auto& cell : lbs_solver_.Grid().local_cells)
  {
    const auto& cell_mapping = lbs_sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t diff_phi_map = diff_sdm.MapDOFLocal(cell, i, diff_uk_man, 0, 0);
      const int64_t lbs_phi_map = lbs_sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      double* output_mapped = &output_phi_local[diff_phi_map];
      const double* phi_in_mapped = &phi_data[lbs_phi_map];

      for (size_t g = 0; g < gss; ++g)
      {
        output_mapped[g] = phi_in_mapped[g];
      } // for g
    }   // for node
  }     // for cell

  return output_phi_local;
}

void
PowerIterationKEigenSCDSA::ProjectBackPhi0(const LBSGroupset& groupset,
                                           const std::vector<double>& input,
                                           std::vector<double>& output)
{
  const auto& lbs_sdm = lbs_solver_.SpatialDiscretization();
  const auto& diff_sdm = diffusion_solver_->SpatialDiscretization();
  const auto& diff_uk_man = diffusion_solver_->UnknownStructure();
  const auto& phi_uk_man = lbs_solver_.UnknownManager();
  const int gsi = groupset.groups.front().id;
  const size_t gss = groupset.groups.size();
  const size_t diff_num_local_dofs = requires_ghosts_
                                       ? diff_sdm.GetNumLocalAndGhostDOFs(diff_uk_man)
                                       : diff_sdm.GetNumLocalDOFs(diff_uk_man);

  OpenSnLogicalErrorIf(input.size() != diff_num_local_dofs, "Vector size mismatch");

  for (const auto& cell : lbs_solver_.Grid().local_cells)
  {
    const auto& cell_mapping = lbs_sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const int64_t diff_phi_map = diff_sdm.MapDOFLocal(cell, i, diff_uk_man, 0, 0);
      const int64_t lbs_phi_map = lbs_sdm.MapDOFLocal(cell, i, phi_uk_man, 0, gsi);

      const double* input_mapped = &input[diff_phi_map];
      double* output_mapped = &output[lbs_phi_map];

      for (int g = 0; g < gss; ++g)
        output_mapped[g] = input_mapped[g];
    } // for dof
  }   // for cell
}

PowerIterationKEigenSCDSA::GhostInfo
PowerIterationKEigenSCDSA::MakePWLDVecGhostCommInfo(const SpatialDiscretization& sdm,
                                                    const UnknownManager& uk_man)
{
  log.Log() << "Making PWLD ghost communicator";

  const size_t num_local_dofs = sdm.GetNumLocalDOFs(uk_man);
  const size_t num_globl_dofs = sdm.GetNumGlobalDOFs(uk_man);

  log.Log() << "Number of global dofs" << num_globl_dofs;

  const size_t num_unknowns = uk_man.unknowns.size();

  // Build a list of global ids
  std::set<int64_t> global_dof_ids_set;

  const auto& grid = lbs_solver_.Grid();
  const auto ghost_cell_ids = grid.cells.GetGhostGlobalIDs();
  for (const auto global_id : ghost_cell_ids)
  {
    const auto& cell = grid.cells[global_id];
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      for (size_t u = 0; u < num_unknowns; ++u)
      {
        const size_t num_comps = uk_man.unknowns[u].num_components;
        for (size_t c = 0; c < num_comps; ++c)
        {
          const int64_t dof_map = sdm.MapDOF(cell, i, uk_man, u, c);
          global_dof_ids_set.insert(dof_map);
        } // for component
      }   // for unknown
    }     // for node i
  }       // for ghost cell

  // Convert the list to a vector
  std::vector<int64_t> global_indices(global_dof_ids_set.begin(), global_dof_ids_set.end());

  // Create the vector ghost communicator
  auto vgc = std::make_shared<VectorGhostCommunicator>(
    num_local_dofs, num_globl_dofs, global_indices, mpi_comm);

  // Create the map
  std::map<int64_t, int64_t> ghost_global_id_2_local_map;
  int64_t k = 0;
  for (const auto ghost_id : global_indices)
  {
    ghost_global_id_2_local_map[ghost_id] = static_cast<int64_t>(num_local_dofs + k++);
  }

  log.Log() << "Done making PWLD ghost communicator";
  return {vgc, ghost_global_id_2_local_map};
}

std::vector<double>
PowerIterationKEigenSCDSA::NodallyAveragedPWLDVector(
  const std::vector<double>& input,
  const SpatialDiscretization& pwld_sdm,
  const SpatialDiscretization& pwlc_sdm,
  const UnknownManager& uk_man,
  const PowerIterationKEigenSCDSA::GhostInfo& ghost_info)
{
  const auto& vgc = ghost_info.vector_ghost_communicator;
  const auto& dfem_dof_global2local_map = ghost_info.ghost_global_id_2_local_map;

  auto input_with_ghosts = vgc->MakeGhostedVector(input);
  vgc->CommunicateGhostEntries(input_with_ghosts);

  const auto& grid = pwld_sdm.Grid();

  const size_t num_unknowns = uk_man.unknowns.size();

  const size_t num_cfem_local_dofs = pwlc_sdm.GetNumLocalAndGhostDOFs(uk_man);

  std::vector<double> cont_input(num_cfem_local_dofs, 0.0);
  std::vector<double> cont_input_ctr(num_cfem_local_dofs, 0.0);

  std::map<int64_t, int64_t> cfem_dof_global2local_map;

  // Local cells first
  std::set<uint64_t> partition_bndry_vertex_id_set;
  for (const auto& cell : grid.local_cells)
  {
    const auto& cell_mapping = pwld_sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      for (size_t u = 0; u < num_unknowns; ++u)
      {
        const size_t num_components = uk_man.unknowns[u].num_components;
        for (size_t c = 0; c < num_components; ++c)
        {
          const int64_t dof_dfem_map = pwld_sdm.MapDOFLocal(cell, i, uk_man, u, c);
          const int64_t dof_cfem_map = pwlc_sdm.MapDOFLocal(cell, i, uk_man, u, c);
          const int64_t dof_cfem_map_globl = pwlc_sdm.MapDOF(cell, i, uk_man, u, c);

          cfem_dof_global2local_map[dof_cfem_map_globl] = dof_cfem_map;

          const double phi_value = input[dof_dfem_map];

          cont_input[dof_cfem_map] += phi_value;
          cont_input_ctr[dof_cfem_map] += 1.0;
        } // for component c
      }   // for unknown u
    }     // for node i

    for (const auto& face : cell.faces)
      if (face.has_neighbor)
        if (not grid.IsCellLocal(face.neighbor_id))
          for (const uint64_t vid : face.vertex_ids)
            partition_bndry_vertex_id_set.insert(vid);
  } // for local cell

  // Ghost cells
  const auto ghost_cell_ids = grid.cells.GetGhostGlobalIDs();
  const auto& vid_set = partition_bndry_vertex_id_set;
  for (const auto global_id : ghost_cell_ids)
  {
    const auto& cell = grid.cells[global_id];
    const auto& cell_mapping = pwld_sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      if (vid_set.find(cell.vertex_ids[i]) == vid_set.end())
        continue;

      for (size_t u = 0; u < num_unknowns; ++u)
      {
        const size_t num_components = uk_man.unknowns[u].num_components;
        for (size_t c = 0; c < num_components; ++c)
        {
          const int64_t dof_dfem_map_globl = pwld_sdm.MapDOF(cell, i, uk_man, u, c);
          const int64_t dof_cfem_map_globl = pwlc_sdm.MapDOF(cell, i, uk_man, u, c);
          if (cfem_dof_global2local_map.count(dof_cfem_map_globl) > 0)
          {
            const int64_t dof_dfem_map = dfem_dof_global2local_map.at(dof_dfem_map_globl);
            const int64_t dof_cfem_map = cfem_dof_global2local_map[dof_cfem_map_globl];

            const double phi_value = input_with_ghosts[dof_dfem_map];

            cont_input[dof_cfem_map] += phi_value;
            cont_input_ctr[dof_cfem_map] += 1.0;
          }
        } // for component
      }   // for unknown
    }     // for node i
  }       // for ghost cell

  // Compute nodal averages
  {
    const size_t num_vals = cont_input.size();
    for (size_t k = 0; k < num_vals; ++k)
      cont_input[k] /= cont_input_ctr[k];
  }

  // Project back to dfem
  auto output = input;
  for (const auto& cell : grid.local_cells)
  {
    const auto& cell_mapping = pwld_sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.NumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      for (size_t u = 0; u < num_unknowns; ++u)
      {
        const size_t num_components = uk_man.unknowns[u].num_components;
        for (size_t c = 0; c < num_components; ++c)
        {
          const int64_t dof_dfem_map = pwld_sdm.MapDOFLocal(cell, i, uk_man, u, c);
          const int64_t dof_cfem_map = pwlc_sdm.MapDOFLocal(cell, i, uk_man, u, c);

          const double phi_value = cont_input[dof_cfem_map];

          output[dof_dfem_map] = phi_value;
        } // for component c
      }   // for unknown u
    }     // for node i
  }       // for local cell

  return output;
}

} // namespace opensn
