// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/point_source_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/point_source/point_source.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_vecops.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/source_functions/source_function.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/logging/log.h"
#include "framework/logging/log_exceptions.h"
#include "framework/utils/timer.h"
#include "framework/utils/utils.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <iomanip>

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, PointSourceProblem);

PointSourceProblem::PointSourceProblem(const std::string& name,
                                       std::shared_ptr<MeshContinuum> grid_ptr)
  : LBSProblem(name, grid_ptr)
{
}

InputParameters
PointSourceProblem::GetInputParameters()
{
  InputParameters params = LBSProblem::GetInputParameters();

  params.SetClassName("PointSourceProblem");

  params.ChangeExistingParamToOptional("name", "LBSPointSourceProblem");

  return params;
}

std::shared_ptr<PointSourceProblem>
PointSourceProblem::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<PointSourceProblem>("lbs::PointSourceProblem", params);
}

PointSourceProblem::PointSourceProblem(const InputParameters& params)
  : LBSProblem(params)
{

}

PointSourceProblem::~PointSourceProblem()
{
}

void
PointSourceProblem::Initialize()
{
  CALI_CXX_MARK_SCOPE("PointSourceProblem::Initialize");

  LBSProblem::Initialize();

  const auto grid_dim = grid_->GetDimension();
}

void
PointSourceProblem::InitializeBoundaries()
{
  CALI_CXX_MARK_SCOPE("PointSourceProblem::InitializeBoundaries");

  // Determine boundary-ids involved in the problem
  std::set<uint64_t> global_unique_bids_set;
  {
    std::set<uint64_t> local_unique_bids_set;
    for (const auto& cell : grid_->local_cells)
      for (const auto& face : cell.faces)
        if (not face.has_neighbor)
          local_unique_bids_set.insert(face.neighbor_id);

    std::vector<uint64_t> local_unique_bids(local_unique_bids_set.begin(),
                                            local_unique_bids_set.end());
    std::vector<uint64_t> recvbuf;
    mpi_comm.all_gather(local_unique_bids, recvbuf);

    global_unique_bids_set = local_unique_bids_set; // give it a head start

    for (uint64_t bid : recvbuf)
      global_unique_bids_set.insert(bid);
  }

  // // Initialize default incident boundary
  // const size_t G = num_groups_;

  // sweep_boundaries_.clear();
  // for (uint64_t bid : global_unique_bids_set)
  // {
  //   const bool has_no_preference = boundary_preferences_.count(bid) == 0;
  //   const bool has_not_been_set = sweep_boundaries_.count(bid) == 0;
  //   if (has_no_preference and has_not_been_set)
  //   {
  //     sweep_boundaries_[bid] = std::make_shared<VacuumBoundary>(G);
  //   } // defaulted
  //   else if (has_not_been_set)
  //   {
  //     const auto& bndry_pref = boundary_preferences_.at(bid);
  //     const auto& mg_q = bndry_pref.isotropic_mg_source;

  //     if (bndry_pref.type == LBSBoundaryType::VACUUM)
  //       sweep_boundaries_[bid] = std::make_shared<VacuumBoundary>(G);
  //     else if (bndry_pref.type == LBSBoundaryType::ISOTROPIC)
  //       sweep_boundaries_[bid] = std::make_shared<IsotropicBoundary>(G, mg_q);
  //     else if (bndry_pref.type == LBSBoundaryType::REFLECTING)
  //     {
  //       // Locally check all faces, that subscribe to this boundary,
  //       // have the same normal
  //       const double EPSILON = 1.0e-12;
  //       std::unique_ptr<Vector3> n_ptr = nullptr;
  //       for (const auto& cell : grid_->local_cells)
  //         for (const auto& face : cell.faces)
  //           if (not face.has_neighbor and face.neighbor_id == bid)
  //           {
  //             if (not n_ptr)
  //               n_ptr = std::make_unique<Vector3>(face.normal);
  //             if (std::fabs(face.normal.Dot(*n_ptr) - 1.0) > EPSILON)
  //               throw std::logic_error(
  //                 "LBSProblem: Not all face normals are, within tolerance, locally the "
  //                 "same for the reflecting boundary condition requested");
  //           }

  //       // Now check globally
  //       const int local_has_bid = n_ptr != nullptr ? 1 : 0;
  //       const Vector3 local_normal = local_has_bid ? *n_ptr : Vector3(0.0, 0.0, 0.0);

  //       std::vector<int> locJ_has_bid(opensn::mpi_comm.size(), 1);
  //       std::vector<double> locJ_n_val(opensn::mpi_comm.size() * 3, 0.0);

  //       mpi_comm.all_gather(local_has_bid, locJ_has_bid);
  //       std::vector<double> lnv = {local_normal.x, local_normal.y, local_normal.z};
  //       mpi_comm.all_gather(lnv.data(), 3, locJ_n_val.data(), 3);

  //       Vector3 global_normal;
  //       for (int j = 0; j < opensn::mpi_comm.size(); ++j)
  //       {
  //         if (locJ_has_bid[j])
  //         {
  //           int offset = 3 * j;
  //           const double* n = &locJ_n_val[offset];
  //           const Vector3 locJ_normal(n[0], n[1], n[2]);

  //           if (local_has_bid)
  //             if (std::fabs(local_normal.Dot(locJ_normal) - 1.0) > EPSILON)
  //               throw std::logic_error(
  //                 "LBSProblem: Not all face normals are, within tolerance, globally the "
  //                 "same for the reflecting boundary condition requested");

  //           global_normal = locJ_normal;
  //         }
  //       }

  //       sweep_boundaries_[bid] = std::make_shared<ReflectingBoundary>(
  //         G, global_normal, MapGeometryTypeToCoordSys(options_.geometry_type));
  //     }
  //   } // non-defaulted
  // } // for bndry id
}

void
PointSourceProblem::ZeroOutflowBalanceVars(LBSGroupset& groupset)
{
  CALI_CXX_MARK_SCOPE("PointSourceProblem::ZeroOutflowBalanceVars");

  for (const auto& cell : grid_->local_cells)
    for (int f = 0; f < cell.faces.size(); ++f)
      for (auto& group : groupset.groups)
        cell_transport_views_[cell.local_id].ZeroOutflow(f, group.id);
}

} // namespace opensn
