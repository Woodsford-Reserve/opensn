#include "modules/linear_boltzmann_solvers/uncollided_problem/uncollided_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/point_source/point_source.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_weight_function.h"
#include "framework/math/quadratures/quadrature_order.h"
#include "framework/math/quadratures/angular/legendre_poly/legendrepoly.h"
#include "framework/logging/log.h"
#include "framework/logging/log_exceptions.h"
#include "framework/utils/timer.h"
#include "framework/utils/utils.h"
#include "framework/utils/hdf_utils.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <iomanip>
#include <utility>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, UncollidedProblem);

UncollidedProblem::UncollidedProblem(const std::string& name,
                                     std::shared_ptr<MeshContinuum> grid_ptr)
  : LBSProblem(name, grid_ptr)
{
}

InputParameters
UncollidedProblem::GetInputParameters()
{
  InputParameters params = LBSProblem::GetInputParameters();

  params.SetClassName("UncollidedProblem");

  params.ChangeExistingParamToOptional("name", "UncollidedProblem");

  params.AddRequiredParameterArray("near_source",
                                   "List of near source region logical volumes.");

  params.AddOptionalParameter("scattering_order",
                              0,
                              "The scattering order of collided flux problem.");

  return params;
}

std::shared_ptr<UncollidedProblem>
UncollidedProblem::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<UncollidedProblem>("lbs::UncollidedProblem", params);
}


UncollidedProblem::UncollidedProblem(const InputParameters& params)
  : LBSProblem(params),
    scattering_order_(params.GetParamValue<size_t>("scattering_order"))
{
  Initialize();

  InitializeNearSourceRegions(params);

  Execute();
}

void
UncollidedProblem::InitializeSpatialDiscretization()
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::InitializeSpatialDiscretization");

  log.Log() << "Initializing spatial discretization.\n";
  discretization_ = PieceWiseLinearDiscontinuous::New(grid_, QuadratureOrder::FOURTH);

  ComputeUnitIntegrals();
}


void 
UncollidedProblem::InitializeNearSourceRegions(const InputParameters& params)
{
  const auto& near_source_param = params.GetParam("near_source");
  near_source_param.RequireBlockTypeIs(ParameterBlockType::ARRAY);

  for (const auto& log_vol : near_source_param)
    near_source_logvols_.push_back(log_vol.GetValue<std::shared_ptr<LogicalVolume>>());
}


UncollidedProblem::~UncollidedProblem() = default;


void 
UncollidedProblem::PrintSimHeader()
{
  if (opensn::mpi_comm.rank() == 0)
  {
    std::stringstream outstr;
    outstr << "\nInitializing " << GetName() << "\n";
    log.Log() << outstr.str() << '\n';
  }
}


void
UncollidedProblem::PopulateCellRelationships(const Vector3& point_source,
                                             std::vector<std::set<std::pair<size_t, double>>>& cell_successors)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::PopulateCellRelationships");

  constexpr double tolerance = 1.0e-16;

  constexpr auto FOPARALLEL = FaceOrientation::PARALLEL;
  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

  cell_face_orientations_.assign(grid_->local_cells.size(), {});
  for (auto& cell : grid_->local_cells)
    cell_face_orientations_[cell.local_id].assign(cell.faces.size(), FOPARALLEL);

  for (auto& cell : grid_->local_cells)
  {
    size_t f = 0;
    for (auto& face : cell.faces)
    {
      // Determine if the face is incident
      FaceOrientation orientation = FOPARALLEL;
      Vector3 omega = ComputeOmega(point_source, face.centroid);
      const double mu = omega.Dot(face.normal);

      bool owns_face = true;
      if (face.has_neighbor and cell.global_id > face.neighbor_id)
        owns_face = false;

      if (owns_face)
      {
        if (mu > tolerance)
          orientation = FOOUTGOING;
        else if (mu < -tolerance)
          orientation = FOINCOMING;

        cell_face_orientations_[cell.local_id][f] = orientation;

        if (face.has_neighbor)
        {
          const auto& adj_cell = grid_->cells[face.neighbor_id];
          const auto adj_face_idx = face.GetNeighborAdjacentFaceIndex(grid_.get());
          auto& adj_face_ori = cell_face_orientations_[adj_cell.local_id][adj_face_idx];

          switch (orientation)
          {
            case FOPARALLEL:
              adj_face_ori = FOPARALLEL;
              break;
            case FOINCOMING:
              adj_face_ori = FOOUTGOING;
              break;
            case FOOUTGOING:
              adj_face_ori = FOINCOMING;
              break;
          }
        }
      }

      ++f;
    } // for face
  }

  // Make directed connections
  for (auto& cell : grid_->local_cells)
  {
    const uint64_t c = cell.local_id;
    size_t f = 0;
    for (auto& face : cell.faces)
    {
      Vector3 omega = ComputeOmega(point_source, face.centroid);
      const double mu = omega.Dot(face.normal);
      // If outgoing determine if it is to a local cell
      if (cell_face_orientations_[cell.local_id][f] == FOOUTGOING)
      {
        // If it is a cell and not bndry
        if (face.has_neighbor)
        {
          const auto weight = 0.;
          cell_successors[c].insert(
            std::make_pair(
                face.GetNeighborLocalID(grid_.get()), 
                weight
              )
            );
        }
      }

      ++f;
    } // for face
  } // for cell
}


void 
UncollidedProblem::Execute()
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::Execute");

  // Create h5 file
  std::string fname = "uncollided.h5";
  auto file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  size_t num_loc_cells = grid_->local_cells.size();

  size_t num_loc_nodes = discretization_->GetNumLocalNodes();
  size_t num_loc_unknowns = num_loc_nodes * num_groups_;

  // Global cell IDs
  std::vector<size_t> global_ids(num_loc_cells);
  for (const auto& cell : grid_->local_cells) global_ids[cell.local_id] = cell.global_id;
  H5WriteDataset1D<size_t>(file, "cell ids", global_ids);

  // Loop over point sources
  for (size_t i = 0; i < GetPointSources().size(); ++i) 
  {
    const auto& point_source = point_sources_[i];
    const auto pt = point_source.get();

    // Ensure point source is inside near-source region
    const auto pt_loc = pt->GetLocation();
    if ( !near_source_logvols_[i]->Inside(pt_loc) )
      throw std::runtime_error("One or more point sources lies outside "
                               "its near-source region.");

    // Initialize uncollided flux and moment vector
    destination_phi_.assign(num_loc_unknowns, 0.);
    flux_moment_.assign(num_loc_unknowns, 0.);

    // Populate uncollided cell relationships
    std::vector<std::set<std::pair<size_t, double>>> cell_successors(num_loc_cells);
    PopulateCellRelationships(pt_loc, cell_successors);

    // Create local cell graph
    Graph local_cell_graph(num_loc_cells);

    for (size_t c = 0; c < num_loc_cells; ++c)
      for (const auto& successor : cell_successors[c])
        boost::add_edge(c, successor.first, successor.second, local_cell_graph);

    // Generate topological ordering
    spls_.clear();
    boost::topological_sort(local_cell_graph, std::back_inserter(spls_)); // NOLINT
    std::reverse(spls_.begin(), spls_.end());
    if (spls_.empty())
    {
      throw std::logic_error("UncollidedProblem: Cyclic dependencies found "
                             "in the local cell graph.");
    }

    // Separate SPLS into near-source and bulk region
    near_spls_.clear(); bulk_spls_.clear();

    for (size_t c : spls_) {
      const auto& cell = grid_->local_cells[c];
      if ( near_source_logvols_[i]->Inside(cell.centroid) ) near_spls_.push_back(c);
      else                                                  bulk_spls_.push_back(c);
    }
    
    // Calculate uncollided flux
    RaytraceNearSourceRegion(pt);
    if (bulk_spls_.size() != 0) SweepBulkRegion(pt_loc);

    // Update balance parameters
    UpdateBalance(pt);

    // Write uncollided flux to h5
    WriteToH5File(file, pt_loc);
  }

  // Finalize balance calculation
  FinalizeBalance(file);

  // Close h5 file
  H5Fclose(file);
}


void 
UncollidedProblem::RaytraceNearSourceRegion(const PointSource* point_source) 
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::RaytraceNearSourceRegion");
  log.Log() << "\nRay-tracing near-source region.\n";

  const auto& sdm = *discretization_;

  // Point source data
  const Vector3& pt_loc = point_source->GetLocation();
  const std::vector<double>& strength = point_source->GetStrength(); 

  // Create raytracer
  RayTracer ray_tracer(grid_);

  // Face leakages
  std::unordered_map<size_t, std::vector<std::vector<double>>> leakages;
  std::vector<std::vector<double>> cell_leakage;
  std::vector<double> face_leakage;

  // Source and sink terms for conservation
  std::vector<double> source, sink;

  // Face orientations
  constexpr auto FOPARALLEL = FaceOrientation::PARALLEL;
  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

  // Ray-trace near-source region cells
  for (size_t c : near_spls_) 
  {
    const Cell& cell = grid_->local_cells[c];

    // Cell mapping
    auto coord_sys = grid_->GetCoordinateSystem();
    auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();


    // Compute leakages
    cell_leakage.resize(cell_num_faces);
    for (size_t f = 0; f < cell_num_faces; ++f) 
    {
      const auto orientation = cell_face_orientations_[c][f];
      face_leakage.assign(num_groups_, 0.);

      // Compute leakage out of outgoing face
      if (orientation == FOOUTGOING)
      {
        // Face data
        const auto& face = cell.faces[f];

        const Vector3& normal = face.normal;
        const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);

        for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
        {
          // Raytrace to point
          Vector3 qp_xyz = fe_srf_data.QPointXYZ(qp);
          Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

          std::vector<double> 
          phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, pt_loc, strength);

          // Compute leakage
          double integrand = (*swf)(fe_srf_data.QPointXYZ(qp))
                           * omega.Dot(normal) 
                           * fe_srf_data.JxW(qp);
          
          for (size_t g = 0; g < num_groups_; ++g)
            face_leakage[g] += phi_qp[g] * integrand;
        }


        if (face.has_neighbor)
        {
          size_t neighbor_id = face.neighbor_id;

          // Near-source/bulk region interface
          if (std::find( bulk_spls_.begin(),
                         bulk_spls_.end(),
                         neighbor_id ) != bulk_spls_.end())
          {
            // Face data
            const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
            const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);

            // Neighbor data
            const Cell& neighbor = grid_->local_cells[neighbor_id];
            const auto& neighbor_mapping = sdm.GetCellMapping(neighbor);

            size_t f_ = face.GetNeighborAdjacentFaceIndex(grid_.get());

            for (size_t fi = 0; fi < num_face_nodes; ++fi)
            {
              const int i = cell_mapping.MapFaceNode(f, fi);

              int j = -1;
              for (size_t fj = 0; fj < num_face_nodes; ++fj)
              {
                j = neighbor_mapping.MapFaceNode(f_, fj);
                if (neighbor.vertex_ids[j] == cell.vertex_ids[i]) break;
              }

              // Compute rhs for bulk region sweep
              const auto jr = sdm.MapDOFLocal(neighbor, j);
              for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
              {
                // Raytrace to quadrature point
                const Vector3& qp_xyz = fe_srf_data.QPointXYZ(qp);
                Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

                std::vector<double> 
                phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, pt_loc, strength);

                // Compute rhs vector
                double integrand = (*swf)(qp_xyz)
                                 * omega.Dot( face.normal )
                                 * fe_srf_data.ShapeValue(i, qp)
                                 * fe_srf_data.JxW(qp);

                for (size_t g = 0; g < num_groups_; ++g)
                  destination_phi_[jr * num_groups_ + g] += phi_qp[g] * integrand;

              } // for qp
            } // for fi
          } // if neighbor_id in bulk_spls_
        } // if face.has_neighbor
      } // if outgoing
      

      // Retrieve leakage in from incoming face
      else if (orientation == FOINCOMING)
      {
        size_t neigh_id = cell.faces[f].neighbor_id;
        size_t neigh_face_ind = cell.faces[f].GetNeighborAdjacentFaceIndex(grid_.get());
        face_leakage = leakages[neigh_id][neigh_face_ind];
      }

      // Save leakage through face
      cell_leakage[f] = face_leakage;
    }

    // Save leakage through cell faces
    leakages.emplace(c, cell_leakage);


    // Mass matrix times least-squares flux vector
    Phi_.assign(num_groups_, Vector<double>(cell_num_nodes, 0.));
    for (const auto& qp : fe_vol_data.GetQuadraturePointIndices()) 
    {
      // Raytrace to point
      Vector3 qp_xyz = fe_vol_data.QPointXYZ(qp);

      std::vector<double> 
      phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, pt_loc, strength);

      for (unsigned int i = 0; i < cell_num_nodes; ++i)
      {
        // Integrand value at quadrature point
        double integrand = (*swf)(fe_vol_data.QPointXYZ(qp))
                         * fe_vol_data.ShapeValue(i, qp)
                         * fe_vol_data.JxW(qp);

        // Compute group-wise least-squares fluxes
        for (size_t g = 0; g < num_groups_; ++g) 
          Phi_[g](i) += phi_qp[g] * integrand;
      }
    } 

    // Invert mass matrix
    for (size_t g = 0; g < num_groups_; ++g) 
    {
      M_ = unit_cell_matrices_[c].intV_shapeI_shapeJ;
      GaussElimination(M_, Phi_[g], static_cast<int>(cell_num_nodes));
    }
    

    // Transport view
    const auto& transport_view = cell_transport_views_[c];
    const auto& xs = transport_view.GetXS();
    const auto& sigma_t = xs.GetSigmaTotal();

    const auto& fe_intgrl_values = unit_cell_matrices_[cell.local_id];
    const auto& IntV_shapeI = fe_intgrl_values.intV_shapeI;

    // Enforce conservation
    std::vector<double> source(num_groups_, 0.);
    std::vector<double> sink(num_groups_, 0.);

    // Point source rate in cell
    if (grid_->CheckPointInsideCell(cell, pt_loc))
    {
      for (const auto& subscriber : point_source->GetSubscribers())
      {
        if (subscriber.cell_local_id == c)
        {
          for (size_t g = 0; g < num_groups_; ++g)
            source[g] += strength[g] * subscriber.volume_weight;
          break;
        }
      }
    }

    // Removal rate in cell
    for (size_t g = 0; g < num_groups_; ++g)
    {
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        sink[g] += sigma_t[g] * Phi_[g](i) * IntV_shapeI(i);
      }
    }

    // Leakage through faces
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      if (cell_face_orientations_[c][f] == FOINCOMING)
        for(size_t g = 0; g < num_groups_; ++g) source[g] += leakages[c][f][g];
        
      else if (cell_face_orientations_[c][f] == FOOUTGOING)
        for(size_t g = 0; g < num_groups_; ++g) sink[g] += leakages[c][f][g];
    }

    // Rescale solution
    for (size_t g = 0; g < num_groups_; ++g)
    {
      double alpha = (sink[g] == 0.) ? 1. : (source[g] / sink[g]);

      // Rescale flux
      for (size_t i = 0; i < cell_num_nodes; ++i) Phi_[g](i) *= alpha;

      // Rescale leakage
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        const auto& face = cell.faces[f];

        if (cell_face_orientations_[c][f] == FOOUTGOING)
        {
          leakages[c][f][g] *= alpha;

          // Near-source/bulk region interface
          if (face.has_neighbor)
          {
            size_t neighbor_id = face.neighbor_id;

            if (std::find( bulk_spls_.begin(),
                           bulk_spls_.end(),
                           neighbor_id ) != bulk_spls_.end())
            {
              // Neighbor data
              const Cell& neighbor = grid_->local_cells[neighbor_id];
              const auto& neighbor_mapping = sdm.GetCellMapping(neighbor);

              // Rescale rhs vector
              size_t num_neighbor_nodes = neighbor_mapping.GetNumNodes();
              for (size_t i = 0; i < num_neighbor_nodes; ++i)
              {
                const auto ir = sdm.MapDOFLocal(neighbor, i);
                destination_phi_[ir * num_groups_ + g] *= alpha;

              } // for i
            } // if neighbor in bulk_spls_
          } // if face.has_neighbor
        } // if outgoing

        // Boundary outflow
        if (not face.has_neighbor)
          out_flow_ += leakages[c][f][g];

      } // for f
    }

    // Update flux solution
    for (size_t i = 0; i < cell_num_nodes; ++i) 
    {
      const auto ir = sdm.MapDOFLocal(cell, i);
      for (size_t g = 0; g < num_groups_; ++g) 
        destination_phi_[ir * num_groups_ + g] = Phi_[g](i);
    }

    for (size_t i = 0; i < cell_num_nodes; ++i) 
    {
      const auto ir = sdm.MapDOFLocal(cell, i);

      double phi_i = 0.;
      for (size_t g = 0; g < num_groups_; ++g) 
        phi_i += destination_phi_[ir * num_groups_ + g];
    }
  }
}


std::vector<double>
UncollidedProblem::RaytraceLine(RayTracer& ray_tracer,
                                const Cell& cell,
                                const Vector3& qp_xyz,
                                const Vector3& pt_loc,
                                const std::vector<double>& strength,
                                const double tolerance)
{
  // Uncollided flux analytical value
  auto phi_ex = [this](double q0, double d, double mfp) 
  {
    if (grid_->GetDimension() == 2)
      return q0 / (2.*M_PI * d) * std::exp(-mfp);

    return q0 / (4.*M_PI * d*d) * std::exp(-mfp);
  };

  // Uncollided flux values at quadrature point
  std::vector<double> phi(num_groups_, 0.);

  // Direction vector
  Vector3 omega = ComputeOmega(qp_xyz, pt_loc);
  if (omega.Norm() == 0.) 
    throw std::runtime_error("Point source lies at cell quadrature point.");
                
  // Starting cell ID and point
  size_t cell_id = cell.local_id;
  Vector3 line_point = qp_xyz;

  // Distance to point source
  double total_length = (pt_loc - qp_xyz).Norm();
  double remaining_distance = total_length;

  // Trace cells along path
  std::vector<std::pair<size_t, double>> segment_lengths;
  while (remaining_distance > tolerance)
  {
    // Trace cell
    RayTracerOutputInformation oi;
    oi = ray_tracer.TraceRay(grid_->local_cells[cell_id], line_point, omega);

    // Distance through cell
    double distance_in_cell = oi.distance_to_surface < remaining_distance
                            ? oi.distance_to_surface
                            : remaining_distance;

    segment_lengths.push_back(std::pair<size_t, double>(cell_id, distance_in_cell));
    remaining_distance -= distance_in_cell;

    // Trace next cell
    cell_id = oi.destination_face_neighbor;
    line_point = oi.pos_f;
  }

  // Compute group-wise uncollided flux values
  std::vector<double> mfp(num_groups_, 0.);
  for (const auto& segment : segment_lengths) 
  {
    size_t cell_id = segment.first;
    double length = segment.second;

    const auto& transport_view = cell_transport_views_[cell_id];
    const auto& xs = transport_view.GetXS();
    const auto& sigma_t = xs.GetSigmaTotal();

    for (size_t g = 0; g < num_groups_; ++g)
      mfp[g] += sigma_t[g] * length;
  }

  for (size_t g = 0; g < num_groups_; ++g) 
    phi[g] = phi_ex(strength[g], total_length, mfp[g]);

  return phi;
}


void 
UncollidedProblem::SweepBulkRegion(const Vector3& pt_loc)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::SweepBulkRegion");
  log.Log() << "Sweeping bulk region.\n";

  const auto& sdm = *discretization_;

  DenseMatrix<double> Amat(max_cell_dof_count_, max_cell_dof_count_);
  DenseMatrix<double> Atemp(max_cell_dof_count_, max_cell_dof_count_);
  std::vector<double> source(max_cell_dof_count_);

  UncollidedMatrices matrices;

  // Sweep bulk region cells
  for (int c : bulk_spls_) 
  {
    const Cell& cell = grid_->local_cells[c];

    // Cell data
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();

    const auto& transport_view = cell_transport_views_[c];
    const auto& xs = transport_view.GetXS();
    const auto& sigma_t = xs.GetSigmaTotal();

    // Compute matrices
    matrices = ComputeUncollidedIntegrals(cell, pt_loc);

    // Zero right-hand side vectors
    for (size_t g = 0; g < num_groups_; ++g)
      for (size_t i = 0; i < cell_num_nodes; ++i)
        Phi_[g](i) = 0.;

    // Gradient matrix
    G_ = matrices.intV_shapeJ_omega_gradshapeI;

    for (size_t i = 0; i < cell_num_nodes; ++i)
      for (size_t j = 0; j < cell_num_nodes; ++j)
        Amat(i, j) = G_(i, j);

    // Surface matrices
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      M_surf_ = matrices.intS_omega_n_shapeI_shapeJ[f];

      // Incoming faces (source terms)
      if (cell_face_orientations_[c][f] == FaceOrientation::INCOMING)
      {
        size_t neighbor_id = cell.faces[f].neighbor_id;

        // Near-source/bulk region interface
        if (std::find( near_spls_.begin(),
                       near_spls_.end(),
                       neighbor_id) != near_spls_.end())
        {
          for (size_t i = 0; i < cell_num_nodes; ++i)
          {
            for (size_t g = 0; g < num_groups_; ++g)
            {
              const auto ir = sdm.MapDOFLocal(cell, i);
              Phi_[g](i) += destination_phi_[ir * num_groups_ + g];
            }
          }
        }
          
        // Bulk region cell neighbor
        else
        {
          size_t f_ = cell.faces[f].GetNeighborAdjacentFaceIndex(grid_.get());

          const Cell& neighbor = grid_->local_cells[neighbor_id];
          const auto& neighbor_mapping = sdm.GetCellMapping(neighbor);

          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            const int i = cell_mapping.MapFaceNode(f, fi);

            for (size_t fj = 0; fj < num_face_nodes; ++fj)
            {
              const int j = cell_mapping.MapFaceNode(f, fj);            

              int k = -1;
              for (size_t fk = 0; fk < num_face_nodes; ++fk)
              {
                k = neighbor_mapping.MapFaceNode(f_, fk);
                if (neighbor.vertex_ids[k] == cell.vertex_ids[j]) break;
              }

              const auto jr = sdm.MapDOFLocal(neighbor, k);
              for (size_t g = 0; g < num_groups_; ++g) 
              {
                double phi_j = destination_phi_[jr * num_groups_ + g];
                Phi_[g](i) -= M_surf_(i, j) * phi_j;
              }
            }
          }
        }
      }

      // Outgoing faces (coefficient matrix)
      if (cell_face_orientations_[c][f] == FaceOrientation::OUTGOING)
      {
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const int i = cell_mapping.MapFaceNode(f, fi);

          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            const int j = cell_mapping.MapFaceNode(f, fj);

            Amat(i, j) += M_surf_(i, j);
          }
        }
      }
    }

    // Construct and solve linear system
    M_ = unit_cell_matrices_[c].intV_shapeI_shapeJ;

    for (size_t g = 0; g < num_groups_; ++g)
    {
      for (size_t i = 0; i < cell_num_nodes; ++i)
        for (size_t j = 0; j < cell_num_nodes; ++j)
          Atemp(i, j) = Amat(i, j) + sigma_t[g] * M_(i, j);

      // Solve system
      GaussElimination(Atemp, Phi_[g], static_cast<int>(cell_num_nodes));
    }

    // Update flux solution
    for (size_t i = 0; i < cell_num_nodes; ++i)
    {
      const auto ir = sdm.MapDOFLocal(cell, i);
      for (size_t g = 0; g < num_groups_; ++g) 
        destination_phi_[ir * num_groups_ + g] = Phi_[g](i);
    }
  }
}


UncollidedMatrices 
UncollidedProblem::ComputeUncollidedIntegrals(const Cell& cell,
                                              const Vector3& pt_loc)
{
  const auto& sdm = *discretization_;

  // Cell mapping
  auto coord_sys = grid_->GetCoordinateSystem();
  auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
  const auto& cell_mapping = sdm.GetCellMapping(cell);
  const size_t cell_num_faces = cell.faces.size();
  const size_t cell_num_nodes = cell_mapping.GetNumNodes();
  const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

  // Matrices
  DenseMatrix<double> IntV_shapeJ_omega_gradshapeI(cell_num_nodes, cell_num_nodes, 0.);
  std::vector<DenseMatrix<double>> IntS_omega_n_shapeI_shapeJ(cell_num_faces);

  // Gradient Matrix
  for (unsigned int i = 0; i < cell_num_nodes; ++i)
  {
    for (unsigned int j = 0; j < cell_num_nodes; ++j)
    {
      for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
      {
        const Vector3& qp_xyz = fe_vol_data.QPointXYZ(qp);
        Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

        IntV_shapeJ_omega_gradshapeI(i, j) -= 
          (*swf)(qp_xyz) * 
          fe_vol_data.ShapeValue(j, qp) * 
          omega.Dot( fe_vol_data.ShapeGrad(i, qp) ) * 
          fe_vol_data.JxW(qp);

      } // for qp
    } // for j
  } // for i

  // Surface matrices
  for (size_t f = 0; f < cell_num_faces; ++f)
  {
    const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
    IntS_omega_n_shapeI_shapeJ[f] = DenseMatrix<double>(cell_num_nodes, cell_num_nodes, 0.0);

    for (unsigned int i = 0; i < cell_num_nodes; ++i)
    {
      for (unsigned int j = 0; j < cell_num_nodes; ++j)
      {
        for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
        {
          const Vector3& qp_xyz = fe_srf_data.QPointXYZ(qp);
          Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

          IntS_omega_n_shapeI_shapeJ[f](i,j) +=
            (*swf)(qp_xyz) *
            omega.Dot( cell.faces[f].normal ) *
            fe_srf_data.ShapeValue(i, qp) *
            fe_srf_data.ShapeValue(j, qp) * 
            fe_srf_data.JxW(qp);

        } // for qp
      } // for j
    } // for i
  } // for f

  return UncollidedMatrices{ IntV_shapeJ_omega_gradshapeI,
                             IntS_omega_n_shapeI_shapeJ };
}


void 
UncollidedProblem::UpdateBalance(const PointSource* point_source)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::UpdateBalance");

  const auto& sdm = *discretization_;

  // Point source data
  const Vector3& pt_loc = point_source->GetLocation();
  const std::vector<double>& strength = point_source->GetStrength();

  // Source rate
  for (size_t g = 0; g < num_groups_; ++g) 
    production_ += strength[g];

  for (const auto& cell : grid_->local_cells) 
  {
    const uint64_t c = cell.local_id;

    // Cell mapping
    auto coord_sys = grid_->GetCoordinateSystem();
    auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

    // Transport view
    const auto& transport_view = cell_transport_views_[c];
    const auto& xs = transport_view.GetXS();
    const auto& sigma_t = xs.GetSigmaTotal();

    const auto& fe_intgrl_values = unit_cell_matrices_[c];
    const auto& IntV_shapeI = fe_intgrl_values.intV_shapeI;

    // Removal rate in cell
    for (size_t g = 0; g < num_groups_; ++g)
    {
      double phi_g = 0.;
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = sdm.MapDOFLocal(cell, i);
        double phi_ig = destination_phi_[ir * num_groups_ + g];

        removal_ += sigma_t[g] * phi_ig * IntV_shapeI(i);
      }
    }

    // Compute outflow
    for (size_t f = 0; f < cell_num_faces; ++f) 
    {
      const auto& face = cell.faces[f];

      // Compute leakage out of outgoing face
      if (not face.has_neighbor)
      { 
        if (std::find( bulk_spls_.begin(),
                       bulk_spls_.end(),
                       cell.local_id ) != bulk_spls_.end())
        {
          // Face data
          const Vector3& normal = cell.faces[f].normal;
          const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
          
          for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
          {
            // Raytrace to point
            Vector3 qp_xyz = fe_srf_data.QPointXYZ(qp);
            Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

            // Compute outflow
            double integrand = (*swf)(fe_srf_data.QPointXYZ(qp))
                             * omega.Dot(normal) 
                             * fe_srf_data.JxW(qp);
            
            for (size_t g = 0; g < num_groups_; ++g)
            {
              // Flux at quadrature point
              for (size_t i = 0; i < cell_num_nodes; ++i)
              {
                const auto ir = sdm.MapDOFLocal(cell, i);
                double phi_ig = destination_phi_[ir * num_groups_ + g];

                out_flow_ += phi_ig * integrand 
                           * fe_srf_data.ShapeValue(i, qp); 
              }
            } // for g
          } // for qp
        } // if cell id in bulk_spls_
      } // if not has_neighbor
    } // for f
  } // for cell
}


void 
UncollidedProblem::WriteToH5File(hid_t file,
                                 const Vector3& pt_loc)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::WriteToH5File");

  // Write uncollided flux data to h5
  if (H5Lexists(file, "0,0", H5P_DEFAULT) > 0)
    OverwriteH5Data(file, "0,0", destination_phi_);

  else H5WriteDataset1D<double>(file, "0,0", destination_phi_);

  // Loop over moments
  for (int ell = 1; ell <= scattering_order_; ++ell)
  {
    for (int m = -ell; m <= ell; ++m)
    {
      std::string name = std::to_string(ell) 
                       + ","
                       + std::to_string(m);

      // Compute l,m harmonic moment of uncollided flux
      ComputeMoment(ell, m, pt_loc);

      // Write flux moment to h5
      if (H5Lexists(file, name.c_str(), H5P_DEFAULT) > 0)
        OverwriteH5Data(file, name, flux_moment_);

      else H5WriteDataset1D<double>(file, name, flux_moment_);
    }
  }
}


void
UncollidedProblem::OverwriteH5Data(hid_t file,
                                   const std::string name,
                                   const std::vector<double>& data)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::OverwriteH5Data");

  // Read data from H5 file
  std::vector<double> data_tmp;
  H5ReadDataset1D<double>(file, name, data_tmp);

  // Add data values
  for (size_t i = 0; i < data_tmp.size(); ++i) data_tmp[i] += data[i];

  // Overwrite data in H5 file
  H5Ldelete(file, name.c_str(), H5P_DEFAULT);
  H5WriteDataset1D<double>(file, name, data_tmp);
}


void 
UncollidedProblem::ComputeMoment(unsigned int ell, 
                                 int m,
                                 const Vector3& pt_loc)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::ComputeMoment");

  const auto& sdm = *discretization_;

  for (const auto& cell : grid_->local_cells) 
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();

    for (size_t i = 0; i < cell_num_nodes; ++i)
    {
      // DOF vertex position
      size_t vertex_id = cell.vertex_ids[i];
      const auto& vertex = grid_->vertices[vertex_id];

      // Vertex direction vector
      Vector3 omega = ComputeOmega(pt_loc, vertex);

      double theta = std::acos(omega.z);

      double sgn = (omega.y > 0.) ? 1. : -1.; 
      double varphi = sgn * std::acos(omega.x 
                    / std::sqrt(omega.x*omega.x 
                              + omega.y*omega.y));
                              
      const auto ir = sdm.MapDOFLocal(cell, i);

      // Compute l,m harmonic moment of uncollided flux
      for (size_t g = 0; g < num_groups_; ++g)
      {
        double phi_ig = destination_phi_[ir * num_groups_ + g];
        double phi_lm = phi_ig * Ylm(ell, m, varphi, theta);

        flux_moment_[ir * num_groups_ + g] = phi_lm;

      } // for g
    } // for i
  } // for cell
}


void 
UncollidedProblem::FinalizeBalance(hid_t file)
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::FinalizeBalance");

  // Finalize balance calulation
  double balance = production_ - (removal_ + out_flow_);
  const double conservation_error = (production_ == 0.0) ? 0.0 : (balance / production_);

  log.Log() << "\nBalance table:\n"
            << std::setprecision(6) << std::scientific
            << " Removal rate                = " << removal_ << "\n"
            << " Production rate             = " << production_ << "\n"
            << " Out-flow rate               = " << out_flow_ << "\n"
            << " Balance (Production - Loss) = " << balance << "\n"
            << " Conservation error          = " << conservation_error << "\n\n";

  // Write balance parameters to h5
  H5CreateAttribute<double>(file, "production", production_);
  H5CreateAttribute<double>(file, "removal", removal_);
  H5CreateAttribute<double>(file, "out-flow", out_flow_);
}

} // namespace opensn