#include "modules/linear_boltzmann_solvers/uncollided_problem/uncollided_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/point_source/point_source.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_weight_function.h"
#include "framework/math/quadratures/quadrature_order.h"
#include "framework/logging/log.h"
#include "framework/logging/log_exceptions.h"
#include "framework/utils/timer.h"
#include "framework/utils/utils.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <iomanip>
#include <utility>
#include <unordered_map>

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

  return params;
}

std::shared_ptr<UncollidedProblem>
UncollidedProblem::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<UncollidedProblem>("lbs::UncollidedProblem", params);
}


UncollidedProblem::UncollidedProblem(const InputParameters& params)
  : LBSProblem(params)
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
          cell_successors[c].insert(std::make_pair(face.GetNeighborLocalID(grid_.get()), weight));
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

  size_t num_loc_cells = grid_->local_cells.size();

  size_t num_loc_nodes = discretization_->GetNumLocalNodes();
  size_t num_loc_unknowns = num_loc_cells * num_groups_;

  // Loop over point sources
  for (size_t i = 0; i < GetPointSources().size(); ++i) {
    const auto pt = GetPointSources()[i];

    // Ensure point source is inside near-source region
    const auto pt_loc = pt->GetLocation();
    if (!near_source_logvols_[i]->Inside(pt_loc))
      throw std::runtime_error("One or more point sources is outside "
                               "its near-source region.");

    // Initialize uncollided flux
    destination_phi_.assign(num_loc_unknowns, 0.);

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
      if (near_source_logvols_[i]->Inside(cell.centroid)) 
        near_spls_.push_back(c);
      else 
        bulk_spls_.push_back(c);
    }
    
    // Calculate uncollided flux
    RaytraceNearSourceRegion(pt.get());
    // SweepBulkRegion();
  }
}


void 
UncollidedProblem::RaytraceNearSourceRegion(const PointSource* point_source) 
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::RaytraceNearSourceRegion");
  log.Log() << "\nRay tracing near-source region.\n";

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

  // Loop over near-source region cells
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


    // Face orientations
    constexpr auto FOPARALLEL = FaceOrientation::PARALLEL;
    constexpr auto FOINCOMING = FaceOrientation::INCOMING;
    constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

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
        const Vector3& normal = cell.faces[f].normal;
        const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);

        double leakage = 0.;
        for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
        {
          // Raytrace to point
          Vector3 qp_xyz = fe_srf_data.QPointXYZ(qp);
          Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

          std::vector<double> phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, 
                                                    pt_loc, strength);

          // Compute leakage
          double integrand = (*swf)(fe_srf_data.QPointXYZ(qp))
                           * omega.Dot(normal) 
                           * fe_srf_data.JxW(qp);
          
          for (size_t g = 0; g < num_groups_; ++g)
            face_leakage[g] += phi_qp[g] * integrand;
        }
      }

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
    for (unsigned int i = 0; i < cell_num_nodes; ++i)
    {
      for (const auto& qp : fe_vol_data.GetQuadraturePointIndices()) 
      {
        // Raytrace to point
        Vector3 qp_xyz = fe_vol_data.QPointXYZ(qp);
        std::vector<double> phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, 
                                                  point_source->GetLocation(), 
                                                  point_source->GetStrength());

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
    M_ = unit_cell_matrices_[c].intV_shapeI_shapeJ;
    for (size_t g = 0; g < num_groups_; ++g) 
      GaussElimination(M_, Phi_[g], static_cast<int>(cell_num_nodes));

    
    // Transport view
    const auto& transport_view = cell_transport_views_[c];
    const auto& xs = transport_view.GetXS();

    // Enforce conservation
    for (size_t g = 0; g < num_groups_; ++g)
    {
      double source = 0., sink = 0.;
      double total_xs = xs.GetSigmaTotal()[g];

      // Point source in cell
      if (grid_->CheckPointInsideCell(cell, pt_loc))
      {
        for (const auto& subscriber : point_source->GetSubscribers())
        {
          if (subscriber.cell_local_id == c)
            source += strength[g] * subscriber.volume_weight;
        }
      }

      // Removal rate in cell
      double phi_avg = 0.;
      for (double phi_val : Phi_[g]) phi_avg += phi_val;
      phi_avg /= cell_num_nodes;

      sink += total_xs * phi_avg * cell.volume;

      // Leakage through faces
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        if (cell_face_orientations_[c][f] == FOINCOMING)
          source += leakages[c][f][g];

        else if (cell_face_orientations_[c][f] == FOOUTGOING)
          sink += leakages[c][f][g];
      }


      // Scaling factor
      double alpha = source / sink;

      for (size_t i = 0; i < cell_num_nodes; ++i) 
        Phi_[g](i) *= alpha;

      for (size_t f = 0; f < cell_num_faces; ++f)
        if (cell_face_orientations_[c][f] == FOOUTGOING)
          leakages[c][f][g] *= alpha;
    }

    // Update flux
    for (size_t i = 0; i < cell_num_nodes; ++i) 
    {
      const auto ir = transport_view.MapDOF(i, 0, 0);
      for (size_t g = 0; g < num_groups_; ++g) destination_phi_[ir + g] = Phi_[g](i);
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
  auto phi_ex = [this](double q0, double d, double mfp) {
    if (grid_->GetDimension() == 2)
      return q0 / (2.*M_PI * d) * std::exp(-mfp);
    return q0 / (4.*M_PI * d*d) * std::exp(-mfp);
  };

  // Uncollided flux values at quadrature point
  std::vector<double> phi (num_groups_, 0.);

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
  for (size_t g = 0; g < num_groups_; ++g) 
  {
    double mfp = 0.;

    for (const auto& segment : segment_lengths) 
    {
      size_t cell_id = segment.first;
      double length = segment.second;

      const auto& transport_view = cell_transport_views_[cell_id];
      const auto& xs = transport_view.GetXS();

      double total_xs = xs.GetSigmaTotal()[g];
      mfp += total_xs * length;
    }

    phi[g] = phi_ex(strength[g], total_length, mfp);
  }

  return phi;
}


void 
UncollidedProblem::SweepBulkRegion()
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::SweepBulkRegion");
  log.Log() << "Sweeping bulk region.\n";

  const auto& sdm = *discretization_;

  // Loop over bulk region cells
  for (int c : bulk_spls_) 
  {
    const Cell& cell = grid_->local_cells[c];

    // Cell mapping
    auto coord_sys = grid_->GetCoordinateSystem();
    auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();


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
  DenseMatrix<double> intV_shapeJ_omega_gradshapeI(cell_num_nodes, cell_num_nodes);
  std::vector<DenseMatrix<double>> IntS_omega_n_shapeI_shapeJ(cell_num_faces);

  // Volume integrals
  for (unsigned int i = 0; i < cell_num_nodes; ++i)
  {
    for (unsigned int j = 0; j < cell_num_nodes; ++j)
    {
      for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
      {
        // IntV_shapeI_omega_gradshapeJ(i, j) +=
        //   (*swf)(fe_vol_data.QPointXYZ(qp)) * fe_vol_data.ShapeValue(i, qp) *

      }
    }
  }
}

} // namespace opensn