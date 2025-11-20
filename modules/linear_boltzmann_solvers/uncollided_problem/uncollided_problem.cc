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
                                             std::vector<std::set<std::pair<int, double>>>& cell_successors)
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

  // Loop over point sources
  for (int i = 0; i < GetPointSources().size(); ++i) {
    const auto pt = GetPointSources()[i];

    // Ensure point source is inside near-source region
    const auto pt_loc = pt->GetLocation();
    if (!near_source_logvols_[i]->Inside(pt_loc))
      throw std::runtime_error("One or more point sources is outside "
                               "its near-source region.");

    // Populate uncollided cell relationships
    std::vector<std::set<std::pair<int, double>>> cell_successors(num_loc_cells);
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
      throw std::logic_error("UncollidedProblem: Cyclic dependencies found in the local cell graph.\n"
                             "Cycles need to be allowed by the calling application.");
    }

    // Separate SPLS into near-source and bulk region
    near_spls_.clear(); bulk_spls_.clear();

    for (int c : spls_) {
      const auto& cell = grid_->local_cells[c];
      if (near_source_logvols_[i]->Inside(cell.centroid)) 
        near_spls_.push_back(c);
      else 
        bulk_spls_.push_back(c);
    }
    
    // Calculate uncollided flux
    RaytraceNearSourceRegion(pt_loc, pt->GetStrength());
    // SweepBulkRegion();
  }
}


void 
UncollidedProblem::RaytraceNearSourceRegion(const Vector3& point_source,
                                            const std::vector<double>& strength) 
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::RaytraceNearSourceRegion");
  log.Log() << "\nRay tracing near-source region.\n";

  const auto& sdm = *discretization_;

  // Create raytracer
  RayTracer ray_tracer(grid_);

  // Loop over near-source region cells
  for (int c : near_spls_) 
  {
    const Cell& cell = grid_->local_cells[c];

    // Cell mapping
    auto coord_sys = grid_->GetCoordinateSystem();
    auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();



    Vector3 centroid = cell.centroid;
    Vector3 omega = ComputeOmega(centroid, point_source);

    RayTracerOutputInformation oi;
    oi = ray_tracer.TraceRay(cell, centroid, omega);

    std::cout << c << ":   " 
              << oi.destination_face_neighbor << ",   "
              << oi.pos_f.PrintStr()          << ",   "
              << oi.distance_to_surface       << std::endl;


    // // Mass matrix times least-squares flux vector
    // Vector<double> MPhi(cell_num_nodes, 0.);
    // for (unsigned int i = 0; i < cell_num_nodes; ++i)
    // {
    //   for (const auto& qp : fe_vol_data.GetQuadraturePointIndices()) 
    //   {
    //     // Raytrace to point
    //     Vector3 qp_xyz = fe_vol_data.QPointXYZ(qp);
    //     double phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, point_source, strength);

    //     // Compute least squares flux
    //     MPhi(i) += (*swf)(fe_vol_data.QPointXYZ(qp))
    //              * fe_vol_data.ShapeValue(i, qp)
    //              * phi_qp * fe_vol_data.JxW(qp);
    //   }
    // }
  }
}


double
UncollidedProblem::RaytraceLine(const RayTracer& ray_tracer,
                                const Cell& cell,
                                const Vector3& qp_xyz,
                                const Vector3& point_source,
                                const std::vector<double>& strength)
{
  // Uncollided flux analytical value
  auto phi_ex = [this](double q0, double d, double mfp) {
    if (grid_->GetDimension() == 2)
      return q0 / (2.*M_PI * d) * std::exp(-mfp);
    return q0 / (4.*M_PI * d*d) * std::exp(-mfp);
  };

  // Direction vector
  Vector3 omega = ComputeOmega(qp_xyz, point_source);
  if (omega.Norm() == 0.) 
    throw std::runtime_error("Point source lies at cell quadrature point.");
                
  
  

  return 0.;
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
                                              const Vector3& point_source)
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
  DenseMatrix<Vector3> IntV_shapeI_omega_gradshapeJ(cell_num_nodes, cell_num_nodes);
  std::vector<DenseMatrix<double>> IntS_omega_shapeI_shapeJ(cell_num_faces);

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


DenseMatrix<double> 
UncollidedProblem::ComputeMassMatrix(const Cell& cell)
{

}

} // namespace opensn