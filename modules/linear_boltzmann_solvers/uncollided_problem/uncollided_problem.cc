#include "modules/linear_boltzmann_solvers/uncollided_problem/uncollided_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
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
  std::cout << "I'm an uncollided problem!" << std::endl;

  // number of mesh cells
  size_t num_loc_cells = grid_->local_cells.size();

  // loop over point sources
  for (auto pt : GetPointSources()) {
    const auto pt_loc = pt->GetLocation();

    // instantiate SPDS object
    const auto sweep_order = std::make_shared<SPDS>(pt_loc, grid_);

    // Populate Uncollided Relationships
    std::vector<std::set<std::pair<int, double>>> cell_successors(num_loc_cells);
    std::set<int> location_successors;
    std::set<int> location_dependencies;

    sweep_order->PopulateUncollidedRelationships(pt_loc, 
                                                 location_dependencies, 
                                                 location_successors, 
                                                 cell_successors);
  }
}



UncollidedProblem::~UncollidedProblem() = default;

void
UncollidedProblem::Initialize()
{
  CALI_CXX_MARK_SCOPE("UncollidedProblem::Initialize");

  LBSProblem::Initialize();
}

} // namespace opensn