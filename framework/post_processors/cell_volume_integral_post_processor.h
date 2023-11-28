#pragma once

#include "framework/post_processors/post_processor.h"
#include "framework/physics/field_function/grid_based_field_function_interface.h"
#include "framework/mesh/logical_volume/logical_volume_interface.h"

namespace chi_mesh
{
class LogicalVolume;
}
namespace chi_physics
{
class FieldFunctionGridBased;
}

namespace chi
{

class CellVolumeIntegralPostProcessor : public PostProcessor,
                                        public chi_physics::GridBasedFieldFunctionInterface,
                                        public chi_mesh::LogicalVolumeInterface
{
public:
  static InputParameters GetInputParameters();
  explicit CellVolumeIntegralPostProcessor(const InputParameters& params);

  void Execute(const Event& event_context) override;

protected:
  void Initialize();

  const bool compute_volume_average_;
  bool initialized_ = false;
  std::vector<uint64_t> cell_local_ids_;
};

} // namespace chi