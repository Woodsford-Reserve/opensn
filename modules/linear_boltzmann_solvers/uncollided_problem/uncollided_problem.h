// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "framework/mesh/logical_volume/logical_volume.h"

namespace opensn
{

class UncollidedProblem : public LBSProblem
{
public:
  explicit UncollidedProblem(const InputParameters& params);

  ~UncollidedProblem() override;

  void Initialize() override;

  void ZeroSolutions() override {}

protected:
  explicit UncollidedProblem(const std::string& name,
                             std::shared_ptr<MeshContinuum> grid_ptr);

  void InitializeNearSourceRegions(const InputParameters& params);

  std::vector<std::shared_ptr<LogicalVolume>> near_source_logvols_;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<UncollidedProblem> Create(const ParameterBlock& params);
};

} // namespace opensn