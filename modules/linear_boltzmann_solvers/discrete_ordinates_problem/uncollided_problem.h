// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <memory>

namespace opensn
{

/**
 * Base class for Uncollided Flux calculation.
 */
class UncollidedProblem : public LBSProblem
{

public:
  /// Static registration based constructor.
  explicit UncollidedProblem(const InputParameters& params);
  ~UncollidedProblem() override;

  void Initialize() override;

  void Execute() {};

  void ReorientAdjointSolution() override {};
  std::pair<size_t, size_t> GetNumPhiIterativeUnknowns() override {};


protected:
  explicit UncollidedProblem(const std::string& name,
                              std::shared_ptr<MeshContinuum> grid_ptr);

  void InitializeBoundaries() override;

  /// Initializes Within-GroupSet solvers.
  void InitializeWGSSolvers() override {};

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<UncollidedProblem> Create(const ParameterBlock& params);
};

} // namespace opensn
