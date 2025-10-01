// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <memory>

namespace opensn
{

/**
 * Base class for Discrete Ordinates solvers. This class mostly establishes utilities related to
 * sweeping. From here we can derive a steady-state, transient, adjoint, and k-eigenvalue solver.
 */
class PointSourceProblem : public LBSProblem
{

public:
  /// Static registration based constructor.
  explicit PointSourceProblem(const InputParameters& params);
  ~PointSourceProblem() override;

  void Initialize() override;

  void ReorientAdjointSolution() override {};
  std::pair<size_t, size_t> GetNumPhiIterativeUnknowns() override {};

  /// Zeroes all the outflow data-structures required to compute balance.
  void ZeroOutflowBalanceVars(LBSGroupset& groupset);

protected:
  explicit PointSourceProblem(const std::string& name,
                              std::shared_ptr<MeshContinuum> grid_ptr);

  void InitializeBoundaries() override;

  /// Initializes Within-GroupSet solvers.
  void InitializeWGSSolvers() override {};

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<PointSourceProblem> Create(const ParameterBlock& params);
};

} // namespace opensn
