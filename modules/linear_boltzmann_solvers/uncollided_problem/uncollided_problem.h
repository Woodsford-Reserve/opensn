// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
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

  /**
   * Populates cell relationships and face orientations for point source calculation.
   *
   * \param point_source The point source position vector.
   * \param cell_successors Cell successors.
   */
  void PopulateCellRelationships(const Vector3& point_source,
                                 std::vector<std::set<std::pair<int, double>>>& cell_successors);

  void InitializeNearSourceRegions(const InputParameters& params);

  void RaytraceNearSourceRegion();

  void RaytraceLine();

  void SweepBulkRegion();

  /// Near source region logical volumes.
  std::vector<std::shared_ptr<LogicalVolume>> near_source_logvols_;
  /// Cell face orientations for the cells in the local cell graph.
  std::vector<std::vector<FaceOrientation>> cell_face_orientations_;
  /// Uncollided sweep-plane local subgrid.
  std::vector<int> spls_;
  /// Near source uncollided sweep-plane local subgrid.
  std::vector<int> near_spls_;
  /// Bulk region uncollided sweep-plane local subgrid.
  std::vector<int> bulk_spls_;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<UncollidedProblem> Create(const ParameterBlock& params);
};

} // namespace opensn