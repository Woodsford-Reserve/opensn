// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/logical_volume/logical_volume.h"
#include "framework/mesh/raytrace/raytracer.h"
#include "framework/mesh/cell/cell.h"
#include "framework/data_types/dense_matrix.h"
#include "framework/data_types/vector.h"
#include "framework/data_types/vector3.h"

namespace opensn
{

struct UncollidedMatrices
{
  DenseMatrix<double> intV_shapeI_omega_gradshapeJ;
  std::vector<DenseMatrix<double>> intS_omega_shapeI_shapeJ;
};


class UncollidedProblem : public LBSProblem
{
public:
  explicit UncollidedProblem(const InputParameters& params);

  ~UncollidedProblem() override;

  void ZeroSolutions() override {}

protected:
  explicit UncollidedProblem(const std::string& name,
                             std::shared_ptr<MeshContinuum> grid_ptr);

  void PrintSimHeader() override;

  void InitializeSpatialDiscretization() override;

  static Vector3 ComputeOmega(const Vector3& point_source,
                              const Vector3& qpoint)
  {
    double norm = (qpoint - point_source).Norm();
    return norm == 0. ? Vector3(0., 0., 0.) 
                      : (qpoint - point_source).Normalized();
  }

  /**
   * Populates cell relationships and face orientations for point source calculation.
   *
   * \param point_source The point source position vector.
   * \param cell_successors Cell successors.
   */
  void PopulateCellRelationships(const Vector3& point_source,
                                 std::vector<std::set<std::pair<int, double>>>& cell_successors);

  void InitializeNearSourceRegions(const InputParameters& params);

  void RaytraceNearSourceRegion(const Vector3& point_source,
                                const std::vector<double>& strength);

  double RaytraceLine(const RayTracer& ray_tracer,
                      const Cell& cell,
                      const Vector3& qp_xyz,
                      const Vector3& point_source,
                      const std::vector<double>& strength);

  void SweepBulkRegion();

  UncollidedMatrices ComputeUncollidedIntegrals(const Cell& cell,
                                                const Vector3& point_source);

  void Execute();

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