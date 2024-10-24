// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/spds/cbc_spds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/graphs/directed_graph.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "caliper/cali.h"

namespace opensn
{

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const MeshContinuum& grid,
                   bool cycle_allowance_flag,
                   bool verbose)
  : SPDS(omega, grid, verbose)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  log.Log0Verbose1() << program_timer.GetTimeString()
                     << " Building sweep ordering for Omega = " << omega.PrintS();

  size_t num_loc_cells = grid.local_cells.size();

  // Populate Cell Relationships
  log.Log0Verbose1() << "Populating cell relationships";
  std::vector<std::set<std::pair<int, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  location_dependencies_.reserve(location_dependencies.size());

  for (auto v : location_successors)
    location_successors_.push_back(v);

  for (auto v : location_dependencies)
    location_dependencies_.push_back(v);

  // Build graph
  DirectedGraph local_DG;

  // Add vertex for each local cell
  for (int c = 0; c < num_loc_cells; ++c)
    local_DG.AddVertex();

  // Create graph edges
  for (int c = 0; c < num_loc_cells; ++c)
    for (auto& successor : cell_successors[c])
      local_DG.AddEdge(c, successor.first, successor.second);

  // Remove local cycles if allowed
  if (verbose_)
    PrintedGhostedGraph();

  if (cycle_allowance_flag)
  {
    log.Log0Verbose1() << program_timer.GetTimeString() << " Removing inter-cell cycles.";

    auto edges_to_remove = local_DG.RemoveCyclicDependencies();

    for (auto& edge_to_remove : edges_to_remove)
    {
      local_cyclic_dependencies_.emplace_back(edge_to_remove.first, edge_to_remove.second);
    }
  }

  // Generate topological sorting
  log.Log0Verbose1() << program_timer.GetTimeString()
                     << " Generating topological sorting for local sweep ordering";
  auto so_temp = local_DG.GenerateTopologicalSort();
  spls_.item_id.clear();
  for (auto v : so_temp)
    spls_.item_id.emplace_back(v);

  if (spls_.item_id.empty())
  {
    log.LogAllError() << "Topological sorting for local sweep-ordering failed. "
                      << "Cyclic dependencies detected. Cycles need to be allowed"
                      << " by calling application.";
    Exit(EXIT_FAILURE);
  }

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create Task
  //                                                        Dependency Graphs
  // All locations will gather other locations' dependencies
  // so that each location has the ability to build
  // the global task graph.

  log.Log0Verbose1() << program_timer.GetTimeString() << " Communicating sweep dependencies.";

  // auto& global_dependencies = sweep_order->global_dependencies;
  std::vector<std::vector<int>> global_dependencies;
  global_dependencies.resize(opensn::mpi_comm.size());

  CommunicateLocationDependencies(location_dependencies_, global_dependencies);

  ////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build task
  ////                                                        dependency graph
  // BuildTaskDependencyGraph(global_dependencies, cycle_allowance_flag);

  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  // For each local cell create a task
  for (const auto& cell : grid_.local_cells)
  {
    const size_t num_faces = cell.faces.size();
    unsigned int num_dependencies = 0;
    std::vector<uint64_t> succesors;

    for (size_t f = 0; f < num_faces; ++f)
      if (cell_face_orientations_[cell.local_id][f] == INCOMING)
      {
        if (cell.faces[f].has_neighbor)
          ++num_dependencies;
      }
      else if (cell_face_orientations_[cell.local_id][f] == OUTGOING)
      {
        const auto& face = cell.faces[f];
        if (face.has_neighbor and grid.IsCellLocal(face.neighbor_id))
          succesors.push_back(grid.cells[face.neighbor_id].local_id);
      }

    task_list_.push_back({num_dependencies, succesors, cell.local_id, &cell, false});
  } // for cell in SPLS

  opensn::mpi_comm.barrier();

  log.Log0Verbose1() << program_timer.GetTimeString() << " Done computing sweep ordering.\n\n";
}

const std::vector<Task>&
CBC_SPDS::TaskList() const
{
  return task_list_;
}

} // namespace opensn
