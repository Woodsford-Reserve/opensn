#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D orthogonal grid uncollided test problem
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import PointSource
    from pyopensn.solver import UncollidedProblem
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    num_procs = 1

    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    # Setup mesh
    nodes = []
    N = 20
    L = 100
    xmin = -L / 2
    dx = L / N
    for i in range(N + 1):
        nodes.append(xmin + i * dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes])
    grid = meshgen.Execute()

    # Set block IDs
    grid.SetUniformBlockID(0)

    # XS data
    num_groups = 168
    xs_graphite = MultiGroupXS()
    xs_graphite.LoadFromOpenSn("xs_graphite_pure.xs")

    # Create point source
    src_strength = [0.0 for _ in range(num_groups)]
    src_strength[0] = 1.0
    loc = [50.0, 50.0, 0.0]
    pt_src0 = PointSource(location=loc, strength=src_strength)

    # Create logical volume
    vol1 = RPPLogicalVolume(
        xmin=30.0,
        xmax=80.0,
        ymin=30.0,
        ymax=80.0,
        infz=True,
    )

    phys = UncollidedProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, num_groups-1],
            },
        ],
        xs_map=[
            {"block_ids": [0], "xs": xs_graphite},
        ],
        point_sources=[pt_src0],
    )