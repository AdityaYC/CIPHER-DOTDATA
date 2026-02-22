"""Builds and updates the 3D point cloud from the world graph.

Uses world_graph.to_3d_pointcloud() for node positions and colors.
When depth maps are stored per node, this module can project depth into 3D
and merge with RGB from frames; for now delegates to world graph.
"""

from typing import Any, List, Tuple

from world_graph import WorldGraph


def build_pointcloud(world_graph: WorldGraph) -> List[Tuple[float, float, float, float, float, float]]:
    """Return list of (x, y, z, r, g, b) from the world graph. RGB in 0..1."""
    if world_graph is None:
        return []
    return world_graph.to_3d_pointcloud()
