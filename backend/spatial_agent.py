"""
Spatial reasoning agent: navigates the world graph step-by-step to answer spatial questions.
Uses CLIP (clip_navigator) to decide which neighbor to move to; bounded to actual graph nodes.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import base64
import io

from clip_navigator import (
    load_clip,
    embed_text,
    embed_frame,
    describe_node,
    find_best_node,
    find_top_k_nodes,
    are_similar,
)


@dataclass
class AgentResult:
    found: bool
    best_node_id: Optional[str]
    confidence: float
    path_taken: List[str]
    steps_taken: int
    description: str
    node_type: str = ""
    coordinates: Optional[Tuple[float, float, float]] = None


def _get_node_frame(node_id: str, world_graph: Any):
    """Return current node frame as PIL Image or None."""
    nodes = getattr(world_graph, "nodes", None) or {}
    node = nodes.get(node_id)
    if not node or not getattr(node, "image_b64", None):
        return None
    try:
        import base64
        from PIL import Image
        raw = base64.b64decode(node.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _neighbor_ids(node_id: str, world_graph: Any) -> List[str]:
    """All neighbors: forward, back, left, right, next, prev (unique)."""
    out = []
    for direction in ("forward", "back", "left", "right", "next", "prev"):
        n = getattr(world_graph, "get_neighbor_direction", None) or getattr(world_graph, "get_neighbor_by_order", None)
        if direction in ("next", "prev"):
            neighbor = getattr(world_graph, "get_neighbor_by_order", lambda i, d: None)(node_id, direction)
        else:
            neighbor = getattr(world_graph, "get_neighbor_direction", lambda i, d: None)(node_id, direction)
        if neighbor and neighbor not in out:
            out.append(neighbor)
    return out


def run_exploration(
    goal_text: str,
    world_graph: Any,
    max_steps: int = 20,
    start_node_id: Optional[str] = None,
    goal_similarity_threshold: float = 0.75,
) -> AgentResult:
    """
    Explore the graph step-by-step until goal is found or max_steps.
    Returns AgentResult with path_taken, best_node_id, confidence.
    """
    nodes = getattr(world_graph, "nodes", None) or {}
    if len(nodes) < 5:
        return AgentResult(
            found=False,
            best_node_id=None,
            confidence=0.0,
            path_taken=[],
            steps_taken=0,
            description="NOT ENOUGH MAP DATA — keep exploring.",
        )
    ordered = sorted(nodes.keys())
    current_id = start_node_id if start_node_id and start_node_id in nodes else ordered[0]
    path: List[str] = [current_id]
    goal_emb = embed_text(goal_text)
    best_node_id = current_id
    best_sim = -2.0

    for step in range(max_steps):
        frame = _get_node_frame(current_id, world_graph)
        desc = describe_node(current_id, world_graph)

        # Check if goal achieved: CLIP similarity between goal and current frame
        if frame is not None and goal_emb is not None:
            frame_emb = embed_frame(frame)
            if frame_emb is not None:
                sim = (goal_emb.flatten() @ frame_emb.flatten()) / (
                    max(1e-9, (goal_emb.flatten() ** 2).sum() ** 0.5 * (frame_emb.flatten() ** 2).sum() ** 0.5)
                )
                sim = float(sim)
                if sim > best_sim:
                    best_sim = sim
                    best_node_id = current_id
                if sim >= goal_similarity_threshold:
                    pos = getattr(world_graph, "get_pose_at_node", lambda _: None)(current_id)
                    return AgentResult(
                        found=True,
                        best_node_id=current_id,
                        confidence=min(1.0, sim),
                        path_taken=path,
                        steps_taken=step + 1,
                        description=desc,
                        coordinates=pos,
                    )

        # Pick best neighbor by description similarity to goal
        neighbors = _neighbor_ids(current_id, world_graph)
        if not neighbors:
            break
        next_id = None
        next_best_sim = -2.0
        if goal_emb is not None:
            for nid in neighbors:
                n_desc = describe_node(nid, world_graph)
                n_emb = embed_text(n_desc)
                if n_emb is not None:
                    s = (goal_emb.flatten() @ n_emb.flatten()) / (
                        max(1e-9, (goal_emb.flatten() ** 2).sum() ** 0.5 * (n_emb.flatten() ** 2).sum() ** 0.5)
                    )
                    if float(s) > next_best_sim:
                        next_best_sim = float(s)
                        next_id = nid
        if next_id is None:
            next_id = neighbors[0]
        current_id = next_id
        path.append(current_id)

    pos = getattr(world_graph, "get_pose_at_node", lambda _: None)(best_node_id)
    return AgentResult(
        found=False,
        best_node_id=best_node_id,
        confidence=max(0.0, min(1.0, best_sim)),
        path_taken=path,
        steps_taken=max_steps,
        description=describe_node(best_node_id, world_graph),
        coordinates=pos,
    )


def run_search(query: str, world_graph: Any) -> AgentResult:
    """Instant best-matching node via CLIP (no step-by-step navigation)."""
    nodes = getattr(world_graph, "nodes", None) or {}
    if len(nodes) < 5:
        return AgentResult(
            found=False,
            best_node_id=None,
            confidence=0.0,
            path_taken=[],
            steps_taken=0,
            description="NOT ENOUGH MAP DATA — keep exploring.",
        )
    best_id = find_best_node(query, world_graph)
    if not best_id:
        ordered = sorted(nodes.keys())
        best_id = ordered[0]
    path = [best_id]
    pos = getattr(world_graph, "get_pose_at_node", lambda _: None)(best_id)
    return AgentResult(
        found=True,
        best_node_id=best_id,
        confidence=0.9,
        path_taken=path,
        steps_taken=1,
        description=describe_node(best_id, world_graph),
        coordinates=pos,
    )
