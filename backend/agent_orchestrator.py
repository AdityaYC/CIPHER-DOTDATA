"""
Orchestrator: classifies query (SPATIAL / KNOWLEDGE / COMBINED) and runs the right agent(s).
All local; no cloud. Returns unified OrchestratorResult for the UI.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from spatial_agent import run_exploration, run_search, AgentResult
from knowledge_agent import answer_question, AnswerResult, load_vector_db, update_graph_embeddings


@dataclass
class OrchestratorResult:
    answer_text: str
    highlighted_node_ids: List[str]
    path_to_navigate: List[str]
    recommended_action: Optional[str]
    confidence: float
    agent_used: str  # "SPATIAL" | "KNOWLEDGE" | "COMBINED"


def classify_query(query_text: str) -> str:
    """Returns 'SPATIAL' | 'KNOWLEDGE' | 'COMBINED'."""
    q = (query_text or "").strip().lower()
    spatial = any(
        x in q
        for x in ("where is", "find", "navigate to", "show me", "locate", "nearest", "which way", "path to")
    )
    knowledge = any(
        x in q
        for x in ("is it safe", "what should i do", "how do i", "recommend", "procedure", "protocol", "triage")
    )
    combined = any(
        x in q
        for x in ("safe to go to", "safe to enter", "where the survivor", "go to where")
    )
    if combined:
        return "COMBINED"
    if spatial and knowledge:
        return "COMBINED"
    if spatial:
        return "SPATIAL"
    if knowledge:
        return "KNOWLEDGE"
    return "SPATIAL"


def run_agent(query_text: str, world_graph: Any) -> OrchestratorResult:
    """
    Classify, run spatial and/or knowledge agent, return unified result.
    If graph has fewer than 5 nodes, returns NOT ENOUGH MAP DATA message.
    """
    q = (query_text or "").strip()
    if not q:
        return OrchestratorResult(
            answer_text="Please enter a question.",
            highlighted_node_ids=[],
            path_to_navigate=[],
            recommended_action=None,
            confidence=0.0,
            agent_used="KNOWLEDGE",
        )
    nodes = getattr(world_graph, "nodes", None) or {}
    if len(nodes) < 5:
        return OrchestratorResult(
            answer_text="NOT ENOUGH MAP DATA — keep exploring or import more video.",
            highlighted_node_ids=[],
            path_to_navigate=[],
            recommended_action=None,
            confidence=0.0,
            agent_used="SPATIAL",
        )
    kind = classify_query(q)
    answer_text = ""
    highlighted_node_ids: List[str] = []
    path_to_navigate: List[str] = []
    recommended_action: Optional[str] = None
    confidence = 0.0

    if kind == "SPATIAL":
        res = run_exploration(q, world_graph, max_steps=20)
        answer_text = res.description
        highlighted_node_ids = [res.best_node_id] if res.best_node_id else []
        path_to_navigate = res.path_taken or []
        confidence = res.confidence
        if not res.found and res.steps_taken >= 20:
            answer_text = f"Best match after {res.steps_taken} steps: {res.description}"

    elif kind == "KNOWLEDGE":
        kr = answer_question(q, world_graph)
        answer_text = kr.answer_text
        highlighted_node_ids = list(kr.referenced_node_ids or [])
        recommended_action = kr.recommended_action
        confidence = kr.confidence

    else:
        # COMBINED: spatial finds location, knowledge advises
        spatial_res = run_exploration(q, world_graph, max_steps=20)
        path_to_navigate = spatial_res.path_taken or []
        highlighted_node_ids = [spatial_res.best_node_id] if spatial_res.best_node_id else []
        kr = answer_question(q, world_graph)
        answer_text = f"{spatial_res.description}\n\n{kr.answer_text}"
        highlighted_node_ids = list(set(highlighted_node_ids + (kr.referenced_node_ids or [])))
        recommended_action = kr.recommended_action
        confidence = (spatial_res.confidence + kr.confidence) / 2.0

    return OrchestratorResult(
        answer_text=answer_text or "No answer.",
        highlighted_node_ids=highlighted_node_ids,
        path_to_navigate=path_to_navigate,
        recommended_action=recommended_action,
        confidence=min(1.0, confidence),
        agent_used=kind,
    )
