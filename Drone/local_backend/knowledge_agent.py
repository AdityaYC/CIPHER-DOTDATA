"""
Disaster knowledge agent: RAG over emergency manuals + world graph, with Genie LLM.
Uses ChromaDB (./chroma_db/), sentence-transformers (./models/embeddings/), Genie (genie_bundle/).
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import os
import sys

# Ensure repo root on path for backend imports
_DRONE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _DRONE2_ROOT not in sys.path:
    sys.path.insert(0, _DRONE2_ROOT)

try:
    from . import emergency_manuals
except ImportError:
    import emergency_manuals


@dataclass
class AnswerResult:
    answer_text: str
    referenced_node_ids: List[str]
    confidence: float
    sources_used: List[str]
    recommended_action: Optional[str] = None


def load_vector_db(world_graph: Any = None) -> bool:
    """
    Load or create ChromaDB at ./chroma_db/. Embed manuals from ./data/emergency_manuals/
    and optionally current graph nodes. Uses all-MiniLM-L6-v2 at ./models/embeddings/.
    """
    try:
        emergency_manuals.ensure_all_manuals()
        from backend.vector_db import load_manuals_from_data_dir, _get_chroma
    except ImportError:
        try:
            from vector_db import load_manuals_from_data_dir, _get_chroma
        except ImportError:
            return False
    try:
        manuals_dir = os.path.join(_DRONE2_ROOT, "data", "emergency_manuals")
        if os.path.isdir(manuals_dir):
            load_manuals_from_data_dir(manuals_dir)
        load_manuals_from_data_dir()  # also data/*.txt
        if world_graph is not None:
            update_graph_embeddings(world_graph)
        return _get_chroma() is not None
    except Exception:
        return False


def update_graph_embeddings(world_graph: Any) -> int:
    """Embed current world graph nodes into vector DB. Returns count synced."""
    try:
        from backend.vector_db import sync_graph_nodes
    except ImportError:
        from vector_db import sync_graph_nodes
    get_graph = getattr(world_graph, "get_graph", None)
    if not get_graph:
        return 0
    return sync_graph_nodes(get_graph)


def retrieve_context(query: str, k: int = 5) -> List[str]:
    """Top k relevant chunks from vector DB (manuals + graph node descriptions)."""
    try:
        from backend.vector_db import query
    except ImportError:
        from vector_db import query
    results = query(query.strip(), top_k=k)
    return [f"[{doc_id}]: {text[:600]}" for doc_id, text, _ in results]


def ask_llm(prompt: str) -> str:
    """Call Llama via Genie (genie-t2t-run.exe), 5s timeout. Template fallback if Genie fails."""
    try:
        from backend.genie_runner import run_genie, is_available
    except ImportError:
        from genie_runner import run_genie, is_available
    if is_available():
        out = run_genie(prompt)
        if out:
            return out.strip()
    # Template fallback: use retrieved context directly
    return "Based on the available manuals and map data, please review the context above and take action as appropriate. If the context does not contain enough information, advise caution and request more data."


def answer_question(question: str, world_graph: Any) -> AnswerResult:
    """
    Full RAG: retrieve context, build prompt, call Genie, return AnswerResult.
    referenced_node_ids are node_ids from vector DB that were used.
    """
    node_ids = []
    context = retrieve_context(question.strip(), k=5)
    for c in context:
        if c.startswith("[node_"):
            end = c.find("]")
            if end > 0:
                node_ids.append(c[1:end])
    system = "You are a disaster response advisor. Use only the provided context and map data. Be concise. If you recommend an action, state it clearly."
    prompt = f"{system}\n\nContext:\n" + "\n\n".join(context[:5]) + f"\n\nQuestion: {question}\n\nAnswer:"
    answer = ask_llm(prompt)
    rec = None
    if "recommend" in answer.lower() or "dispatch" in answer.lower() or "should" in answer.lower():
        rec = answer
    return AnswerResult(
        answer_text=answer,
        referenced_node_ids=node_ids,
        confidence=0.8 if answer else 0.0,
        sources_used=context[:3],
        recommended_action=rec,
    )
