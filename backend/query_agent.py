"""
PHANTOM CODE — Agentic query: vector DB search + local Llama (Genie) for structured answers.
No cloud, no API keys. Fallback to keyword answer from graph if Genie fails or times out.
"""

import logging
from typing import List, Optional, Callable

try:
    from .vector_db import query as vector_query
    from .genie_runner import run_genie, is_available as genie_available
except ImportError:
    from backend.vector_db import query as vector_query
    from backend.genie_runner import run_genie, is_available as genie_available

logger = logging.getLogger(__name__)


def _format_prompt(question: str, context_chunks: List[str]) -> str:
    """Format context + question for Llama chat (simple template)."""
    context = "\n\n".join(context_chunks[:3])
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Use the following context to answer the question. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _keyword_fallback(question: str, get_graph_callback: Optional[Callable]) -> str:
    """Fast fallback: search graph nodes by keyword if no Genie. Never crashes."""
    if get_graph_callback:
        try:
            data = get_graph_callback()
            nodes = data.get("nodes", []) if isinstance(data, dict) else []
            q = question.lower()
            out = []
            for node in nodes[:5]:
                node_id = node.get("node_id", "")
                dets = node.get("detections", [])
                for d in dets:
                    cls = (d.get("class_name") or d.get("class", "")).lower()
                    if cls in q or q in cls or ("survivor" in q and cls == "person"):
                        out.append(f"Node {node_id}: {cls} detected.")
            if out:
                return " ".join(out[:3])
            return "No matching nodes in current map. Try refining your question or wait for more detections."
        except Exception:
            pass
    return (
        "No on-device LLM (Genie) or map graph is available. "
        "You can still get answers by adding .txt emergency manuals to the data/ folder and restarting the backend (vector search will use them). "
        "For full answers, set up the Genie bundle (genie_bundle/ with genie-t2t-run.exe and config) in the project root."
    )


def query_agent(
    question: str,
    top_k: int = 3,
    get_graph_callback: Optional[Callable] = None,
) -> dict:
    """
    Natural language question -> vector search -> Genie -> structured response.
    Returns {"answer": str, "node_ids": List[str]}.
    node_ids are the doc IDs from vector DB that were referenced (for map highlight).
    """
    node_ids = []
    if not question or not question.strip():
        return {"answer": "Please ask a question.", "node_ids": []}

    # 1) Vector search
    results = vector_query(question.strip(), top_k=top_k)
    context_chunks = []
    for doc_id, text, _ in results:
        context_chunks.append(f"[{doc_id}]: {text[:500]}")
        if doc_id.startswith("node_"):
            node_ids.append(doc_id)

    # 2) Genie if available
    if context_chunks and genie_available():
        prompt = _format_prompt(question, context_chunks)
        answer = run_genie(prompt)
        if answer:
            return {"answer": answer.strip(), "node_ids": node_ids}

    # 3) If we have vector search results, return them as the answer (no Genie needed)
    if context_chunks:
        summary = "\n\n".join(f"• {chunk}" for chunk in context_chunks[:3])
        return {
            "answer": f"From manuals / map:\n\n{summary}",
            "node_ids": node_ids,
        }

    # 4) No context: try graph keyword fallback, then friendly message
    answer = _keyword_fallback(question, get_graph_callback)
    return {"answer": answer, "node_ids": node_ids}
