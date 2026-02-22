"""
PHANTOM CODE — Agentic query: vector DB search + local Llama for structured answers.
Uses Ollama (recommended) or Genie when available. No cloud, no API keys.

Recommended for agentic talk: install Ollama and run:
    ollama run llama3.2
Use llama3.2 (3B) for fast answers; llama3.1 (8B) for higher quality.
"""

import logging
from typing import List, Optional, Callable

try:
    from .vector_db import query as vector_query
    from .genie_runner import run_genie, is_available as genie_available
    from .ollama_runner import run_ollama, is_available as ollama_available
except ImportError:
    from backend.vector_db import query as vector_query
    from backend.genie_runner import run_genie, is_available as genie_available
    from backend.ollama_runner import run_ollama, is_available as ollama_available

logger = logging.getLogger(__name__)


def _format_prompt(question: str, context_chunks: List[str]) -> str:
    """Format context + question for Llama (Ollama or Genie)."""
    context = "\n\n".join(context_chunks[:3])
    return (
        "Use the following context to answer the question. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _format_prompt_genie(question: str, context_chunks: List[str]) -> str:
    """Genie expects Llama chat template."""
    context = "\n\n".join(context_chunks[:3])
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Use the following context to answer the question. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _doc_id_to_title(doc_id: str) -> str:
    """Turn manual_foo.txt or node_0001 into a readable title."""
    if doc_id.startswith("manual_"):
        name = doc_id.replace("manual_", "", 1)
        if name.endswith(".txt"):
            name = name[:-4]
        return name.replace("_", " ").strip().title()
    if doc_id.startswith("node_"):
        return f"Map location {doc_id}"
    return doc_id


def _format_manual_fallback(
    results: List,
    question: str,
    max_chars_per_chunk: int = 2000,
) -> str:
    """Format vector search results as a readable answer when Genie is not used."""
    if not results:
        return "No relevant manuals or map data found for your question."
    lines = ["Based on the emergency manuals and map:\n"]
    seen = set()
    for doc_id, text, _ in results:
        if not text or doc_id in seen:
            continue
        seen.add(doc_id)
        title = _doc_id_to_title(doc_id)
        excerpt = (text[:max_chars_per_chunk] + "…") if len(text) > max_chars_per_chunk else text
        excerpt = excerpt.strip()
        lines.append(f"{title}\n{excerpt}\n")
    return "\n".join(lines).strip()


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
        "No on-device LLM is available. For agentic answers: (1) Install Ollama and run 'ollama run llama3.2', then restart the backend; "
        "or (2) set up the Genie bundle (genie_bundle/) in the project root. "
        "You can still get answers from manuals by adding .txt files to the data/ folder."
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

    # 2) Ollama (recommended for agentic talk: ollama run llama3.2)
    if context_chunks and ollama_available():
        prompt = _format_prompt(question, context_chunks)
        answer = run_ollama(prompt)
        if answer:
            return {"answer": answer.strip(), "node_ids": node_ids, "confidence": 0.85, "recommended_action": ""}

    # 3) Genie if available (Qualcomm Genie bundle)
    if context_chunks and genie_available():
        prompt = _format_prompt_genie(question, context_chunks)
        answer = run_genie(prompt)
        if answer:
            return {"answer": answer.strip(), "node_ids": node_ids, "confidence": 0.85, "recommended_action": ""}

    # 4) If we have vector search results, return a readable answer (no LLM or LLM returned empty)
    if context_chunks:
        answer = _format_manual_fallback(results[:5], question)
        return {
            "answer": answer,
            "node_ids": node_ids,
            "confidence": 0.75,
            "recommended_action": "",
        }

    # 5) No context: try Ollama with just the question (agentic talk), then graph keyword fallback
    if ollama_available():
        answer = run_ollama(
            f"Question: {question}\n\nAnswer briefly:",
            system="You are a helpful disaster response assistant. Answer concisely.",
        )
        if answer:
            return {"answer": answer.strip(), "node_ids": node_ids, "confidence": 0.6, "recommended_action": ""}
    answer = _keyword_fallback(question, get_graph_callback)
    return {"answer": answer, "node_ids": node_ids, "confidence": 0.5, "recommended_action": ""}
