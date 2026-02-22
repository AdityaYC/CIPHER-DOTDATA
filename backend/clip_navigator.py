"""
CLIP-based vision reasoning for spatial agent.
Uses openai/clip-vit-base-patch32 from local cache (./models/clip/).
Falls back to keyword matching when CLIP is unavailable.
"""

from typing import Any, List, Optional, Tuple
import base64
import re
import io
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRONE2_ROOT = _HERE.parent
CLIP_CACHE = _DRONE2_ROOT / "models" / "clip"

_model = None
_processor = None


def load_clip() -> bool:
    """Load CLIP (openai/clip-vit-base-patch32) from local cache. Returns True if loaded."""
    global _model, _processor
    if _model is not None:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        os.makedirs(CLIP_CACHE, exist_ok=True)
        cache_dir = str(CLIP_CACHE)
        # Load from local cache only (no internet). Cache at ./models/clip/ via first-time download elsewhere if needed.
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir, local_files_only=True)
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir, local_files_only=True)
        _model.eval()
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"CLIP not loaded: {e}")
        return False


def embed_text(text: str):
    """CLIP text embedding as numpy array (1D). Returns None if CLIP unavailable."""
    if not load_clip():
        return None
    import torch
    try:
        inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
        return out[0].numpy()
    except Exception:
        return None


def embed_frame(frame) -> Optional[Any]:
    """CLIP image embedding. frame: numpy RGB (H,W,3) or PIL Image. Returns numpy 1D or None."""
    if not load_clip():
        return None
    import torch
    try:
        from PIL import Image
        if hasattr(frame, "shape"):
            frame = Image.fromarray(frame)
        inputs = _processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            out = _model.get_image_features(**inputs)
        return out[0].numpy()
    except Exception:
        return None


def _cosine(a, b) -> float:
    import numpy as np
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n == 0:
        return 0.0
    return float(np.dot(a, b) / n)


def find_best_node(query_text: str, world_graph: Any) -> Optional[str]:
    """Node_id with highest cosine similarity between query embedding and node frame embeddings."""
    q_emb = embed_text(query_text)
    if q_emb is None:
        return _keyword_best_node(query_text, world_graph)
    best_id = None
    best_sim = -2.0
    for node_id, node in (world_graph.nodes or {}).items():
        if not getattr(node, "image_b64", None):
            continue
        try:
            raw = base64.b64decode(node.image_b64)
            from PIL import Image
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_emb = embed_frame(img)
            if img_emb is not None:
                sim = _cosine(q_emb, img_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = node_id
        except Exception:
            continue
    if best_id is not None:
        return best_id
    return _keyword_best_node(query_text, world_graph)


def find_top_k_nodes(query_text: str, world_graph: Any, k: int = 3) -> List[str]:
    """List of k most relevant node_ids by CLIP similarity (or keyword fallback)."""
    q_emb = embed_text(query_text)
    if q_emb is None:
        return _keyword_top_nodes(query_text, world_graph, k)
    scored = []
    for node_id, node in (world_graph.nodes or {}).items():
        if not getattr(node, "image_b64", None):
            continue
        try:
            raw = base64.b64decode(node.image_b64)
            from PIL import Image
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_emb = embed_frame(img)
            if img_emb is not None:
                sim = _cosine(q_emb, img_emb)
                scored.append((node_id, sim))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[1])
    return [nid for nid, _ in scored[:k]]


def are_similar(frame_a, frame_b, threshold: float = 0.85) -> bool:
    """Whether two frames are similar (CLIP image similarity >= threshold). Used for goal-found check."""
    ea, eb = embed_frame(frame_a), embed_frame(frame_b)
    if ea is None or eb is None:
        return False
    return _cosine(ea, eb) >= threshold


def describe_node(node_id: str, world_graph: Any) -> str:
    """String description from node semantic labels and type. Fallback to keyword if CLIP unavailable."""
    nodes = getattr(world_graph, "nodes", None) or {}
    node = nodes.get(node_id)
    if not node:
        return f"Unknown node {node_id}."
    parts = []
    dets = getattr(node, "detections", []) or []
    for d in dets[:8]:
        cls = getattr(d, "class_name", None) or getattr(d, "class", "object")
        conf = getattr(d, "confidence", 0) or 0
        dist = getattr(d, "distance_meters", None)
        cat = getattr(d, "category", None)
        cat_str = str(cat).split(".")[-1] if cat else ""
        s = f"{cls} at {conf*100:.0f}% confidence"
        if dist is not None:
            s += f", {dist:.1f}m"
        if cat_str:
            s += f", {cat_str}"
        parts.append(s)
    if not parts:
        parts.append("No detections.")
    node_type = "IMPORTED" if getattr(node, "source", "") == "imported" else "LIVE"
    text = " ".join(parts) + f". Node type: {node_type}. Structural risk: LOW."
    return f"This location shows: {text}"


def find_nodes_by_detection_class(query_text: str, world_graph: Any) -> List[Tuple[str, float]]:
    """Find nodes that have YOLO detections matching the query (e.g. 'fire extinguisher' -> nodes with that class).
    Returns [(node_id, max_confidence), ...] sorted by confidence descending. Uses Manual/import YOLO detection frames."""
    nodes = getattr(world_graph, "nodes", None) or {}
    if not nodes:
        return []
    words = re.sub(r"[^\w\s]", " ", query_text.lower()).split()
    words = [w for w in words if len(w) > 1]
    if not words:
        return []
    out: List[Tuple[str, float]] = []
    for node_id, node in nodes.items():
        dets = getattr(node, "detections", []) or []
        best_conf = 0.0
        for d in dets:
            cls = (getattr(d, "class_name", None) or getattr(d, "class", "") or "").lower()
            conf = float(getattr(d, "confidence", 0) or 0)
            for w in words:
                if w in cls or cls in w:
                    best_conf = max(best_conf, conf)
                    break
            if "person" in words and cls == "person":
                best_conf = max(best_conf, conf)
            if "exit" in words and "door" in cls:
                best_conf = max(best_conf, conf)
        if best_conf > 0:
            out.append((node_id, best_conf))
    out.sort(key=lambda x: -x[1])
    return out


def _keyword_best_node(query_text: str, world_graph: Any) -> Optional[str]:
    """Fallback: pick first node whose detections match query keywords."""
    by_det = find_nodes_by_detection_class(query_text, world_graph)
    if by_det:
        return by_det[0][0]
    q = query_text.lower()
    nodes = getattr(world_graph, "nodes", None) or {}
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        for d in getattr(node, "detections", []) or []:
            cls = (getattr(d, "class_name", None) or getattr(d, "class", "") or "").lower()
            if cls in q or q in cls or ("person" in q and cls == "person") or ("exit" in q and "door" in cls):
                return node_id
    return list(nodes.keys())[0] if nodes else None


def _keyword_top_nodes(query_text: str, world_graph: Any, k: int) -> List[str]:
    """Fallback: nodes that match query keywords (by detection class), then rest."""
    by_det = find_nodes_by_detection_class(query_text, world_graph)
    if by_det:
        matched_ids = [nid for nid, _ in by_det]
        nodes = getattr(world_graph, "nodes", None) or {}
        rest = [nid for nid in sorted(nodes.keys()) if nid not in matched_ids]
        return (matched_ids + rest)[:k]
    q = query_text.lower()
    nodes = getattr(world_graph, "nodes", None) or {}
    matched = []
    rest = []
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        for d in getattr(node, "detections", []) or []:
            cls = (getattr(d, "class_name", None) or getattr(d, "class", "") or "").lower()
            if cls in q or q in cls or ("person" in q and cls == "person"):
                matched.append(node_id)
                break
        else:
            rest.append(node_id)
    return (matched + rest)[:k]
