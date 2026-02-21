"""
PHANTOM CODE — Local vector database for semantic search over world graph nodes and emergency manuals.
Uses ChromaDB with persistent storage and local sentence-transformers. No cloud, no API keys.
"""

import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Paths relative to project root (parent of backend/)
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
CHROMA_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")
EMBEDDING_CACHE = os.path.join(_PROJECT_ROOT, "models", "embeddings")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_collection = None
_embed_fn = None


def _get_embedding_model():
    """Load local sentence-transformers model. Falls back to None if unavailable."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        import sentence_transformers  # type: ignore
        os.makedirs(EMBEDDING_CACHE, exist_ok=True)
        # Small local model; no API, no internet required after first download
        model = sentence_transformers.SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=EMBEDDING_CACHE,
        )
        def embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            return model.encode(texts, show_progress_bar=False)
        _embed_fn = embed
        return _embed_fn
    except Exception as e:
        logger.warning(f"Embedding model not available: {e}. Vector search will use keyword fallback.")
        return None


def _get_chroma():
    """Get or create ChromaDB collection. Persistent at CHROMA_DIR. No cloud sync."""
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb  # type: ignore
        os.makedirs(CHROMA_DIR, exist_ok=True)
        # Use path only to avoid Settings() config/Pydantic issues (e.g. chroma_server_nofile type inference)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        coll = client.get_or_create_collection(
            name="phantom_nodes_and_manuals",
            metadata={"description": "World graph nodes and emergency manuals"},
        )
        _collection = coll
        return _collection
    except Exception as e:
        logger.warning(f"ChromaDB not available: {e}. Vector search disabled.")
        return None


def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Return list of embedding vectors or None if embedding failed."""
    fn = _get_embedding_model()
    if fn is None:
        return None
    try:
        return fn(texts).tolist() if hasattr(fn(texts), "tolist") else list(fn(texts))
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def add_document(doc_id: str, text: str, metadata: Optional[dict] = None) -> bool:
    """Add a single document (e.g. graph node or manual excerpt) to the vector DB."""
    coll = _get_chroma()
    if coll is None:
        return False
    emb = _embed_texts([text])
    if emb is None:
        return False
    meta = metadata or {}
    try:
        coll.upsert(ids=[doc_id], embeddings=emb, documents=[text], metadatas=[meta])
        return True
    except Exception as e:
        logger.warning(f"Vector DB add failed: {e}")
        return False


def add_node_text(node_id: str, text: str) -> bool:
    """Embed a world graph node description. E.g. 'SURVIVOR at position X confidence 91% depth 2.3m'."""
    return add_document(doc_id=node_id, text=text, metadata={"type": "node", "node_id": node_id})


def load_manuals_from_data_dir(data_dir: Optional[str] = None) -> int:
    """Load all .txt files from data_dir (default PROJECT_ROOT/data) into the vector DB."""
    dir_path = data_dir or DATA_DIR
    if not os.path.isdir(dir_path):
        logger.warning(f"Data dir not found: {dir_path}")
        return 0
    coll = _get_chroma()
    if coll is None:
        return 0
    count = 0
    for name in os.listdir(dir_path):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue
        doc_id = f"manual_{name}"
        if add_document(doc_id=doc_id, text=text, metadata={"type": "manual", "source": name}):
            count += 1
    return count


def query(question: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """
    Semantic search. Returns list of (doc_id, text, score).
    Falls back to empty list if Chroma or embedding fails.
    """
    coll = _get_chroma()
    if coll is None:
        return []
    emb = _embed_texts([question])
    if emb is None:
        return []
    try:
        results = coll.query(query_embeddings=emb, n_results=top_k, include=["documents", "metadatas"])
        out = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            for i, doc_id in enumerate(ids):
                text = docs[i] if i < len(docs) else ""
                # Chroma returns distance; lower can mean more similar depending on config
                score = 1.0 - (dists[i] / 2.0) if dists[i] is not None else 0.0
                score = max(0.0, min(1.0, score))
                out.append((doc_id, text, float(score)))
        return out
    except Exception as e:
        logger.warning(f"Vector query failed: {e}")
        return []


def sync_graph_nodes(get_graph_callback):
    """
    Call get_graph_callback() to get { 'nodes': [ { node_id, gps_lat, gps_lon, detections: [...] } ] }.
    Embed each node as text and upsert into vector DB. Safe no-op if callback or DB fails.
    """
    try:
        data = get_graph_callback()
        nodes = data.get("nodes") if isinstance(data, dict) else []
    except Exception as e:
        logger.warning(f"get_graph_callback failed: {e}")
        return 0
    count = 0
    for node in nodes:
        node_id = node.get("node_id", "")
        dets = node.get("detections", [])
        lat = node.get("gps_lat", 0)
        lon = node.get("gps_lon", 0)
        parts = [f"Position ({lat:.5f}, {lon:.5f})."]
        for d in dets[:5]:
            cls = d.get("class_name", d.get("class", "object"))
            conf = d.get("confidence", 0)
            depth = d.get("distance_meters")
            cat = d.get("category", "")
            s = f"{cls.upper()} detected with confidence {int(conf*100)}%"
            if depth is not None:
                s += f" at depth {depth:.1f}m"
            if cat:
                s += f" category {cat}"
            parts.append(s)
        text = " ".join(parts)
        if add_node_text(node_id, text):
            count += 1
    return count
