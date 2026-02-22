"""
Run Llama (or other chat models) via Ollama for agentic Q&A.
Recommended for Agent tab: install Ollama, then run:
    ollama run llama3.2
Use llama3.2 for fast agentic talk; use llama3.1 for higher quality if your machine can run it.
"""

import logging
import os
import urllib.request
import urllib.error
import json

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Model for agentic talk: llama3.2 (3B) is fast; llama3.1 (8B) is higher quality
OLLAMA_AGENT_MODEL = os.environ.get("OLLAMA_AGENT_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SEC = 30


def _api_url(path: str) -> str:
    base = OLLAMA_HOST.rstrip("/")
    return f"{base}{path}"


def is_available(model: str = OLLAMA_AGENT_MODEL) -> bool:
    """Return True if Ollama is running and the model is available."""
    try:
        req = urllib.request.Request(
            _api_url("/api/tags"),
            method="GET",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        models = data.get("models", [])
        names = [m.get("name", "") for m in models]
        # Match exact or with :tag
        for n in names:
            if n == model or n.startswith(model + ":"):
                return True
        if names:
            logger.info(f"Ollama running but model {model} not in {names}; will try anyway.")
        return True  # Ollama is up; /api/generate may still pull the model
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")
        return False


def run_ollama(
    prompt: str,
    system: str = "You are a helpful disaster response assistant. Use only the provided context. Be concise.",
    model: str = OLLAMA_AGENT_MODEL,
    timeout: int = OLLAMA_TIMEOUT_SEC,
) -> str:
    """
    Call Ollama /api/generate with the given prompt. Returns response text or empty string on failure.
    """
    url = _api_url("/api/generate")
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        response = out.get("response", "").strip()
        return response if response else ""
    except urllib.error.HTTPError as e:
        logger.warning(f"Ollama HTTP error: {e.code} {e.reason}")
        return ""
    except Exception as e:
        logger.warning(f"Ollama run failed: {e}")
        return ""
