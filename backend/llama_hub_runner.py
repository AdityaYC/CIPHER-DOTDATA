"""
Local LLM runner — HuggingFace transformers, CPU-optimized.
Model: HuggingFaceTB/SmolLM2-360M-Instruct
  - 360M params → fast on CPU (~20s first load, ~3-5s per query)
  - Free, no token, no gating, already cached
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
_pipe = None
_load_lock = threading.Lock()
_load_failed = False  # only set True after 3 consecutive failures


def _load() -> bool:
    global _pipe, _load_failed
    if _pipe is not None:
        return True
    if _load_failed:
        return False
    with _load_lock:
        if _pipe is not None:
            return True
        try:
            from transformers import pipeline
            import torch
            logger.info(f"Loading {_MODEL_ID}...")
            _pipe = pipeline(
                "text-generation",
                model=_MODEL_ID,
                torch_dtype=torch.float32,
                device=-1,  # CPU
            )
            logger.info(f"{_MODEL_ID} ready.")
            return True
        except Exception as e:
            logger.warning(f"LLM load failed: {e}")
            _pipe = None
            return False


def run_llama_1b(
    prompt: str,
    system: Optional[str] = None,
    max_new_tokens: int = 128,
) -> str:
    print(f"[LLM] run_llama_1b called, prompt len={len(prompt)}", flush=True)
    if not _load():
        print("[LLM] _load() returned False — model unavailable", flush=True)
        return ""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt[:1500]})  # cap prompt length
        print(f"[LLM] Running inference...", flush=True)
        result = _pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    answer = msg.get("content", "").strip()
                    print(f"[LLM] Answer: {answer[:80]}", flush=True)
                    return answer
        answer = str(generated).strip()
        print(f"[LLM] Answer (str): {answer[:80]}", flush=True)
        return answer
    except Exception as e:
        print(f"[LLM] inference error: {e}", flush=True)
        logger.warning(f"LLM inference error: {e}")
        return ""


def is_available() -> bool:
    return _load()
