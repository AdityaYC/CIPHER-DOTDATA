"""
PHANTOM CODE — LLM advisory via local Ollama. No cloud, no API keys.
"""

import logging
import time
import httpx

import config

logger = logging.getLogger(__name__)

OFFLINE_MESSAGE = "LLM OFFLINE — Advisory unavailable"


def get_advisory(detection_summary: str, mission_key: str) -> str:
    """
    Send detection summary + mission prompt to local Ollama.
    Returns tactical advice text or OFFLINE_MESSAGE if unreachable.
    """
    mission = config.MISSIONS.get(mission_key, list(config.MISSIONS.values())[0])
    system = mission["prompt"]
    user = f"Current drone observations:\n{detection_summary}\n\nWhat should the operator do?"

    payload = {
        "model": config.LLM_MODEL,
        "prompt": user,
        "system": system,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(config.LLM_ENDPOINT, json=payload)
            r.raise_for_status()
            data = r.json()
            elapsed = time.perf_counter() - start
            logger.info(f"LLM response in {elapsed:.1f}s")
            return data.get("response", OFFLINE_MESSAGE).strip()
    except Exception as e:
        logger.warning(f"Ollama unreachable: {e}")
        return OFFLINE_MESSAGE
