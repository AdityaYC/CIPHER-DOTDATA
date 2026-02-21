"""
PHANTOM CODE — Run Llama via Qualcomm Genie SDK (genie-t2t-run) subprocess.
Local only; no cloud. Handles timeouts with keyword fallback.
"""

import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
GENIE_BUNDLE = os.path.join(_PROJECT_ROOT, "genie_bundle")
GENIE_EXE = "genie-t2t-run.exe"
GENIE_CONFIG = "genie_config.json"
GENIE_TIMEOUT_SEC = 5


def _exe_path() -> Optional[str]:
    exe = os.path.join(GENIE_BUNDLE, GENIE_EXE)
    if os.path.isfile(exe):
        return exe
    return None


def run_genie(prompt: str, config_path: Optional[str] = None) -> str:
    """
    Run Genie with the given prompt. Returns response text or empty string on failure.
    If Genie takes longer than GENIE_TIMEOUT_SEC, returns empty string (caller can fallback).
    """
    exe = _exe_path()
    if not exe:
        logger.warning("genie-t2t-run.exe not found in genie_bundle")
        return ""
    config = config_path or os.path.join(GENIE_BUNDLE, GENIE_CONFIG)
    if not os.path.isfile(config):
        logger.warning(f"Genie config not found: {config}")
        return ""
    # Genie expects prompt on -p; format as required by the model (e.g. chat template)
    try:
        proc = subprocess.run(
            [exe, "-c", config, "-p", prompt],
            cwd=GENIE_BUNDLE,
            capture_output=True,
            text=True,
            timeout=GENIE_TIMEOUT_SEC,
            env=os.environ.copy(),
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.strip()
        if proc.stderr:
            logger.debug(f"Genie stderr: {proc.stderr[:200]}")
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("Genie timed out")
        return ""
    except Exception as e:
        logger.warning(f"Genie run failed: {e}")
        return ""


def is_available() -> bool:
    """Return True if Genie bundle and exe are present."""
    return _exe_path() is not None and os.path.isfile(os.path.join(GENIE_BUNDLE, GENIE_CONFIG))
