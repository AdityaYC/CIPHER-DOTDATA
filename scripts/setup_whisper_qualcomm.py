"""
Download and export Whisper-Base from Qualcomm AI Hub for on-device voice (NPU).

- Downloads openai/whisper-base.en and caches in models/whisper_hf/ (for tokenizer + mel).
- Exports encoder and decoder to ONNX for Qualcomm (QNN) via qai_hub_models.
- Output: models/whisper_qualcomm/ (encoder + decoder ONNX for use with voice_input).

Run from repo root (Python 3.10+):
    pip install qai-hub "qai-hub-models[whisper]"
    python -m qai_hub configure --api_token YOUR_TOKEN   # if using AI Hub cloud
    python scripts/setup_whisper_qualcomm.py

For local-only transcription without Qualcomm export, the app uses Hugging Face
Whisper (openai/whisper-base.en) automatically; install:
    pip install transformers soundfile
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
WHISPER_HF = MODELS_DIR / "whisper_hf"
WHISPER_QUALCOMM = MODELS_DIR / "whisper_qualcomm"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_QUALCOMM.mkdir(parents=True, exist_ok=True)

    # 1) Pre-download Hugging Face Whisper (tokenizer + model for cache)
    print("Downloading Whisper (openai/whisper-base.en) to cache...")
    try:
        from transformers import pipeline, WhisperProcessor
        os.environ.setdefault("TRANSFORMERS_CACHE", str(WHISPER_HF))
        WhisperProcessor.from_pretrained("openai/whisper-base.en", cache_dir=str(WHISPER_HF))
        pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=-1, model_kwargs={"cache_dir": str(WHISPER_HF)})
        print("  Whisper HF cache ready at", WHISPER_HF)
    except Exception as e:
        print(f"  HF Whisper skip: {e}")

    # 2) Export Whisper via Qualcomm AI Hub (optional; requires qai_hub token)
    print("Exporting Whisper-Base for Qualcomm...")
    try:
        from qai_hub_models.models.whisper_base import Model
        from qai_hub_models.utils.export import export_without_hub_access

        model = Model.from_pretrained()
        export_without_hub_access(
            model,
            str(WHISPER_QUALCOMM),
            skip_inferencing=True,
            skip_profiling=True,
        )
        print("  Qualcomm Whisper export saved to", WHISPER_QUALCOMM)
    except ImportError as e:
        print("  qai_hub_models not installed. pip install 'qai-hub-models[whisper]'")
    except Exception as e:
        print(f"  Export failed: {e}")
        print("  Voice will still work using Hugging Face Whisper (CPU).")

    print("Done. Install for voice: pip install transformers soundfile sounddevice")


if __name__ == "__main__":
    main()
