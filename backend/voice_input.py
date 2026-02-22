"""
PHANTOM CODE — Record audio and transcribe with Whisper (Qualcomm Hub export or Hugging Face).
Local only; no cloud. Uses ONNX with QNN when Qualcomm model is present, else transformers.
"""

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
WHISPER_ONNX_DIR = os.path.join(_PROJECT_ROOT, "models", "whisper_qualcomm")
WHISPER_CACHE = os.path.join(_PROJECT_ROOT, "models", "whisper_hf")
RECORD_SECONDS = 5

_whisper_pipeline = None


def _load_transformers_whisper():
    """Load Hugging Face Whisper pipeline (fallback when Qualcomm ONNX not used)."""
    global _whisper_pipeline
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    try:
        from transformers import pipeline
        os.makedirs(WHISPER_CACHE, exist_ok=True)
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            device=-1,
            model_kwargs={"cache_dir": WHISPER_CACHE},
        )
        return _whisper_pipeline
    except Exception as e:
        logger.warning(f"Whisper transformers load failed: {e}")
        return None


def _wav_to_array(wav_bytes: bytes):
    """Convert WAV bytes to float32 mono array and sample rate."""
    try:
        import soundfile as sf
        import io
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except ImportError:
        try:
            import wave
            import numpy as np
            import io
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                n = w.getnframes()
                buf = w.readframes(n)
                data = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                sr = w.getframerate()
            return data, sr
        except Exception as e:
            logger.warning(f"WAV read failed: {e}")
            return None, None


def _audio_bytes_to_wav(raw: bytes) -> Optional[bytes]:
    """Convert WebM/other browser recording to WAV bytes if needed. Returns WAV bytes or None."""
    if not raw or len(raw) < 12:
        return raw if raw else None
    # WebM / Matroska magic
    if raw[:4] == b"\x1aE\xdf\xa3" or (raw[:3] == b"ID3") or b"webm" in raw[:32]:
        try:
            import io
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(raw))
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"WebM/audio convert failed: {e}")
            return None
    return raw


def transcribe_audio(audio_wav_bytes: Optional[bytes]) -> str:
    """
    Transcribe WAV or WebM bytes using Whisper. Accepts browser MediaRecorder WebM.
    Uses Hugging Face transformers Whisper (openai/whisper-base.en).
    """
    if not audio_wav_bytes:
        return ""
    wav_bytes = _audio_bytes_to_wav(audio_wav_bytes) or audio_wav_bytes
    data, sr = _wav_to_array(wav_bytes)
    if data is None or sr is None:
        return ""

    # Transformers Whisper (openai/whisper-base.en). For Qualcomm NPU, run scripts/setup_whisper_qualcomm.py
    # and place encoder/decoder ONNX in models/whisper_qualcomm/ for future use.
    pipe = _load_transformers_whisper()
    if pipe is None:
        return "[Voice: Whisper not available. Install: pip install transformers soundfile]"
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_wav_bytes)
            path = f.name
        try:
            out = pipe(path, generate_kwargs={"max_new_tokens": 200})
            text = (out.get("text") or "").strip()
            return text if text else "[No speech detected]"
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Whisper inference failed: {e}")
        return ""


def record_audio(seconds: float = RECORD_SECONDS) -> Optional[bytes]:
    """Record from default microphone for `seconds`. Returns WAV bytes or None."""
    try:
        import sounddevice as sd
        import numpy as np
        import io
        import wave
        sample_rate = 16000
        samples = int(seconds * sample_rate)
        rec = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes((rec * 32767).astype(np.int16).tobytes())
        return buf.getvalue()
    except ImportError:
        logger.warning("sounddevice not installed. pip install sounddevice")
        return None
    except Exception as e:
        logger.warning(f"Recording failed: {e}")
        return None


def record_and_transcribe(seconds: float = RECORD_SECONDS) -> str:
    """Record for `seconds` and return transcribed text. Safe fallback returns empty string."""
    wav = record_audio(seconds)
    return transcribe_audio(wav)
