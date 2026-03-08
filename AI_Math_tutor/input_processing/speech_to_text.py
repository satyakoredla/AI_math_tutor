"""
input_processing/speech_to_text.py
Transcribes audio files to text using OpenAI Whisper.
Falls back to openai.Audio API if local whisper is unavailable.
"""
from __future__ import annotations
import os
from typing import Dict, Any

# ── Local Whisper (optional) ─────────────────────────────────────────────────
try:
    import whisper as _local_whisper
    _LOCAL_WHISPER_AVAILABLE = True
except ImportError:
    _local_whisper = None
    _LOCAL_WHISPER_AVAILABLE = False

# Lazy-load model
_whisper_model = None


def _get_local_model(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        if not _LOCAL_WHISPER_AVAILABLE:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
        _whisper_model = _local_whisper.load_model(model_name)
    return _whisper_model


def transcribe_audio(audio_path: str, model_name: str = "base") -> Dict[str, Any]:
    """
    Transcribe an audio file to text.

    Parameters
    ----------
    audio_path : str
        Absolute path to .wav / .mp3 / .m4a / .ogg file.
    model_name : str
        Whisper model size: tiny | base | small | medium | large

    Returns
    -------
    dict with keys:
        text        – transcribed text (str)
        confidence  – estimated confidence [0.0 – 1.0] (float)
        language    – detected language (str)
        error       – error message if failed (str | None)
    """
    if not os.path.isfile(audio_path):
        return _error_result(f"File not found: {audio_path}")

    # --- Try local whisper ---
    if _LOCAL_WHISPER_AVAILABLE:
        return _transcribe_local(audio_path, model_name)

    # --- Try OpenAI API whisper ---
    return _transcribe_openai_api(audio_path)


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".wav", model_name: str = "base") -> Dict[str, Any]:
    """
    Transcribe audio from raw bytes (e.g. from st.file_uploader).
    """
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = transcribe_audio(tmp_path, model_name)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return result

    except Exception as exc:
        return _error_result(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_local(audio_path: str, model_name: str) -> Dict[str, Any]:
    """Use locally installed openai-whisper."""
    try:
        model = _get_local_model(model_name)
        result = model.transcribe(audio_path)

        text = result.get("text", "").strip()
        language = result.get("language", "en")

        # Whisper doesn't expose a single confidence score per file;
        # use segment-level log probabilities as a proxy.
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            # map logprob [-inf, 0] → [0, 1]
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob / 5.0))
        else:
            confidence = 0.8 if text else 0.0

        return {"text": text, "confidence": confidence, "language": language, "error": None}

    except Exception as exc:
        return _error_result(str(exc))


def _transcribe_openai_api(audio_path: str) -> Dict[str, Any]:
    """Use OpenAI Whisper API (requires OPENAI_API_KEY)."""
    try:
        from openai import OpenAI
        client = OpenAI()

        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json",
            )

        text = response.text.strip() if hasattr(response, "text") else ""
        return {"text": text, "confidence": 0.9, "language": "en", "error": None}

    except Exception as exc:
        return _error_result(f"OpenAI API transcription failed: {exc}")


def _error_result(message: str) -> Dict[str, Any]:
    return {"text": "", "confidence": 0.0, "language": "unknown", "error": message}
