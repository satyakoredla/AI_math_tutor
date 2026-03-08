"""
input_processing/image_ocr.py
Extracts text from uploaded math problem images using EasyOCR.
Falls back gracefully if EasyOCR is not installed.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Any

# ── EasyOCR (optional) ──────────────────────────────────────────────────────
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

# ── Pillow ───────────────────────────────────────────────────────────────────
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Lazy-load reader to avoid long startup time
_reader = None

def _get_reader():
    """Initialise EasyOCR reader once."""
    global _reader
    if _reader is None:
        if not _EASYOCR_AVAILABLE:
            raise ImportError("easyocr is not installed. Run: pip install easyocr")
        _reader = easyocr.Reader(['en'], verbose=False)
    return _reader


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract text from an image file using EasyOCR.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file.

    Returns
    -------
    dict with keys:
        text        – extracted text (str)
        confidence  – average confidence [0.0 – 1.0] (float)
        raw_results – list of (bbox, text, conf) tuples
        error       – error message if extraction failed (str | None)
    """
    if not os.path.isfile(image_path):
        return _error_result(f"File not found: {image_path}")

    if not _EASYOCR_AVAILABLE:
        return _error_result(
            "easyocr is not installed. Install it with: pip install easyocr"
        )

    try:
        reader = _get_reader()
        raw = reader.readtext(image_path)

        if not raw:
            return {
                "text": "",
                "confidence": 0.0,
                "raw_results": [],
                "error": "No text detected in image.",
            }

        texts = [item[1] for item in raw]
        confidences = [float(item[2]) for item in raw]

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return {
            "text": combined_text,
            "confidence": avg_confidence,
            "raw_results": raw,
            "error": None,
        }

    except Exception as exc:
        return _error_result(str(exc))


def extract_text_from_pil_image(pil_image) -> Dict[str, Any]:
    """
    Extract text from a PIL Image object (e.g. from st.file_uploader).
    Saves to a temp file and calls extract_text_from_image.
    """
    import tempfile
    if not _PIL_AVAILABLE:
        return _error_result("Pillow is not installed.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name

        result = extract_text_from_image(tmp_path)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return result

    except Exception as exc:
        return _error_result(str(exc))


def _error_result(message: str) -> Dict[str, Any]:
    return {
        "text": "",
        "confidence": 0.0,
        "raw_results": [],
        "error": message,
    }
