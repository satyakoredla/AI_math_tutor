"""
utils/config.py
Centralized configuration for MathMentor AI.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────── API ─────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
print("GEMINI_API_KEY", GEMINI_API_KEY)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# ──────────────────────── Paths ───────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_DOCS_DIR = os.path.join(BASE_DIR, "data", "math_docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
MEMORY_DB_PATH = os.path.join(BASE_DIR, "data", "memory.db")

# ─────────────────────── RAG ───────────────────────────
RAG_TOP_K: int = 3
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ──────────────────── Thresholds ───────────────────────
OCR_CONFIDENCE_THRESHOLD: float = 0.6      # below this → trigger HITL
VERIFIER_CONFIDENCE_THRESHOLD: float = 0.7  # below this → trigger HITL

# ─────────────────── Math Topics ──────────────────────
MATH_TOPICS = [
    "algebra",
    "calculus",
    "trigonometry",
    "probability",
    "statistics",
    "matrices",
    "determinants",
    "limits",
    "continuity",
    "coordinate_geometry",
    "complex_numbers",
    "sequences",
    "series",
    "quadratic_equations",
    "binomial_theorem",
    "permutations",
    "combinations",
    "vectors",
    "differential_equations",
    "integration",
    "differentiation",
]
