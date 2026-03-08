"""
rag/retriever.py
Provides the RAG retrieval interface used by the Solver Agent.
"""
from __future__ import annotations
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import CHROMA_DB_DIR, RAG_TOP_K, GEMINI_API_KEY

_vectorstore = None


def _get_vectorstore():
    """Lazy-load the Chroma vector store, building it if missing."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from rag.create_embeddings import build_vector_store, _get_embeddings
    if os.path.exists(CHROMA_DB_DIR):
        from langchain_community.vectorstores import Chroma
        embeddings = _get_embeddings()
        _vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        _vectorstore = build_vector_store(force_rebuild=False)
    return _vectorstore


def get_relevant_docs(query: str, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Parameters
    ----------
    query : str
        The math problem or topic query.
    k : int
        Number of documents to retrieve.

    Returns
    -------
    list of dicts, each with:
        content  – text of the chunk
        source   – filename of the source document
        score    – similarity score (higher = more relevant)
    """
    try:
        store = _get_vectorstore()
        results_with_scores = store.similarity_search_with_score(query, k=k)

        docs = []
        for doc, score in results_with_scores:
            docs.append({
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "score": float(score),
            })
        return docs

    except Exception as e:
        # Return keyword-based fallback if vector store unavailable
        return _keyword_fallback(query, k, str(e))


def format_docs_for_prompt(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved docs into a string for LLM prompt context."""
    if not docs:
        return "No relevant formulas found."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Reference {i} — {doc['source']}]\n{doc['content']}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based fallback (works entirely offline)
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_KEYWORDS = {
    "calculus_derivatives.txt": ["derivative", "differentiat", "d/dx", "slope", "tangent"],
    "calculus_integrals.txt": ["integral", "integrat", "antiderivative", "area under"],
    "algebra_formulas.txt": ["algebra", "quadratic", "factor", "expand", "polynomial", "log", "identit"],
    "trigonometry.txt": ["sin", "cos", "tan", "trig", "angle", "radian", "degree"],
    "limits_continuity.txt": ["limit", "continuity", "lim", "approach", "l'hopital"],
    "probability_statistics.txt": ["probability", "permutation", "combination", "statistic", "random"],
    "matrices_determinants.txt": ["matrix", "matrices", "determinant", "inverse", "eigenvalue"],
    "coordinate_geometry.txt": ["parabola", "ellipse", "hyperbola", "circle", "line", "distance", "slope"],
    "complex_numbers.txt": ["complex", "imaginary", "real part", "modulus", "argument", "euler"],
    "sequences_series.txt": ["sequence", "series", "ap", "gp", "arithmetic", "geometric", "sum"],
    "quadratic_equations.txt": ["quadratic", "roots", "discriminant", "vieta"],
    "binomial_theorem.txt": ["binomial", "expansion", "coefficient", "pascal", "nCr"],
    "permutations_combinations.txt": ["permutation", "combination", "factorial", "arrangement"],
    "vectors_3d.txt": ["vector", "dot product", "cross product", "3d", "plane", "line in space"],
    "differential_equations.txt": ["differential equation", "ode", "dy/dx", "order", "degree"],
}


def _keyword_fallback(query: str, k: int, err_msg: str) -> List[Dict[str, Any]]:
    """Return docs based on keyword matching when vector store is unavailable."""
    query_lower = query.lower()
    scores: Dict[str, int] = {}

    for filename, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[filename] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "math_docs")
    results = []
    for filename, score in ranked:
        path = os.path.join(docs_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            results.append({
                "content": content[:800],   # first 800 chars
                "source": filename,
                "score": float(score),
            })

    if not results:
        # Generic algebra fallback
        path = os.path.join(docs_dir, "algebra_formulas.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                results.append({"content": f.read()[:800], "source": "algebra_formulas.txt", "score": 0.1})

    return results
