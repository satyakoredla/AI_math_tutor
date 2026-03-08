"""
agents/router_agent.py
Intent Router Agent — classifies math topic and selects retrieval strategy.
"""
from __future__ import annotations
import os
import sys
import re
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MATH_TOPICS


ROUTING_TABLE = {
    "calculus": {
        "subtopics": ["differentiation", "integration", "limits", "series expansion"],
        "primary_docs": ["calculus_derivatives.txt", "calculus_integrals.txt", "limits_continuity.txt"],
        "solver_mode": "symbolic",
    },
    "trigonometry": {
        "subtopics": ["identities", "equations", "inverse trig", "heights and distances"],
        "primary_docs": ["trigonometry.txt"],
        "solver_mode": "symbolic",
    },
    "algebra": {
        "subtopics": ["identities", "polynomials", "logarithms", "AM-GM"],
        "primary_docs": ["algebra_formulas.txt", "quadratic_equations.txt"],
        "solver_mode": "symbolic",
    },
    "quadratic_equations": {
        "subtopics": ["roots", "discriminant", "nature of roots", "Vieta"],
        "primary_docs": ["quadratic_equations.txt", "algebra_formulas.txt"],
        "solver_mode": "symbolic",
    },
    "matrices": {
        "subtopics": ["determinants", "inverse", "rank", "Cramer's rule"],
        "primary_docs": ["matrices_determinants.txt"],
        "solver_mode": "numerical",
    },
    "probability": {
        "subtopics": ["basic probability", "Bayes", "distributions", "expectation"],
        "primary_docs": ["probability_statistics.txt", "permutations_combinations.txt"],
        "solver_mode": "analytical",
    },
    "sequences": {
        "subtopics": ["AP", "GP", "HP", "sum formulas"],
        "primary_docs": ["sequences_series.txt"],
        "solver_mode": "analytical",
    },
    "complex_numbers": {
        "subtopics": ["modulus", "argument", "De Moivre", "locus"],
        "primary_docs": ["complex_numbers.txt"],
        "solver_mode": "symbolic",
    },
    "vectors": {
        "subtopics": ["dot product", "cross product", "3D geometry", "planes"],
        "primary_docs": ["vectors_3d.txt"],
        "solver_mode": "analytical",
    },
    "coordinate_geometry": {
        "subtopics": ["lines", "circles", "conics", "distance"],
        "primary_docs": ["coordinate_geometry.txt"],
        "solver_mode": "analytical",
    },
    "limits": {
        "subtopics": ["standard limits", "L'Hopital", "continuity"],
        "primary_docs": ["limits_continuity.txt", "calculus_derivatives.txt"],
        "solver_mode": "symbolic",
    },
    "differential_equations": {
        "subtopics": ["separable", "linear", "exact", "Bernoulli"],
        "primary_docs": ["differential_equations.txt", "calculus_integrals.txt"],
        "solver_mode": "symbolic",
    },
    "binomial_theorem": {
        "subtopics": ["general term", "coefficient", "middle term"],
        "primary_docs": ["binomial_theorem.txt"],
        "solver_mode": "analytical",
    },
    "permutations_combinations": {
        "subtopics": ["permutations", "combinations", "derangements"],
        "primary_docs": ["permutations_combinations.txt"],
        "solver_mode": "analytical",
    },
}


def run_router_agent(parsed_problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route the parsed problem to the appropriate solver strategy.

    Parameters
    ----------
    parsed_problem : dict  — output of ParserAgent

    Returns
    -------
    dict with routing information
    """
    topic = parsed_problem.get("topic", "algebra").lower()

    # Normalise topic
    if topic not in ROUTING_TABLE:
        topic = _fuzzy_match_topic(topic)

    route = ROUTING_TABLE.get(topic, ROUTING_TABLE["algebra"])

    return {
        "agent": "RouterAgent",
        "detected_topic": topic,
        "subtopics": route["subtopics"],
        "primary_docs": route["primary_docs"],
        "solver_mode": route["solver_mode"],
        "rag_query": _build_rag_query(parsed_problem, topic),
        "confidence": 0.9 if topic in ROUTING_TABLE else 0.5,
    }


def _build_rag_query(parsed: Dict[str, Any], topic: str) -> str:
    """Build a targeted RAG query from problem context."""
    parts = [topic]
    problem_text = parsed.get("problem_text", "")
    if problem_text:
        # Extract key math expressions (first 120 chars)
        parts.append(problem_text[:120])
    return " ".join(parts)


def _fuzzy_match_topic(topic: str) -> str:
    """Best-effort topic matching for partial / variant topic names."""
    topic_lower = topic.lower()
    matches = {
        "deriv": "calculus", "integr": "calculus", "differen": "calculus",
        "trig": "trigonometry", "sin ": "trigonometry", "cos ": "trigonometry",
        "matri": "matrices", "determin": "matrices",
        "prob": "probability", "statistic": "probability",
        "sequ": "sequences", "series": "sequences", "progres": "sequences",
        "complex": "complex_numbers", "imagin": "complex_numbers",
        "vector": "vectors", "dot prod": "vectors",
        "circle": "coordinate_geometry", "conic": "coordinate_geometry",
        "limit": "limits", "continui": "limits",
        "binomi": "binomial_theorem",
        "permut": "permutations_combinations", "combin": "permutations_combinations",
        "quadrati": "quadratic_equations",
        "differ eq": "differential_equations",
    }
    for key, mapped in matches.items():
        if key in topic_lower:
            return mapped
    return "algebra"
