"""
agents/parser_agent.py
Parser Agent — takes raw math text and returns structured problem JSON.
"""
from __future__ import annotations
import re
import json
import os
import sys
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import GEMINI_API_KEY, GEMINI_MODEL


PARSER_PROMPT_TEMPLATE = """\
You are a JEE mathematics parser. Given a raw math question, extract structured information.

Raw question:
{raw_text}

Return ONLY a valid JSON object with these exact fields:
{{
  "problem_text": "cleaned, well-formatted problem statement",
  "topic": "one of: algebra, calculus, trigonometry, probability, statistics, matrices, limits, coordinate_geometry, complex_numbers, sequences, quadratic_equations, binomial_theorem, permutations_combinations, vectors, differential_equations, arithmetic",
  "variables": ["list", "of", "variables", "used"],
  "constraints": ["any constraints like domain, range, n>0, etc"],
  "needs_clarification": false,
  "clarification_reason": ""
}}

If the text is unclear or not a math problem, set needs_clarification to true and explain in clarification_reason.
Return ONLY the JSON, no other text.
"""


def run_parser_agent(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw math input into structured problem representation.

    Parameters
    ----------
    raw_text : str  — raw text from OCR / speech / keyboard

    Returns
    -------
    dict with parsed fields + agent metadata
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return _build_result(
            problem_text="",
            topic="unknown",
            variables=[],
            constraints=[],
            needs_clarification=True,
            clarification_reason="Empty input received.",
            method="none",
        )

    # ── Try LLM first ────────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        result = _parse_with_llm(raw_text)
        if result:
            return result

    # ── Rule-based fallback ───────────────────────────────────────────────────
    return _parse_with_rules(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_with_llm(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = PARSER_PROMPT_TEMPLATE.format(raw_text=raw_text)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0}
        )
        raw_json = response.text.strip()
        # Strip markdown code fences if present
        raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
        raw_json = re.sub(r"\s*```$", "", raw_json)
        parsed = json.loads(raw_json)
        parsed["method"] = "llm"
        return parsed
    except Exception as e:
        print(f"[ParserAgent] Gemini generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based Parser (offline fallback)
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_PATTERNS = {
    "calculus": r"\b(deriv|differenti|integr|d/dx|limit|continuity|chain rule|product rule)\b",
    "trigonometry": r"\b(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|trig)\b",
    "algebra": r"\b(factor|expand|simplify|polynomial|log|logarithm|exponent|identit)\b",
    "quadratic_equations": r"\b(quadratic|roots?|discriminant|ax\^2|ax2)\b",
    "matrices": r"\b(matrix|matrices|determinant|eigenvalue|transpose|inverse)\b",
    "probability": r"\b(probability|chance|random|event|sample space|dice|coin)\b",
    "sequences": r"\b(sequence|series|ap|gp|arithmetic|geometric|sum of|progression)\b",
    "complex_numbers": r"\b(complex|imaginary|real part|im part|modulus|argument|argand)\b",
    "vectors": r"\b(vector|dot product|cross product|scalar|magnitude|unit vector)\b",
    "coordinate_geometry": r"\b(distance|midpoint|slope|circle|parabola|ellipse|hyperbola|line equation)\b",
    "limits": r"\b(limit|lim|l'h[oô]pital|approaches|tends to)\b",
    "differential_equations": r"\b(ode|differential equation|dy/dx|d\^2y|order of equation)\b",
    "binomial_theorem": r"\b(binomial|ncr|nCr|coefficient of|general term|pascal)\b",
    "permutations_combinations": r"\b(permutation|combination|factorial|arrange|select)\b",
}

_VAR_PATTERN = re.compile(r"\b([a-zA-Z])\b(?!\s*[=<>])")


def _parse_with_rules(raw_text: str) -> Dict[str, Any]:
    text_lower = raw_text.lower()

    # Detect topic
    topic = "algebra"
    for t, pat in _TOPIC_PATTERNS.items():
        if re.search(pat, text_lower):
            topic = t
            break

    # Extract variables
    vars_found = list(dict.fromkeys(
        v for v in _VAR_PATTERN.findall(raw_text)
        if v not in ("I", "a", "i") and len(v) == 1
    ))[:5]

    # Clean problem text
    problem_text = raw_text.strip()
    if not problem_text.endswith("?"):
        problem_text = problem_text.rstrip(".") + "."

    needs_clarification = len(problem_text) < 5

    return _build_result(
        problem_text=problem_text,
        topic=topic,
        variables=vars_found,
        constraints=[],
        needs_clarification=needs_clarification,
        clarification_reason="Problem text is too short or unclear." if needs_clarification else "",
        method="rules",
    )


def _build_result(**kwargs) -> Dict[str, Any]:
    return {
        "problem_text": kwargs.get("problem_text", ""),
        "topic": kwargs.get("topic", "unknown"),
        "variables": kwargs.get("variables", []),
        "constraints": kwargs.get("constraints", []),
        "needs_clarification": kwargs.get("needs_clarification", False),
        "clarification_reason": kwargs.get("clarification_reason", ""),
        "agent": "ParserAgent",
        "method": kwargs.get("method", "rules"),
    }
