"""
agents/verifier_agent.py
Verifier Agent — checks if the solver's answer is mathematically correct.
Triggers HITL when confidence is low.
"""
from __future__ import annotations
import os
import sys
import re
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import GEMINI_API_KEY, GEMINI_MODEL, VERIFIER_CONFIDENCE_THRESHOLD

VERIFIER_PROMPT = """\
You are a strict JEE mathematics verifier. Your job is to check if the given solution is correct.

ORIGINAL PROBLEM:
{problem_text}

PROPOSED SOLUTION:
{solution}

FINAL ANSWER:
{final_answer}

Please verify by:
1. Re-solving the problem independently (briefly)
2. Checking for algebraic / arithmetic errors
3. Checking domain restrictions (log, sqrt, etc.)
4. Checking units if applicable

Respond with ONLY a valid JSON object:
{{
  "is_correct": true or false,
  "confidence": 0.0 to 1.0,
  "verified_answer": "the correct answer",
  "issues": ["list of issues found, or empty if none"],
  "verification_steps": "brief re-derivation or check",
  "needs_human_review": true or false
}}
"""


def run_verifier_agent(
    parsed_problem: Dict[str, Any],
    solver_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Verify the solver's answer.

    Returns
    -------
    dict with verification result and HITL flag
    """
    problem_text = parsed_problem.get("problem_text", "")
    solution = solver_result.get("solution", "")
    final_answer = solver_result.get("final_answer", "")
    solver_confidence = solver_result.get("confidence", 0.5)
    solver_method = solver_result.get("method_used", "unknown")

    # ── SymPy solutions can be auto-verified ─────────────────────────────────
    if solver_method == "sympy":
        return _sympy_verified_result(final_answer, solver_confidence)

    # ── LLM verification ─────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        result = _verify_with_llm(problem_text, solution, final_answer)
        if result:
            return result

    # ── Heuristic verification ────────────────────────────────────────────────
    return _heuristic_verify(problem_text, solution, final_answer, solver_confidence)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Verifier
# ─────────────────────────────────────────────────────────────────────────────

def _verify_with_llm(problem_text: str, solution: str, final_answer: str) -> Optional[Dict[str, Any]]:
    try:
        import json
        import google.generativeai as genai
        import sys
        import importlib.util

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = VERIFIER_PROMPT.format(
            problem_text=problem_text,
            solution=solution[:2000],
            final_answer=final_answer,
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0}
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        confidence = float(parsed.get("confidence", 0.8))
        needs_review = not parsed.get("is_correct", True) or confidence < VERIFIER_CONFIDENCE_THRESHOLD

        return {
            "agent": "VerifierAgent",
            "is_correct": parsed.get("is_correct", True),
            "confidence": confidence,
            "verified_answer": parsed.get("verified_answer", final_answer),
            "issues": parsed.get("issues", []),
            "verification_steps": parsed.get("verification_steps", ""),
            "needs_human_review": needs_review,
            "method": "llm",
        }
    except Exception as e:
        print(f"[VerifierAgent] Gemini verification failed: {e}")
        return None


def _sympy_verified_result(final_answer: str, confidence: float) -> Dict[str, Any]:
    """SymPy solutions are algorithmically correct — high confidence."""
    return {
        "agent": "VerifierAgent",
        "is_correct": True,
        "confidence": confidence,
        "verified_answer": final_answer,
        "issues": [],
        "verification_steps": "Solved using SymPy symbolic computation — mathematically exact.",
        "needs_human_review": False,
        "method": "sympy_auto",
    }


def _heuristic_verify(
    problem_text: str,
    solution: str,
    final_answer: str,
    solver_confidence: float,
) -> Dict[str, Any]:
    """Rule-based heuristic verification."""
    issues = []

    # Check for obvious error markers in solution text (whole-word match only)
    error_patterns = [
        (r"\berror\b", "error"),
        (r"\bundefined\b", "undefined"),
        (r"\binvalid\b", "invalid"),
        (r"\bdivision by zero\b", "division by zero"),
        (r"\bnan\b", "NaN value"),          # whole-word only — avoids 'cannot', 'plan'
        (r"\binfinity\b", "infinity result"),
    ]
    for pat, label in error_patterns:
        if re.search(pat, solution.lower()):
            issues.append(f"Potential issue detected: '{label}' found in solution")

    # Check domain issues
    if re.search(r"sqrt\s*\(?\s*-", final_answer):
        issues.append("Square root of negative number — check domain")
    if re.search(r"log\s*\(?\s*0", final_answer):
        issues.append("Logarithm of zero is undefined")
    if re.search(r"1/0|/\s*0\b", final_answer):
        issues.append("Division by zero detected")

    confidence = solver_confidence * (0.7 if issues else 1.0)
    needs_review = confidence < VERIFIER_CONFIDENCE_THRESHOLD or bool(issues)

    return {
        "agent": "VerifierAgent",
        "is_correct": len(issues) == 0,
        "confidence": confidence,
        "verified_answer": final_answer,
        "issues": issues,
        "verification_steps": "Heuristic check: scanned for common mathematical errors.",
        "needs_human_review": needs_review,
        "method": "heuristic",
    }
