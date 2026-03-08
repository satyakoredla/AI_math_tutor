"""
agents/explainer_agent.py
Explainer Agent — generates a teacher-style step-by-step explanation.
"""
from __future__ import annotations
import os
import sys
import re
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import GEMINI_API_KEY, GEMINI_MODEL

EXPLAINER_PROMPT = """\
You are a friendly and encouraging JEE mathematics teacher. Explain the following solved problem
to a student in a clear, engaging, step-by-step manner.

PROBLEM:
{problem_text}

TOPIC: {topic}

SOLUTION STEPS:
{solution}

FINAL ANSWER: {final_answer}

RELEVANT FORMULAS USED:
{formulas}

Write an explanation that:
1. Starts with a 1-2 sentence overview of the approach
2. Lists each step clearly (Step 1:, Step 2:, etc.)
3. Explains WHY each step is taken (not just what)
4. References the specific formula used
5. Gives a "Key Insight" or "Shortcut" tip at the end
6. Is encouraging and supportive in tone

Keep the explanation thorough but concise. Use plain text with minimal LaTeX.
"""


def run_explainer_agent(
    parsed_problem: Dict[str, Any],
    solver_result: Dict[str, Any],
    verifier_result: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a detailed teacher-style explanation of the solution.

    Returns
    -------
    dict with explanation, key_insight, and difficulty assessment
    """
    problem_text = parsed_problem.get("problem_text", "")
    topic = parsed_problem.get("topic", "algebra")
    solution = solver_result.get("solution", "")
    final_answer = verifier_result.get("verified_answer", solver_result.get("final_answer", ""))
    steps = solver_result.get("steps", [])

    # Pull key formula references
    formulas = _extract_formulas(retrieved_docs)

    # ── LLM explanation ───────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        result = _explain_with_llm(problem_text, topic, solution, final_answer, formulas)
        if result:
            return result

    # ── Template-based explanation ────────────────────────────────────────────
    return _explain_with_template(problem_text, topic, steps, final_answer, formulas)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Explainer
# ─────────────────────────────────────────────────────────────────────────────

def _explain_with_llm(
    problem_text: str, topic: str, solution: str,
    final_answer: str, formulas: str,
) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai
        import sys
        import importlib.util
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = EXPLAINER_PROMPT.format(
            problem_text=problem_text,
            topic=topic,
            solution=solution[:1500],
            final_answer=final_answer,
            formulas=formulas[:600],
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3}
        )
        explanation = response.text.strip()
        key_insight = _extract_key_insight(explanation)
        difficulty = _assess_difficulty(topic, problem_text)

        return {
            "agent": "ExplainerAgent",
            "explanation": explanation,
            "key_insight": key_insight,
            "difficulty": difficulty,
            "topic": topic,
            "formulas_used": formulas,
            "method": "llm",
        }
    except Exception as e:
        print(f"[ExplainerAgent] Gemini explanation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Template-based Explainer (offline fallback)
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_TIPS = {
    "calculus": "💡 Key Insight: When differentiating, always apply chain rule for composite functions: d/dx[f(g(x))] = f'(g(x)) · g'(x)",
    "trigonometry": "💡 Key Insight: Always remember sin²θ + cos²θ = 1. This identity is the foundation of most trig simplifications.",
    "algebra": "💡 Key Insight: When factoring, look for common factors first, then try special identities like (a²-b²) = (a+b)(a-b).",
    "quadratic_equations": "💡 Key Insight: The discriminant D = b²-4ac tells you everything about the nature of roots before solving!",
    "matrices": "💡 Key Insight: Always check if a matrix is singular (det = 0) before attempting to find its inverse.",
    "probability": "💡 Key Insight: For conditional probability, Bayes' theorem P(A|B) = P(B|A)·P(A)/P(B) is very powerful.",
    "sequences": "💡 Key Insight: In AP/GP, always identify the first term and common difference/ratio first.",
    "complex_numbers": "💡 Key Insight: Convert to polar form r·e^(iθ) for multiplication/powers — much simpler!",
    "vectors": "💡 Key Insight: If a·b = 0, vectors are perpendicular. If a×b = 0, they are parallel.",
    "limits": "💡 Key Insight: lim(x→0) sin(x)/x = 1 is one of the most important standard limits in JEE!",
    "coordinate_geometry": "💡 Key Insight: Always find centre and radius first for circle problems — it simplifies everything.",
}


def _explain_with_template(
    problem_text: str, topic: str,
    steps: List[str], final_answer: str, formulas: str,
) -> Dict[str, Any]:
    """Build a structured explanation without LLM."""
    topic_clean = topic.replace("_", " ").title()
    tip = _TOPIC_TIPS.get(topic, "💡 Key Insight: Practice is the key to mastering JEE math!")

    step_text = "\n".join(
        f"**Step {i+1}:** {s}" for i, s in enumerate(steps)
    ) if steps else "See solution above for detailed steps."

    explanation = (
        f"## 📚 Topic: {topic_clean}\n\n"
        f"**Overview:**\nThis problem involves {topic_clean.lower()}. "
        f"We will apply the relevant formulas step by step.\n\n"
        f"**Relevant Formulas:**\n{formulas[:400]}\n\n"
        f"**Step-by-Step Walkthrough:**\n{step_text}\n\n"
        f"**✅ Final Answer:** {final_answer}\n\n"
        f"{tip}"
    )

    return {
        "agent": "ExplainerAgent",
        "explanation": explanation,
        "key_insight": tip,
        "difficulty": _assess_difficulty(topic, problem_text),
        "topic": topic,
        "formulas_used": formulas,
        "method": "template",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_formulas(docs: List[Dict[str, Any]]) -> str:
    """Pull first 200 chars from each retrieved doc as formula references."""
    parts = []
    for doc in docs[:2]:
        lines = doc.get("content", "").split("\n")
        formula_lines = [l for l in lines if re.search(r"[=+\-/^√∫∑]", l)][:4]
        parts.append("\n".join(formula_lines))
    return "\n".join(parts) or "See knowledge base for formulas."


def _extract_key_insight(explanation: str) -> str:
    """Extract 'Key Insight' tip from LLM explanation."""
    match = re.search(r"[Kk]ey [Ii]nsight[:\s]+(.+?)(?:\n|$)", explanation)
    if match:
        return "💡 " + match.group(1).strip()
    match = re.search(r"[Ss]hortcut[:\s]+(.+?)(?:\n|$)", explanation)
    if match:
        return "⚡ " + match.group(1).strip()
    return "💡 Practice similar problems to build speed and confidence for JEE!"


def _assess_difficulty(topic: str, problem_text: str) -> str:
    """Rough difficulty classification."""
    hard_topics = {"differential_equations", "complex_numbers", "vectors", "matrices"}
    medium_topics = {"calculus", "trigonometry", "probability", "coordinate_geometry"}

    if topic in hard_topics:
        return "⭐⭐⭐ Hard"
    elif topic in medium_topics:
        return "⭐⭐ Medium"
    else:
        return "⭐ Easy"
