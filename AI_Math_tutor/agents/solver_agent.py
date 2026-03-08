"""
agents/solver_agent.py
Solver Agent — uses RAG context + LLM (or sympy fallback) to solve math problems.
"""
from __future__ import annotations
import os
import sys
import re
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import GEMINI_API_KEY, GEMINI_MODEL
# print("GEMINI_API_KEY", GEMINI_API_KEY)
# print("GEMINI_MODEL", GEMINI_MODEL)

SOLVER_PROMPT = """\
You are an expert JEE mathematics tutor. Solve the following math problem step-by-step.

PROBLEM:
{problem_text}

TOPIC: {topic}

RELEVANT FORMULAS AND REFERENCES (from knowledge base):
{references}

Instructions:
1. Identify the relevant formula/method from the references above.
2. Show every algebraic step clearly.
3. Box or clearly state the FINAL ANSWER at the end.
4. If the problem has multiple parts, solve each part separately.
5. Use standard mathematical notation.

Provide a clear, structured solution:
"""


def run_solver_agent(
    parsed_problem: Dict[str, Any],
    route: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Solve the math problem using RAG context and LLM / symbolic solver.

    Returns
    -------
    dict with:
        solution        – full solution text
        final_answer    – extracted final answer
        steps           – list of solution steps
        method_used     – "llm" | "sympy" | "fallback"
        confidence      – estimated confidence [0..1]
    """
    problem_text = parsed_problem.get("problem_text", "")
    topic = parsed_problem.get("topic", "algebra")

    from rag.retriever import format_docs_for_prompt
    references = format_docs_for_prompt(retrieved_docs)

    # ── Try LLM ──────────────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        result = _solve_with_llm(problem_text, topic, references)
        if result:
            return result

    # ── Try SymPy symbolic solver ─────────────────────────────────────────────
    sympy_result = _solve_with_sympy(problem_text, topic)
    if sympy_result:
        return sympy_result

    # ── Fallback: rule-based ──────────────────────────────────────────────────
    return _solve_fallback(problem_text, topic, references)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_with_llm(problem_text: str, topic: str, references: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai
        import sys
        import importlib.util
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = SOLVER_PROMPT.format(
            problem_text=problem_text,
            topic=topic,
            references=references,
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1}
        )
        solution = response.text.strip()
        steps = _extract_steps(solution)
        final_answer = _extract_final_answer(solution)

        return {
            "agent": "SolverAgent",
            "solution": solution,
            "final_answer": final_answer,
            "steps": steps,
            "method_used": "llm",
            "confidence": 0.92,
        }
    except Exception as e:
        # print(f"[SolverAgent] Gemini solving failed with error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SymPy Solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_with_sympy(problem_text: str, topic: str) -> Optional[Dict[str, Any]]:
    """Attempt symbolic solution using SymPy for common problem types."""
    try:
        import sympy as sp

        text = problem_text.lower()

        # ── Derivative problems ───────────────────────────────────────────────
        deriv_match = re.search(
            r"deriv(?:ative)?\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?$",
            text, re.IGNORECASE
        )
        if deriv_match or "d/dx" in text:
            expr_str = _extract_expression(problem_text)
            if expr_str:
                x = sp.Symbol('x')
                expr = sp.sympify(expr_str, locals={"x": x, "e": sp.E, "pi": sp.pi})
                derivative = sp.diff(expr, x)
                simplified = sp.simplify(derivative)
                solution = (
                    f"**Problem:** Find the derivative of {expr_str}\n\n"
                    f"**Using Power/Chain Rule:**\n"
                    f"f(x) = {sp.latex(expr)}\n\n"
                    f"**Step-by-Step:**\n"
                    f"d/dx [{sp.latex(expr)}] = {sp.latex(simplified)}\n\n"
                    f"**Final Answer:** f'(x) = {simplified}"
                )
                return {
                    "agent": "SolverAgent",
                    "solution": solution,
                    "final_answer": str(simplified),
                    "steps": [
                        f"Identify the expression: f(x) = {expr}",
                        f"Apply differentiation rules",
                        f"Result: f'(x) = {simplified}",
                    ],
                    "method_used": "sympy",
                    "confidence": 0.95,
                }

        # ── Linear equation solving ──────────────────────────────────────────
        eq_match = re.search(r"(\d*x\s*[+\-]\s*\d+\s*=\s*\d+)", problem_text, re.IGNORECASE)
        if eq_match or ("solve" in text and "=" in problem_text):
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            transformations = (standard_transformations + (implicit_multiplication_application,))
            
            eq_str = eq_match.group(1) if eq_match else problem_text
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=")
                try:
                    x = sp.Symbol('x')
                    lhs_sym = parse_expr(lhs.strip(), transformations=transformations)
                    rhs_sym = parse_expr(rhs.strip(), transformations=transformations)
                    eq = sp.Eq(lhs_sym, rhs_sym)
                    solution_vals = sp.solve(eq, x)
                    sol_text = (
                        f"**Problem:** Solve {eq_str}\n\n"
                        f"**Steps:**\n"
                        f"rearrange: {lhs.strip()} - ({rhs.strip()}) = 0\n"
                        f"Solve for x\n\n"
                        f"**Final Answer:** x = {solution_vals}"
                    )
                    return {
                        "agent": "SolverAgent",
                        "solution": sol_text,
                        "final_answer": f"x = {solution_vals}",
                        "steps": [
                            f"Write equation: {eq}",
                            f"Solve algebraically",
                            f"x = {solution_vals}",
                        ],
                        "method_used": "sympy",
                        "confidence": 0.95,
                    }
                except Exception:
                    pass

        # ── Integrate problems ───────────────────────────────────────────────
        int_match = re.search(r"integr(?:al|ate)?\s+(?:of\s+)?(.+)", text, re.IGNORECASE)
        if int_match:
            expr_str = _extract_expression(problem_text)
            if expr_str:
                x = sp.Symbol('x')
                expr = sp.sympify(expr_str, locals={"x": x})
                integral = sp.integrate(expr, x)
                sol_text = (
                    f"**Problem:** Integrate {expr_str}\n\n"
                    f"∫ {sp.latex(expr)} dx = {sp.latex(integral)} + C\n\n"
                    f"**Final Answer:** {integral} + C"
                )
                return {
                    "agent": "SolverAgent",
                    "solution": sol_text,
                    "final_answer": f"{integral} + C",
                    "steps": [
                        f"Expression: {expr}",
                        f"Apply integration rule",
                        f"Result: {integral} + C",
                    ],
                    "method_used": "sympy",
                    "confidence": 0.95,
                }

        # ── Multi-variable expression evaluation ─────────────────────────────
        # Handles: "evaluate x/y + y/z for x=2, y=-1, z=3"
        val_match = re.search(
            r"for\s+([a-z])\s*=\s*(-?\d+(?:\.\d+)?)[,\s]+([a-z])\s*=\s*(-?\d+(?:\.\d+)?)(?:[,\s]+([a-z])\s*=\s*(-?\d+(?:\.\d+)?))?",
            problem_text, re.IGNORECASE
        )
        if val_match or re.search(r"(?:evaluate|find|calculate).+for\s+\w\s*=", text, re.IGNORECASE):
            subs = _extract_variable_values(problem_text)
            if subs:
                syms = {v: sp.Symbol(v) for v in subs}
                # Try to extract each sub-expression (i), (ii) etc.
                sub_parts = re.findall(r'\([iv]+\)\s*(.+?)(?=\s*\([iv]+\)|\s*$)', problem_text, re.IGNORECASE)
                if not sub_parts:
                    sub_parts = [problem_text]

                steps = []
                results = []
                for part in sub_parts:
                    part_clean = part.strip().rstrip('.,')
                    try:
                        expr_str = _clean_for_sympy(part_clean, list(syms.keys()))
                        expr = sp.sympify(expr_str, locals=syms)
                        val = expr.subs(list(subs.items()))
                        val_simplified = sp.simplify(val)
                        steps.append(f"Expression: {part_clean}")
                        steps.append(f"  Substitute {', '.join(f'{k}={v}' for k,v in subs.items())}")
                        steps.append(f"  = {val_simplified}")
                        results.append(f"{part_clean} = **{val_simplified}**")
                    except Exception:
                        results.append(f"{part_clean} = (could not evaluate automatically)")

                subs_str = ", ".join(f"{k} = {v}" for k, v in subs.items())
                result_str = "\n".join(results)
                sol_text = (
                    f"**Problem:** Evaluate the expression for {subs_str}\n\n"
                    f"**Given values:** {subs_str}\n\n"
                    f"**Step-by-Step Substitution:**\n"
                )
                for s in steps:
                    sol_text += f"\n{s}"
                sol_text += f"\n\n**Final Answers:**\n{result_str}"

                all_answers = "; ".join(results)
                return {
                    "agent": "SolverAgent",
                    "solution": sol_text,
                    "final_answer": all_answers,
                    "steps": steps,
                    "method_used": "sympy",
                    "confidence": 0.95,
                }

        # ── Combinatorics: Letter arrangements ─────────────────────────────
        # Handles: "number of ways to arrange the letters in the word MATHEMATICS"
        comb_match = re.search(
            r"(?:number of ways to arrange|how many ways to arrange|permutations of|arrang\w*)\s+.*?letters\s+in\s+(?:the\s+word\s+)?[\"']?(\w+)[\"']?",
            problem_text, re.IGNORECASE
        )
        if comb_match:
            word = comb_match.group(1)
            if word and len(word) > 1:
                from collections import Counter
                import math
                word_clean = re.sub(r'[^a-zA-Z]', '', word).upper()
                counts = Counter(word_clean)
                n = len(word_clean)
                denom = 1
                for char, count in counts.items():
                    denom *= math.factorial(count)
                
                total = math.factorial(n) // denom
                
                steps = [f"Total letters in '{word_clean}': {n}"]
                repeat_desc = []
                for char, count in counts.items():
                    if count > 1:
                        repeat_desc.append(f"{char} repeats {count} times")
                
                if repeat_desc:
                    steps.append("Repeated letters: " + ", ".join(repeat_desc))
                    steps.append(f"Calculation: {n}! / (" + " * ".join([f"{count}!" for char, count in counts.items() if count > 1]) + ")")
                else:
                    steps.append(f"Calculation: {n}!")
                
                steps.append(f"Final calculation: {math.factorial(n)} / {denom} = {total}")

                sol_text = (
                    f"**Problem:** Find the number of ways to arrange the letters in '{word_clean}'\n\n"
                    f"**Steps:**\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)]) +
                    f"\n\n**Final Answer:** {total}"
                )
                return {
                    "agent": "SolverAgent",
                    "solution": sol_text,
                    "final_answer": str(total),
                    "steps": steps,
                    "method_used": "sympy",
                    "confidence": 0.98,
                }

        # ── Combinatorics: nCr / nPr ──────────────────────────────────────────
        # Handles: "Evaluate 10C3" or "Find 10P3"
        ncr_match = re.search(r"(\d+)\s*([CP])\s*(\d+)", problem_text, re.IGNORECASE)
        if ncr_match:
            n = int(ncr_match.group(1))
            op = ncr_match.group(2).upper()
            r = int(ncr_match.group(3))
            
            import math
            if op == 'C':
                val = math.comb(n, r)
                method = "Combination (nCr)"
                formula = f"{n}! / ({r}! * ({n}-{r})!)"
            else:
                val = math.perm(n, r)
                method = "Permutation (nPr)"
                formula = f"{n}! / ({n}-{r})!"
            
            sol_text = (
                f"**Problem:** Evaluate {n}{op}{r}\n\n"
                f"**Method:** {method}\n"
                f"**Formula:** {formula}\n\n"
                f"**Final Answer:** {val}"
            )
            return {
                "agent": "SolverAgent",
                "solution": sol_text,
                "final_answer": str(val),
                "steps": [f"Identify n={n}, r={r}", f"Apply {op} formula", f"Result: {val}"],
                "method_used": "sympy",
                "confidence": 0.98,
            }

    except Exception:
        pass

    return None


def _extract_variable_values(text: str) -> dict:
    """Extract variable=value pairs from problem text."""
    import sympy as sp
    pattern = re.finditer(r'\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)', text)
    result = {}
    for m in pattern:
        var, val = m.group(1), m.group(2)
        if var.lower() not in ('e',):   # skip Euler's number
            result[sp.Symbol(var)] = sp.Rational(val)
    return result


def _clean_for_sympy(expr: str, var_names: list) -> str:
    """Prepare extracted expression string for SymPy parsing."""
    # Replace ^ with **
    expr = expr.replace('^', '**')
    # Expand xy → x*y, xz → x*z etc. for single-letter vars
    for v in var_names:
        for u in var_names:
            if v != u:
                expr = re.sub(rf'\b{v}{u}\b', f'{v}*{u}', expr)
    # x2 → x**2 etc.
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1**\2', expr)
    return expr.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_fallback(problem_text: str, topic: str, references: str) -> Dict[str, Any]:
    """Basic template-based solution when LLM and SymPy both fail."""
    solution = (
        f"**Problem Analysis:**\n{problem_text}\n\n"
        f"**Topic:** {topic.replace('_', ' ').title()}\n\n"
        f"**Relevant Formulas:**\n{references[:600]}\n\n"
        f"**Note:** Full step-by-step solution requires a Gemini API key or correct "
        f"problem format. Please set your GEMINI_API_KEY in the .env file for complete solutions.\n\n"
        f"**Approach:**\n"
        f"1. Identify relevant formula from the references above\n"
        f"2. Substitute the given values\n"
        f"3. Simplify step by step\n"
        f"4. Verify the answer"
    )
    return {
        "agent": "SolverAgent",
        "solution": solution,
        "final_answer": "See solution above (API key needed for full computation)",
        "steps": ["Identify formula", "Substitute values", "Simplify", "Verify"],
        "method_used": "fallback",
        "confidence": 0.3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_expression(text: str) -> Optional[str]:
    """Try to extract a mathematical expression from the problem text."""
    # Remove common words
    expr = re.sub(
        r"\b(find|the|derivative|of|integral|integrate|differentiate|calculate|evaluate|solve|for|value|of)\b",
        "", text, flags=re.IGNORECASE
    ).strip()
    # Normalise power notation
    expr = expr.replace("^", "**").replace("x2", "x**2").replace("x3", "x**3")
    expr = re.sub(r"x(\d)", r"x**\1", expr)
    # Remove trailing punctuation
    expr = expr.rstrip(".,?!").strip()
    return expr if expr else None


def _extract_steps(solution: str) -> List[str]:
    """Extract numbered steps from a solution string."""
    steps = re.findall(r"(?:Step\s*\d+[:.]\s*)(.+?)(?=Step\s*\d+[:.:]|\Z)", solution, re.DOTALL)
    if steps:
        return [s.strip() for s in steps]
    lines = [l.strip() for l in solution.split("\n") if l.strip()]
    return lines[:8]


def _extract_final_answer(solution: str) -> str:
    """Extract the final answer from a solution string."""
    patterns = [
        r"[Ff]inal [Aa]nswer[:\s]+(.+?)(?:\n|$)",
        r"[Aa]nswer[:\s]+(.+?)(?:\n|$)",
        r"=\s*(.+?)(?:\n|$)",
        r"∴\s*(.+?)(?:\n|$)",
        r"Therefore[,:\s]+(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        match = re.search(pat, solution)
        if match:
            return match.group(1).strip()
    # Return last non-empty line
    lines = [l.strip() for l in solution.split("\n") if l.strip()]
    return lines[-1] if lines else "See solution above"
