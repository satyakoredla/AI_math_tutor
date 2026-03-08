"""
memory/memory_store.py
SQLite-backed memory system for storing and retrieving solved problems.
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MEMORY_DB_PATH

# ── SQLAlchemy ────────────────────────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, String, Text, Float,
    Integer, DateTime, Boolean, text
)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()


class SolvedProblem(Base):
    __tablename__ = "solved_problems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_hash = Column(String(64), unique=True, index=True)
    input_question = Column(Text, nullable=False)
    topic = Column(String(100))
    parsed_problem = Column(Text)       # JSON
    retrieved_docs = Column(Text)       # JSON
    solution = Column(Text)
    final_answer = Column(Text)
    explanation = Column(Text)
    confidence = Column(Float, default=0.0)
    is_verified = Column(Boolean, default=False)
    user_feedback = Column(Text)        # "correct" | "incorrect" | corrected text
    timestamp = Column(DateTime, default=datetime.utcnow)
    solve_count = Column(Integer, default=1)


# ── Engine & Session ──────────────────────────────────────────────────────────
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        _engine = create_engine(f"sqlite:///{MEMORY_DB_PATH}", echo=False)
        Base.metadata.create_all(_engine)
    return _engine


def _hash_question(question: str) -> str:
    """Stable hash for deduplication."""
    normalised = " ".join(question.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_solution(
    input_question: str,
    parsed_problem: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]],
    solver_result: Dict[str, Any],
    explainer_result: Dict[str, Any],
    verifier_result: Dict[str, Any],
) -> int:
    """
    Save a solved problem to memory. Updates count if already exists.

    Returns
    -------
    int — database row id
    """
    q_hash = _hash_question(input_question)
    engine = _get_engine()

    with Session(engine) as session:
        existing = session.query(SolvedProblem).filter_by(question_hash=q_hash).first()
        if existing:
            existing.solve_count += 1
            existing.timestamp = datetime.utcnow()
            session.commit()
            return existing.id

        record = SolvedProblem(
            question_hash=q_hash,
            input_question=input_question,
            topic=parsed_problem.get("topic", "unknown"),
            parsed_problem=json.dumps(parsed_problem),
            retrieved_docs=json.dumps(retrieved_docs),
            solution=solver_result.get("solution", ""),
            final_answer=verifier_result.get("verified_answer", solver_result.get("final_answer", "")),
            explanation=explainer_result.get("explanation", ""),
            confidence=verifier_result.get("confidence", 0.0),
            is_verified=verifier_result.get("is_correct", False),
        )
        session.add(record)
        session.commit()
        return record.id


def find_similar(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Find previously solved similar problems.
    Uses keyword overlap as a similarity heuristic.

    Returns
    -------
    list of dicts (most relevant first)
    """
    engine = _get_engine()
    query_words = set(query.lower().split())

    with Session(engine) as session:
        all_records = session.query(SolvedProblem).all()
        scored = []
        for rec in all_records:
            rec_words = set(rec.input_question.lower().split())
            overlap = len(query_words & rec_words)
            if overlap > 2:
                scored.append((overlap, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, rec in scored[:max_results]:
            results.append({
                "id": rec.id,
                "question": rec.input_question,
                "topic": rec.topic,
                "answer": rec.final_answer,
                "explanation": rec.explanation[:300] if rec.explanation else "",
                "confidence": rec.confidence,
                "solve_count": rec.solve_count,
                "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M") if rec.timestamp else "",
            })
        return results


def get_exact_match(query: str) -> Optional[Dict[str, Any]]:
    """Check if this exact question was solved before."""
    q_hash = _hash_question(query)
    engine = _get_engine()

    with Session(engine) as session:
        rec = session.query(SolvedProblem).filter_by(question_hash=q_hash).first()
        if rec:
            return {
                "id": rec.id,
                "question": rec.input_question,
                "topic": rec.topic,
                "solution": rec.solution,
                "answer": rec.final_answer,
                "explanation": rec.explanation,
                "confidence": rec.confidence,
                "solve_count": rec.solve_count,
                "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M") if rec.timestamp else "",
            }
    return None


def update_feedback(record_id: int, feedback: str) -> bool:
    """Store user feedback: 'correct', 'incorrect', or corrected answer text."""
    engine = _get_engine()
    with Session(engine) as session:
        rec = session.query(SolvedProblem).filter_by(id=record_id).first()
        if rec:
            rec.user_feedback = feedback
            session.commit()
            return True
    return False


def get_recent_problems(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recently solved problems."""
    engine = _get_engine()
    with Session(engine) as session:
        records = (
            session.query(SolvedProblem)
            .order_by(SolvedProblem.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "question": r.input_question[:80],
                "topic": r.topic,
                "answer": r.final_answer[:60] if r.final_answer else "",
                "solve_count": r.solve_count,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "",
            }
            for r in records
        ]


def get_stats() -> Dict[str, Any]:
    """Return memory statistics."""
    engine = _get_engine()
    with Session(engine) as session:
        total = session.query(SolvedProblem).count()
        topics = session.execute(
            text("SELECT topic, COUNT(*) as cnt FROM solved_problems GROUP BY topic ORDER BY cnt DESC LIMIT 5")
        ).fetchall()
        return {
            "total_problems": total,
            "top_topics": [(row[0], row[1]) for row in topics],
        }
