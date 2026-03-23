"""
Task 1: Confidence Calibration (Metacognitive Monitoring — Retrospective)

Cognitive science basis: Nelson & Narens (1990) confidence judgments;
Lichtenstein, Fischhoff & Phillips (1982) overconfidence literature;
Fleming & Lau (2014) metacognitive sensitivity.

The model answers a question and rates its confidence 0-100.
Per-item score: 1 - |confidence/100 - correctness|
Aggregate: ECE (10 bins), Brier Score, AUROC2
"""
from src.schemas import AnswerWithConfidence
from src.utils import check_answer


def confidence_calibration(llm, question: str, correct_answer: str,
                           difficulty: str) -> float:
    """
    Ask model a question, require confidence rating, measure calibration.

    In Kaggle, this is decorated with @kbench.task and uses llm.prompt().
    Locally, this function defines the scoring logic only.

    Returns:
        float: Per-item calibration score in [0, 1].
               1.0 = confidence perfectly matches correctness.
    """
    response = llm.prompt(
        f"""Answer the following question. After your answer, rate your confidence from 0 to 100 (where 0 = certainly wrong, 100 = certainly correct).

Question: {question}

Respond with your answer and confidence level.""",
        schema=AnswerWithConfidence
    )

    is_correct = check_answer(response.answer, correct_answer)
    confidence = max(0, min(100, response.confidence)) / 100.0
    correctness = 1.0 if is_correct else 0.0

    calibration_error = abs(confidence - correctness)
    return round(1.0 - calibration_error, 4)


def confidence_calibration_inner(llm, question: str, correct_answer: str) -> float:
    """Simplified version for composite task dispatch."""
    return confidence_calibration(llm, question, correct_answer, difficulty="")
