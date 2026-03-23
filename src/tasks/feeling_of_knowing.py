"""
Task 2: Feeling-of-Knowing (Metacognitive Monitoring — Prospective)

Cognitive science basis: Hart (1965) RJR paradigm; Nelson (1984) gamma
correlation for FOK accuracy; Nelson & Narens (1990) monitoring framework.

Key difference from Task 1: The model predicts success BEFORE answering
(prospective), whereas Task 1 rates confidence AFTER answering (retrospective).
Delayed JOLs are more accurate than immediate JOLs (Nelson & Dunlosky, 1991).

Aggregate: Gamma correlation, AUROC2
"""
from src.schemas import FOKResponse
from src.utils import check_answer


def feeling_of_knowing(llm, question: str, correct_answer: str,
                       difficulty: str) -> float:
    """
    Two-phase FOK test:
    Phase 1: Predict whether you can answer correctly (0-100)
    Phase 2: Actually answer

    Returns:
        float: Per-item FOK accuracy in [0, 1].
               1.0 = prediction perfectly matches correctness.
    """
    fok_response = llm.prompt(
        f"""I'm going to ask you a question. Before you answer, predict how likely you are to get it correct on a scale of 0-100.

Then provide your best answer.

Question: {question}""",
        schema=FOKResponse
    )

    is_correct = check_answer(fok_response.answer, correct_answer)
    prediction = max(0, min(100, fok_response.prediction)) / 100.0
    correctness = 1.0 if is_correct else 0.0

    fok_error = abs(prediction - correctness)
    return round(1.0 - fok_error, 4)


def feeling_of_knowing_inner(llm, question: str, correct_answer: str) -> float:
    """Simplified version for composite task dispatch."""
    return feeling_of_knowing(llm, question, correct_answer, difficulty="")
