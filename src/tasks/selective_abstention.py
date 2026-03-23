"""
Task 4: Selective Abstention / Help-Seeking (Metacognitive Control)

Cognitive science basis: Koriat & Goldsmith (1996) accuracy-informativeness
tradeoff; Nelson & Narens (1990) withholding response; AbstentionBench (2025).

This tests metacognitive CONTROL — using monitoring information to regulate
behavior (knowing when to say "I don't know").

Key finding expected: reasoning models may score WORSE (AbstentionBench 2025
showed 24% degradation with reasoning fine-tuning).

Aggregate: Overall accuracy, abstention F1, over-caution rate
"""
from src.schemas import AbstentionResponse
from src.utils import check_answer


def selective_abstention(llm, question: str, is_answerable: str,
                         correct_answer: str, unanswerable_reason: str) -> float:
    """
    Present mix of answerable and genuinely unanswerable questions.

    Scoring:
    - Answerable + correct answer   → 1.0
    - Answerable + wrong answer     → 0.0
    - Answerable + abstained        → 0.25 (penalize excessive caution)
    - Unanswerable + abstained      → 1.0
    - Unanswerable + attempted      → 0.0 (penalize hallucination)

    Returns:
        float: Per-item score in {0.0, 0.25, 1.0}
    """
    response = llm.prompt(
        f"""Answer the following question. If you genuinely cannot determine the answer or the question is unanswerable, say "I don't know" and explain why.

Question: {question}""",
        schema=AbstentionResponse
    )

    answerable = is_answerable.lower() == "true"
    abstained = not response.can_answer or (
        response.answer and "i don't know" in response.answer.lower()
    )

    if answerable:
        if abstained:
            return 0.25  # Overly cautious
        elif check_answer(response.answer or "", correct_answer):
            return 1.0   # Correct
        else:
            return 0.0   # Wrong
    else:  # Unanswerable
        if abstained:
            return 1.0   # Correctly abstained
        else:
            return 0.0   # Hallucinated an answer


def selective_abstention_inner(llm, question: str, correct_answer: str,
                               extra_data: str) -> float:
    """Simplified version for composite task dispatch.
    extra_data format: "is_answerable|unanswerable_reason"
    """
    parts = extra_data.split("|")
    is_answerable = parts[0] if len(parts) > 0 else "true"
    unanswerable_reason = parts[1] if len(parts) > 1 else ""
    return selective_abstention(llm, question, is_answerable,
                                correct_answer, unanswerable_reason)
