"""
Task 5: Metacognitive Knowledge (Self-Knowledge Sub-ability)

Cognitive science basis: Flavell (1979) person knowledge;
Dunning & Kruger (1999) metacognitive miscalibration;
DeepMind paper's "metacognitive knowledge" sub-ability.

Multi-turn task:
Phase 1: Model predicts its accuracy for a domain
Phase 2: Model answers 10 domain questions
Phase 3: Compare prediction to reality

Expected finding: Dunning-Kruger pattern — models overestimate on hard
domains and may underestimate on easy ones.

Aggregate: Mean prediction error, correlation, Dunning-Kruger pattern analysis
"""
from src.schemas import DomainPrediction
from src.utils import check_answer


def metacognitive_knowledge(llm, domain: str, domain_questions: str,
                            domain_answers: str) -> float:
    """
    Phase 1: Ask model to predict its accuracy in a domain.
    Phase 2: Test it on 10 questions from that domain.
    Phase 3: Compare prediction to reality.

    Returns:
        float: 1 - |predicted_accuracy/100 - actual_accuracy| in [0, 1]
    """
    # Phase 1: Self-assessment
    prediction = llm.prompt(
        f"""I'm going to test you on 10 questions about {domain}. Before I do, predict what percentage (0-100) you'll get correct. Also tell me what you think will be hardest and easiest.""",
        schema=DomainPrediction
    )

    # Phase 2: Answer domain questions sequentially
    questions = domain_questions.split("|||")
    answers = domain_answers.split("|||")
    correct_count = 0

    for q, a in zip(questions, answers):
        resp = llm.prompt(f"Answer briefly: {q}")
        if check_answer(str(resp), a):
            correct_count += 1

    actual_accuracy = correct_count / len(questions)
    predicted = max(0, min(100, prediction.predicted_accuracy)) / 100.0

    error = abs(predicted - actual_accuracy)
    return round(1.0 - error, 4)


def metacognitive_knowledge_inner(llm, domain: str, domain_questions: str,
                                  extra_data: str) -> float:
    """Simplified version for composite task dispatch.
    extra_data = domain_answers
    """
    return metacognitive_knowledge(llm, domain, domain_questions, extra_data)
