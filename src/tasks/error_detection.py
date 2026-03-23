"""
Task 3: Error Detection (Error Detection Sub-ability)

Cognitive science basis: Yeung & Summerfield (2012) error monitoring;
conflict monitoring theory (Botvinick et al., 2001); error signaling paradigms.

The model reviews a presented solution and determines if it contains an error,
WITHOUT being told that errors may exist.

Aggregate: Balanced accuracy, F1, per-error-type breakdown
"""
from src.schemas import ErrorReview
from src.utils import check_answer, validate_bool


def error_detection(llm, problem: str, presented_solution: str,
                    solution_has_error: str, error_type: str) -> float:
    """
    Present a solution (correct or with planted error).
    Ask model to review without hinting that errors may exist.
    Score: ability to correctly classify solutions as right/wrong.

    Returns:
        float: 1.0 for correct classification, 0.0 for wrong.
    """
    try:
        response = llm.prompt(
            f"""Here is a problem and a proposed solution. Please review this solution and explain your assessment of it.

Problem: {problem}

Proposed Solution: {presented_solution}

Provide your assessment. If you find any issues, explain them and provide the correct answer.""",
            schema=ErrorReview
        )

        has_error = solution_has_error.lower() == "true"
        model_detected_error = validate_bool(response.has_error)

        if has_error and model_detected_error:
            return 1.0  # True positive: correctly detected error
        elif not has_error and not model_detected_error:
            return 1.0  # True negative: correctly identified correct solution
        else:
            return 0.0  # Misclassification
    except Exception as e:
        print(f"[WARN] error_detection failed: {e}")
        return 0.0


def error_detection_inner(llm, problem: str, presented_solution: str,
                          extra_data: str) -> float:
    """Simplified version for composite task dispatch.
    extra_data format: "solution_has_error|error_type"
    """
    parts = extra_data.split("|")
    solution_has_error = parts[0] if len(parts) > 0 else "false"
    error_type = parts[1] if len(parts) > 1 else "none"
    return error_detection(llm, problem, presented_solution,
                           solution_has_error, error_type)
