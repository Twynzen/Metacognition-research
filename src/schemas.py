"""
Structured output schemas for kaggle-benchmarks SDK.

These dataclasses are used with `llm.prompt(..., schema=SchemaName)` to force
structured JSON output from the model being evaluated.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnswerWithConfidence:
    """Task 1: Confidence Calibration — retrospective monitoring."""
    answer: str
    confidence: int  # 0-100


@dataclass
class FOKResponse:
    """Task 2: Feeling-of-Knowing — prospective monitoring."""
    prediction: int  # 0-100 likelihood of answering correctly
    answer: str


@dataclass
class ErrorReview:
    """Task 3: Error Detection — error monitoring."""
    has_error: bool
    error_explanation: str
    corrected_answer: str


@dataclass
class AbstentionResponse:
    """Task 4: Selective Abstention — metacognitive control."""
    can_answer: bool
    answer: Optional[str]
    confidence: int  # 0-100


@dataclass
class DomainPrediction:
    """Task 5: Metacognitive Knowledge — self-knowledge."""
    predicted_accuracy: int  # 0-100
    hardest_aspect: str
    easiest_aspect: str
