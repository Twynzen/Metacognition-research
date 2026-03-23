"""
Deterministic answer checker shared across all tasks.

CRITICAL: False negatives (marking correct answers wrong) corrupt ALL metrics.
This function must handle: exact match, case insensitive, numerical tolerance,
containment, yes/no keywords, and common normalizations.
"""
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove articles/punctuation."""
    text = text.strip().lower()
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove articles
    text = re.sub(r"\b(the|a|an)\b", "", text)
    # Remove punctuation except decimal points in numbers
    text = re.sub(r"[^\w\s.]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_number(text: str):
    """Extract the first number from text, including negatives and decimals."""
    match = re.search(r"-?\d+\.?\d*", text.replace(",", ""))
    if match:
        return float(match.group())
    return None


def check_answer(model_answer: str, correct_answer: str) -> bool:
    """
    Deterministic answer checker with multiple matching strategies.

    Rules (applied in order):
    1. Normalize both strings and check exact match
    2. Check if correct answer is contained in model answer
    3. Numerical comparison with tolerance (±0.01)
    4. Yes/No keyword matching for boolean questions
    5. Single-letter answer matching (e.g., "A", "B", "C", "D")

    Returns True if the model answer is considered correct.
    """
    if not model_answer or not correct_answer:
        return False

    model_norm = normalize_text(model_answer)
    correct_norm = normalize_text(correct_answer)

    # 1. Exact match after normalization
    if model_norm == correct_norm:
        return True

    # 2. Containment: correct answer appears in model's response
    if correct_norm and correct_norm in model_norm:
        return True

    # 3. Numerical comparison with tolerance
    model_num = extract_number(model_answer)
    correct_num = extract_number(correct_answer)
    if model_num is not None and correct_num is not None:
        if abs(model_num - correct_num) < 0.01:
            return True

    # 4. Yes/No matching
    yes_words = {"yes", "true", "correct", "right", "affirmative"}
    no_words = {"no", "false", "incorrect", "wrong", "negative"}
    model_tokens = set(model_norm.split())
    if correct_norm in yes_words:
        if model_tokens & yes_words:
            return True
    if correct_norm in no_words:
        if model_tokens & no_words:
            return True

    # 5. Single-letter matching (for multiple choice)
    if len(correct_norm) == 1 and correct_norm.isalpha():
        # Check if model answer starts with or contains just that letter
        first_word = model_norm.split()[0] if model_norm.split() else ""
        if first_word == correct_norm:
            return True

    return False
