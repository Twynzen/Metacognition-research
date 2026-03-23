"""
Comprehensive tests for the answer checker.

CRITICAL: False negatives corrupt ALL metrics. These tests ensure robust matching.
"""
import pytest
from src.utils import check_answer, normalize_text, extract_number


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_lowercases(self):
        assert normalize_text("HELLO") == "hello"

    def test_removes_articles(self):
        assert normalize_text("The quick brown fox") == "quick brown fox"
        assert normalize_text("a cat") == "cat"
        assert normalize_text("an apple") == "apple"

    def test_removes_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"


class TestExtractNumber:
    def test_integer(self):
        assert extract_number("42") == 42.0

    def test_negative(self):
        assert extract_number("-7") == -7.0

    def test_decimal(self):
        assert extract_number("3.14") == 3.14

    def test_embedded_in_text(self):
        assert extract_number("The answer is 256 meters") == 256.0

    def test_with_commas(self):
        assert extract_number("1,234,567") == 1234567.0

    def test_no_number(self):
        assert extract_number("no numbers here") is None


class TestCheckAnswer:
    # --- Exact match ---
    def test_exact_match(self):
        assert check_answer("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert check_answer("paris", "Paris") is True
        assert check_answer("PARIS", "paris") is True

    def test_with_whitespace(self):
        assert check_answer("  Paris  ", "Paris") is True

    # --- Containment ---
    def test_containment(self):
        assert check_answer("The answer is 42", "42") is True

    def test_containment_phrase(self):
        assert check_answer("I believe the answer is Paris, France", "Paris") is True

    # --- Numerical matching ---
    def test_numerical_exact(self):
        assert check_answer("130", "130") is True

    def test_numerical_with_text(self):
        assert check_answer("The result is 130", "130") is True

    def test_numerical_tolerance(self):
        assert check_answer("3.14159", "3.14159") is True

    def test_numerical_close(self):
        assert check_answer("3.141", "3.141") is True

    def test_numerical_reject_off_by_one(self):
        assert check_answer("131", "130") is False

    def test_numerical_reject_wrong(self):
        assert check_answer("391", "381") is False

    # --- Yes/No matching ---
    def test_yes_match(self):
        assert check_answer("Yes, Rex is a mammal", "Yes") is True

    def test_no_match(self):
        assert check_answer("No, that is incorrect", "No") is True

    def test_true_for_yes(self):
        assert check_answer("True", "Yes") is True

    def test_correct_for_yes(self):
        assert check_answer("That is correct", "Yes") is True

    # --- Edge cases ---
    def test_empty_model(self):
        assert check_answer("", "Paris") is False

    def test_empty_correct(self):
        assert check_answer("Paris", "") is False

    def test_both_empty(self):
        assert check_answer("", "") is False

    # --- Should REJECT ---
    def test_reject_partial_wrong(self):
        # "12" should not match "124"
        assert check_answer("12", "124") is False

    def test_reject_different_numbers(self):
        assert check_answer("500", "391") is False

    def test_reject_unrelated(self):
        assert check_answer("London", "Paris") is False

    # --- Single letter matching ---
    def test_single_letter_match(self):
        assert check_answer("C is the answer", "C") is True

    def test_single_letter_exact(self):
        assert check_answer("B", "B") is True

    # --- Real benchmark scenarios ---
    def test_math_answer(self):
        assert check_answer("The square root of 144 is 12", "12") is True

    def test_factual_answer(self):
        assert check_answer("Au", "Au") is True

    def test_year_answer(self):
        assert check_answer("The treaty was signed in 1648", "1648") is True

    def test_prime_factor(self):
        assert check_answer("67", "67") is True

    def test_modular_arithmetic(self):
        result = (347 * 29) % 17
        assert check_answer(str(result), str(result)) is True
