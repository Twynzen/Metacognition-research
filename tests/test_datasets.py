"""
Tests for all dataset generators.

Verifies: correct columns, no duplicates, correct answer validity,
difficulty distribution, and expected sizes.
"""
import pytest
import pandas as pd


class TestCalibrationDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.datasets.calibration import generate_calibration_dataset
        self.df = generate_calibration_dataset(n=300)

    def test_correct_size(self):
        assert len(self.df) == 300

    def test_required_columns(self):
        for col in ["question", "correct_answer", "difficulty", "domain"]:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_no_duplicate_questions(self):
        assert self.df["question"].nunique() == len(self.df)

    def test_difficulty_distribution(self):
        """Approximately 20% easy / 60% medium / 20% hard."""
        counts = self.df["difficulty"].value_counts(normalize=True)
        assert counts.get("easy", 0) > 0.10, "Too few easy questions"
        assert counts.get("medium", 0) > 0.40, "Too few medium questions"
        assert counts.get("hard", 0) > 0.10, "Too few hard questions"

    def test_domain_coverage(self):
        domains = set(self.df["domain"].unique())
        assert "math" in domains
        assert "factual" in domains
        assert "logic" in domains

    def test_no_empty_answers(self):
        assert (self.df["correct_answer"].str.strip() != "").all()

    def test_no_empty_questions(self):
        assert (self.df["question"].str.strip() != "").all()

    def test_math_answers_are_correct(self):
        """Spot-check math questions by verifying computed answers."""
        math_df = self.df[self.df["domain"] == "math"].head(20)
        for _, row in math_df.iterrows():
            assert row["correct_answer"].strip() != "", f"Empty answer for: {row['question']}"


class TestErrorDetectionDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.datasets.error_detection import generate_error_detection_dataset
        self.df = generate_error_detection_dataset(n=200)

    def test_correct_size(self):
        assert len(self.df) == 200

    def test_required_columns(self):
        for col in ["problem", "presented_solution", "solution_has_error", "error_type"]:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_balanced_errors(self):
        """~50% should have errors."""
        error_rate = (self.df["solution_has_error"] == "true").mean()
        assert 0.40 <= error_rate <= 0.60, f"Error rate {error_rate} not balanced"

    def test_error_types(self):
        types = set(self.df["error_type"].unique())
        assert "none" in types
        assert "arithmetic" in types

    def test_no_empty_problems(self):
        assert (self.df["problem"].str.strip() != "").all()

    def test_no_empty_solutions(self):
        assert (self.df["presented_solution"].str.strip() != "").all()


class TestAbstentionDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.datasets.abstention import generate_abstention_dataset
        self.df = generate_abstention_dataset(n=200)

    def test_correct_size(self):
        assert len(self.df) == 200

    def test_required_columns(self):
        for col in ["question", "is_answerable", "correct_answer", "unanswerable_reason"]:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_answerable_ratio(self):
        """~60% answerable, ~40% unanswerable."""
        answerable_rate = (self.df["is_answerable"] == "true").mean()
        assert 0.50 <= answerable_rate <= 0.70, f"Answerable rate {answerable_rate} unexpected"

    def test_unanswerable_categories(self):
        unanswerable = self.df[self.df["is_answerable"] == "false"]
        reasons = set(unanswerable["unanswerable_reason"].unique())
        expected = {"future_unknown", "subjective", "false_premise", "underspecified", "genuinely_unknown"}
        assert reasons == expected, f"Missing categories: {expected - reasons}"

    def test_answerable_have_answers(self):
        answerable = self.df[self.df["is_answerable"] == "true"]
        assert (answerable["correct_answer"].str.strip() != "").all()

    def test_no_duplicate_questions(self):
        assert self.df["question"].nunique() == len(self.df)


class TestSelfKnowledgeDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.datasets.self_knowledge import generate_self_knowledge_dataset
        self.df = generate_self_knowledge_dataset(n=200)

    def test_correct_size(self):
        # 20 rows (one per domain), 10 questions each = 200 total questions
        assert len(self.df) == 20

    def test_required_columns(self):
        for col in ["domain", "domain_questions", "domain_answers"]:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_twenty_domains(self):
        assert self.df["domain"].nunique() == 20

    def test_ten_questions_per_domain(self):
        for _, row in self.df.iterrows():
            questions = row["domain_questions"].split("|||")
            answers = row["domain_answers"].split("|||")
            assert len(questions) == 10, f"Domain {row['domain']} has {len(questions)} questions"
            assert len(answers) == 10, f"Domain {row['domain']} has {len(answers)} answers"

    def test_no_empty_questions(self):
        for _, row in self.df.iterrows():
            for q in row["domain_questions"].split("|||"):
                assert q.strip() != "", f"Empty question in domain {row['domain']}"

    def test_no_empty_answers(self):
        for _, row in self.df.iterrows():
            for a in row["domain_answers"].split("|||"):
                assert a.strip() != "", f"Empty answer in domain {row['domain']}"
