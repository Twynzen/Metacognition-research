"""
Tests for all metric implementations.

Verifies against known values from the literature and hand-computed examples.
"""
import pytest
import numpy as np
from src.metrics.calibration import compute_ece, compute_brier, compute_brier_decomposition
from src.metrics.discrimination import compute_auroc2, compute_gamma
from src.metrics.aggregate import geometric_mean, bootstrap_ci


class TestECE:
    def test_perfect_calibration(self):
        """Well-calibrated: 90% confident correct, 10% confident wrong → ECE = 0.1 (bin-edge effect)."""
        confidences = [0.9, 0.9, 0.1, 0.1]
        correctness = [1, 1, 0, 0]
        result = compute_ece(confidences, correctness)
        # With 10 bins and 4 items: each bin has |1.0 - 0.9| or |0.0 - 0.1| = 0.1 gap
        assert result["ece"] <= 0.11, f"Well-calibrated ECE should be ≈ 0.1, got {result['ece']}"

    def test_truly_perfect_calibration(self):
        """Truly perfect: confidence exactly matches empirical accuracy → ECE = 0."""
        # 50 items at 80% confidence, 40 correct (80% accuracy) → perfect for that bin
        confidences = [0.8] * 50
        correctness = [1] * 40 + [0] * 10
        result = compute_ece(confidences, correctness)
        assert result["ece"] < 0.01, f"Truly perfect calibration should have ECE ≈ 0, got {result['ece']}"

    def test_miscalibrated(self):
        """All 90% confident but only 50% correct → ECE ≈ 0.4"""
        confidences = [0.9, 0.9, 0.9, 0.9]
        correctness = [1, 0, 1, 0]
        result = compute_ece(confidences, correctness)
        assert 0.35 <= result["ece"] <= 0.45, f"Miscalibrated ECE should be ≈ 0.4, got {result['ece']}"

    def test_completely_wrong(self):
        """100% confident but always wrong → ECE = 1.0"""
        confidences = [1.0, 1.0, 1.0, 1.0]
        correctness = [0, 0, 0, 0]
        result = compute_ece(confidences, correctness)
        assert result["ece"] >= 0.95, f"Completely wrong should have ECE ≈ 1.0, got {result['ece']}"

    def test_empty_input(self):
        result = compute_ece([], [])
        assert result["ece"] == 0.0

    def test_bin_data_structure(self):
        confidences = [0.1, 0.2, 0.5, 0.8, 0.9]
        correctness = [0, 0, 1, 1, 1]
        result = compute_ece(confidences, correctness)
        for b in result["bin_data"]:
            assert "bin_center" in b
            assert "accuracy" in b
            assert "confidence" in b
            assert "count" in b
            assert "gap" in b

    def test_mce_is_max_gap(self):
        """MCE should be the maximum calibration error across bins."""
        confidences = [0.9, 0.9, 0.1, 0.1]
        correctness = [1, 1, 0, 0]
        result = compute_ece(confidences, correctness)
        assert result["mce"] >= result["ece"]

    def test_larger_dataset(self):
        """ECE on a realistic-sized dataset."""
        np.random.seed(42)
        n = 300
        confidences = np.random.uniform(0, 1, n)
        # Generate outcomes correlated with confidence
        correctness = (np.random.uniform(0, 1, n) < confidences).astype(float)
        result = compute_ece(confidences.tolist(), correctness.tolist())
        assert 0 <= result["ece"] <= 1


class TestBrier:
    def test_perfect(self):
        """Perfect predictions → BS = 0"""
        bs = compute_brier([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
        assert bs == 0.0

    def test_worst_case(self):
        """Maximally wrong → BS = 1.0"""
        bs = compute_brier([1.0, 1.0, 0.0, 0.0], [0, 0, 1, 1])
        assert bs == 1.0

    def test_no_skill(self):
        """All 0.5 confidence on balanced → BS = 0.25"""
        bs = compute_brier([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert bs == 0.25

    def test_decomposition_sums(self):
        """BS ≈ Reliability - Resolution + Uncertainty"""
        np.random.seed(42)
        conf = np.random.uniform(0, 1, 100)
        outcomes = (np.random.uniform(0, 1, 100) < conf).astype(float)
        result = compute_brier_decomposition(conf.tolist(), outcomes.tolist())
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(result["brier_score"] - reconstructed) < 0.02


class TestAUROC2:
    def test_perfect_discrimination(self):
        """Correct trials always have higher confidence → AUROC2 = 1.0"""
        confidences = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        correctness = [1, 1, 1, 0, 0, 0]
        assert compute_auroc2(confidences, correctness) == 1.0

    def test_random_discrimination(self):
        """Random confidence assignment → AUROC2 ≈ 0.5"""
        np.random.seed(42)
        n = 1000
        confidences = np.random.uniform(0, 1, n).tolist()
        correctness = np.random.randint(0, 2, n).tolist()
        auroc = compute_auroc2(confidences, correctness)
        assert 0.4 <= auroc <= 0.6, f"Random should give AUROC ≈ 0.5, got {auroc}"

    def test_all_correct(self):
        """All correct → returns 0.5 (undefined)"""
        assert compute_auroc2([0.9, 0.8, 0.7], [1, 1, 1]) == 0.5

    def test_all_incorrect(self):
        """All incorrect → returns 0.5 (undefined)"""
        assert compute_auroc2([0.9, 0.8, 0.7], [0, 0, 0]) == 0.5

    def test_inverse_discrimination(self):
        """Higher confidence on wrong answers → AUROC2 < 0.5"""
        confidences = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        correctness = [1, 1, 1, 0, 0, 0]
        assert compute_auroc2(confidences, correctness) < 0.5


class TestGamma:
    def test_perfect_gamma(self):
        """Perfect concordance → gamma = 1.0"""
        confidences = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        correctness = [1, 1, 1, 0, 0, 0]
        assert compute_gamma(confidences, correctness) == 1.0

    def test_inverse_gamma(self):
        """Perfect discordance → gamma = -1.0"""
        confidences = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        correctness = [1, 1, 1, 0, 0, 0]
        assert compute_gamma(confidences, correctness) == -1.0

    def test_no_discrimination(self):
        """All same confidence → gamma = 0.0"""
        confidences = [0.5, 0.5, 0.5, 0.5]
        correctness = [1, 0, 1, 0]
        assert compute_gamma(confidences, correctness) == 0.0

    def test_all_correct(self):
        """All correct → gamma = 0.0 (no pairs to compare)"""
        assert compute_gamma([0.9, 0.8], [1, 1]) == 0.0

    def test_relationship_to_auroc(self):
        """gamma ≈ 2 * AUROC - 1"""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05]
        correctness = [1, 1, 1, 0, 0, 0, 0, 0]
        gamma = compute_gamma(confidences, correctness)
        auroc = compute_auroc2(confidences, correctness)
        # Approximate relationship (exact only without ties)
        assert abs(gamma - (2 * auroc - 1)) < 0.15


class TestGeometricMean:
    def test_equal_scores(self):
        assert geometric_mean([0.8, 0.8, 0.8]) == 0.8

    def test_varied_scores(self):
        gm = geometric_mean([0.9, 0.1])
        assert gm == 0.3  # sqrt(0.09) = 0.3

    def test_zero_score_penalty(self):
        """Zero drives the geometric mean to near-zero."""
        gm = geometric_mean([0.0, 0.9, 0.9])
        assert gm < 0.01

    def test_all_ones(self):
        assert geometric_mean([1.0, 1.0, 1.0]) == 1.0

    def test_single_score(self):
        assert geometric_mean([0.75]) == 0.75


class TestBootstrapCI:
    def test_returns_three_values(self):
        lower, upper, mean = bootstrap_ci([0.5, 0.6, 0.7, 0.8], seed=42)
        assert lower < mean < upper

    def test_narrow_ci_for_constant(self):
        """Constant values → CI width ≈ 0"""
        lower, upper, mean = bootstrap_ci([0.5] * 100, seed=42)
        assert abs(upper - lower) < 0.01

    def test_wider_ci_for_variable(self):
        """High variance → wider CI"""
        lower_const, upper_const, _ = bootstrap_ci([0.5] * 100, seed=42)
        lower_var, upper_var, _ = bootstrap_ci([0.0, 1.0] * 50, seed=42)
        assert (upper_var - lower_var) > (upper_const - lower_const)

    def test_seed_reproducibility(self):
        r1 = bootstrap_ci([0.3, 0.5, 0.7, 0.9], seed=42)
        r2 = bootstrap_ci([0.3, 0.5, 0.7, 0.9], seed=42)
        assert r1 == r2
