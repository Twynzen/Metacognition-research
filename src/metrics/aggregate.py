"""
Aggregate scoring: geometric mean composite and bootstrap confidence intervals.

The geometric mean prevents compensation between tasks — a model cannot
make up for poor calibration with good error detection.
"""
import numpy as np


def geometric_mean(sub_scores):
    """
    Geometric mean of sub-task scores.

    Prevents compensation: a zero on any task drives the composite to zero.
    Epsilon floor (1e-10) avoids log(0) while preserving near-zero penalty.

    Args:
        sub_scores: list/array of float in [0, 1]

    Returns:
        float: geometric mean composite score
    """
    values = [max(v, 1e-10) for v in sub_scores]
    return round(float(np.exp(np.mean(np.log(values)))), 4)


def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95, seed=None):
    """
    Nonparametric bootstrap confidence intervals for any aggregate metric.

    Args:
        scores: array-like of per-item scores
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95 = 95% CI)
        seed: random seed for reproducibility

    Returns:
        tuple: (lower_bound, upper_bound, mean)
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores, dtype=float)

    boot_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    mean = float(np.mean(boot_means))

    return round(lower, 4), round(upper, 4), round(mean, 4)
