"""
Calibration metrics: ECE, MCE, Brier Score with Murphy decomposition.

References:
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Murphy (1973). A new vector partition of the probability score.
- Naeini et al. (2015). Obtaining well calibrated probabilities using Bayesian binning.
"""
import numpy as np


def compute_ece(confidences, correctness, n_bins=10):
    """
    Expected Calibration Error with equal-width bins.

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: array-like of float in [0, 1]
        correctness: array-like of int/float in {0, 1}
        n_bins: number of equal-width bins (default 10)

    Returns:
        dict with 'ece', 'mce', 'bin_data' (list of dicts for reliability diagram)
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

    if len(conf) == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_data": []}

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        # Use >= for the first bin to include confidence=0
        if i == 0:
            mask = (conf >= bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        else:
            mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])

        count = mask.sum()
        if count > 0:
            bin_acc = corr[mask].mean()
            bin_conf = conf[mask].mean()
            bin_size = count / len(conf)
            gap = abs(bin_acc - bin_conf)
            ece += bin_size * gap
            mce = max(mce, gap)
            bin_data.append({
                "bin_lower": float(bin_boundaries[i]),
                "bin_upper": float(bin_boundaries[i + 1]),
                "bin_center": float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "count": int(count),
                "gap": float(gap),
            })

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "bin_data": bin_data,
    }


def compute_brier(confidences, outcomes):
    """
    Brier Score: BS = (1/N) * sum((f_i - o_i)^2)

    BS = 0 is perfect; BS = 0.25 is no-skill baseline for balanced datasets.

    Args:
        confidences: array-like of float in [0, 1]
        outcomes: array-like of int/float in {0, 1}

    Returns:
        float: Brier score
    """
    conf = np.array(confidences, dtype=float)
    out = np.array(outcomes, dtype=float)
    return round(float(np.mean((conf - out) ** 2)), 4)


def compute_brier_decomposition(confidences, outcomes, n_bins=10):
    """
    Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty

    Args:
        confidences: array-like of float in [0, 1]
        outcomes: array-like of int/float in {0, 1}
        n_bins: number of bins

    Returns:
        dict with 'brier_score', 'reliability', 'resolution', 'uncertainty'
    """
    conf = np.array(confidences, dtype=float)
    out = np.array(outcomes, dtype=float)
    n = len(conf)

    bs = float(np.mean((conf - out) ** 2))
    base_rate = float(out.mean())
    uncertainty = base_rate * (1 - base_rate)

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i == 0:
            mask = (conf >= bin_bounds[i]) & (conf <= bin_bounds[i + 1])
        else:
            mask = (conf > bin_bounds[i]) & (conf <= bin_bounds[i + 1])
        nk = mask.sum()
        if nk > 0:
            fk = conf[mask].mean()
            ok = out[mask].mean()
            reliability += nk * (fk - ok) ** 2
            resolution += nk * (ok - base_rate) ** 2

    reliability /= n
    resolution /= n

    return {
        "brier_score": round(bs, 4),
        "reliability": round(reliability, 4),
        "resolution": round(resolution, 4),
        "uncertainty": round(uncertainty, 4),
    }
