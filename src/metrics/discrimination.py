"""
Discrimination metrics: AUROC2 (metacognitive sensitivity) and Goodman-Kruskal Gamma.

References:
- Fleming & Lau (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8:443.
- Nelson (1984). A comparison of current measures of the accuracy of FOK judgments.
- Goodman & Kruskal (1954). Measures of association for cross classifications.
"""
import numpy as np


def compute_auroc2(confidences, correctness):
    """
    Type 2 AUROC: metacognitive sensitivity (bias-free).

    Measures whether the model assigns higher confidence to correct responses
    than to incorrect ones. This is the SINGLE MOST IMPORTANT METRIC because
    it is unaffected by overall over/underconfidence tendency.

    Args:
        confidences: array-like of float (any scale, typically [0, 1])
        correctness: array-like of int/float in {0, 1}

    Returns:
        float: AUROC2 score. 0.5 = random, >0.7 = good, 1.0 = perfect.
    """
    from sklearn.metrics import roc_auc_score

    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

    if len(np.unique(corr)) < 2:
        return 0.5  # Cannot compute without both correct and incorrect trials

    return round(float(roc_auc_score(corr, conf)), 4)


def compute_gamma(confidences, correctness):
    """
    Goodman-Kruskal Gamma correlation for FOK accuracy.

    gamma = (concordant - discordant) / (concordant + discordant)

    A concordant pair: one correct and one incorrect trial where the correct
    trial has higher confidence. A discordant pair: the opposite.

    This is the standard measure for metamemory resolution (Nelson, 1984).
    Mathematical relationship: gamma = 2 * AUROC - 1

    Args:
        confidences: array-like of float
        correctness: array-like of int/float in {0, 1}

    Returns:
        float: Gamma in [-1, 1]. Typical human FOK: 0.30-0.60.
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

    # Optimized: separate correct and incorrect trials, then count pairs
    correct_conf = conf[corr == 1]
    incorrect_conf = conf[corr == 0]

    if len(correct_conf) == 0 or len(incorrect_conf) == 0:
        return 0.0

    concordant = 0
    discordant = 0

    for c in correct_conf:
        for ic in incorrect_conf:
            if c > ic:
                concordant += 1
            elif c < ic:
                discordant += 1
            # ties are excluded (standard practice)

    denom = concordant + discordant
    if denom == 0:
        return 0.0

    return round(float((concordant - discordant) / denom), 4)
