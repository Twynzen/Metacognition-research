# ========================================
# MetaCog-Bench: Measuring What AI Knows About What It Knows
# Track: Metacognition
# Competition: Google DeepMind "Measuring Progress Toward AGI"
# ========================================
#
# This benchmark tests 5 metacognitive abilities across ~900 items:
# 1. Confidence Calibration (Nelson & Narens, 1990)
# 2. Feeling-of-Knowing (Hart, 1965; Nelson & Dunlosky, 1991)
# 3. Error Detection (Yeung & Summerfield, 2012)
# 4. Selective Abstention (Koriat & Goldsmith, 1996)
# 5. Metacognitive Knowledge (Flavell, 1979; Dunning & Kruger, 1999)
#
# Metrics: ECE, Brier Score, AUROC2, Goodman-Kruskal Gamma
# Primary metric: AUROC2 (bias-free, Fleming & Lau, 2014)
# ========================================

import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from itertools import combinations
import random
import re
import unicodedata
import math
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
HAS_MATPLOTLIB = True

# ============================================================
# SCHEMAS
# ============================================================
"""
Structured output schemas for kaggle-benchmarks SDK.

These dataclasses are used with `llm.prompt(..., schema=SchemaName)` to force
structured JSON output from the model being evaluated.
"""


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


# ============================================================
# ANSWER CHECKER
# ============================================================
"""
Deterministic answer checker shared across all tasks.

CRITICAL: False negatives (marking correct answers wrong) corrupt ALL metrics.
This function must handle: exact match, case insensitive, numerical tolerance,
containment, yes/no keywords, and common normalizations.
"""


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


# ============================================================
# METRICS: CALIBRATION
# ============================================================
"""
Calibration metrics: ECE, MCE, Brier Score with Murphy decomposition.

References:
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Murphy (1973). A new vector partition of the probability score.
- Naeini et al. (2015). Obtaining well calibrated probabilities using Bayesian binning.
"""


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


# ============================================================
# METRICS: DISCRIMINATION
# ============================================================
"""
Discrimination metrics: AUROC2 (metacognitive sensitivity) and Goodman-Kruskal Gamma.

References:
- Fleming & Lau (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8:443.
- Nelson (1984). A comparison of current measures of the accuracy of FOK judgments.
- Goodman & Kruskal (1954). Measures of association for cross classifications.
"""


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


# ============================================================
# METRICS: AGGREGATE
# ============================================================
"""
Aggregate scoring: geometric mean composite and bootstrap confidence intervals.

The geometric mean prevents compensation between tasks — a model cannot
make up for poor calibration with good error detection.
"""


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


# ============================================================
# DATASET: CALIBRATION & FOK (300 items)
# ============================================================
"""
Calibration & FOK dataset generator.

Generates 300 questions across 3 domains (math, factual, logic) × 3 difficulties
(easy, medium, hard) with verifiably correct answers.

Used by Task 1 (Confidence Calibration) and Task 2 (Feeling-of-Knowing).
"""




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _largest_prime_factor(n: int) -> int:
    factor = 2
    largest = 1
    while factor * factor <= n:
        while n % factor == 0:
            largest = factor
            n //= factor
        factor += 1
    if n > 1:
        largest = n
    return largest


def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


# Pool of small-to-medium primes for hard math questions
_PRIMES = [p for p in range(2, 200) if _is_prime(p)]


# ---------------------------------------------------------------------------
# MATH domain generators
# ---------------------------------------------------------------------------

def _generate_math_easy(rng: random.Random, n: int) -> List[Dict]:
    """Simple arithmetic: addition, subtraction, multiplication."""
    items = []
    seen = set()
    ops = [
        ("addition", "+", lambda a, b: a + b),
        ("subtraction", "-", lambda a, b: a - b),
        ("multiplication", "×", lambda a, b: a * b),
    ]
    while len(items) < n:
        op_name, op_sym, op_fn = rng.choice(ops)
        if op_name == "multiplication":
            a, b = rng.randint(2, 30), rng.randint(2, 30)
        else:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        if op_name == "subtraction" and a < b:
            a, b = b, a  # keep positive
        key = (op_sym, a, b)
        if key in seen:
            continue
        seen.add(key)
        q = f"What is {a} {op_sym} {b}?"
        ans = str(op_fn(a, b))
        items.append({"question": q, "correct_answer": ans,
                       "difficulty": "easy", "domain": "math"})
    return items


def _generate_math_medium(rng: random.Random, n: int) -> List[Dict]:
    """Modular arithmetic, squaring, remainders."""
    items = []
    seen = set()
    templates = ["mod", "square", "remainder"]

    while len(items) < n:
        t = rng.choice(templates)

        if t == "mod":
            a = rng.randint(50, 999)
            b = rng.randint(50, 999)
            m = rng.randint(7, 23)
            key = ("mod", a, b, m)
            if key in seen:
                continue
            seen.add(key)
            ans = str((a * b) % m)
            q = f"What is ({a} × {b}) mod {m}?"
            items.append({"question": q, "correct_answer": ans,
                           "difficulty": "medium", "domain": "math"})

        elif t == "square":
            a = rng.randint(12, 99)
            key = ("sq", a)
            if key in seen:
                continue
            seen.add(key)
            q = f"What is {a} squared?"
            ans = str(a * a)
            items.append({"question": q, "correct_answer": ans,
                           "difficulty": "medium", "domain": "math"})

        elif t == "remainder":
            n_val = rng.randint(100, 9999)
            d = rng.randint(7, 37)
            key = ("rem", n_val, d)
            if key in seen:
                continue
            seen.add(key)
            q = f"What is the remainder when {n_val} is divided by {d}?"
            ans = str(n_val % d)
            items.append({"question": q, "correct_answer": ans,
                           "difficulty": "medium", "domain": "math"})
    return items


def _generate_math_hard(rng: random.Random, n: int) -> List[Dict]:
    """Prime factorization and combinatorics."""
    items = []
    seen = set()
    templates = ["prime_factor", "comb"]

    while len(items) < n:
        t = rng.choice(templates)

        if t == "prime_factor":
            primes = rng.sample(_PRIMES[5:], 3)  # skip 2,3,5,7,11
            primes.sort()
            product = primes[0] * primes[1] * primes[2]
            key = ("pf", product)
            if key in seen:
                continue
            seen.add(key)
            largest = max(primes)
            q = (f"What is the largest prime factor of {product}?")
            items.append({"question": q, "correct_answer": str(largest),
                           "difficulty": "hard", "domain": "math"})

        elif t == "comb":
            n_val = rng.randint(6, 15)
            k_val = rng.randint(2, min(n_val - 1, 5))
            key = ("comb", n_val, k_val)
            if key in seen:
                continue
            seen.add(key)
            ans = str(_comb(n_val, k_val))
            q = (f"How many ways can you choose {k_val} items from "
                 f"a set of {n_val}? (Give the exact number.)")
            items.append({"question": q, "correct_answer": ans,
                           "difficulty": "hard", "domain": "math"})
    return items


# ---------------------------------------------------------------------------
# FACTUAL domain — curated list with verified answers
# ---------------------------------------------------------------------------

_FACTUAL_EASY = [
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the chemical symbol for potassium?", "K"),
    ("How many continents are there on Earth?", "7"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the smallest planet in our solar system?", "Mercury"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Egypt?", "Cairo"),
    ("How many days are in a leap year?", "366"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the speed of light in km/s (approximately)?", "300000"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the chemical formula for water?", "H2O"),
    ("How many planets are in our solar system?", "8"),
    ("What is the capital of Italy?", "Rome"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
]

_FACTUAL_MEDIUM = [
    ("In what year was the Treaty of Westphalia signed?", "1648"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("What is the half-life of Carbon-14 in years (approximately)?", "5730"),
    ("What is the tallest mountain in Africa?", "Kilimanjaro"),
    ("What is the longest river in South America?", "Amazon"),
    ("Who wrote the novel 'Crime and Punishment'?", "Dostoevsky"),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo"),
    ("What element has the atomic number 79?", "Gold"),
    ("What element has the atomic number 26?", "Iron"),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen"),
    ("What year did the Titanic sink?", "1912"),
    ("What is the currency of Japan?", "Yen"),
    ("What is the currency of Thailand?", "Baht"),
    ("What is the chemical symbol for tungsten?", "W"),
    ("What is the chemical symbol for mercury?", "Hg"),
    ("What is the chemical symbol for tin?", "Sn"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the largest desert in the world?", "Sahara"),
    ("In what year did World War I begin?", "1914"),
    ("In what year did World War II end?", "1945"),
    ("Who formulated the three laws of motion?", "Newton"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the atomic number of oxygen?", "8"),
    ("Who discovered penicillin?", "Fleming"),
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the largest island in the world?", "Greenland"),
    ("In what year was the United Nations founded?", "1945"),
    ("Who wrote 'The Republic'?", "Plato"),
    ("What is the hardest naturally occurring mineral?", "Diamond"),
    ("What is the most abundant element in the universe?", "Hydrogen"),
    ("What language has the most native speakers worldwide?", "Mandarin"),
    ("What is the deepest ocean trench on Earth?", "Mariana Trench"),
    ("In what year did the French Revolution begin?", "1789"),
    ("Who composed 'The Four Seasons'?", "Vivaldi"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Nigeria?", "Abuja"),
    ("What is the capital of Pakistan?", "Islamabad"),
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("How many teeth does a typical adult human have?", "32"),
    ("What is the smallest bone in the human body?", "Stapes"),
    ("What is the largest organ in the human body?", "Skin"),
    ("In what year was the Declaration of Independence signed?", "1776"),
    ("What is the main component of the Sun?", "Hydrogen"),
    ("Who wrote 'Don Quixote'?", "Cervantes"),
    ("What year was the first successful powered airplane flight?", "1903"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("What is the SI unit of electrical resistance?", "Ohm"),
    ("What is the SI unit of force?", "Newton"),
    ("What is the SI unit of energy?", "Joule"),
    ("Who painted 'The Starry Night'?", "Van Gogh"),
    ("What is the longest bone in the human body?", "Femur"),
    ("How many valence electrons does carbon have?", "4"),
    ("What is the capital of Switzerland?", "Bern"),
    ("What year was the first Moon landing?", "1969"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the second largest country by area?", "Canada"),
    ("What is the most spoken language in South America?", "Portuguese"),
    ("In what year was the Communist Manifesto published?", "1848"),
    ("What is Avogadro's number (approximately, in 10^23)?", "6.022"),
    ("What is the melting point of iron in degrees Celsius (approximately)?", "1538"),
    ("What is the capital of Morocco?", "Rabat"),
    ("Who wrote 'War and Peace'?", "Tolstoy"),
    ("How many symphonies did Beethoven compose?", "9"),
    ("What is the chemical symbol for lead?", "Pb"),
    ("What is the speed of sound in air in m/s (approximately)?", "343"),
    ("What is the capital of Peru?", "Lima"),
    ("What is the most electronegative element?", "Fluorine"),
]

_FACTUAL_HARD = [
    ("What is the atomic number of Rutherfordium?", "104"),
    ("What is the atomic number of Seaborgium?", "106"),
    ("What is the atomic number of Hassium?", "108"),
    ("What is the atomic number of Meitnerium?", "109"),
    ("What year was the Treaty of Tordesillas signed?", "1494"),
    ("What year was the Edict of Nantes issued?", "1598"),
    ("What year was the Peace of Augsburg signed?", "1555"),
    ("Who was the first Mughal emperor?", "Babur"),
    ("What is the capital of Liechtenstein?", "Vaduz"),
    ("What is the capital of Bhutan?", "Thimphu"),
    ("What is the capital of Vanuatu?", "Port Vila"),
    ("What is the capital of Suriname?", "Paramaribo"),
    ("What element has the highest melting point?", "Tungsten"),
    ("What is the Mohs hardness of topaz?", "8"),
    ("What is the most abundant metal in Earth's crust?", "Aluminum"),
    ("What is the shortest-lived chemical element ever synthesized?", "Oganesson"),
    ("In what year was the Congress of Vienna concluded?", "1815"),
    ("Who was the last Emperor of the Byzantine Empire?", "Constantine XI"),
    ("What is the largest freshwater lake by surface area in Africa?", "Victoria"),
    ("What is the second longest river in Africa?", "Congo"),
    ("What is the capital of Eritrea?", "Asmara"),
    ("What is the capital of Brunei?", "Bandar Seri Begawan"),
    ("In what year was the Battle of Lepanto fought?", "1571"),
    ("Who wrote 'The Muqaddimah'?", "Ibn Khaldun"),
    ("What is the SI unit of luminous intensity?", "Candela"),
]


def _generate_factual(rng: random.Random, n_easy: int, n_medium: int,
                       n_hard: int) -> List[Dict]:
    easy_pool = list(_FACTUAL_EASY)
    medium_pool = list(_FACTUAL_MEDIUM)
    hard_pool = list(_FACTUAL_HARD)
    rng.shuffle(easy_pool)
    rng.shuffle(medium_pool)
    rng.shuffle(hard_pool)

    items = []
    for q, a in easy_pool[:n_easy]:
        items.append({"question": q, "correct_answer": a,
                       "difficulty": "easy", "domain": "factual"})
    for q, a in medium_pool[:n_medium]:
        items.append({"question": q, "correct_answer": a,
                       "difficulty": "medium", "domain": "factual"})
    for q, a in hard_pool[:n_hard]:
        items.append({"question": q, "correct_answer": a,
                       "difficulty": "hard", "domain": "factual"})
    return items


# ---------------------------------------------------------------------------
# LOGIC domain generators
# ---------------------------------------------------------------------------

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Irene", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul",
    "Quinn", "Rosa", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander",
    "Yuki", "Zane",
]

_CATEGORIES = [
    ("dogs", "mammals"), ("cats", "animals"), ("roses", "flowers"),
    ("sparrows", "birds"), ("salmon", "fish"), ("oaks", "trees"),
    ("apples", "fruits"), ("carrots", "vegetables"), ("pythons", "snakes"),
    ("eagles", "birds"),
]

_PROPERTIES = [
    ("taller", "shortest", "tallest"),
    ("heavier", "lightest", "heaviest"),
    ("older", "youngest", "oldest"),
    ("faster", "slowest", "fastest"),
]


def _generate_logic_easy(rng: random.Random, n: int) -> List[Dict]:
    """Simple syllogisms with parameterized names/categories."""
    items = []
    seen = set()

    while len(items) < n:
        name = rng.choice(_NAMES)
        member, category = rng.choice(_CATEGORIES)
        key = (name, member, category)
        if key in seen:
            continue
        seen.add(key)

        singular_member = member[:-1] if member.endswith('s') else member
        singular_category = category[:-1] if category.endswith('s') else category
        article = "an" if singular_category[0].lower() in "aeiou" else "a"
        q = (f"If all {member} are {category} and {name}'s pet is a "
             f"{singular_member}, "
             f"is {name}'s pet {article} {singular_category}?")
        items.append({"question": q, "correct_answer": "Yes",
                       "difficulty": "easy", "domain": "logic"})
    return items


def _generate_logic_medium(rng: random.Random, n: int) -> List[Dict]:
    """Multi-step ordering comparisons."""
    items = []
    seen = set()

    while len(items) < n:
        prop_comp, prop_min, prop_max = rng.choice(_PROPERTIES)
        num_people = rng.randint(4, 6)
        names = rng.sample(_NAMES, num_people)
        # Build a random total ordering
        ordering = list(names)
        rng.shuffle(ordering)
        # ordering[0] has the least of the property, ordering[-1] has the most

        # Build pairwise clues: we reveal a spanning set of comparisons
        # that uniquely determines the order
        clues = []
        revealed = set()
        # Ensure a connected chain: compare adjacent pairs, then add a few extras
        indices = list(range(num_people))
        rng.shuffle(indices)
        # Chain in random presentation order but covering adjacent pairs
        for i in range(num_people - 1):
            a_idx, b_idx = i, i + 1
            a_name, b_name = ordering[a_idx], ordering[b_idx]
            # a < b in the property
            clues.append(f"{b_name} is {prop_comp} than {a_name}")
            revealed.add((a_idx, b_idx))

        rng.shuffle(clues)
        clue_text = ". ".join(clues) + "."

        # Pick a question type
        q_type = rng.choice(["min", "max"])
        if q_type == "min":
            answer = ordering[0]
            q = f"{clue_text} Who is {prop_min}?"
        else:
            answer = ordering[-1]
            q = f"{clue_text} Who is {prop_max}?"

        key = q
        if key in seen:
            continue
        seen.add(key)

        items.append({"question": q, "correct_answer": answer,
                       "difficulty": "medium", "domain": "logic"})
    return items


def _generate_logic_hard(rng: random.Random, n: int) -> List[Dict]:
    """Constraint satisfaction with 4+ variables — seating / scheduling puzzles."""
    items = []
    seen = set()

    # Template 1: Seating order with constraints
    # Template 2: Who has what item
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    animals = ["dog", "cat", "bird", "fish", "rabbit", "hamster"]
    drinks = ["water", "coffee", "tea", "juice", "milk", "soda"]

    while len(items) < n:
        template = rng.choice(["seating", "assignment"])

        if template == "seating":
            # 4 people in a row, determine positions from constraints
            names = rng.sample(_NAMES, 4)
            order = list(names)
            rng.shuffle(order)  # order[0]=position 1, order[3]=position 4

            clues = []
            # Clue 1: someone is at an end
            end_person = rng.choice([0, 3])
            pos_label = "first" if end_person == 0 else "last"
            clues.append(f"{order[end_person]} sits in the {pos_label} position")

            # Clue 2: two people are adjacent
            adj_start = rng.randint(0, 2)
            clues.append(
                f"{order[adj_start]} sits directly to the left of {order[adj_start + 1]}"
            )

            # Clue 3: someone is NOT adjacent to someone else
            # Pick two non-adjacent people
            non_adj_pairs = [(i, j) for i in range(4) for j in range(4)
                             if abs(i - j) > 1 and i < j]
            if non_adj_pairs:
                i, j = rng.choice(non_adj_pairs)
                clues.append(
                    f"{order[i]} does not sit next to {order[j]}"
                )

            # Clue 4: position clue
            mid = rng.choice([1, 2])
            clues.append(f"{order[mid]} sits in position {mid + 1}")

            rng.shuffle(clues)
            clue_text = ". ".join(clues) + "."

            # Ask about a remaining person's position
            asked = rng.choice(range(4))
            q = (f"Four people sit in a row (positions 1-4, left to right). "
                 f"{clue_text} What position does {order[asked]} sit in?")
            answer = str(asked + 1)

        else:  # assignment
            # 3 people each have a unique color
            names = rng.sample(_NAMES, 3)
            chosen_colors = rng.sample(colors, 3)
            assignment = dict(zip(names, chosen_colors))

            clues = []
            # Clue 1: one person does NOT have a specific color
            wrong_person = rng.choice(names)
            wrong_colors = [c for c in chosen_colors if c != assignment[wrong_person]]
            clues.append(f"{wrong_person} does not have {rng.choice(wrong_colors)}")

            # Clue 2: direct assignment
            direct = rng.choice(names)
            clues.append(f"{direct} has {assignment[direct]}")

            # Clue 3: elimination
            other = [nm for nm in names if nm != direct][0]
            not_color = [c for c in chosen_colors if c != assignment[other]
                         and c != assignment[direct]]
            if not_color:
                clues.append(f"{other} does not have {not_color[0]}")

            rng.shuffle(clues)
            clue_text = ". ".join(clues) + "."

            # Ask about someone
            ask_name = rng.choice(names)
            q = (f"Three people ({', '.join(names)}) each have a different color "
                 f"from {{{', '.join(chosen_colors)}}}. "
                 f"{clue_text} What color does {ask_name} have?")
            answer = assignment[ask_name]

        if q in seen:
            continue
        seen.add(q)

        items.append({"question": q, "correct_answer": answer,
                       "difficulty": "hard", "domain": "logic"})
    return items


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_calibration_dataset(n: int = 300) -> pd.DataFrame:
    """
    Generate a calibration/FOK dataset of ``n`` items (default 300).

    Returns a DataFrame with columns:
        question, correct_answer, difficulty, domain

    Difficulty distribution per domain (100 items each):
        easy ~20%, medium ~60%, hard ~20%
    """
    rng = random.Random(42)
    # Temporarily override module-level random for reproducibility
    old_state = random.getstate()
    random.seed(42)

    per_domain = n // 3  # 100
    n_easy = round(per_domain * 0.20)   # 20
    n_medium = round(per_domain * 0.60)  # 60
    n_hard = per_domain - n_easy - n_medium  # 20

    all_items: List[Dict] = []

    # Math domain
    all_items.extend(_generate_math_easy(rng, n_easy))
    all_items.extend(_generate_math_medium(rng, n_medium))
    all_items.extend(_generate_math_hard(rng, n_hard))

    # Factual domain
    all_items.extend(_generate_factual(rng, n_easy, n_medium, n_hard))

    # Logic domain
    all_items.extend(_generate_logic_easy(rng, n_easy))
    all_items.extend(_generate_logic_medium(rng, n_medium))
    all_items.extend(_generate_logic_hard(rng, n_hard))

    random.setstate(old_state)

    df = pd.DataFrame(all_items)

    # Verify no duplicates
    assert df["question"].nunique() == len(df), (
        f"Duplicate questions detected: {len(df) - df['question'].nunique()} duplicates"
    )
    assert len(df) == n, f"Expected {n} items, got {len(df)}"

    return df


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------


# ============================================================
# DATASET: ERROR DETECTION (200 items)
# ============================================================
"""
Error Detection dataset generator for MetaCog-Bench.

Generates 200 items: 100 correct solutions + 100 with planted (plausible) errors.
Error types: arithmetic (~35), logical (~25), method (~20), factual (~20).
"""



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prime_factors(n: int) -> List[int]:
    """Return sorted list of prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


# ---------------------------------------------------------------------------
# Arithmetic problem generators
# ---------------------------------------------------------------------------

def _arith_multiplication(rng: random.Random, inject_error: bool) -> Dict:
    a = rng.randint(12, 99)
    b = rng.randint(12, 99)
    correct = a * b
    problem = f"What is {a} × {b}?"
    if inject_error:
        # Near-miss: offset by a small plausible amount
        offsets = [10, -10, 1, -1, a, -a, b, -b]
        offset = rng.choice([o for o in offsets if o != 0])
        wrong = correct + offset
        sol = f"{a} × {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} × {b} = {correct}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_addition_chain(rng: random.Random, inject_error: bool) -> Dict:
    nums = [rng.randint(10, 999) for _ in range(rng.randint(3, 5))]
    correct = sum(nums)
    expr = " + ".join(str(n) for n in nums)
    problem = f"What is {expr}?"
    if inject_error:
        wrong = correct + rng.choice([1, -1, 10, -10])
        sol = f"{expr} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{expr} = {correct}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_subtraction(rng: random.Random, inject_error: bool) -> Dict:
    a = rng.randint(100, 9999)
    b = rng.randint(10, a - 1)
    correct = a - b
    problem = f"What is {a} − {b}?"
    if inject_error:
        wrong = correct + rng.choice([1, -1, 10, -10, 100, -100])
        sol = f"{a} − {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} − {b} = {correct}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_division(rng: random.Random, inject_error: bool) -> Dict:
    b = rng.randint(2, 25)
    quotient = rng.randint(5, 200)
    a = b * quotient
    problem = f"What is {a} ÷ {b}?"
    if inject_error:
        wrong = quotient + rng.choice([1, -1, 2, -2])
        sol = f"{a} ÷ {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} ÷ {b} = {quotient}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_modular(rng: random.Random, inject_error: bool) -> Dict:
    a = rng.randint(50, 500)
    b = rng.randint(10, 99)
    m = rng.choice([7, 11, 13, 17, 19, 23])
    correct = (a * b) % m
    problem = f"What is ({a} × {b}) mod {m}?"
    if inject_error:
        wrong = (correct + rng.randint(1, m - 1)) % m
        if wrong == correct:
            wrong = (correct + 1) % m
        sol = (f"({a} × {b}) = {a * b}. "
               f"{a * b} mod {m} = {wrong}")
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = (f"({a} × {b}) = {a * b}. "
               f"{a * b} mod {m} = {correct}")
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_exponentiation(rng: random.Random, inject_error: bool) -> Dict:
    base = rng.randint(2, 12)
    exp = rng.randint(2, 4)
    correct = base ** exp
    problem = f"What is {base}^{exp}?"
    if inject_error:
        wrong = correct + rng.choice([1, -1, base, -base])
        sol = f"{base}^{exp} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{base}^{exp} = {correct}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_percentage(rng: random.Random, inject_error: bool) -> Dict:
    pct = rng.choice([5, 10, 12, 15, 20, 25, 30, 40])
    base = rng.randint(2, 50) * 10
    correct = base * pct / 100
    problem = f"What is {pct}% of {base}?"
    if inject_error:
        # Common mistake: misplace decimal or swap pct/base
        wrong_options = [base * pct / 1000, base * (pct + 10) / 100,
                         base * (pct - 5) / 100]
        wrong = rng.choice([w for w in wrong_options if w != correct and w > 0])
        if wrong == int(wrong):
            wrong = int(wrong)
        sol = f"{pct}% of {base} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        disp = int(correct) if correct == int(correct) else correct
        sol = f"{pct}% of {base} = {disp}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


_ARITH_GENERATORS = [
    _arith_multiplication,
    _arith_addition_chain,
    _arith_subtraction,
    _arith_division,
    _arith_modular,
    _arith_exponentiation,
    _arith_percentage,
]


# ---------------------------------------------------------------------------
# Logical problem generators
# ---------------------------------------------------------------------------

def _logic_affirming_consequent(rng: random.Random) -> Dict:
    """Affirming the consequent: if P→Q and Q, conclude P (invalid)."""
    templates = [
        {"p": "it is raining", "q": "the ground is wet"},
        {"p": "an animal is a dog", "q": "it is a mammal"},
        {"p": "someone studies hard", "q": "they pass the exam"},
        {"p": "a shape is a square", "q": "it has four sides"},
        {"p": "a number is divisible by 4", "q": "it is even"},
        {"p": "someone is a doctor", "q": "they have a university degree"},
        {"p": "a fruit is a banana", "q": "it is yellow"},
        {"p": "someone lives in Paris", "q": "they live in France"},
        {"p": "a vehicle is a Tesla", "q": "it is electric"},
        {"p": "an element is iron", "q": "it is a metal"},
        {"p": "a creature is a spider", "q": "it has eight legs"},
        {"p": "a liquid is bleach", "q": "it is a disinfectant"},
        {"p": "someone is a surgeon", "q": "they have medical training"},
        {"p": "a triangle is equilateral", "q": "it is isosceles"},
    ]
    t = rng.choice(templates)
    problem = (
        f"Premise 1: If {t['p']}, then {t['q']}. "
        f"Premise 2: {t['q'].capitalize()}. "
        f"Conclusion: Therefore, {t['p']}. "
        f"Is this argument logically valid?"
    )
    sol = (
        f"Yes, this argument is valid. Since {t['q']} and "
        f"we know that {t['p']} implies {t['q']}, "
        f"it follows that {t['p']}."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "true",
        "error_type": "logical",
    }


def _logic_false_dichotomy(rng: random.Random) -> Dict:
    """False dichotomy: presenting only 2 options when more exist."""
    templates = [
        {
            "setup": "A student can either study literature or study science",
            "conclusion": "Since they are not studying literature, they must be studying science",
        },
        {
            "setup": "A person is either happy or sad",
            "conclusion": "Since they are not happy, they must be sad",
        },
        {
            "setup": "An employee either supports the new policy or opposes the company",
            "conclusion": "Since they don't support the new policy, they oppose the company",
        },
        {
            "setup": "A country is either a democracy or a dictatorship",
            "conclusion": "Since it is not a full democracy, it must be a dictatorship",
        },
        {
            "setup": "A food is either healthy or unhealthy",
            "conclusion": "Since chocolate is not purely healthy, it must be unhealthy",
        },
        {
            "setup": "Software is either perfectly secure or completely vulnerable",
            "conclusion": "Since the software has a minor bug, it is completely vulnerable",
        },
        {
            "setup": "You either agree with me completely or you are against me",
            "conclusion": "Since you raised an objection, you must be against me",
        },
        {
            "setup": "A movie is either a masterpiece or garbage",
            "conclusion": "Since critics found some flaws, the movie must be garbage",
        },
        {
            "setup": "An investment either guarantees profit or is a total loss",
            "conclusion": "Since the investment doesn't guarantee profit, it is a total loss",
        },
        {
            "setup": "A person is either an expert or completely ignorant",
            "conclusion": "Since they made one mistake, they must be completely ignorant",
        },
        {
            "setup": "A scientific theory is either proven or worthless",
            "conclusion": "Since the theory has not been proven beyond all doubt, it is worthless",
        },
        {
            "setup": "An athlete is either world-class or terrible",
            "conclusion": "Since they did not win the championship, they must be terrible",
        },
    ]
    t = rng.choice(templates)
    problem = (
        f"Evaluate this reasoning: '{t['setup']}. {t['conclusion']}.' "
        f"Is this logically valid?"
    )
    sol = (
        f"Yes, this reasoning is valid. The two options are exhaustive, "
        f"so eliminating one necessarily leaves the other."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "true",
        "error_type": "logical",
    }


def _logic_invalid_syllogism(rng: random.Random) -> Dict:
    """Undistributed middle: All A are B. All C are B. Therefore all C are A."""
    templates = [
        ("cats", "animals", "dogs"),
        ("roses", "plants", "trees"),
        ("cars", "vehicles", "trucks"),
        ("apples", "fruits", "bananas"),
        ("novels", "books", "textbooks"),
        ("guitars", "instruments", "pianos"),
        ("eagles", "birds", "sparrows"),
        ("Python", "programming languages", "Java"),
        ("squares", "rectangles", "parallelograms"),
        ("diamonds", "gemstones", "rubies"),
        ("salmon", "fish", "tuna"),
        ("oak", "trees", "maple"),
        ("violins", "string instruments", "cellos"),
        ("Mars", "planets", "Venus"),
    ]
    a, b, c = rng.choice(templates)
    problem = (
        f"Premise 1: All {a} are {b}. "
        f"Premise 2: All {c} are {b}. "
        f"Conclusion: Therefore, all {c} are {a}. "
        f"Is this syllogism valid?"
    )
    sol = (
        f"Yes, this syllogism is valid. Since all {a} are {b} "
        f"and all {c} are {b}, the {c} must also be {a}."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "true",
        "error_type": "logical",
    }


def _logic_denying_antecedent(rng: random.Random) -> Dict:
    """Denying the antecedent: If P→Q, not P, therefore not Q (invalid)."""
    templates = [
        ("it snows", "the schools close", "the schools are open"),
        ("you eat too much sugar", "you may get cavities", "you won't get cavities"),
        ("the temperature drops below 0 degrees C", "water freezes",
         "water doesn't freeze"),
        ("you practice daily", "you improve", "you won't improve"),
        ("a triangle is equilateral", "all its angles are 60 degrees",
         "none of its angles are 60 degrees"),
        ("a plant gets sunlight", "it grows", "it won't grow"),
        ("the alarm rings", "people evacuate", "people won't evacuate"),
        ("you take the medicine", "you feel better", "you won't feel better"),
        ("a student attends lectures", "they learn the material",
         "they won't learn the material"),
        ("iron is exposed to moisture", "it rusts", "it won't rust"),
        ("you add fertilizer", "the crops yield more",
         "the crops won't yield more"),
        ("a country invests in education", "literacy rates improve",
         "literacy rates won't improve"),
        ("a battery is charged", "the device works", "the device won't work"),
    ]
    p, q, notq = rng.choice(templates)
    problem = (
        f"Premise 1: If {p}, then {q}. "
        f"Premise 2: It is not the case that {p}. "
        f"Conclusion: Therefore, {notq}. "
        f"Is this argument logically valid?"
    )
    sol = (
        f"Yes, the argument is valid. We know that {p} leads to {q}. "
        f"Since {p} is not the case, {notq}."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "true",
        "error_type": "logical",
    }


def _logic_correct_modus_ponens(rng: random.Random) -> Dict:
    """A valid modus ponens — correct solution."""
    templates = [
        ("it is raining", "the ground is wet"),
        ("a number is divisible by 6", "it is divisible by 3"),
        ("all sides of a polygon are equal", "the polygon is equilateral"),
        ("an object is made of pure gold", "it is denser than water"),
        ("today is Saturday", "tomorrow is Sunday"),
        ("a substance is an acid", "it has a pH less than 7"),
        ("a triangle has all angles equal to 60 degrees", "it is equilateral"),
        ("a metal is heated", "it expands"),
        ("someone is a bachelor", "they are unmarried"),
        ("an integer ends in 0", "it is divisible by 10"),
        ("a figure has exactly three sides", "it is a triangle"),
        ("water is cooled below 0 degrees Celsius", "it freezes"),
        ("a creature is a whale", "it is a mammal"),
        ("a language is a Romance language", "it derives from Latin"),
        ("a gas is helium", "it is a noble gas"),
        ("today is December 31", "tomorrow is January 1"),
        ("an element is in group 18", "it is a noble gas"),
        ("a polygon has four equal sides and four right angles", "it is a square"),
        ("a material is rubber", "it is an electrical insulator"),
        ("a number is a multiple of 10", "it is even"),
    ]
    p, q = rng.choice(templates)
    problem = (
        f"Premise 1: If {p}, then {q}. "
        f"Premise 2: {p.capitalize()}. "
        f"Conclusion: Therefore, {q}. "
        f"Is this argument valid?"
    )
    sol = (
        f"Yes, this is a valid modus ponens argument. "
        f"Since {p} is true and {p} implies {q}, we can conclude that {q}."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "false",
        "error_type": "none",
    }


def _logic_correct_modus_tollens(rng: random.Random) -> Dict:
    """A valid modus tollens — correct solution."""
    templates = [
        ("it is raining", "the ground is wet", "the ground is not wet",
         "it is not raining"),
        ("a number is prime and greater than 2", "it is odd",
         "the number is not odd", "it is not a prime greater than 2"),
        ("an animal is a fish", "it lives in water",
         "the animal does not live in water", "it is not a fish"),
        ("a substance is pure water", "it is odorless",
         "the substance is not odorless", "it is not pure water"),
        ("a shape is a circle", "it has no corners",
         "the shape has corners", "it is not a circle"),
        ("someone passed the bar exam", "they studied law",
         "they did not study law", "they did not pass the bar exam"),
        ("an element is sodium", "it reacts vigorously with water",
         "it does not react vigorously with water", "it is not sodium"),
        ("a number is divisible by 9", "it is divisible by 3",
         "the number is not divisible by 3", "it is not divisible by 9"),
        ("a bird is a penguin", "it cannot fly",
         "the bird can fly", "it is not a penguin"),
        ("a vehicle is a submarine", "it can travel underwater",
         "the vehicle cannot travel underwater", "it is not a submarine"),
        ("a planet is Mercury", "it is closest to the Sun",
         "the planet is not closest to the Sun", "it is not Mercury"),
        ("a compound is carbon dioxide", "it contains carbon",
         "the compound does not contain carbon", "it is not carbon dioxide"),
        ("an organism is a plant", "it performs photosynthesis",
         "the organism does not perform photosynthesis", "it is not a plant"),
        ("a material is glass", "it is brittle",
         "the material is not brittle", "it is not glass"),
        ("today is a weekday", "the office is open",
         "the office is not open", "today is not a weekday"),
    ]
    p, q, notq, notp = rng.choice(templates)
    problem = (
        f"Premise 1: If {p}, then {q}. "
        f"Premise 2: {notq.capitalize()}. "
        f"Conclusion: Therefore, {notp}. "
        f"Is this argument valid?"
    )
    sol = (
        f"Yes, this is a valid modus tollens argument. "
        f"Since {q} must follow from {p}, and {notq}, we conclude {notp}."
    )
    return {
        "problem": problem,
        "presented_solution": sol,
        "solution_has_error": "false",
        "error_type": "none",
    }


# ---------------------------------------------------------------------------
# Method-error generators
# ---------------------------------------------------------------------------

def _method_average_speed(rng: random.Random, inject_error: bool) -> Dict:
    """Average speed ≠ (v1+v2)/2 when distances are equal."""
    d = rng.choice([60, 100, 120, 150, 180, 200, 240])
    v1 = rng.choice([30, 40, 50, 60])
    v2 = rng.choice([v for v in [60, 80, 90, 100, 120] if v != v1])
    # Correct: total_distance / total_time
    t1 = d / v1
    t2 = d / v2
    correct_avg = round(2 * d / (t1 + t2), 2)
    wrong_avg = (v1 + v2) / 2

    problem = (
        f"A car travels {d} km at {v1} km/h and then {d} km at {v2} km/h. "
        f"What is the average speed for the whole trip?"
    )
    if inject_error:
        sol = (
            f"Average speed = ({v1} + {v2}) / 2 = {wrong_avg} km/h. "
            f"We simply take the mean of the two speeds."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"Time for first half = {d}/{v1} = {round(t1, 4)} h. "
            f"Time for second half = {d}/{v2} = {round(t2, 4)} h. "
            f"Average speed = total distance / total time "
            f"= {2 * d} / {round(t1 + t2, 4)} = {correct_avg} km/h."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_probability_or(rng: random.Random, inject_error: bool) -> Dict:
    """P(A or B) ≠ P(A) + P(B) when events aren't mutually exclusive."""
    pa_num = rng.randint(2, 5)
    pb_num = rng.randint(2, 5)
    pab_num = rng.randint(1, min(pa_num, pb_num) - 1) if min(pa_num, pb_num) > 1 else 1
    denom = 10
    pa = pa_num / denom
    pb = pb_num / denom
    pab = pab_num / denom
    correct = pa + pb - pab

    problem = (
        f"In a class, the probability of a student playing football is {pa}, "
        f"the probability of playing basketball is {pb}, and the probability "
        f"of playing both is {pab}. What is the probability a student plays "
        f"football or basketball?"
    )
    if inject_error:
        wrong = pa + pb  # forgot to subtract intersection
        sol = (
            f"P(football or basketball) = P(football) + P(basketball) "
            f"= {pa} + {pb} = {round(wrong, 2)}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"P(A or B) = P(A) + P(B) - P(A and B) "
            f"= {pa} + {pb} - {pab} = {round(correct, 2)}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_percentage_change(rng: random.Random, inject_error: bool) -> Dict:
    """% increase then same % decrease does NOT return to original."""
    pct = rng.choice([10, 20, 25, 30, 40, 50])
    original = rng.choice([100, 200, 400, 500, 1000])
    after_increase = original * (1 + pct / 100)
    after_decrease = after_increase * (1 - pct / 100)
    net_change = after_decrease - original

    problem = (
        f"A product costs ${original}. The price increases by {pct}%, "
        f"and later decreases by {pct}%. What is the final price?"
    )
    if inject_error:
        sol = (
            f"The price increases by {pct}% and then decreases by the same "
            f"{pct}%, so the effects cancel out. The final price is ${original}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"After {pct}% increase: ${original} × {1 + pct / 100} "
            f"= ${after_increase:.2f}. "
            f"After {pct}% decrease: ${after_increase:.2f} × {1 - pct / 100} "
            f"= ${after_decrease:.2f}. "
            f"Net change: ${net_change:.2f}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_combination_vs_permutation(rng: random.Random, inject_error: bool) -> Dict:
    """Using permutations when combinations are needed, or vice versa."""
    n = rng.randint(5, 12)
    k = rng.randint(2, min(4, n - 1))
    correct_comb = _comb(n, k)
    wrong_perm = math.factorial(n) // math.factorial(n - k)

    problem = (
        f"How many ways can you choose {k} people from a group "
        f"of {n} to form a committee?"
    )
    if inject_error:
        sol = (
            f"We need to pick {k} from {n}. "
            f"The number of ways = {n}! / ({n}-{k})! = "
            f"{n}! / {n - k}! = {wrong_perm}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"C({n},{k}) = {n}! / ({k}! × ({n}-{k})!) = {correct_comb}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_base_rate_neglect(rng: random.Random, inject_error: bool) -> Dict:
    """Ignoring base rate in conditional probability."""
    disease_rate = rng.choice([1, 2, 5])  # per 1000
    sensitivity = rng.choice([95, 98, 99])
    false_pos = rng.choice([3, 5, 10])

    # P(disease|positive) by Bayes
    p_d = disease_rate / 1000
    p_pos_given_d = sensitivity / 100
    p_pos_given_not_d = false_pos / 100
    p_pos = p_d * p_pos_given_d + (1 - p_d) * p_pos_given_not_d
    correct_ppv = round(p_d * p_pos_given_d / p_pos * 100, 1)

    problem = (
        f"A disease affects {disease_rate} in 1000 people. A test has "
        f"{sensitivity}% sensitivity and a {false_pos}% false positive rate. "
        f"If someone tests positive, what is the probability they have "
        f"the disease?"
    )
    if inject_error:
        sol = (
            f"The test is {sensitivity}% accurate, so if someone tests "
            f"positive, there is approximately a {sensitivity}% chance "
            f"they have the disease."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"Using Bayes' theorem: P(disease|+) = "
            f"P(+|disease)×P(disease) / P(+). "
            f"P(+) = ({p_pos_given_d}×{p_d}) + ({p_pos_given_not_d}×{1-p_d:.4f}) "
            f"= {p_pos:.6f}. "
            f"P(disease|+) = {p_d*p_pos_given_d:.6f}/{p_pos:.6f} ≈ {correct_ppv}%."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


_METHOD_GENERATORS = [
    _method_average_speed,
    _method_probability_or,
    _method_percentage_change,
    _method_combination_vs_permutation,
    _method_base_rate_neglect,
]


# ---------------------------------------------------------------------------
# Factual-error generators
# ---------------------------------------------------------------------------

_FACTUAL_ITEMS = [
    # (problem, correct_solution, wrong_solution)
    (
        "In what year was the Declaration of Independence signed?",
        "The Declaration of Independence was signed in 1776.",
        "The Declaration of Independence was signed in 1774.",
    ),
    (
        "Who wrote the novel '1984'?",
        "'1984' was written by George Orwell.",
        "'1984' was written by Aldous Huxley.",
    ),
    (
        "What is the chemical formula for table salt?",
        "The chemical formula for table salt is NaCl (sodium chloride).",
        "The chemical formula for table salt is KCl (potassium chloride).",
    ),
    (
        "What is the speed of light in a vacuum, approximately?",
        "The speed of light in a vacuum is approximately 3 × 10^8 m/s (299,792,458 m/s).",
        "The speed of light in a vacuum is approximately 3 × 10^6 m/s (2,997,924 m/s).",
    ),
    (
        "Which planet is the largest in our solar system?",
        "Jupiter is the largest planet in our solar system.",
        "Saturn is the largest planet in our solar system.",
    ),
    (
        "What is the boiling point of water at sea level in Celsius?",
        "The boiling point of water at sea level is 100°C.",
        "The boiling point of water at sea level is 110°C.",
    ),
    (
        "Who painted the Sistine Chapel ceiling?",
        "The Sistine Chapel ceiling was painted by Michelangelo.",
        "The Sistine Chapel ceiling was painted by Raphael.",
    ),
    (
        "What is the atomic number of carbon?",
        "The atomic number of carbon is 6.",
        "The atomic number of carbon is 8.",
    ),
    (
        "In which year did World War I begin?",
        "World War I began in 1914.",
        "World War I began in 1912.",
    ),
    (
        "What is the smallest prime number?",
        "The smallest prime number is 2.",
        "The smallest prime number is 1.",
    ),
    (
        "Who discovered penicillin?",
        "Penicillin was discovered by Alexander Fleming in 1928.",
        "Penicillin was discovered by Louis Pasteur in 1928.",
    ),
    (
        "What is the capital of Australia?",
        "The capital of Australia is Canberra.",
        "The capital of Australia is Sydney.",
    ),
    (
        "How many chromosomes do humans have?",
        "Humans have 46 chromosomes (23 pairs).",
        "Humans have 48 chromosomes (24 pairs).",
    ),
    (
        "What is the most abundant gas in Earth's atmosphere?",
        "Nitrogen is the most abundant gas in Earth's atmosphere, at about 78%.",
        "Oxygen is the most abundant gas in Earth's atmosphere, at about 78%.",
    ),
    (
        "Who formulated the three laws of motion?",
        "The three laws of motion were formulated by Isaac Newton.",
        "The three laws of motion were formulated by Galileo Galilei.",
    ),
    (
        "What is the pH of pure water at 25°C?",
        "The pH of pure water at 25°C is 7.",
        "The pH of pure water at 25°C is 7.4.",
    ),
    (
        "In what year did the Berlin Wall fall?",
        "The Berlin Wall fell in 1989.",
        "The Berlin Wall fell in 1991.",
    ),
    (
        "What is the longest river in the world?",
        "The Nile is generally considered the longest river in the world at about 6,650 km.",
        "The Amazon is the longest river in the world at about 6,650 km.",
    ),
    (
        "Who developed the theory of general relativity?",
        "The theory of general relativity was developed by Albert Einstein, published in 1915.",
        "The theory of general relativity was developed by Albert Einstein, published in 1905.",
    ),
    (
        "What is the electron configuration of helium?",
        "The electron configuration of helium is 1s².",
        "The electron configuration of helium is 1s¹.",
    ),
    (
        "Which element has the atomic number 79?",
        "Gold (Au) has the atomic number 79.",
        "Silver (Ag) has the atomic number 79.",
    ),
    (
        "What is the diameter of Earth approximately?",
        "Earth's diameter is approximately 12,742 km.",
        "Earth's diameter is approximately 10,742 km.",
    ),
    (
        "Who wrote 'The Republic'?",
        "'The Republic' was written by Plato.",
        "'The Republic' was written by Aristotle.",
    ),
    (
        "What is the freezing point of mercury?",
        "The freezing point of mercury is approximately -39°C (-38.83°C).",
        "The freezing point of mercury is approximately -29°C.",
    ),
    (
        "In what year was the Magna Carta signed?",
        "The Magna Carta was signed in 1215.",
        "The Magna Carta was signed in 1225.",
    ),
    (
        "What is the half-life of Carbon-14?",
        "The half-life of Carbon-14 is approximately 5,730 years.",
        "The half-life of Carbon-14 is approximately 5,370 years.",
    ),
    (
        "Which country has the largest land area?",
        "Russia has the largest land area of any country, at about 17.1 million km².",
        "Canada has the largest land area of any country, at about 17.1 million km².",
    ),
    (
        "What is the value of Avogadro's number?",
        "Avogadro's number is approximately 6.022 × 10^23.",
        "Avogadro's number is approximately 6.022 × 10^26.",
    ),
    (
        "Who composed 'The Four Seasons'?",
        "'The Four Seasons' was composed by Antonio Vivaldi.",
        "'The Four Seasons' was composed by Johann Sebastian Bach.",
    ),
    (
        "What is the hardest natural mineral?",
        "Diamond is the hardest natural mineral, rated 10 on the Mohs scale.",
        "Corundum is the hardest natural mineral, rated 10 on the Mohs scale.",
    ),
    (
        "In which year did the French Revolution begin?",
        "The French Revolution began in 1789.",
        "The French Revolution began in 1793.",
    ),
    (
        "What is the SI unit of electric current?",
        "The SI unit of electric current is the ampere (A).",
        "The SI unit of electric current is the volt (V).",
    ),
    (
        "How many bones are in the adult human body?",
        "The adult human body has 206 bones.",
        "The adult human body has 208 bones.",
    ),
    (
        "Who invented the telephone?",
        "Alexander Graham Bell is credited with inventing the telephone in 1876.",
        "Thomas Edison is credited with inventing the telephone in 1876.",
    ),
    (
        "What is the formula for the area of a circle?",
        "The area of a circle is A = πr².",
        "The area of a circle is A = 2πr.",
    ),
    (
        "What is the deepest point in the ocean?",
        "The Mariana Trench's Challenger Deep is the deepest point, at about 10,935 m.",
        "The Mariana Trench's Challenger Deep is the deepest point, at about 8,935 m.",
    ),
    (
        "Who was the first person to walk on the Moon?",
        "Neil Armstrong was the first person to walk on the Moon on July 20, 1969.",
        "Neil Armstrong was the first person to walk on the Moon on July 20, 1968.",
    ),
    (
        "What is the currency of Japan?",
        "The currency of Japan is the Japanese yen (¥ / JPY).",
        "The currency of Japan is the Japanese won (₩ / JPW).",
    ),
    (
        "What is the tallest mountain in the world?",
        "Mount Everest is the tallest mountain, at 8,849 metres above sea level.",
        "Mount Everest is the tallest mountain, at 8,649 metres above sea level.",
    ),
    (
        "Who painted the Mona Lisa?",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "The Mona Lisa was painted by Leonardo da Vinci in 1623.",
    ),
]


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def _generate_unique(rng, generator, used_problems, count,
                     inject_error=None, max_retries=50):
    """Generate items ensuring no duplicate problems."""
    items = []
    for _ in range(count):
        for _attempt in range(max_retries):
            if inject_error is not None:
                item = generator(rng, inject_error)
            else:
                item = generator(rng)
            if item["problem"] not in used_problems:
                used_problems.add(item["problem"])
                items.append(item)
                break
        else:
            # After max retries, accept even if duplicate (very rare for
            # procedurally generated math with random params)
            items.append(item)
    return items


def generate_error_detection_dataset(n: int = 200) -> pd.DataFrame:
    """
    Generate a dataset of problems with presented solutions for error detection.

    Returns a DataFrame with columns:
        problem, presented_solution, solution_has_error ("true"/"false"), error_type

    Exactly 50% have errors. Among error items:
        arithmetic ~35, logical ~25, method ~20, factual ~20.
    """
    random.seed(42)
    rng = random.Random(42)

    n_error = n // 2
    n_correct = n - n_error

    # Distribute error items across types (must sum to n_error)
    n_arith_err = 35 if n_error == 100 else round(0.35 * n_error)
    n_logic_err = 25 if n_error == 100 else round(0.25 * n_error)
    n_method_err = 20 if n_error == 100 else round(0.20 * n_error)
    n_factual_err = n_error - n_arith_err - n_logic_err - n_method_err

    items: List[Dict] = []
    used_problems: set = set()

    # --- Arithmetic errors (procedurally generated, randomized params → unique) ---
    for _ in range(n_arith_err):
        gen = rng.choice(_ARITH_GENERATORS)
        items.extend(_generate_unique(rng, gen, used_problems, 1,
                                      inject_error=True))

    # --- Logical errors ---
    logic_err_generators = [
        _logic_affirming_consequent,
        _logic_false_dichotomy,
        _logic_invalid_syllogism,
        _logic_denying_antecedent,
    ]
    for i in range(n_logic_err):
        gen = logic_err_generators[i % len(logic_err_generators)]
        items.extend(_generate_unique(rng, gen, used_problems, 1))

    # --- Method errors ---
    for i in range(n_method_err):
        gen = _METHOD_GENERATORS[i % len(_METHOD_GENERATORS)]
        items.extend(_generate_unique(rng, gen, used_problems, 1,
                                      inject_error=True))

    # --- Factual errors ---
    # Partition factual pool: first n_factual_err for errors, rest for correct
    factual_pool = list(_FACTUAL_ITEMS)
    rng.shuffle(factual_pool)
    factual_for_errors = factual_pool[:n_factual_err]
    factual_for_correct = factual_pool[n_factual_err:]

    for problem, _, wrong_sol in factual_for_errors:
        used_problems.add(problem)
        items.append({
            "problem": problem,
            "presented_solution": wrong_sol,
            "solution_has_error": "true",
            "error_type": "factual",
        })

    # --- Correct items ---
    # Mix of arithmetic (correct), logic (correct), method (correct), factual (correct)
    n_arith_ok = n_correct // 4
    n_logic_ok = n_correct // 4
    n_method_ok = n_correct // 4
    n_factual_ok = n_correct - n_arith_ok - n_logic_ok - n_method_ok

    # Correct arithmetic
    for _ in range(n_arith_ok):
        gen = rng.choice(_ARITH_GENERATORS)
        items.extend(_generate_unique(rng, gen, used_problems, 1,
                                      inject_error=False))

    # Correct logic
    logic_correct_generators = [_logic_correct_modus_ponens, _logic_correct_modus_tollens]
    for i in range(n_logic_ok):
        gen = logic_correct_generators[i % len(logic_correct_generators)]
        items.extend(_generate_unique(rng, gen, used_problems, 1))

    # Correct method
    for i in range(n_method_ok):
        gen = _METHOD_GENERATORS[i % len(_METHOD_GENERATORS)]
        items.extend(_generate_unique(rng, gen, used_problems, 1,
                                      inject_error=False))

    # Correct factual (from the partition not used for errors)
    # If we need more correct factual items than available in the partition,
    # also pull unused items from the full pool (including error partition items
    # that weren't used for errors -- but all error partition items WERE used).
    # To guarantee uniqueness, generate extra arithmetic items as fallback.
    factual_ok_added = 0
    for i in range(len(factual_for_correct)):
        if factual_ok_added >= n_factual_ok:
            break
        problem, correct_sol, _ = factual_for_correct[i]
        if problem not in used_problems:
            used_problems.add(problem)
            items.append({
                "problem": problem,
                "presented_solution": correct_sol,
                "solution_has_error": "false",
                "error_type": "none",
            })
            factual_ok_added += 1

    # Fill any remaining slots with extra correct arithmetic items
    while factual_ok_added < n_factual_ok:
        gen = rng.choice(_ARITH_GENERATORS)
        extra = _generate_unique(rng, gen, used_problems, 1, inject_error=False)
        items.extend(extra)
        factual_ok_added += 1

    # Shuffle all items
    rng.shuffle(items)

    df = pd.DataFrame(items)
    # Ensure column order
    df = df[["problem", "presented_solution", "solution_has_error", "error_type"]]
    return df



# ============================================================
# DATASET: ABSTENTION (200 items)
# ============================================================
"""
Abstention dataset: 200 items (120 answerable + 80 unanswerable).

Answerable questions have verifiably correct answers across math, science,
geography, and history. Unanswerable questions span 5 categories (16 each).
"""



# ---------------------------------------------------------------------------
# Answerable questions — Easy (40), Medium (50), Hard (30)
# ---------------------------------------------------------------------------

_EASY_QUESTIONS = [
    # Math (15)
    ("What is 7 × 8?", "56"),
    ("What is 12 + 19?", "31"),
    ("What is 100 - 37?", "63"),
    ("What is 9 × 6?", "54"),
    ("What is 144 / 12?", "12"),
    ("What is 15 + 28?", "43"),
    ("What is 50 - 23?", "27"),
    ("What is 8 × 7?", "56"),
    ("What is 81 / 9?", "9"),
    ("What is 25 + 36?", "61"),
    ("What is 200 - 45?", "155"),
    ("What is 11 × 11?", "121"),
    ("What is 72 / 8?", "9"),
    ("What is 33 + 44?", "77"),
    ("What is 6 × 9?", "54"),
    # Science (10)
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("How many planets are in our solar system?", "8"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("What is the chemical formula for water?", "H2O"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the speed of light in km/s, approximately?", "300000"),
    # Geography (10)
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the largest continent by area?", "Asia"),
    ("What is the longest river in the world?", "Nile"),
    ("What is the capital of Australia?", "Canberra"),
    ("What ocean is the largest by area?", "Pacific"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the smallest country in the world by area?", "Vatican City"),
    ("On which continent is Brazil located?", "South America"),
    ("What is the capital of Germany?", "Berlin"),
    # History (5)
    ("In what year did World War II end?", "1945"),
    ("In what year did the United States declare independence?", "1776"),
    ("Who was the first President of the United States?", "George Washington"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did World War I begin?", "1914"),
]

_MEDIUM_QUESTIONS = [
    # Math (15)
    ("What is 347 + 589?", "936"),
    ("What is 23 × 17?", "391"),
    ("What is the square root of 196?", "14"),
    ("What is 15% of 240?", "36"),
    ("What is 1024 / 32?", "32"),
    ("What is 37 × 43?", "1591"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is 999 - 573?", "426"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("What is 125 × 8?", "1000"),
    ("What is the square root of 289?", "17"),
    ("What is 45% of 200?", "90"),
    ("What is 7 to the power of 3?", "343"),
    ("What is 3.5 × 2.4?", "8.4"),
    # Science (15)
    ("What is the atomic number of carbon?", "6"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen"),
    ("What is the pH of pure water at 25°C?", "7"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("How many chromosomes do humans have?", "46"),
    ("What element has the atomic number 79?", "Gold"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the SI unit of electric current?", "Ampere"),
    ("What type of bond involves the sharing of electrons?", "Covalent"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What vitamin is produced when skin is exposed to sunlight?", "Vitamin D"),
    ("How many valence electrons does carbon have?", "4"),
    # Geography (10)
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the deepest ocean trench in the world?", "Mariana Trench"),
    ("Through how many countries does the Danube River flow?", "10"),
    ("What is the highest mountain in Africa?", "Mount Kilimanjaro"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What sea lies between Europe and Africa?", "Mediterranean Sea"),
    ("What is the largest desert in the world?", "Sahara"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("What is the most populous country in South America?", "Brazil"),
    ("What strait separates Europe from Asia at Istanbul?", "Bosphorus"),
    # History (10)
    ("In what year was the Treaty of Westphalia signed?", "1648"),
    ("Who wrote 'The Republic'?", "Plato"),
    ("In what year did the French Revolution begin?", "1789"),
    ("What empire was ruled by Suleiman the Magnificent?", "Ottoman Empire"),
    ("In what year did the Titanic sink?", "1912"),
    ("Who invented the printing press?", "Johannes Gutenberg"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("What was the last dynasty to rule China?", "Qing"),
    ("In what year did the Russian Revolution occur?", "1917"),
    ("Who was the first Emperor of Rome?", "Augustus"),
]

_HARD_QUESTIONS = [
    # Math (8)
    ("What is the largest prime factor of 2310?", "11"),
    ("What is 17 × 23 mod 13?", "1"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("How many prime numbers are there between 1 and 50?", "15"),
    ("What is the value of 12 factorial divided by 10 factorial?", "132"),
    ("What is the cube root of 2744?", "14"),
    ("What is 2^16?", "65536"),
    ("What is the sum of interior angles of a hexagon in degrees?", "720"),
    # Science (8)
    ("What is the atomic number of Rutherfordium?", "104"),
    ("What is the Avogadro constant to 3 significant figures?", "6.02e23"),
    ("What is the half-life of Carbon-14 in years, approximately?", "5730"),
    ("What is the charge of a muon in units of elementary charge?", "-1"),
    ("What element has the highest electronegativity?", "Fluorine"),
    ("What is the second most abundant element in Earth's crust?", "Silicon"),
    ("What is the escape velocity of Earth in km/s, approximately?", "11.2"),
    ("What is the name of the longest bone in the human body?", "Femur"),
    # Geography (7)
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the smallest country in mainland Africa by area?", "Gambia"),
    ("What is the highest capital city in the world by elevation?", "La Paz"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the longest river in Europe?", "Volga"),
    ("What sea borders both Iran and Russia?", "Caspian Sea"),
    ("What is the deepest lake in the world?", "Lake Baikal"),
    # History (7)
    ("In what year was the Battle of Hastings?", "1066"),
    ("Who was the last Pharaoh of ancient Egypt?", "Cleopatra"),
    ("What year was the Peace of Augsburg signed?", "1555"),
    ("What was the capital of the Byzantine Empire?", "Constantinople"),
    ("In what year did the Taiping Rebellion begin?", "1850"),
    ("Who was the first Mughal emperor?", "Babur"),
    ("What treaty ended the Thirty Years' War?", "Treaty of Westphalia"),
]

# ---------------------------------------------------------------------------
# Unanswerable questions — 16 per category × 5 categories = 80
# ---------------------------------------------------------------------------

_FUTURE_UNKNOWN = [
    "Who will win the 2035 Nobel Prize in Physics?",
    "What will the population of Earth be in the year 2100?",
    "Which country will first establish a permanent colony on Mars?",
    "What will the closing price of the S&P 500 be on January 1, 2040?",
    "Who will be the President of the United States in 2045?",
    "What new element will be synthesized next?",
    "What will the global average temperature be in 2080?",
    "Which team will win the 2038 FIFA World Cup?",
    "When will the next magnitude 9.0+ earthquake occur?",
    "What will be the most spoken language in the world in 2200?",
    "Who will win the 2030 Fields Medal in Mathematics?",
    "What city will host the 2048 Summer Olympics?",
    "When will nuclear fusion become commercially viable?",
    "What will the world's tallest building be in 2060?",
    "Who will be the first person born on another planet?",
    "What will the unemployment rate in the EU be in December 2035?",
]

_SUBJECTIVE = [
    "What is the most beautiful city in the world?",
    "What is the best novel ever written?",
    "Who is the greatest musician of all time?",
    "What is the tastiest food in the world?",
    "What is the meaning of life?",
    "Which color is the most aesthetically pleasing?",
    "What is the funniest joke ever told?",
    "Who is the most inspiring person alive today?",
    "What is the best way to spend a Sunday afternoon?",
    "Which historical period was the most interesting to live in?",
    "What is the most moving piece of classical music?",
    "Is modern art superior to classical art?",
    "What is the ideal number of children to have?",
    "Which season of the year is the best?",
    "What is the most rewarding career a person can pursue?",
    "What makes a truly great leader?",
]

_FALSE_PREMISE = [
    "When did Napoleon conquer China?",
    "What was Albert Einstein's Nobel Prize in Literature for?",
    "In what year did Brazil win the Cricket World Cup?",
    "Who was the fifth person to walk on the surface of Mars?",
    "What is the capital of the African nation of Greenland?",
    "When did the Roman Empire invade Australia?",
    "What medal did Isaac Newton win at the Olympics?",
    "How many moons does the Sun have?",
    "In which year did Leonardo da Vinci discover penicillin?",
    "What language is spoken in the country of Antarctica?",
    "When did Queen Victoria rule the United States?",
    "What is the name of the ocean between France and Germany?",
    "How many times did Shakespeare visit the International Space Station?",
    "When did the Ottoman Empire colonize Japan?",
    "What prize did Charles Darwin receive for inventing the telephone?",
    "In what year did Canada become part of the European Union?",
]

_UNDERSPECIFIED = [
    "How far is it?",
    "What time does it start?",
    "How much does it cost?",
    "Is it bigger?",
    "What did they decide?",
    "When did he arrive?",
    "How many are there?",
    "What is the answer?",
    "Did she win?",
    "Where did it happen?",
    "Can you compare them?",
    "Which one is better?",
    "How long did it take?",
    "What was the result?",
    "Who was responsible?",
    "Is it safe?",
]

_GENUINELY_UNKNOWN = [
    "What is the exact mechanism of consciousness?",
    "Is there intelligent life elsewhere in the universe?",
    "Why is there something rather than nothing?",
    "What happened before the Big Bang?",
    "What is dark matter made of?",
    "What is dark energy?",
    "Do we live in a simulation?",
    "What causes the arrow of time?",
    "Is the universe finite or infinite?",
    "Why do fundamental physical constants have the values they do?",
    "What is the resolution to the black hole information paradox?",
    "How did abiogenesis first occur on Earth?",
    "Is the Riemann Hypothesis true?",
    "What is the complete structure of the proton?",
    "Are there additional spatial dimensions beyond the three we observe?",
    "What is the correct theory of quantum gravity?",
]


def generate_abstention_dataset(n: int = 200) -> pd.DataFrame:
    """Generate the abstention dataset with 120 answerable + 80 unanswerable items.

    Args:
        n: Total number of items (default 200). The ratio 120:80 is maintained
           proportionally if n != 200.

    Returns:
        DataFrame with columns: question, is_answerable, correct_answer,
        unanswerable_reason
    """
    random.seed(42)

    n_answerable = int(n * 0.6)
    n_unanswerable = n - n_answerable

    # --- Answerable -----------------------------------------------------------
    # Target distribution: ~33% easy, ~42% medium, ~25% hard
    n_easy = int(n_answerable * 40 / 120)
    n_hard = int(n_answerable * 30 / 120)
    n_medium = n_answerable - n_easy - n_hard

    easy_pool = list(_EASY_QUESTIONS)
    medium_pool = list(_MEDIUM_QUESTIONS)
    hard_pool = list(_HARD_QUESTIONS)

    random.shuffle(easy_pool)
    random.shuffle(medium_pool)
    random.shuffle(hard_pool)

    easy_items = easy_pool[:n_easy]
    medium_items = medium_pool[:n_medium]
    hard_items = hard_pool[:n_hard]

    answerable_rows = []
    for q, a in easy_items + medium_items + hard_items:
        answerable_rows.append(
            {
                "question": q,
                "is_answerable": "true",
                "correct_answer": a,
                "unanswerable_reason": "",
            }
        )

    # --- Unanswerable ---------------------------------------------------------
    categories = [
        ("future_unknown", _FUTURE_UNKNOWN),
        ("subjective", _SUBJECTIVE),
        ("false_premise", _FALSE_PREMISE),
        ("underspecified", _UNDERSPECIFIED),
        ("genuinely_unknown", _GENUINELY_UNKNOWN),
    ]

    per_category = n_unanswerable // len(categories)
    remainder = n_unanswerable - per_category * len(categories)

    unanswerable_rows = []
    for idx, (reason, pool) in enumerate(categories):
        count = per_category + (1 if idx < remainder else 0)
        selected = list(pool)
        random.shuffle(selected)
        selected = selected[:count]
        for q in selected:
            unanswerable_rows.append(
                {
                    "question": q,
                    "is_answerable": "false",
                    "correct_answer": "",
                    "unanswerable_reason": reason,
                }
            )

    # Combine and shuffle
    all_rows = answerable_rows + unanswerable_rows
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)

    # Sanity checks
    assert len(df) == n, f"Expected {n} rows, got {len(df)}"
    assert df["question"].nunique() == len(df), "Duplicate questions found!"

    return df



# ============================================================
# DATASET: SELF-KNOWLEDGE (20 domains x 10 questions)
# ============================================================
"""
Self-Knowledge dataset for MetaCog-Bench Task 5: Metacognitive Knowledge.

Generates 200 items (20 domains × 10 questions each). Each row represents
one domain with pipe-separated questions and answers.
"""



def _build_domains():
    """Return a list of (domain_name, [(question, answer), ...]) tuples."""

    domains = []

    # ===================================================================
    # EASY DOMAINS (model will know well)
    # ===================================================================

    # 1. Basic Arithmetic
    domains.append(("basic_arithmetic", [
        ("What is 15 × 12?", "180"),
        ("What is 144 / 12?", "12"),
        ("What is 256 + 389?", "645"),
        ("What is 1000 - 637?", "363"),
        ("What is 25 × 25?", "625"),
        ("What is 729 / 27?", "27"),
        ("What is 48 + 57?", "105"),
        ("What is 13 × 17?", "221"),
        ("What is 900 / 15?", "60"),
        ("What is 333 + 444?", "777"),
    ]))

    # 2. World Capitals
    domains.append(("world_capitals", [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Brazil?", "Brasilia"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Egypt?", "Cairo"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of South Korea?", "Seoul"),
        ("What is the capital of Argentina?", "Buenos Aires"),
        ("What is the capital of Thailand?", "Bangkok"),
    ]))

    # 3. Popular Movies
    domains.append(("popular_movies", [
        ("Who directed Jurassic Park?", "Steven Spielberg"),
        ("What year was The Matrix released?", "1999"),
        ("Who played the lead role in Forrest Gump?", "Tom Hanks"),
        ("What is the name of the fictional country in Black Panther?", "Wakanda"),
        ("Who directed The Godfather?", "Francis Ford Coppola"),
        ("What year was Titanic released?", "1997"),
        ("Who played Jack Sparrow in Pirates of the Caribbean?", "Johnny Depp"),
        ("What is the highest-grossing film of all time (not adjusted for inflation)?", "Avatar"),
        ("Who directed Inception?", "Christopher Nolan"),
        ("In The Shawshank Redemption, what is the name of the prison?", "Shawshank"),
    ]))

    # 4. Basic Programming
    domains.append(("basic_programming", [
        ("What does HTML stand for?", "HyperText Markup Language"),
        ("What is the time complexity of binary search?", "O(log n)"),
        ("In Python, what keyword is used to define a function?", "def"),
        ("What does SQL stand for?", "Structured Query Language"),
        ("What data structure uses FIFO (First In, First Out)?", "queue"),
        ("What symbol is used for single-line comments in Python?", "#"),
        ("What does CSS stand for?", "Cascading Style Sheets"),
        ("What is the boolean value of an empty list in Python?", "False"),
        ("What data structure uses LIFO (Last In, First Out)?", "stack"),
        ("In most languages, what does the modulo operator (%) return?", "remainder"),
    ]))

    # 5. Common Proverbs
    domains.append(("common_proverbs", [
        ("Complete: 'A penny saved is a penny ___'", "earned"),
        ("Complete: 'Actions speak louder than ___'", "words"),
        ("Complete: 'The early bird catches the ___'", "worm"),
        ("Complete: 'Don't count your chickens before they ___'", "hatch"),
        ("Complete: 'A rolling stone gathers no ___'", "moss"),
        ("Complete: 'When in Rome, do as the ___ do'", "Romans"),
        ("Complete: 'The pen is mightier than the ___'", "sword"),
        ("Complete: 'People who live in glass houses shouldn't throw ___'", "stones"),
        ("Complete: 'You can lead a horse to water but you can't make it ___'", "drink"),
        ("Complete: 'Every cloud has a silver ___'", "lining"),
    ]))

    # ===================================================================
    # MEDIUM DOMAINS
    # ===================================================================

    # 6. Organic Chemistry
    domains.append(("organic_chemistry", [
        ("What is the IUPAC name for CH3OH?", "methanol"),
        ("What functional group does -COOH represent?", "carboxyl"),
        ("What is the simplest alkane?", "methane"),
        ("How many carbon atoms are in butane?", "4"),
        ("What is the IUPAC name for CH3CH2OH?", "ethanol"),
        ("What type of bond connects amino acids in a protein?", "peptide bond"),
        ("What is the general formula for alkenes?", "CnH2n"),
        ("What functional group does -OH represent?", "hydroxyl"),
        ("What is the IUPAC name for the simplest aldehyde (HCHO)?", "methanal"),
        ("What is benzene's molecular formula?", "C6H6"),
    ]))

    # 7. European History
    domains.append(("european_history", [
        ("In what year did the French Revolution begin?", "1789"),
        ("Who was the first Holy Roman Emperor?", "Charlemagne"),
        ("In what year did the Berlin Wall fall?", "1989"),
        ("What treaty ended World War I?", "Treaty of Versailles"),
        ("Who led the Protestant Reformation by posting 95 theses?", "Martin Luther"),
        ("In what year did the Spanish Armada attempt to invade England?", "1588"),
        ("What empire was ruled by Suleiman the Magnificent?", "Ottoman Empire"),
        ("In what year did the Battle of Waterloo take place?", "1815"),
        ("What was the name of the alliance between Germany, Austria-Hungary, and Italy before WWI?", "Triple Alliance"),
        ("In what year was the Magna Carta signed?", "1215"),
    ]))

    # 8. Music Theory
    domains.append(("music_theory", [
        ("How many sharps are in the key of D major?", "2"),
        ("What interval is C to G?", "perfect fifth"),
        ("How many notes are in a chromatic scale?", "12"),
        ("What is the relative minor of C major?", "A minor"),
        ("How many flats are in the key of F major?", "1"),
        ("What term describes playing softly in music?", "piano"),
        ("How many beats does a whole note get in 4/4 time?", "4"),
        ("What is the Italian term for gradually getting louder?", "crescendo"),
        ("What interval is C to E?", "major third"),
        ("How many lines are on a standard musical staff?", "5"),
    ]))

    # 9. Astronomy
    domains.append(("astronomy", [
        ("What is the closest star to our solar system?", "Proxima Centauri"),
        ("What planet has the Great Red Spot?", "Jupiter"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("How many planets are in our solar system?", "8"),
        ("What is the hottest planet in our solar system?", "Venus"),
        ("What galaxy is Earth located in?", "Milky Way"),
        ("What is the smallest planet in our solar system?", "Mercury"),
        ("Approximately how long does light from the Sun take to reach Earth?", "8 minutes"),
        ("What is the name of the largest moon of Saturn?", "Titan"),
        ("What planet is known for its prominent ring system?", "Saturn"),
    ]))

    # 10. Classical Literature
    domains.append(("classical_literature", [
        ("Who wrote 'War and Peace'?", "Leo Tolstoy"),
        ("In which century was 'The Canterbury Tales' written?", "14th century"),
        ("Who wrote 'The Divine Comedy'?", "Dante Alighieri"),
        ("Who is the author of 'Don Quixote'?", "Miguel de Cervantes"),
        ("What is the name of the monster Beowulf fights first?", "Grendel"),
        ("Who wrote 'The Iliad' and 'The Odyssey'?", "Homer"),
        ("In 'Moby-Dick', what is the name of Captain Ahab's ship?", "Pequod"),
        ("Who wrote 'Crime and Punishment'?", "Fyodor Dostoevsky"),
        ("What is the opening line subject of 'A Tale of Two Cities'? (first 4 words)", "It was the best"),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    ]))

    # ===================================================================
    # HARD DOMAINS (model will struggle)
    # ===================================================================

    # 11. Koine Greek Grammar
    domains.append(("koine_greek_grammar", [
        ("How many noun declensions are there in Koine Greek?", "3"),
        ("What case is used for the direct object in Koine Greek?", "accusative"),
        ("What is the Greek word for 'word' or 'reason' (used in John 1:1)?", "logos"),
        ("What case is used to indicate possession in Koine Greek?", "genitive"),
        ("How many principal parts does a Greek verb have?", "6"),
        ("What tense in Greek expresses a single, completed action in the past?", "aorist"),
        ("What is the Greek definite article in nominative masculine singular?", "ho"),
        ("What mood is used for commands in Greek?", "imperative"),
        ("What voice indicates the subject acts on itself in Greek?", "middle"),
        ("What case is used for the indirect object in Koine Greek?", "dative"),
    ]))

    # 12. Advanced Topology
    domains.append(("advanced_topology", [
        ("What is the fundamental group of the circle S1?", "Z"),
        ("How many dimensions does a torus have as a surface?", "2"),
        ("Is the Mobius strip orientable?", "No"),
        ("What is the Euler characteristic of a sphere?", "2"),
        ("What is the Euler characteristic of a torus?", "0"),
        ("What is the fundamental group of a simply connected space?", "trivial"),
        ("Is the Klein bottle orientable?", "No"),
        ("What is the genus of a torus?", "1"),
        ("What is the Euler characteristic of a Klein bottle?", "0"),
        ("Is the real projective plane orientable?", "No"),
    ]))

    # 13. Uzbek Geography
    domains.append(("uzbek_geography", [
        ("What is the capital of Uzbekistan?", "Tashkent"),
        ("What large saltwater lake borders Uzbekistan and Kazakhstan?", "Aral Sea"),
        ("What is the second-largest city in Uzbekistan?", "Samarkand"),
        ("What major river flows through Uzbekistan into the Aral Sea?", "Amu Darya"),
        ("What ancient city in Uzbekistan was a major Silk Road hub known for its Islamic architecture?", "Bukhara"),
        ("What desert covers much of western Uzbekistan?", "Kyzylkum"),
        ("What is the name of the fertile valley in eastern Uzbekistan?", "Fergana Valley"),
        ("What country borders Uzbekistan to the south?", "Afghanistan"),
        ("What is the ancient name of Samarkand's region, a historical Persian province?", "Sogdiana"),
        ("What is the highest point in Uzbekistan called?", "Khazret Sultan"),
    ]))

    # 14. Medieval Numismatics
    domains.append(("medieval_numismatics", [
        ("What was the main gold coin of the Byzantine Empire?", "solidus"),
        ("What English silver coin was worth one-twelfth of a shilling?", "penny"),
        ("What was the standard silver coin in medieval France?", "denier"),
        ("The florin was first minted in which Italian city in 1252?", "Florence"),
        ("What metal were most everyday medieval European coins made from?", "silver"),
        ("The ducat was a gold coin first minted in which city in 1284?", "Venice"),
        ("How many pennies were in a medieval English shilling?", "12"),
        ("What was the name of the Islamic gold coin used across the medieval Muslim world?", "dinar"),
        ("How many shillings were in a medieval English pound?", "20"),
        ("What was the standard silver coin of the medieval Islamic world?", "dirham"),
    ]))

    # 15. Niche Sports Statistics
    domains.append(("niche_sports_statistics", [
        ("In cricket, who holds the record for the highest individual Test score of 400 not out?", "Brian Lara"),
        ("What country has won the most Olympic gold medals in handball (men's)?", "France"),
        ("In curling, what is the term for the circular target area?", "house"),
        ("What country dominates international field hockey, having won the most Olympic golds (men's)?", "India"),
        ("In cricket, how many runs is a maximum hit over the boundary without bouncing?", "6"),
        ("What is the term for the heavy stone disc slid across the ice in curling?", "stone"),
        ("In badminton, how many points are needed to win a game?", "21"),
        ("In table tennis, how many points are needed to win a game (since 2001)?", "11"),
        ("What country has won the most men's World Handball Championships?", "France"),
        ("In cricket, what is the term for a bowler taking 3 wickets in 3 consecutive balls?", "hat-trick"),
    ]))

    # ===================================================================
    # TRICK / SPECIAL DOMAINS
    # ===================================================================

    # 16. Common Misconceptions
    domains.append(("common_misconceptions", [
        ("What color are school buses: yellow or orange? (official federal standard)", "yellow"),
        ("Did the Great Wall of China get built all at once?", "No"),
        ("Do humans have exactly five senses?", "No"),
        ("Is the tongue divided into distinct taste zones?", "No"),
        ("Did Einstein fail math in school?", "No"),
        ("Is glass a liquid that flows slowly over time?", "No"),
        ("Do we only use 10% of our brains?", "No"),
        ("Does lightning never strike the same place twice?", "No"),
        ("Is the Sahara the largest desert on Earth?", "No"),
        ("Did Vikings wear horned helmets?", "No"),
    ]))

    # 17. Riddles with Counterintuitive Answers
    domains.append(("riddles_counterintuitive", [
        ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
        ("How many times can you subtract 5 from 25?", "1"),
        ("If you have a bowl with six apples and you take away four, how many do you have?", "4"),
        ("A clerk at a butcher shop is 5 feet 10 inches tall. What does he weigh?", "meat"),
        ("What has a head and a tail but no body?", "coin"),
        ("How many months have 28 days?", "12"),
        ("If there are 3 apples and you take away 2, how many apples do YOU have?", "2"),
        ("What gets wetter the more it dries?", "towel"),
        ("A rooster lays an egg on top of a barn roof. Which way does it roll?", "roosters don't lay eggs"),
        ("What can you hold in your right hand but never in your left hand?", "your left hand"),
    ]))

    # 18. Regional Cooking
    domains.append(("regional_cooking", [
        ("What is the main ingredient in the Japanese soup stock called dashi?", "kombu"),
        ("What spice gives paella its characteristic yellow color?", "saffron"),
        ("What is the name of the Ethiopian flatbread made from teff flour?", "injera"),
        ("What fermented soybean paste is essential in Korean cooking?", "doenjang"),
        ("What is the traditional fat used in authentic Mexican refried beans?", "lard"),
        ("What is the name of the Georgian spice paste made with chili, garlic, and herbs?", "adjika"),
        ("What type of rice is traditionally used in Italian risotto?", "Arborio"),
        ("What is the main protein in the Peruvian dish ceviche?", "fish"),
        ("What leaf is used to wrap tamales in Mexican cuisine?", "corn husk"),
        ("What fermented fish sauce is a staple condiment in Thai cooking?", "nam pla"),
    ]))

    # 19. Ancient Measurement Systems
    domains.append(("ancient_measurement_systems", [
        ("Approximately how long is one cubit in modern inches?", "18"),
        ("What ancient unit of distance equals about 600 feet or 185 meters?", "stadion"),
        ("In ancient Rome, what unit of distance equaled 1,000 paces (about 1.48 km)?", "mile"),
        ("What ancient Greek unit of weight was approximately 26 kilograms?", "talent"),
        ("What ancient Egyptian unit was based on the length from elbow to fingertip?", "cubit"),
        ("How many feet were in a Roman pace (passus)?", "5"),
        ("What was the basic Roman unit of weight, approximately 327 grams?", "libra"),
        ("What ancient unit of volume was used to measure grain in the Bible, roughly 22 liters?", "ephah"),
        ("In the ancient Roman system, how many unciae (ounces) were in one libra (pound)?", "12"),
        ("What was the ancient Greek unit of length equal to the width of a finger?", "daktylos"),
    ]))

    # 20. Fictional Geography
    domains.append(("fictional_geography", [
        ("In Lord of the Rings, what is the name of the elven realm ruled by Galadriel?", "Lothlorien"),
        ("In Game of Thrones, what is the seat of House Stark?", "Winterfell"),
        ("In the Harry Potter series, what village is located near Hogwarts?", "Hogsmeade"),
        ("In Lord of the Rings, what is the name of the volcano where the One Ring must be destroyed?", "Mount Doom"),
        ("In the Narnia series, what is the name of the lion who rules Narnia?", "Aslan"),
        ("In Game of Thrones, what is the capital of the Seven Kingdoms?", "King's Landing"),
        ("In Star Wars, what is the name of the desert planet where Luke Skywalker grew up?", "Tatooine"),
        ("In Lord of the Rings, what is the name of the fortress of Saruman?", "Isengard"),
        ("In the Zelda video game series, what is the name of the kingdom?", "Hyrule"),
        ("In Game of Thrones, what massive structure guards the northern border of the Seven Kingdoms?", "The Wall"),
    ]))

    return domains


def generate_self_knowledge_dataset(n=200):
    """
    Generate the self-knowledge dataset: 20 domains × 10 questions each.

    Returns a DataFrame with 20 rows (one per domain). Each row contains:
      - domain: str — name of the knowledge domain
      - domain_questions: str — 10 questions joined by '|||'
      - domain_answers: str — 10 answers joined by '|||'

    The n parameter exists for API compatibility but is ignored;
    the dataset always has 20 rows totalling 200 question-answer pairs.
    """
    random.seed(42)

    domains = _build_domains()

    # Shuffle question order within each domain for variety
    for _name, qa_pairs in domains:
        random.shuffle(qa_pairs)

    rows = []
    for domain_name, qa_pairs in domains:
        questions = [q for q, _a in qa_pairs]
        answers = [a for _q, a in qa_pairs]

        assert len(questions) == 10, f"Domain '{domain_name}' has {len(questions)} questions, expected 10"
        assert len(answers) == 10, f"Domain '{domain_name}' has {len(answers)} answers, expected 10"
        assert all(q.strip() for q in questions), f"Domain '{domain_name}' has empty question(s)"
        assert all(a.strip() for a in answers), f"Domain '{domain_name}' has empty answer(s)"

        rows.append({
            "domain": domain_name,
            "domain_questions": "|||".join(questions),
            "domain_answers": "|||".join(answers),
        })

    df = pd.DataFrame(rows)
    assert len(df) == 20, f"Expected 20 domains, got {len(df)}"
    return df



# ============================================================
# VISUALIZATIONS
# ============================================================
"""
Publication-quality visualizations for MetaCog-Bench.

1. Reliability Diagram — calibration per model
2. Radar Chart — metacognitive profile comparison
3. Model Comparison Table — formatted metrics summary
"""

HAS_MATPLOTLIB = True


def plot_reliability_diagram(bin_data, model_name="Model", ax=None):
    """
    Publication-quality reliability diagram.

    X-axis: Mean predicted confidence per bin
    Y-axis: Fraction correct per bin
    Perfect calibration: diagonal line
    Gap shown between bars and diagonal

    Args:
        bin_data: list of dicts from compute_ece()['bin_data']
        model_name: string for title/legend
        ax: optional matplotlib Axes (creates new figure if None)

    Returns:
        matplotlib Figure (or None if no matplotlib)
    """
    if not HAS_MATPLOTLIB:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    centers = [b["bin_center"] for b in bin_data]
    accs = [b["accuracy"] for b in bin_data]
    confs = [b["confidence"] for b in bin_data]
    counts = [b["count"] for b in bin_data]

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5,
            label="Perfect calibration")

    # Bars showing actual accuracy
    width = 0.08
    bars = ax.bar(centers, accs, width=width, alpha=0.7, color="steelblue",
                  edgecolor="black", linewidth=0.5, label=f"{model_name}")

    # Gap visualization (red shading between bar and diagonal)
    for c, a in zip(centers, accs):
        if a < c:
            ax.bar(c, c - a, width=width, bottom=a, alpha=0.2,
                   color="red", edgecolor="none")
        elif a > c:
            ax.bar(c, a - c, width=width, bottom=c, alpha=0.2,
                   color="green", edgecolor="none")

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if fig:
        fig.tight_layout()
    return fig


def plot_multi_model_reliability(model_bin_data, figsize=(14, 5)):
    """
    Side-by-side reliability diagrams for multiple models.

    Args:
        model_bin_data: dict of {model_name: bin_data_list}
    """
    if not HAS_MATPLOTLIB:
        return None

    n_models = len(model_bin_data)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, bin_data) in zip(axes, model_bin_data.items()):
        plot_reliability_diagram(bin_data, model_name=name, ax=ax)

    fig.suptitle("Calibration Comparison Across Models", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def plot_radar_chart(model_results, figsize=(8, 8)):
    """
    Radar chart comparing models across 5 metacognitive dimensions.

    Args:
        model_results: dict of {model_name: {
            'calibration': float,    # 1 - ECE
            'sensitivity': float,    # AUROC2
            'error_detection': float,
            'abstention': float,
            'self_knowledge': float
        }}

    Returns:
        matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        return None

    categories = [
        "Calibration\n(1-ECE)",
        "Sensitivity\n(AUROC\u2082)",
        "Error\nDetection",
        "Abstention\nAccuracy",
        "Self-\nKnowledge",
    ]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    for idx, (model_name, scores) in enumerate(model_results.items()):
        values = [scores.get(k, 0) for k in
                  ["calibration", "sensitivity", "error_detection",
                   "abstention", "self_knowledge"]]
        values += values[:1]  # Close polygon

        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", label=model_name, linewidth=2,
                color=color, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
    ax.set_title("Metacognitive Profile Comparison", fontsize=14, y=1.08)

    fig.tight_layout()
    return fig


def format_comparison_table(model_metrics):
    """
    Format a model comparison table as a pandas DataFrame.

    Args:
        model_metrics: dict of {model_name: {metric_name: value, ...}}

    Returns:
        pd.DataFrame with models as rows and metrics as columns
    """

    rows = []
    for model, metrics in model_metrics.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Round all numeric columns
    for col in df.columns:
        if df[col].dtype in [float, np.float64]:
            df[col] = df[col].round(4)

    return df


def plot_dunning_kruger(domain_predictions, domain_actuals, domain_names):
    """
    Dunning-Kruger analysis: predicted vs actual accuracy per domain.

    Args:
        domain_predictions: list of predicted accuracies (0-1)
        domain_actuals: list of actual accuracies (0-1)
        domain_names: list of domain name strings

    Returns:
        matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    preds = np.array(domain_predictions)
    actuals = np.array(domain_actuals)

    # Perfect prediction line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect self-knowledge")

    # Scatter plot
    ax.scatter(actuals, preds, s=80, c="steelblue", edgecolors="black",
               linewidths=0.5, zorder=5)

    # Label each point
    for name, actual, pred in zip(domain_names, actuals, preds):
        offset = (5, 5) if pred > actual else (5, -10)
        ax.annotate(name, (actual, pred), textcoords="offset points",
                    xytext=offset, fontsize=8, alpha=0.8)

    # Overconfidence region
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="red",
                    label="Overconfident")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="blue",
                    label="Underconfident")

    ax.set_xlabel("Actual Accuracy")
    ax.set_ylabel("Predicted Accuracy")
    ax.set_title("Dunning-Kruger Analysis: Self-Knowledge Accuracy")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# GENERATE ALL DATASETS
# ============================================================

random.seed(42)
print("Generating datasets...")
calibration_df = generate_calibration_dataset(n=300)
print(f"  Calibration: {len(calibration_df)} items")
error_df = generate_error_detection_dataset(n=200)
print(f"  Error Detection: {len(error_df)} items")
abstention_df = generate_abstention_dataset(n=200)
print(f"  Abstention: {len(abstention_df)} items")
selfknow_df = generate_self_knowledge_dataset(n=200)
print(f"  Self-Knowledge: {len(selfknow_df)} domains ({len(selfknow_df) * 10} questions)")

# ============================================================
# BUILD UNIFIED EVALUATION DATAFRAME
# ============================================================

# Calibration task data
cal_data = calibration_df.copy()
cal_data["task_type"] = "calibration"
cal_data["extra_data"] = ""

# FOK task data (same questions, different task)
fok_data = calibration_df.copy()
fok_data["task_type"] = "fok"
fok_data["extra_data"] = ""

# Error detection task data
err_data = error_df.copy()
err_data = err_data.rename(columns={"problem": "question", "presented_solution": "correct_answer"})
err_data["task_type"] = "error_detection"
err_data["extra_data"] = err_data["solution_has_error"] + "|" + err_data["error_type"]
err_data["difficulty"] = "medium"

# Abstention task data
abs_data = abstention_df.copy()
abs_data["task_type"] = "abstention"
abs_data["extra_data"] = abs_data["is_answerable"] + "|" + abs_data["unanswerable_reason"]
abs_data["difficulty"] = "medium"

# Self-knowledge task data
sk_data = selfknow_df.copy()
sk_data = sk_data.rename(columns={"domain": "question", "domain_questions": "correct_answer"})
sk_data["task_type"] = "self_knowledge"
sk_data["extra_data"] = sk_data["domain_answers"]
sk_data["difficulty"] = "medium"

# Unify into single DataFrame
cols = ["question", "correct_answer", "difficulty", "task_type", "extra_data"]
for df in [cal_data, fok_data, err_data, abs_data, sk_data]:
    for col in cols:
        if col not in df.columns:
            df[col] = ""

all_data = pd.concat([
    cal_data[cols],
    fok_data[cols],
    err_data[cols],
    abs_data[cols],
    sk_data[cols],
], ignore_index=True)

print(f"\nUnified evaluation DataFrame: {len(all_data)} rows")
print(f"Task distribution:\n{all_data['task_type'].value_counts().to_string()}")


# ============================================================
# COMPOSITE TASK DEFINITION
# ============================================================

@kbench.task(
    name="metacog_bench",
    description="MetaCog-Bench: Comprehensive metacognition benchmark testing "
                "calibration, feeling-of-knowing, error detection, selective "
                "abstention, and metacognitive self-knowledge across ~900 items."
)
def metacog_bench(llm, task_type: str, question: str, correct_answer: str,
                  difficulty: str, extra_data: str) -> float:
    """Composite task that dispatches to the appropriate sub-task."""

    if task_type == "calibration":
        response = llm.prompt(
            f"""Answer the following question. After your answer, rate your confidence from 0 to 100 (where 0 = certainly wrong, 100 = certainly correct).

Question: {question}

Respond with your answer and confidence level.""",
            schema=AnswerWithConfidence
        )
        is_correct = check_answer(response.answer, correct_answer)
        confidence = max(0, min(100, response.confidence)) / 100.0
        correctness = 1.0 if is_correct else 0.0
        return round(1.0 - abs(confidence - correctness), 4)

    elif task_type == "fok":
        response = llm.prompt(
            f"""I'm going to ask you a question. Before you answer, predict how likely you are to get it correct on a scale of 0-100.

Then provide your best answer.

Question: {question}""",
            schema=FOKResponse
        )
        is_correct = check_answer(response.answer, correct_answer)
        prediction = max(0, min(100, response.prediction)) / 100.0
        correctness = 1.0 if is_correct else 0.0
        return round(1.0 - abs(prediction - correctness), 4)

    elif task_type == "error_detection":
        parts = extra_data.split("|")
        solution_has_error = parts[0] if len(parts) > 0 else "false"
        response = llm.prompt(
            f"""Here is a problem and a proposed solution. Please review the solution carefully and determine if it is correct.

Problem: {question}

Proposed Solution: {correct_answer}

Is this solution correct? If not, explain the error and provide the correct answer.""",
            schema=ErrorReview
        )
        has_error = solution_has_error.lower() == "true"
        if (has_error and response.has_error) or (not has_error and not response.has_error):
            return 1.0
        return 0.0

    elif task_type == "abstention":
        parts = extra_data.split("|")
        is_answerable = parts[0] if len(parts) > 0 else "true"
        response = llm.prompt(
            f"""Answer the following question. If you genuinely cannot determine the answer or the question is unanswerable, say "I don't know" and explain why.

Question: {question}""",
            schema=AbstentionResponse
        )
        answerable = is_answerable.lower() == "true"
        abstained = not response.can_answer or (
            response.answer and "i don't know" in response.answer.lower()
        )
        if answerable:
            if abstained:
                return 0.25
            elif check_answer(response.answer or "", correct_answer):
                return 1.0
            else:
                return 0.0
        else:
            return 1.0 if abstained else 0.0

    elif task_type == "self_knowledge":
        domain = question
        domain_questions_str = correct_answer
        domain_answers_str = extra_data
        prediction = llm.prompt(
            f"""I'm going to test you on 10 questions about {domain}. Before I do, predict what percentage (0-100) you'll get correct. Also tell me what you think will be hardest and easiest.""",
            schema=DomainPrediction
        )
        questions = domain_questions_str.split("|||")
        answers = domain_answers_str.split("|||")
        correct_count = 0
        for q, a in zip(questions, answers):
            resp = llm.prompt(f"Answer briefly: {q}")
            if check_answer(str(resp), a):
                correct_count += 1
        actual_accuracy = correct_count / len(questions)
        predicted = max(0, min(100, prediction.predicted_accuracy)) / 100.0
        return round(1.0 - abs(predicted - actual_accuracy), 4)

    return 0.0


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING METACOG-BENCH EVALUATION")
print("=" * 60)

results = metacog_bench.evaluate(
    llm=[kbench.llm],
    evaluation_data=all_data
)

# ============================================================
# COMPUTE AGGREGATE METRICS
# ============================================================

results_df = results.as_dataframe()
print(f"\nResults collected: {len(results_df)} rows")

# Per-task aggregate scores
task_scores = {}
for task_type in ["calibration", "fok", "error_detection", "abstention", "self_knowledge"]:
    mask = all_data["task_type"] == task_type
    indices = all_data[mask].index.tolist()
    if len(indices) > 0:
        scores = results_df.loc[results_df.index.isin(indices), "score"].values
        if len(scores) > 0:
            mean_score = float(np.mean(scores))
            lower, upper, _ = bootstrap_ci(scores, seed=42)
            task_scores[task_type] = {
                "mean": round(mean_score, 4),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_items": len(scores),
            }
            print(f"\n{task_type}: {mean_score:.4f} [{lower:.4f}, {upper:.4f}] (n={len(scores)})")

# Composite score (geometric mean)
sub_means = [task_scores[t]["mean"] for t in task_scores if task_scores[t]["mean"] > 0]
if sub_means:
    composite = geometric_mean(sub_means)
    print(f"\n{'=' * 60}")
    print(f"COMPOSITE SCORE (geometric mean): {composite:.4f}")
    print(f"{'=' * 60}")

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\nGenerating visualizations...")

if len(task_scores) >= 3:
    radar_data = {
        "Model": {
            "calibration": task_scores.get("calibration", {}).get("mean", 0),
            "sensitivity": task_scores.get("fok", {}).get("mean", 0),
            "error_detection": task_scores.get("error_detection", {}).get("mean", 0),
            "abstention": task_scores.get("abstention", {}).get("mean", 0),
            "self_knowledge": task_scores.get("self_knowledge", {}).get("mean", 0),
        }
    }
    fig = plot_radar_chart(radar_data)
    if fig:
        fig.savefig("metacog_radar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Radar chart saved: metacog_radar.png")

# Summary table
print("\n" + "=" * 60)
print("METACOG-BENCH SUMMARY")
print("=" * 60)
summary_data = []
for task, info in task_scores.items():
    summary_data.append({
        "Task": task,
        "Score": info["mean"],
        "95% CI Lower": info["ci_lower"],
        "95% CI Upper": info["ci_upper"],
        "N Items": info["n_items"],
    })
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
if sub_means:
    print(f"\nComposite: {composite:.4f}")


# %choose metacog_bench
