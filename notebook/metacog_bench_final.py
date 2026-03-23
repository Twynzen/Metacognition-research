# ========================================
# MetaCog-Bench: Measuring What AI Knows About What It Knows
# Track: Metacognition
# Authors: [Team]
# ========================================
#
# This benchmark tests 5 metacognitive abilities:
# 1. Confidence Calibration (Nelson & Narens, 1990)
# 2. Feeling-of-Knowing (Hart, 1965)
# 3. Error Detection (Yeung & Summerfield, 2012)
# 4. Selective Abstention (Koriat & Goldsmith, 1996)
# 5. Metacognitive Knowledge (Flavell, 1979)
#
# Metrics: ECE, Brier Score, AUROC2, Goodman-Kruskal Gamma
# ========================================

import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import random
import re
import unicodedata
import math
from itertools import combinations

# ========================================
# --- SCHEMAS ---
# ========================================

@dataclass
class AnswerWithConfidence:
    """Task 1: Confidence Calibration -- retrospective monitoring."""
    answer: str
    confidence: int  # 0-100

@dataclass
class FOKResponse:
    """Task 2: Feeling-of-Knowing -- prospective monitoring."""
    prediction: int  # 0-100 likelihood of answering correctly
    answer: str

@dataclass
class ErrorReview:
    """Task 3: Error Detection -- error monitoring."""
    has_error: bool
    error_explanation: str
    corrected_answer: str

@dataclass
class AbstentionResponse:
    """Task 4: Selective Abstention -- metacognitive control."""
    can_answer: bool
    answer: Optional[str]
    confidence: int  # 0-100

@dataclass
class DomainPrediction:
    """Task 5: Metacognitive Knowledge -- self-knowledge."""
    predicted_accuracy: int  # 0-100
    hardest_aspect: str
    easiest_aspect: str

# ========================================
# --- UTILITIES ---
# ========================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove articles/punctuation."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\b(the|a|an)\b", "", text)
    text = re.sub(r"[^\w\s.]", "", text)
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
    3. Numerical comparison with tolerance (+/-0.01)
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
        first_word = model_norm.split()[0] if model_norm.split() else ""
        if first_word == correct_norm:
            return True

    return False

# ========================================
# --- METRICS ---
# ========================================

def compute_ece(confidences, correctness, n_bins=10):
    """
    Expected Calibration Error with equal-width bins.
    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
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
    """Brier Score: BS = (1/N) * sum((f_i - o_i)^2)"""
    conf = np.array(confidences, dtype=float)
    out = np.array(outcomes, dtype=float)
    return round(float(np.mean((conf - out) ** 2)), 4)


def compute_brier_decomposition(confidences, outcomes, n_bins=10):
    """Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty"""
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


def compute_auroc2(confidences, correctness):
    """
    Type 2 AUROC: metacognitive sensitivity (bias-free).
    Measures whether the model assigns higher confidence to correct responses
    than to incorrect ones.
    """
    from sklearn.metrics import roc_auc_score

    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

    if len(np.unique(corr)) < 2:
        return 0.5

    return round(float(roc_auc_score(corr, conf)), 4)


def compute_gamma(confidences, correctness):
    """
    Goodman-Kruskal Gamma correlation for FOK accuracy.
    gamma = (concordant - discordant) / (concordant + discordant)
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)

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

    denom = concordant + discordant
    if denom == 0:
        return 0.0

    return round(float((concordant - discordant) / denom), 4)


def geometric_mean(sub_scores):
    """Geometric mean of sub-task scores. Prevents compensation."""
    values = [max(v, 1e-10) for v in sub_scores]
    return round(float(np.exp(np.mean(np.log(values)))), 4)


def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95, seed=None):
    """Nonparametric bootstrap confidence intervals for any aggregate metric."""
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

# ========================================
# --- DATASET GENERATION: Calibration & FOK ---
# ========================================

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


_PRIMES = [p for p in range(2, 200) if _is_prime(p)]


def _generate_math_easy(rng: random.Random, n: int) -> List[Dict]:
    """Simple arithmetic: addition, subtraction, multiplication."""
    items = []
    seen = set()
    ops = [
        ("addition", "+", lambda a, b: a + b),
        ("subtraction", "-", lambda a, b: a - b),
        ("multiplication", "\u00d7", lambda a, b: a * b),
    ]
    while len(items) < n:
        op_name, op_sym, op_fn = rng.choice(ops)
        if op_name == "multiplication":
            a, b = rng.randint(2, 30), rng.randint(2, 30)
        else:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        if op_name == "subtraction" and a < b:
            a, b = b, a
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
            q = f"What is ({a} \u00d7 {b}) mod {m}?"
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
            primes = rng.sample(_PRIMES[5:], 3)
            primes.sort()
            product = primes[0] * primes[1] * primes[2]
            key = ("pf", product)
            if key in seen:
                continue
            seen.add(key)
            largest = max(primes)
            q = f"What is the largest prime factor of {product}?"
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
        ordering = list(names)
        rng.shuffle(ordering)

        clues = []
        revealed = set()
        for i in range(num_people - 1):
            a_idx, b_idx = i, i + 1
            a_name, b_name = ordering[a_idx], ordering[b_idx]
            clues.append(f"{b_name} is {prop_comp} than {a_name}")
            revealed.add((a_idx, b_idx))

        rng.shuffle(clues)
        clue_text = ". ".join(clues) + "."

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
    """Constraint satisfaction with 4+ variables."""
    items = []
    seen = set()

    colors = ["red", "blue", "green", "yellow", "purple", "orange"]

    while len(items) < n:
        template = rng.choice(["seating", "assignment"])

        if template == "seating":
            names = rng.sample(_NAMES, 4)
            order = list(names)
            rng.shuffle(order)

            clues = []
            end_person = rng.choice([0, 3])
            pos_label = "first" if end_person == 0 else "last"
            clues.append(f"{order[end_person]} sits in the {pos_label} position")

            adj_start = rng.randint(0, 2)
            clues.append(
                f"{order[adj_start]} sits directly to the left of {order[adj_start + 1]}"
            )

            non_adj_pairs = [(i, j) for i in range(4) for j in range(4)
                             if abs(i - j) > 1 and i < j]
            if non_adj_pairs:
                i, j = rng.choice(non_adj_pairs)
                clues.append(
                    f"{order[i]} does not sit next to {order[j]}"
                )

            mid = rng.choice([1, 2])
            clues.append(f"{order[mid]} sits in position {mid + 1}")

            rng.shuffle(clues)
            clue_text = ". ".join(clues) + "."

            asked = rng.choice(range(4))
            q = (f"Four people sit in a row (positions 1-4, left to right). "
                 f"{clue_text} What position does {order[asked]} sit in?")
            answer = str(asked + 1)

        else:
            names = rng.sample(_NAMES, 3)
            chosen_colors = rng.sample(colors, 3)
            assignment = dict(zip(names, chosen_colors))

            clues = []
            wrong_person = rng.choice(names)
            wrong_colors = [c for c in chosen_colors if c != assignment[wrong_person]]
            clues.append(f"{wrong_person} does not have {rng.choice(wrong_colors)}")

            direct = rng.choice(names)
            clues.append(f"{direct} has {assignment[direct]}")

            other = [nm for nm in names if nm != direct][0]
            not_color = [c for c in chosen_colors if c != assignment[other]
                         and c != assignment[direct]]
            if not_color:
                clues.append(f"{other} does not have {not_color[0]}")

            rng.shuffle(clues)
            clue_text = ". ".join(clues) + "."

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


def generate_calibration_dataset(n: int = 300) -> pd.DataFrame:
    """Generate calibration/FOK dataset of n items (default 300)."""
    rng = random.Random(42)
    old_state = random.getstate()
    random.seed(42)

    per_domain = n // 3
    n_easy = round(per_domain * 0.20)
    n_medium = round(per_domain * 0.60)
    n_hard = per_domain - n_easy - n_medium

    all_items: List[Dict] = []

    all_items.extend(_generate_math_easy(rng, n_easy))
    all_items.extend(_generate_math_medium(rng, n_medium))
    all_items.extend(_generate_math_hard(rng, n_hard))
    all_items.extend(_generate_factual(rng, n_easy, n_medium, n_hard))
    all_items.extend(_generate_logic_easy(rng, n_easy))
    all_items.extend(_generate_logic_medium(rng, n_medium))
    all_items.extend(_generate_logic_hard(rng, n_hard))

    random.setstate(old_state)

    df = pd.DataFrame(all_items)
    assert df["question"].nunique() == len(df), "Duplicate questions detected"
    assert len(df) == n, f"Expected {n} items, got {len(df)}"
    return df


# ========================================
# --- DATASET GENERATION: Error Detection ---
# ========================================

def _prime_factors_ed(n: int) -> List[int]:
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


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _comb_ed(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def _arith_multiplication(rng: random.Random, inject_error: bool) -> Dict:
    a = rng.randint(12, 99)
    b = rng.randint(12, 99)
    correct = a * b
    problem = f"What is {a} \u00d7 {b}?"
    if inject_error:
        offsets = [10, -10, 1, -1, a, -a, b, -b]
        offset = rng.choice([o for o in offsets if o != 0])
        wrong = correct + offset
        sol = f"{a} \u00d7 {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} \u00d7 {b} = {correct}"
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
    problem = f"What is {a} \u2212 {b}?"
    if inject_error:
        wrong = correct + rng.choice([1, -1, 10, -10, 100, -100])
        sol = f"{a} \u2212 {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} \u2212 {b} = {correct}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_division(rng: random.Random, inject_error: bool) -> Dict:
    b = rng.randint(2, 25)
    quotient = rng.randint(5, 200)
    a = b * quotient
    problem = f"What is {a} \u00f7 {b}?"
    if inject_error:
        wrong = quotient + rng.choice([1, -1, 2, -2])
        sol = f"{a} \u00f7 {b} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = f"{a} \u00f7 {b} = {quotient}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _arith_modular(rng: random.Random, inject_error: bool) -> Dict:
    a = rng.randint(50, 500)
    b = rng.randint(10, 99)
    m = rng.choice([7, 11, 13, 17, 19, 23])
    correct = (a * b) % m
    problem = f"What is ({a} \u00d7 {b}) mod {m}?"
    if inject_error:
        wrong = (correct + rng.randint(1, m - 1)) % m
        if wrong == correct:
            wrong = (correct + 1) % m
        sol = (f"({a} \u00d7 {b}) = {a * b}. "
               f"{a * b} mod {m} = {wrong}")
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        sol = (f"({a} \u00d7 {b}) = {a * b}. "
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
    base_val = rng.randint(2, 50) * 10
    correct = base_val * pct / 100
    problem = f"What is {pct}% of {base_val}?"
    if inject_error:
        wrong_options = [base_val * pct / 1000, base_val * (pct + 10) / 100,
                         base_val * (pct - 5) / 100]
        wrong = rng.choice([w for w in wrong_options if w != correct and w > 0])
        if wrong == int(wrong):
            wrong = int(wrong)
        sol = f"{pct}% of {base_val} = {wrong}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "arithmetic"}
    else:
        disp = int(correct) if correct == int(correct) else correct
        sol = f"{pct}% of {base_val} = {disp}"
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


_ARITH_GENERATORS = [
    _arith_multiplication, _arith_addition_chain, _arith_subtraction,
    _arith_division, _arith_modular, _arith_exponentiation, _arith_percentage,
]


def _logic_affirming_consequent(rng: random.Random) -> Dict:
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "true", "error_type": "logical"}


def _logic_false_dichotomy(rng: random.Random) -> Dict:
    templates = [
        {"setup": "A student can either study literature or study science",
         "conclusion": "Since they are not studying literature, they must be studying science"},
        {"setup": "A person is either happy or sad",
         "conclusion": "Since they are not happy, they must be sad"},
        {"setup": "An employee either supports the new policy or opposes the company",
         "conclusion": "Since they don't support the new policy, they oppose the company"},
        {"setup": "A country is either a democracy or a dictatorship",
         "conclusion": "Since it is not a full democracy, it must be a dictatorship"},
        {"setup": "A food is either healthy or unhealthy",
         "conclusion": "Since chocolate is not purely healthy, it must be unhealthy"},
        {"setup": "Software is either perfectly secure or completely vulnerable",
         "conclusion": "Since the software has a minor bug, it is completely vulnerable"},
        {"setup": "You either agree with me completely or you are against me",
         "conclusion": "Since you raised an objection, you must be against me"},
        {"setup": "A movie is either a masterpiece or garbage",
         "conclusion": "Since critics found some flaws, the movie must be garbage"},
        {"setup": "An investment either guarantees profit or is a total loss",
         "conclusion": "Since the investment doesn't guarantee profit, it is a total loss"},
        {"setup": "A person is either an expert or completely ignorant",
         "conclusion": "Since they made one mistake, they must be completely ignorant"},
        {"setup": "A scientific theory is either proven or worthless",
         "conclusion": "Since the theory has not been proven beyond all doubt, it is worthless"},
        {"setup": "An athlete is either world-class or terrible",
         "conclusion": "Since they did not win the championship, they must be terrible"},
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "true", "error_type": "logical"}


def _logic_invalid_syllogism(rng: random.Random) -> Dict:
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "true", "error_type": "logical"}


def _logic_denying_antecedent(rng: random.Random) -> Dict:
    templates = [
        ("it snows", "the schools close", "the schools are open"),
        ("you eat too much sugar", "you may get cavities", "you won't get cavities"),
        ("the temperature drops below 0\u00b0C", "water freezes", "water doesn't freeze"),
        ("you practice daily", "you improve", "you won't improve"),
        ("a triangle is equilateral", "all its angles are 60\u00b0",
         "none of its angles are 60\u00b0"),
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "true", "error_type": "logical"}


def _logic_correct_modus_ponens(rng: random.Random) -> Dict:
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "false", "error_type": "none"}


def _logic_correct_modus_tollens(rng: random.Random) -> Dict:
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
    return {"problem": problem, "presented_solution": sol,
            "solution_has_error": "false", "error_type": "none"}


def _method_average_speed(rng: random.Random, inject_error: bool) -> Dict:
    d = rng.choice([60, 100, 120, 150, 180, 200, 240])
    v1 = rng.choice([30, 40, 50, 60])
    v2 = rng.choice([v for v in [60, 80, 90, 100, 120] if v != v1])
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
        wrong = pa + pb
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
            f"After {pct}% increase: ${original} \u00d7 {1 + pct / 100} "
            f"= ${after_increase:.2f}. "
            f"After {pct}% decrease: ${after_increase:.2f} \u00d7 {1 - pct / 100} "
            f"= ${after_decrease:.2f}. "
            f"Net change: ${net_change:.2f}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_combination_vs_permutation(rng: random.Random, inject_error: bool) -> Dict:
    n_val = rng.randint(5, 12)
    k = rng.randint(2, min(4, n_val - 1))
    correct_comb = _comb_ed(n_val, k)
    wrong_perm = math.factorial(n_val) // math.factorial(n_val - k)

    problem = (
        f"How many ways can you choose {k} people from a group "
        f"of {n_val} to form a committee?"
    )
    if inject_error:
        sol = (
            f"We need to pick {k} from {n_val}. "
            f"The number of ways = {n_val}! / ({n_val}-{k})! = "
            f"{n_val}! / {n_val - k}! = {wrong_perm}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "true", "error_type": "method"}
    else:
        sol = (
            f"C({n_val},{k}) = {n_val}! / ({k}! \u00d7 ({n_val}-{k})!) = {correct_comb}."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


def _method_base_rate_neglect(rng: random.Random, inject_error: bool) -> Dict:
    disease_rate = rng.choice([1, 2, 5])
    sensitivity = rng.choice([95, 98, 99])
    false_pos = rng.choice([3, 5, 10])

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
            f"P(+|disease)\u00d7P(disease) / P(+). "
            f"P(+) = ({p_pos_given_d}\u00d7{p_d}) + ({p_pos_given_not_d}\u00d7{1-p_d:.4f}) "
            f"= {p_pos:.6f}. "
            f"P(disease|+) = {p_d*p_pos_given_d:.6f}/{p_pos:.6f} \u2248 {correct_ppv}%."
        )
        return {"problem": problem, "presented_solution": sol,
                "solution_has_error": "false", "error_type": "none"}


_METHOD_GENERATORS = [
    _method_average_speed, _method_probability_or, _method_percentage_change,
    _method_combination_vs_permutation, _method_base_rate_neglect,
]


_FACTUAL_ITEMS_ED = [
    ("In what year was the Declaration of Independence signed?",
     "The Declaration of Independence was signed in 1776.",
     "The Declaration of Independence was signed in 1774."),
    ("Who wrote the novel '1984'?",
     "'1984' was written by George Orwell.",
     "'1984' was written by Aldous Huxley."),
    ("What is the chemical formula for table salt?",
     "The chemical formula for table salt is NaCl (sodium chloride).",
     "The chemical formula for table salt is KCl (potassium chloride)."),
    ("What is the speed of light in a vacuum, approximately?",
     "The speed of light in a vacuum is approximately 3 \u00d7 10^8 m/s (299,792,458 m/s).",
     "The speed of light in a vacuum is approximately 3 \u00d7 10^6 m/s (2,997,924 m/s)."),
    ("Which planet is the largest in our solar system?",
     "Jupiter is the largest planet in our solar system.",
     "Saturn is the largest planet in our solar system."),
    ("What is the boiling point of water at sea level in Celsius?",
     "The boiling point of water at sea level is 100\u00b0C.",
     "The boiling point of water at sea level is 110\u00b0C."),
    ("Who painted the Sistine Chapel ceiling?",
     "The Sistine Chapel ceiling was painted by Michelangelo.",
     "The Sistine Chapel ceiling was painted by Raphael."),
    ("What is the atomic number of carbon?",
     "The atomic number of carbon is 6.",
     "The atomic number of carbon is 8."),
    ("In which year did World War I begin?",
     "World War I began in 1914.",
     "World War I began in 1912."),
    ("What is the smallest prime number?",
     "The smallest prime number is 2.",
     "The smallest prime number is 1."),
    ("Who discovered penicillin?",
     "Penicillin was discovered by Alexander Fleming in 1928.",
     "Penicillin was discovered by Louis Pasteur in 1928."),
    ("What is the capital of Australia?",
     "The capital of Australia is Canberra.",
     "The capital of Australia is Sydney."),
    ("How many chromosomes do humans have?",
     "Humans have 46 chromosomes (23 pairs).",
     "Humans have 48 chromosomes (24 pairs)."),
    ("What is the most abundant gas in Earth's atmosphere?",
     "Nitrogen is the most abundant gas in Earth's atmosphere, at about 78%.",
     "Oxygen is the most abundant gas in Earth's atmosphere, at about 78%."),
    ("Who formulated the three laws of motion?",
     "The three laws of motion were formulated by Isaac Newton.",
     "The three laws of motion were formulated by Galileo Galilei."),
    ("What is the pH of pure water at 25\u00b0C?",
     "The pH of pure water at 25\u00b0C is 7.",
     "The pH of pure water at 25\u00b0C is 7.4."),
    ("In what year did the Berlin Wall fall?",
     "The Berlin Wall fell in 1989.",
     "The Berlin Wall fell in 1991."),
    ("What is the longest river in the world?",
     "The Nile is generally considered the longest river in the world at about 6,650 km.",
     "The Amazon is the longest river in the world at about 6,650 km."),
    ("Who developed the theory of general relativity?",
     "The theory of general relativity was developed by Albert Einstein, published in 1915.",
     "The theory of general relativity was developed by Albert Einstein, published in 1905."),
    ("What is the electron configuration of helium?",
     "The electron configuration of helium is 1s\u00b2.",
     "The electron configuration of helium is 1s\u00b9."),
    ("Which element has the atomic number 79?",
     "Gold (Au) has the atomic number 79.",
     "Silver (Ag) has the atomic number 79."),
    ("What is the diameter of Earth approximately?",
     "Earth's diameter is approximately 12,742 km.",
     "Earth's diameter is approximately 10,742 km."),
    ("Who wrote 'The Republic'?",
     "'The Republic' was written by Plato.",
     "'The Republic' was written by Aristotle."),
    ("What is the freezing point of mercury?",
     "The freezing point of mercury is approximately -39\u00b0C (-38.83\u00b0C).",
     "The freezing point of mercury is approximately -29\u00b0C."),
    ("In what year was the Magna Carta signed?",
     "The Magna Carta was signed in 1215.",
     "The Magna Carta was signed in 1225."),
    ("What is the half-life of Carbon-14?",
     "The half-life of Carbon-14 is approximately 5,730 years.",
     "The half-life of Carbon-14 is approximately 5,370 years."),
    ("Which country has the largest land area?",
     "Russia has the largest land area of any country, at about 17.1 million km\u00b2.",
     "Canada has the largest land area of any country, at about 17.1 million km\u00b2."),
    ("What is the value of Avogadro's number?",
     "Avogadro's number is approximately 6.022 \u00d7 10^23.",
     "Avogadro's number is approximately 6.022 \u00d7 10^26."),
    ("Who composed 'The Four Seasons'?",
     "'The Four Seasons' was composed by Antonio Vivaldi.",
     "'The Four Seasons' was composed by Johann Sebastian Bach."),
    ("What is the hardest natural mineral?",
     "Diamond is the hardest natural mineral, rated 10 on the Mohs scale.",
     "Corundum is the hardest natural mineral, rated 10 on the Mohs scale."),
    ("In which year did the French Revolution begin?",
     "The French Revolution began in 1789.",
     "The French Revolution began in 1793."),
    ("What is the SI unit of electric current?",
     "The SI unit of electric current is the ampere (A).",
     "The SI unit of electric current is the volt (V)."),
    ("How many bones are in the adult human body?",
     "The adult human body has 206 bones.",
     "The adult human body has 208 bones."),
    ("Who invented the telephone?",
     "Alexander Graham Bell is credited with inventing the telephone in 1876.",
     "Thomas Edison is credited with inventing the telephone in 1876."),
    ("What is the formula for the area of a circle?",
     "The area of a circle is A = \u03c0r\u00b2.",
     "The area of a circle is A = 2\u03c0r."),
    ("What is the deepest point in the ocean?",
     "The Mariana Trench's Challenger Deep is the deepest point, at about 10,935 m.",
     "The Mariana Trench's Challenger Deep is the deepest point, at about 8,935 m."),
    ("Who was the first person to walk on the Moon?",
     "Neil Armstrong was the first person to walk on the Moon on July 20, 1969.",
     "Neil Armstrong was the first person to walk on the Moon on July 20, 1968."),
    ("What is the currency of Japan?",
     "The currency of Japan is the Japanese yen (\u00a5 / JPY).",
     "The currency of Japan is the Japanese won (\u20a9 / JPW)."),
    ("What is the tallest mountain in the world?",
     "Mount Everest is the tallest mountain, at 8,849 metres above sea level.",
     "Mount Everest is the tallest mountain, at 8,649 metres above sea level."),
    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci.",
     "The Mona Lisa was painted by Leonardo da Vinci in 1623."),
]


def _generate_unique_ed(rng, generator, used_problems, count,
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
            items.append(item)
    return items


def generate_error_detection_dataset(n: int = 200) -> pd.DataFrame:
    """Generate error detection dataset: 50% correct, 50% with planted errors."""
    random.seed(42)
    rng = random.Random(42)

    n_error = n // 2
    n_correct = n - n_error

    n_arith_err = 35 if n_error == 100 else round(0.35 * n_error)
    n_logic_err = 25 if n_error == 100 else round(0.25 * n_error)
    n_method_err = 20 if n_error == 100 else round(0.20 * n_error)
    n_factual_err = n_error - n_arith_err - n_logic_err - n_method_err

    items: List[Dict] = []
    used_problems: set = set()

    # Arithmetic errors
    for _ in range(n_arith_err):
        gen = rng.choice(_ARITH_GENERATORS)
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1, inject_error=True))

    # Logical errors
    logic_err_generators = [
        _logic_affirming_consequent, _logic_false_dichotomy,
        _logic_invalid_syllogism, _logic_denying_antecedent,
    ]
    for i in range(n_logic_err):
        gen = logic_err_generators[i % len(logic_err_generators)]
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1))

    # Method errors
    for i in range(n_method_err):
        gen = _METHOD_GENERATORS[i % len(_METHOD_GENERATORS)]
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1, inject_error=True))

    # Factual errors
    factual_pool = list(_FACTUAL_ITEMS_ED)
    rng.shuffle(factual_pool)
    factual_for_errors = factual_pool[:n_factual_err]
    factual_for_correct = factual_pool[n_factual_err:]

    for problem, _, wrong_sol in factual_for_errors:
        used_problems.add(problem)
        items.append({"problem": problem, "presented_solution": wrong_sol,
                      "solution_has_error": "true", "error_type": "factual"})

    # Correct items
    n_arith_ok = n_correct // 4
    n_logic_ok = n_correct // 4
    n_method_ok = n_correct // 4
    n_factual_ok = n_correct - n_arith_ok - n_logic_ok - n_method_ok

    for _ in range(n_arith_ok):
        gen = rng.choice(_ARITH_GENERATORS)
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1, inject_error=False))

    logic_correct_generators = [_logic_correct_modus_ponens, _logic_correct_modus_tollens]
    for i in range(n_logic_ok):
        gen = logic_correct_generators[i % len(logic_correct_generators)]
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1))

    for i in range(n_method_ok):
        gen = _METHOD_GENERATORS[i % len(_METHOD_GENERATORS)]
        items.extend(_generate_unique_ed(rng, gen, used_problems, 1, inject_error=False))

    for i in range(n_factual_ok):
        problem, correct_sol, _ = factual_for_correct[i % len(factual_for_correct)]
        if problem not in used_problems:
            used_problems.add(problem)
        items.append({"problem": problem, "presented_solution": correct_sol,
                      "solution_has_error": "false", "error_type": "none"})

    rng.shuffle(items)
    df = pd.DataFrame(items)
    df = df[["problem", "presented_solution", "solution_has_error", "error_type"]]
    return df


# ========================================
# --- DATASET GENERATION: Abstention ---
# ========================================

_ABS_EASY_QUESTIONS = [
    ("What is 7 \u00d7 8?", "56"), ("What is 12 + 19?", "31"),
    ("What is 100 - 37?", "63"), ("What is 9 \u00d7 6?", "54"),
    ("What is 144 / 12?", "12"), ("What is 15 + 28?", "43"),
    ("What is 50 - 23?", "27"), ("What is 8 \u00d7 7?", "56"),
    ("What is 81 / 9?", "9"), ("What is 25 + 36?", "61"),
    ("What is 200 - 45?", "155"), ("What is 11 \u00d7 11?", "121"),
    ("What is 72 / 8?", "9"), ("What is 33 + 44?", "77"),
    ("What is 6 \u00d7 9?", "54"),
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
    ("In what year did World War II end?", "1945"),
    ("In what year did the United States declare independence?", "1776"),
    ("Who was the first President of the United States?", "George Washington"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did World War I begin?", "1914"),
]

_ABS_MEDIUM_QUESTIONS = [
    ("What is 347 + 589?", "936"), ("What is 23 \u00d7 17?", "391"),
    ("What is the square root of 196?", "14"), ("What is 15% of 240?", "36"),
    ("What is 1024 / 32?", "32"), ("What is 37 \u00d7 43?", "1591"),
    ("What is 2 to the power of 10?", "1024"), ("What is 999 - 573?", "426"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("What is 125 \u00d7 8?", "1000"), ("What is the square root of 289?", "17"),
    ("What is 45% of 200?", "90"), ("What is 7 to the power of 3?", "343"),
    ("What is 3.5 \u00d7 2.4?", "8.4"),
    ("What is the atomic number of carbon?", "6"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen"),
    ("What is the pH of pure water at 25\u00b0C?", "7"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("How many chromosomes do humans have?", "46"),
    ("What element has the atomic number 79?", "Gold"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the SI unit of electric current?", "Ampere"),
    ("What type of bond involves the sharing of electrons?", "Covalent"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What vitamin is produced when skin is exposed to sunlight?", "Vitamin D"),
    ("How many valence electrons does carbon have?", "4"),
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

_ABS_HARD_QUESTIONS = [
    ("What is the largest prime factor of 2310?", "11"),
    ("What is 17 \u00d7 23 mod 13?", "1"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("How many prime numbers are there between 1 and 50?", "15"),
    ("What is the value of 12 factorial divided by 10 factorial?", "132"),
    ("What is the cube root of 2744?", "14"),
    ("What is 2^16?", "65536"),
    ("What is the sum of interior angles of a hexagon in degrees?", "720"),
    ("What is the atomic number of Rutherfordium?", "104"),
    ("What is the Avogadro constant to 3 significant figures?", "6.02e23"),
    ("What is the half-life of Carbon-14 in years, approximately?", "5730"),
    ("What is the charge of a muon in units of elementary charge?", "-1"),
    ("What element has the highest electronegativity?", "Fluorine"),
    ("What is the second most abundant element in Earth's crust?", "Silicon"),
    ("What is the escape velocity of Earth in km/s, approximately?", "11.2"),
    ("What is the name of the longest bone in the human body?", "Femur"),
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the smallest country in mainland Africa by area?", "Gambia"),
    ("What is the highest capital city in the world by elevation?", "La Paz"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the longest river in Europe?", "Volga"),
    ("What sea borders both Iran and Russia?", "Caspian Sea"),
    ("What is the deepest lake in the world?", "Lake Baikal"),
    ("In what year was the Battle of Hastings?", "1066"),
    ("Who was the last Pharaoh of ancient Egypt?", "Cleopatra"),
    ("What year was the Peace of Augsburg signed?", "1555"),
    ("What was the capital of the Byzantine Empire?", "Constantinople"),
    ("In what year did the Taiping Rebellion begin?", "1850"),
    ("Who was the first Mughal emperor?", "Babur"),
    ("What treaty ended the Thirty Years' War?", "Treaty of Westphalia"),
]

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
    "How far is it?", "What time does it start?", "How much does it cost?",
    "Is it bigger?", "What did they decide?", "When did he arrive?",
    "How many are there?", "What is the answer?", "Did she win?",
    "Where did it happen?", "Can you compare them?", "Which one is better?",
    "How long did it take?", "What was the result?", "Who was responsible?",
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
    """Generate abstention dataset: 120 answerable + 80 unanswerable."""
    random.seed(42)

    n_answerable = int(n * 0.6)
    n_unanswerable = n - n_answerable

    n_easy = int(n_answerable * 40 / 120)
    n_hard = int(n_answerable * 30 / 120)
    n_medium = n_answerable - n_easy - n_hard

    easy_pool = list(_ABS_EASY_QUESTIONS)
    medium_pool = list(_ABS_MEDIUM_QUESTIONS)
    hard_pool = list(_ABS_HARD_QUESTIONS)

    random.shuffle(easy_pool)
    random.shuffle(medium_pool)
    random.shuffle(hard_pool)

    easy_items = easy_pool[:n_easy]
    medium_items = medium_pool[:n_medium]
    hard_items = hard_pool[:n_hard]

    answerable_rows = []
    for q, a in easy_items + medium_items + hard_items:
        answerable_rows.append({
            "question": q, "is_answerable": "true",
            "correct_answer": a, "unanswerable_reason": "",
        })

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
            unanswerable_rows.append({
                "question": q, "is_answerable": "false",
                "correct_answer": "", "unanswerable_reason": reason,
            })

    all_rows = answerable_rows + unanswerable_rows
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)
    assert len(df) == n, f"Expected {n} rows, got {len(df)}"
    assert df["question"].nunique() == len(df), "Duplicate questions found!"
    return df


# ========================================
# --- DATASET GENERATION: Self-Knowledge ---
# ========================================

def _build_domains():
    """Return a list of (domain_name, [(question, answer), ...]) tuples."""
    domains = []

    # EASY DOMAINS
    domains.append(("basic_arithmetic", [
        ("What is 15 \u00d7 12?", "180"), ("What is 144 / 12?", "12"),
        ("What is 256 + 389?", "645"), ("What is 1000 - 637?", "363"),
        ("What is 25 \u00d7 25?", "625"), ("What is 729 / 27?", "27"),
        ("What is 48 + 57?", "105"), ("What is 13 \u00d7 17?", "221"),
        ("What is 900 / 15?", "60"), ("What is 333 + 444?", "777"),
    ]))

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

    # MEDIUM DOMAINS
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

    # HARD DOMAINS
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

    # TRICK / SPECIAL DOMAINS
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
    """Generate self-knowledge dataset: 20 domains x 10 questions each."""
    random.seed(42)

    domains = _build_domains()

    for _name, qa_pairs in domains:
        random.shuffle(qa_pairs)

    rows = []
    for domain_name, qa_pairs in domains:
        questions = [q for q, _a in qa_pairs]
        answers = [a for _q, a in qa_pairs]

        assert len(questions) == 10
        assert len(answers) == 10

        rows.append({
            "domain": domain_name,
            "domain_questions": "|||".join(questions),
            "domain_answers": "|||".join(answers),
        })

    df = pd.DataFrame(rows)
    assert len(df) == 20
    return df


# ========================================
# --- GENERATE ALL DATASETS ---
# ========================================

random.seed(42)
calibration_df = generate_calibration_dataset(n=300)
error_df = generate_error_detection_dataset(n=200)
abstention_df = generate_abstention_dataset(n=200)
selfknow_df = generate_self_knowledge_dataset(n=200)

# ========================================
# --- BUILD UNIFIED EVALUATION DATA ---
# ========================================

# Calibration task data
cal_data = calibration_df.copy()
cal_data['task_type'] = 'calibration'
cal_data['extra_data'] = ''

# FOK task data (same questions, different task)
fok_data = calibration_df.copy()
fok_data['task_type'] = 'fok'
fok_data['extra_data'] = ''

# Error detection task data
err_data = error_df.copy()
err_data = err_data.rename(columns={'problem': 'question', 'presented_solution': 'correct_answer'})
err_data['task_type'] = 'error_detection'
err_data['extra_data'] = err_data['solution_has_error'] + '|' + err_data['error_type']
err_data['difficulty'] = 'medium'

# Abstention task data
abs_data = abstention_df.copy()
abs_data['task_type'] = 'abstention'
abs_data['extra_data'] = abs_data['is_answerable'] + '|' + abs_data['unanswerable_reason']
abs_data['difficulty'] = 'medium'

# Self-knowledge task data
sk_data = selfknow_df.copy()
sk_data = sk_data.rename(columns={'domain': 'question', 'domain_questions': 'correct_answer'})
sk_data['task_type'] = 'self_knowledge'
sk_data['extra_data'] = sk_data['domain_answers']
sk_data['difficulty'] = 'medium'

# Unify columns
for df in [cal_data, fok_data, err_data, abs_data, sk_data]:
    for col in ['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']:
        if col not in df.columns:
            df[col] = ''

all_data = pd.concat([
    cal_data[['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']],
    fok_data[['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']],
    err_data[['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']],
    abs_data[['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']],
    sk_data[['question', 'correct_answer', 'difficulty', 'task_type', 'extra_data']],
], ignore_index=True)

print(f"Total evaluation items: {len(all_data)}")
print(f"Task distribution:\n{all_data['task_type'].value_counts().to_string()}")

# ========================================
# --- COMPOSITE TASK ---
# ========================================

@kbench.task(
    name="metacog_bench",
    description="MetaCog-Bench: Comprehensive metacognition benchmark testing "
                "calibration, FOK, error detection, abstention, and self-knowledge. "
                "Based on Nelson & Narens (1990), Fleming & Lau (2014), and "
                "Flavell (1979) metacognitive frameworks."
)
def metacog_bench(llm, task_type: str, question: str, correct_answer: str,
                  difficulty: str, extra_data: str) -> float:
    """
    Unified metacognition benchmark dispatcher.
    Routes each item to the appropriate task implementation based on task_type.
    """

    if task_type == "calibration":
        # ---- Task 1: Confidence Calibration ----
        response = llm.prompt(
            f"Answer the following question. After your answer, rate your "
            f"confidence from 0 to 100 (where 0 = certainly wrong, "
            f"100 = certainly correct).\n\n"
            f"Question: {question}\n\n"
            f"Respond with your answer and confidence level.",
            schema=AnswerWithConfidence
        )
        is_correct = check_answer(response.answer, correct_answer)
        confidence = max(0, min(100, response.confidence)) / 100.0
        correctness = 1.0 if is_correct else 0.0
        calibration_error = abs(confidence - correctness)
        return round(1.0 - calibration_error, 4)

    elif task_type == "fok":
        # ---- Task 2: Feeling-of-Knowing ----
        fok_response = llm.prompt(
            f"I'm going to ask you a question. Before you answer, predict "
            f"how likely you are to get it correct on a scale of 0-100.\n\n"
            f"Then provide your best answer.\n\n"
            f"Question: {question}",
            schema=FOKResponse
        )
        is_correct = check_answer(fok_response.answer, correct_answer)
        prediction = max(0, min(100, fok_response.prediction)) / 100.0
        correctness = 1.0 if is_correct else 0.0
        fok_error = abs(prediction - correctness)
        return round(1.0 - fok_error, 4)

    elif task_type == "error_detection":
        # ---- Task 3: Error Detection ----
        # question = problem, correct_answer = presented_solution
        # extra_data = "solution_has_error|error_type"
        parts = extra_data.split("|")
        solution_has_error = parts[0] if len(parts) > 0 else "false"

        response = llm.prompt(
            f"Here is a problem and a proposed solution. Please review the "
            f"solution carefully and determine if it is correct.\n\n"
            f"Problem: {question}\n\n"
            f"Proposed Solution: {correct_answer}\n\n"
            f"Is this solution correct? If not, explain the error and "
            f"provide the correct answer.",
            schema=ErrorReview
        )

        has_error = solution_has_error.lower() == "true"

        if has_error and response.has_error:
            return 1.0  # True positive
        elif not has_error and not response.has_error:
            return 1.0  # True negative
        else:
            return 0.0  # Misclassification

    elif task_type == "abstention":
        # ---- Task 4: Selective Abstention ----
        # extra_data = "is_answerable|unanswerable_reason"
        parts = extra_data.split("|")
        is_answerable = parts[0] if len(parts) > 0 else "true"

        response = llm.prompt(
            f"Answer the following question. If you genuinely cannot "
            f"determine the answer or the question is unanswerable, "
            f"say \"I don't know\" and explain why.\n\n"
            f"Question: {question}",
            schema=AbstentionResponse
        )

        answerable = is_answerable.lower() == "true"
        abstained = not response.can_answer or (
            response.answer and "i don't know" in response.answer.lower()
        )

        if answerable:
            if abstained:
                return 0.25  # Overly cautious
            elif check_answer(response.answer or "", correct_answer):
                return 1.0   # Correct
            else:
                return 0.0   # Wrong
        else:  # Unanswerable
            if abstained:
                return 1.0   # Correctly abstained
            else:
                return 0.0   # Hallucinated an answer

    elif task_type == "self_knowledge":
        # ---- Task 5: Metacognitive Knowledge ----
        # question = domain name, correct_answer = pipe-separated questions
        # extra_data = pipe-separated answers
        domain = question
        domain_questions = correct_answer
        domain_answers = extra_data

        # Phase 1: Self-assessment
        prediction = llm.prompt(
            f"I'm going to test you on 10 questions about {domain}. "
            f"Before I do, predict what percentage (0-100) you'll get "
            f"correct. Also tell me what you think will be hardest and "
            f"easiest.",
            schema=DomainPrediction
        )

        # Phase 2: Answer domain questions sequentially
        questions = domain_questions.split("|||")
        answers = domain_answers.split("|||")
        correct_count = 0

        for q, a in zip(questions, answers):
            resp = llm.prompt(f"Answer briefly: {q}")
            if check_answer(str(resp), a):
                correct_count += 1

        actual_accuracy = correct_count / len(questions)
        predicted = max(0, min(100, prediction.predicted_accuracy)) / 100.0

        error = abs(predicted - actual_accuracy)
        return round(1.0 - error, 4)

    # Fallback (should not reach here)
    return 0.0


# ========================================
# --- RUN EVALUATION ---
# ========================================

results = metacog_bench.evaluate(
    llm=[kbench.llm],
    evaluation_data=all_data
)

# ========================================
# --- COMPUTE AGGREGATE METRICS ---
# ========================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

results_df = results.as_dataframe()
print("\n=== Raw Results Summary ===")
print(f"Total results: {len(results_df)}")
print(results_df.head())

# Merge task_type info back into results
results_df = results_df.merge(
    all_data[['question', 'task_type', 'extra_data', 'correct_answer']],
    on='question', how='left', suffixes=('', '_orig')
)

# Compute per-task-type metrics
task_types = ['calibration', 'fok', 'error_detection', 'abstention', 'self_knowledge']
model_metrics = {}

for model_name in results_df['model'].unique() if 'model' in results_df.columns else ['default']:
    if 'model' in results_df.columns:
        model_df = results_df[results_df['model'] == model_name]
    else:
        model_df = results_df

    metrics = {}

    for task in task_types:
        task_df = model_df[model_df['task_type'] == task]
        if len(task_df) == 0:
            continue

        scores = task_df['score'].values if 'score' in task_df.columns else []
        if len(scores) == 0:
            continue

        mean_score = float(np.mean(scores))
        metrics[f'{task}_mean'] = round(mean_score, 4)

        # Bootstrap CI
        if len(scores) > 1:
            lo, hi, boot_mean = bootstrap_ci(scores, n_bootstrap=1000, seed=42)
            metrics[f'{task}_ci_lower'] = lo
            metrics[f'{task}_ci_upper'] = hi

    # Compute composite score
    task_means = [metrics.get(f'{t}_mean', 0) for t in task_types if f'{t}_mean' in metrics]
    if task_means:
        metrics['composite'] = geometric_mean(task_means)

    model_metrics[model_name] = metrics

# Print results
print("\n=== MetaCog-Bench Results ===")
for model_name, metrics in model_metrics.items():
    print(f"\n--- {model_name} ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

# ========================================
# --- VISUALIZATIONS ---
# ========================================

if HAS_MATPLOTLIB:
    # Radar chart of model performance across tasks
    radar_data = {}
    for model_name, metrics in model_metrics.items():
        radar_data[model_name] = {
            'calibration': metrics.get('calibration_mean', 0),
            'sensitivity': metrics.get('fok_mean', 0),
            'error_detection': metrics.get('error_detection_mean', 0),
            'abstention': metrics.get('abstention_mean', 0),
            'self_knowledge': metrics.get('self_knowledge_mean', 0),
        }

    # Plot radar chart
    categories = [
        "Calibration\n(1-ECE)",
        "Sensitivity\n(AUROC\u2082)",
        "Error\nDetection",
        "Abstention\nAccuracy",
        "Self-\nKnowledge",
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    for idx, (model_name, scores) in enumerate(radar_data.items()):
        values = [scores.get(k, 0) for k in
                  ["calibration", "sensitivity", "error_detection",
                   "abstention", "self_knowledge"]]
        values += values[:1]

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
    ax.set_title("MetaCog-Bench: Metacognitive Profile", fontsize=14, y=1.08)
    fig.tight_layout()
    plt.savefig("metacog_radar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Radar chart saved to metacog_radar.png")

    # Model comparison table
    print("\n=== Model Comparison Table ===")
    table_rows = []
    for model_name, metrics in model_metrics.items():
        row = {"Model": model_name}
        row.update(metrics)
        table_rows.append(row)
    comparison_df = pd.DataFrame(table_rows)
    if len(comparison_df) > 0:
        print(comparison_df.to_string(index=False))

# %choose metacog_bench
