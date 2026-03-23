"""
Calibration & FOK dataset generator.

Generates 300 questions across 3 domains (math, factual, logic) × 3 difficulties
(easy, medium, hard) with verifiably correct answers.

Used by Task 1 (Confidence Calibration) and Task 2 (Feeling-of-Knowing).
"""

import random
import math
from itertools import combinations
from typing import List, Dict, Tuple

import pandas as pd


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
    # Novel composition questions (combine known facts for contamination resistance)
    ("What is the sum of the number of continents and the number of planets in our solar system?", "15"),  # 7+8
    ("What is the product of the number of sides on a triangle and the number of sides on a pentagon?", "15"),  # 3*5
    ("How many years between the signing of the Magna Carta (1215) and the French Revolution (1789)?", "574"),
    ("What is the sum of the atomic number of Helium and the atomic number of Carbon?", "8"),  # 2+6
    ("How many total letters are in the chemical symbols for Gold (Au) and Silver (Ag)?", "4"),
    ("What is the sum of the number of US states and the number of Canadian provinces?", "60"),  # 50+10
    ("How many years between Columbus reaching the Americas (1492) and the Moon landing (1969)?", "477"),
    ("What is the product of the number of vowels in English and the number of primary colors?", "15"),  # 5*3
    ("What is the sum of the boiling point and freezing point of water in Celsius?", "100"),  # 100+0
    ("How many total strings are on a standard guitar and a standard violin?", "10"),  # 6+4
    ("What is the square of the number of Harry Potter books?", "49"),  # 7^2
    ("How many total wheels on a bicycle and a tricycle?", "5"),  # 2+3
    ("What is the sum of the number of bones in the adult human body and the number of teeth?", "238"),  # 206+32
    ("How many years between the end of WWI (1918) and the start of WWII (1939)?", "21"),
    ("What is the product of the number of Great Lakes and the number of oceans?", "25"),  # 5*5
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

if __name__ == "__main__":
    df = generate_calibration_dataset()
    print(f"Total items: {len(df)}")
    print(f"\nDomain counts:\n{df['domain'].value_counts().to_string()}")
    print(f"\nDifficulty counts:\n{df['difficulty'].value_counts().to_string()}")
    print(f"\nDomain × Difficulty:\n{df.groupby(['domain', 'difficulty']).size().unstack(fill_value=0)}")
    print(f"\nUnique questions: {df['question'].nunique()}")
    print(f"\nSample items:")
    for domain in ["math", "factual", "logic"]:
        for diff in ["easy", "medium", "hard"]:
            row = df[(df["domain"] == domain) & (df["difficulty"] == diff)].iloc[0]
            print(f"  [{domain}/{diff}] Q: {row['question']}")
            print(f"              A: {row['correct_answer']}")
