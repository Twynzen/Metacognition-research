"""
Error Detection dataset generator for MetaCog-Bench.

Generates 200 items: 100 correct solutions + 100 with planted (plausible) errors.
Error types: arithmetic (~35), logical (~25), method (~20), factual (~20).
"""

import random
import math
import pandas as pd
from typing import List, Dict, Tuple


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

# Curated subtle method errors — these require metacognitive reasoning to detect
_STATIC_METHOD_ERRORS = [
    {
        "problem": "A store offers 20% off, then an additional 15% off the reduced price. What is the total discount?",
        "presented_solution": "Total discount = 20% + 15% = 35%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "You drive 60 miles at 30 mph, then 60 miles at 60 mph. What is your average speed for the whole trip?",
        "presented_solution": "Average speed = (30 + 60) / 2 = 45 mph.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "A ball is thrown upward at 20 m/s. What is the maximum height reached? (Use g = 10 m/s².)",
        "presented_solution": "Maximum height = v × t = 20 × 2 = 40 m. (Time to reach top = v/g = 20/10 = 2 s.)",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "What is the probability of getting at least one head in 3 coin flips?",
        "presented_solution": "There are 3 flips and 6 possible outcomes. Favorable outcomes (at least one head) = 3. So probability = 3/6 = 50%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "In a room of 23 people, what is the probability that at least two people share a birthday? (Assume 365 days in a year.)",
        "presented_solution": "There are 23 people and 365 possible birthdays. The probability is approximately 23/365 ≈ 6.3%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "A store offers 30% off, then an additional 20% off the reduced price. What is the total discount?",
        "presented_solution": "Total discount = 30% + 20% = 50%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "You drive 100 miles at 25 mph, then 100 miles at 75 mph. What is your average speed for the whole trip?",
        "presented_solution": "Average speed = (25 + 75) / 2 = 50 mph.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "A ball is thrown upward at 30 m/s. What is the maximum height reached? (Use g = 10 m/s².)",
        "presented_solution": "Maximum height = v × t = 30 × 3 = 90 m. (Time to reach top = v/g = 30/10 = 3 s.)",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "What is the probability of getting at least one six in 4 rolls of a fair die?",
        "presented_solution": "There are 4 rolls, each with a 1/6 chance of six. So the probability = 4 × (1/6) = 4/6 ≈ 66.7%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
    {
        "problem": "In a room of 30 people, what is the probability that at least two people share a birthday? (Assume 365 days in a year.)",
        "presented_solution": "There are 30 people and 365 possible birthdays. The probability is approximately 30/365 ≈ 8.2%.",
        "solution_has_error": "true",
        "error_type": "method",
    },
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
    # First, include curated static method errors (shuffle and pick up to available)
    static_method_pool = list(_STATIC_METHOD_ERRORS)
    rng.shuffle(static_method_pool)
    n_static_method = min(len(static_method_pool), n_method_err)
    for item in static_method_pool[:n_static_method]:
        if item["problem"] not in used_problems:
            used_problems.add(item["problem"])
            items.append(item)
    # Fill remaining method error slots with procedural generators
    n_method_remaining = n_method_err - n_static_method
    for i in range(n_method_remaining):
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


if __name__ == "__main__":
    df = generate_error_detection_dataset()
    print(f"Total items: {len(df)}")
    print(f"\nError distribution:")
    print(df["solution_has_error"].value_counts())
    print(f"\nError type distribution:")
    print(df["error_type"].value_counts())
    print(f"\nSample items:")
    for _, row in df.head(5).iterrows():
        print(f"\n  Problem: {row['problem'][:80]}...")
        print(f"  Solution: {row['presented_solution'][:80]}...")
        print(f"  Has error: {row['solution_has_error']}, Type: {row['error_type']}")
