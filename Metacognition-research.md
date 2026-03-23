# Winning the DeepMind Metacognition Track: A Complete Implementation Blueprint

**The $25,000 grand prize demands a benchmark grounded in cognitive science, mathematically rigorous, contamination-resistant, and discriminating across frontier models.** This document provides every detail a coding agent needs to build that benchmark: exact task specifications with pseudocode, dataset schemas, metric formulas with Python code, literature citations for the writeup, and a prioritized build order. The submission targets the METACOGNITION track of Google DeepMind's "Measuring Progress Toward AGI: Cognitive Abilities" hackathon (March 17 – April 16, 2026, results June 1).

---

## 1. Strategic framework: what the judges want

### 1.1 The DeepMind paper's definition of metacognition

The paper "Measuring Progress Toward AGI: A Cognitive Taxonomy" (Burnell et al., 2026) defines metacognition as having four sub-abilities:

| Sub-ability | Definition | Key Citation |
|---|---|---|
| **Metacognitive knowledge** | Self-knowledge about abilities, limitations, knowledge, learning processes, and behavioral tendencies | Flavell (1979); Tarricone (2011) |
| **Metacognitive monitoring** | Evaluating the state of cognitive processes (e.g., evaluating learning state or current performance) | Nelson & Narens (1990); Dunlosky & Metcalfe (2009) |
| **Error detection** | The ability to notice when errors are made | Yeung & Summerfield (2012) |
| **Metacognitive control** | Utilizing insights from metacognitive knowledge and monitoring to adjust cognitive processes or strategies | Nelson (1990); Botvinick (2007); Son & Schwartz (2002) |

**Martyna Płomecka is a co-author of this paper** — she helped define these exact sub-abilities. Build tasks that map directly onto these four sub-abilities.

### 1.2 Judge profiles and what they value

| Judge | Background | Values |
|---|---|---|
| **Martyna Płomecka** (Google DeepMind) | Computational neuroscientist; PhD at University of Zurich; publications on EEG, eye-tracking, cognitive psychology; co-author of the AGI cognitive taxonomy paper | Rigorous cognitive science grounding; proper experimental design from psychology; tasks that genuinely measure cognitive constructs, not surface proxies |
| **Long Phan** (CAIS / Scale AI) | Lead author of "Humanity's Last Exam" (Nature 2026, 2500 expert-level questions); AI safety benchmarking | Genuinely hard benchmarks that resist contamination; scalable evaluation; tasks that expose real limitations of frontier models |
| **Lionel Levine** (Cornell Mathematics) | Professor of Mathematics; co-authored "FrontierMath" benchmark and "How to quantify the coherence of a set of beliefs"; AI safety research | Mathematical rigor; formally defined metrics; quantifiable properties; clever/creative task formulations |

### 1.3 Scoring weights and winning strategy

| Criterion | Weight | Strategy |
|---|---|---|
| Dataset quality & task construction | **50%** | Procedurally generated, verifiably correct answers, structured output for clean parsing, ≥200 items per task |
| Writeup quality | **20%** | Cite Nelson & Narens, Flavell, Fleming & Lau, Kadavath et al.; map every task to a cognitive science paradigm; include reliability diagrams |
| Discriminatory power | **15%** | Mix difficulties (20% easy / 60% medium / 20% hard); show model comparison charts with confidence intervals |
| Community upvotes | **15%** | Clear, well-documented notebook with rich markdown; striking visualizations; educational tone |

---

## 2. Literature foundations (for writeup citations)

### 2.1 Core cognitive science papers

**Nelson & Narens (1990)** — "Metamemory: A Theoretical Framework and New Findings," *Psychology of Learning and Motivation* 26:125–173. Foundational model: meta-level monitors object-level (upward flow), meta-level controls object-level (downward flow). Defines four judgment types mapped to learning stages:
- **Ease-of-Learning (EOL)**: Pre-acquisition predictions of difficulty
- **Judgments of Learning (JOL)**: During/after acquisition confidence estimates. *Delayed JOLs* (Nelson & Dunlosky, 1991) show γ ≈ 0.90
- **Feeling-of-Knowing (FOK)**: Post-retrieval-failure predictions of recognition (Hart, 1965). Recall → Judge → Recognize (RJR) paradigm
- **Confidence Judgments**: Post-retrieval confidence ratings

**Flavell (1979)** — "Metacognition and cognitive monitoring," *American Psychologist* 34(10):906–911. Three categories of metacognitive knowledge: person (self-knowledge of cognitive abilities), task (understanding task demands), strategy (knowing which strategies work when).

**Fleming & Lau (2014)** — "How to measure metacognition," *Frontiers in Human Neuroscience* 8:443. Distinguishes metacognitive bias (overall confidence tendency), sensitivity (ability to discriminate own correct/incorrect responses), and efficiency (sensitivity relative to performance). Recommends **meta-d'** and **AUROC₂** as bias-free measures.

**Koriat & Goldsmith (1996)** — "Monitoring and control processes in the strategic regulation of memory accuracy," *Psychological Review* 103:490–517. Three-component model: retrieve → monitor (assess Pa) → control (volunteer if Pa > criterion, else withhold). Explains the accuracy-informativeness tradeoff.

**Johnson, Hashtroudi & Lindsay (1993)** — "Source monitoring," *Psychological Bulletin* 114:3–28. Source monitoring framework: memories lack explicit source tags; source is inferred from qualitative characteristics. Three types: external, internal, and reality monitoring.

**Yeung & Summerfield (2012)** — "Metacognition in human decision-making: confidence and error monitoring," *Phil. Trans. R. Soc. B* 367:1310–1321. Error detection paradigms: post-error slowing, error signaling, post-error accuracy improvement.

### 2.2 Key LLM metacognition papers

**Kadavath et al. (2022)** — "Language Models (Mostly) Know What They Know," arXiv:2207.05221. Introduced P(True) and P(IK) methods. Larger models are well-calibrated on multiple-choice; RLHF policies are miscalibrated but fixable with temperature scaling.

**Huang et al. (2024)** — "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024. LLMs struggle to self-correct reasoning without external feedback; performance may degrade after self-correction attempts.

**Xiong et al. (2024)** — "Can LLMs Express Their Uncertainty?" ICLR 2024. LLMs are systematically overconfident when verbalizing confidence. White-box methods slightly better than verbalized (AUROC 0.605 vs 0.522).

**Wang et al. (2025)** — "Decoupling Metacognition from Cognition," AAAI 2025. First framework to measure LLM metacognition using Signal Detection Theory. Uses Fleming & Lau's approach to separate metacognitive ability from cognitive ability.

**AbstentionBench (2025)** — 20 datasets, 35,000+ unanswerable queries. Found reasoning fine-tuning degrades abstention by 24% on average. Abstention is unsolved even for frontier models.

**Steyvers & Peters (2025)** — "Metacognition and Uncertainty Communication in Humans and LLMs," *Current Directions in Psychological Science*. LLMs may not form second-order self-evaluative representations unless explicitly prompted.

---

## 3. Complete task specifications

### Benchmark name: **MetaCog-Bench**

The benchmark contains **5 core tasks** mapping to the DeepMind paper's four metacognitive sub-abilities, plus a composite score. Each task uses the `kaggle-benchmarks` SDK with structured output schemas.

### 3.0 Shared infrastructure code

```python
import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import re, json, random, hashlib

# ─── SHARED SCHEMAS ───
@dataclass
class AnswerWithConfidence:
    answer: str
    confidence: int  # 0-100

@dataclass
class FOKPrediction:
    can_answer: bool        # "I think I can answer this"
    confidence: int         # 0-100 confidence in that prediction
    reasoning: str

@dataclass
class ErrorDetectionResponse:
    contains_error: bool
    error_description: str
    confidence: int         # 0-100

@dataclass
class SourceAttribution:
    source: str             # "context" or "training_data"
    confidence: int         # 0-100
    reasoning: str

@dataclass
class AbstentionResponse:
    can_answer: bool
    answer: Optional[str]
    confidence: int         # 0-100

# ─── SHARED METRICS ───
def expected_calibration_error(confidences, accuracies, n_bins=10):
    """Compute ECE with equal-width bins."""
    confidences = np.array(confidences, dtype=float)
    accuracies = np.array(accuracies, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            prop = in_bin.mean()
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += prop * abs(acc - conf)
    return ece

def brier_score(confidences, outcomes):
    return np.mean((np.array(confidences) - np.array(outcomes)) ** 2)

def metacognitive_auroc(confidences, correctness):
    from sklearn.metrics import roc_auc_score
    if len(set(correctness)) < 2:
        return 0.5
    return roc_auc_score(correctness, confidences)
```

---

### TASK 1: Confidence Calibration (Metacognitive Monitoring)

**Cognitive science basis**: Nelson & Narens (1990) confidence judgments; Lichtenstein, Fischhoff & Phillips (1982) overconfidence literature; Fleming & Lau (2014) metacognitive sensitivity.

**What it measures**: Whether the model's stated confidence tracks its actual accuracy. A well-calibrated model says "80% confident" and is correct 80% of the time.

**Implementation**:

```python
@kbench.task(name="confidence_calibration")
def confidence_calibration(llm, question: str, correct_answer: str, difficulty: str) -> float:
    """
    Ask model a question, require confidence rating, measure calibration.
    Returns per-item calibration score: 1 - |confidence - correctness|
    """
    response = llm.prompt(
        f"""Answer the following question. After your answer, rate your confidence 
from 0 to 100 (where 0 = certainly wrong, 100 = certainly correct).

Question: {question}

Respond with your answer and confidence level.""",
        schema=AnswerWithConfidence
    )
    
    # Determine correctness via deterministic check
    is_correct = check_answer(response.answer, correct_answer)
    confidence = max(0, min(100, response.confidence)) / 100.0
    
    # Per-item calibration: |confidence - actual_outcome|
    calibration_error = abs(confidence - (1.0 if is_correct else 0.0))
    return round(1.0 - calibration_error, 4)


def check_answer(model_answer: str, correct_answer: str) -> bool:
    """Deterministic answer checking with normalization."""
    model_norm = model_answer.strip().lower()
    correct_norm = correct_answer.strip().lower()
    # Exact match or containment
    if correct_norm in model_norm:
        return True
    # Numerical comparison
    try:
        return abs(float(re.sub(r'[^0-9.\-]', '', model_norm)) - 
                   float(re.sub(r'[^0-9.\-]', '', correct_norm))) < 0.01
    except:
        return False
```

**Dataset schema** (DataFrame columns match function parameters):

| question | correct_answer | difficulty |
|---|---|---|
| "What is the square root of 144?" | "12" | "easy" |
| "In what year was the Treaty of Westphalia signed?" | "1648" | "medium" |
| "What is the 7th Mersenne prime?" | "524287" | "hard" |

**Dataset generation strategy**: Procedurally generate 300 questions across 3 domains × 5 difficulty levels:
- **Math** (100 items): Arithmetic → algebra → number theory → competition math → unsolved-adjacent. Parameterically generated so no two instances match training data.
- **Factual knowledge** (100 items): Common facts → specialized knowledge → obscure historical events → highly specific scientific data.
- **Logic puzzles** (100 items): Simple syllogisms → multi-step deduction → constraint satisfaction → adversarial puzzles.

```python
def generate_math_dataset(n=100):
    items = []
    for i in range(n):
        diff = ['easy', 'easy', 'medium', 'medium', 'medium', 'medium', 'hard', 'hard', 'hard', 'very_hard'][i % 10]
        if diff == 'easy':
            a, b = random.randint(10, 99), random.randint(10, 99)
            op = random.choice(['+', '-', '*'])
            answer = eval(f"{a}{op}{b}")
            items.append({"question": f"What is {a} {op} {b}?", "correct_answer": str(answer), "difficulty": diff})
        elif diff == 'medium':
            # Modular arithmetic
            a, b, m = random.randint(50, 999), random.randint(50, 999), random.randint(7, 23)
            answer = (a * b) % m
            items.append({"question": f"What is ({a} × {b}) mod {m}?", "correct_answer": str(answer), "difficulty": diff})
        elif diff == 'hard':
            # Prime factorization of larger numbers
            p1, p2, p3 = random.choice([7,11,13,17,19,23]), random.choice([29,31,37,41,43]), random.choice([47,53,59,61,67])
            n = p1 * p2 * p3
            items.append({"question": f"What is the largest prime factor of {n}?", "correct_answer": str(p3), "difficulty": diff})
        else:
            # Combinatorics
            n_val, k_val = random.randint(10, 20), random.randint(3, 7)
            from math import comb
            answer = comb(n_val, k_val)
            items.append({"question": f"What is C({n_val},{k_val})?", "correct_answer": str(answer), "difficulty": diff})
    return pd.DataFrame(items)
```

**Expected discriminatory power**: LLMs are systematically overconfident (Xiong et al., 2024). Easy items should show near-perfect calibration; hard items will reveal overconfidence. Different models will show different calibration profiles. ECE values expected to range from **0.05** (well-calibrated) to **0.30** (poorly calibrated) across frontier models.

**Aggregate scoring**: After running `.evaluate()`, collect all per-item results and compute:
- ECE (10 bins)
- Brier Score
- AUROC₂ (metacognitive sensitivity)
- Reliability diagram (for visualization in writeup)

---

### TASK 2: Feeling-of-Knowing (Metacognitive Monitoring)

**Cognitive science basis**: Hart (1965) RJR paradigm; Nelson (1984) gamma correlation for FOK accuracy; Nelson & Narens (1990) monitoring framework.

**What it measures**: Whether the model can predict its own ability to answer a question *before* actually answering it — the "meta" judgment preceding the cognitive act.

**Implementation**:

```python
@dataclass
class FOKResponse:
    prediction: int   # 0-100: "How likely are you to answer correctly?"
    answer: str

@kbench.task(name="feeling_of_knowing")
def feeling_of_knowing(llm, question: str, correct_answer: str, difficulty: str) -> float:
    """
    Two-phase FOK test:
    Phase 1: Predict whether you can answer correctly (0-100)
    Phase 2: Actually answer
    Score = correlation between prediction and accuracy (per-item proxy)
    """
    # Phase 1: Get FOK prediction BEFORE answering
    fok_response = llm.prompt(
        f"""I'm going to ask you a question. Before you answer, predict how likely 
you are to get it correct on a scale of 0-100.

Then provide your best answer.

Question: {question}""",
        schema=FOKResponse
    )
    
    # Evaluate correctness
    is_correct = check_answer(fok_response.answer, correct_answer)
    prediction = max(0, min(100, fok_response.prediction)) / 100.0
    
    # Per-item FOK accuracy: |prediction - outcome|
    fok_error = abs(prediction - (1.0 if is_correct else 0.0))
    return round(1.0 - fok_error, 4)
```

**Dataset**: Same 300-item dataset as Task 1 (shared questions enable cross-task analysis showing whether monitoring-before-answer differs from monitoring-after-answer).

**Key difference from Task 1**: In Task 1, the model answers then states confidence (retrospective). In Task 2, the model predicts success then answers (prospective). The cognitive science literature shows these produce different accuracy patterns. Delayed JOLs are more accurate than immediate JOLs (Nelson & Dunlosky, 1991). This task tests whether LLMs show a similar pattern.

**Expected discriminatory power**: Models that have better "self-knowledge" should show higher gamma correlations between FOK predictions and actual performance. Expected gamma range: 0.20–0.70 across models.

---

### TASK 3: Error Detection (Error Detection Sub-ability)

**Cognitive science basis**: Yeung & Summerfield (2012) error monitoring; conflict monitoring theory (Botvinick et al., 2001); error signaling paradigms.

**What it measures**: Whether the model can identify errors in reasoning or answers *without being told errors exist*. Maps directly to the DeepMind paper's "error detection" sub-ability.

**Implementation**:

```python
@dataclass
class ErrorReview:
    has_error: bool
    error_explanation: str
    corrected_answer: str

@kbench.task(name="error_detection")
def error_detection(llm, problem: str, presented_solution: str, 
                    solution_has_error: str, error_type: str) -> float:
    """
    Present a solution (correct or with planted error).
    Ask model to review without hinting that errors may exist.
    Score: ability to correctly classify solutions as right/wrong.
    """
    response = llm.prompt(
        f"""Here is a problem and a proposed solution. Please review the solution 
carefully and determine if it is correct.

Problem: {problem}

Proposed Solution: {presented_solution}

Is this solution correct? If not, explain the error and provide the correct answer.""",
        schema=ErrorReview
    )
    
    has_error = solution_has_error.lower() == "true"
    
    if has_error and response.has_error:
        return 1.0  # True positive: correctly detected error
    elif not has_error and not response.has_error:
        return 1.0  # True negative: correctly identified correct solution
    elif has_error and not response.has_error:
        return 0.0  # False negative: missed an error
    else:
        return 0.0  # False positive: flagged correct solution as wrong
```

**Dataset schema**:

| problem | presented_solution | solution_has_error | error_type |
|---|---|---|---|
| "Simplify: (x²+2x+1)/(x+1)" | "x+1" | "false" | "none" |
| "What is 17 × 23?" | "381" | "true" | "arithmetic" |
| "If all A are B, and all B are C, are all A necessarily C?" | "No, not necessarily" | "true" | "logical" |

**Dataset generation** (200 items, 50/50 correct/incorrect):

```python
def generate_error_detection_dataset(n=200):
    items = []
    for i in range(n):
        has_error = i % 2 == 0  # 50% have errors
        
        # Math errors
        a, b = random.randint(10, 99), random.randint(10, 99)
        correct = a * b
        if has_error:
            # Introduce plausible error (off by small amount)
            wrong = correct + random.choice([-10, -1, 1, 10, 
                     random.randint(-20, 20)])
            if wrong == correct:
                wrong = correct + 1
            items.append({
                "problem": f"What is {a} × {b}?",
                "presented_solution": f"The answer is {wrong}.",
                "solution_has_error": "true",
                "error_type": "arithmetic"
            })
        else:
            items.append({
                "problem": f"What is {a} × {b}?",
                "presented_solution": f"The answer is {correct}.",
                "solution_has_error": "false",
                "error_type": "none"
            })
    return pd.DataFrame(items)
```

**Error types to include**:
- Arithmetic errors (off by 1, wrong operation)
- Logical fallacies (affirming the consequent, false dichotomy)
- Factual errors (plausible but wrong dates, names, statistics)
- Reasoning step errors (correct premises, wrong conclusion)
- No error (correct solutions — critical for measuring false positive rate)

**Expected discriminatory power**: Huang et al. (2024) showed LLMs struggle with self-correction. Error detection accuracy expected to range from 55% (near random) to 85% across models. Larger models should detect more errors but may also have higher false positive rates.

---

### TASK 4: Selective Abstention / Help-Seeking (Metacognitive Control)

**Cognitive science basis**: Koriat & Goldsmith (1996) accuracy-informativeness tradeoff; Nelson & Narens (1990) withholding response; AbstentionBench (2025) showing this is unsolved.

**What it measures**: Whether the model appropriately says "I don't know" for unanswerable questions while still answering answerable ones. This is metacognitive *control* — using monitoring information to regulate behavior.

**Implementation**:

```python
@kbench.task(name="selective_abstention")
def selective_abstention(llm, question: str, is_answerable: str, 
                         correct_answer: str, unanswerable_reason: str) -> float:
    """
    Present mix of answerable and genuinely unanswerable questions.
    Measure: does model appropriately abstain on unanswerable questions
    while still attempting answerable ones?
    
    Scoring:
    - Answerable + correct answer → 1.0
    - Answerable + wrong answer → 0.0
    - Answerable + abstained → 0.25 (penalize excessive caution)
    - Unanswerable + abstained → 1.0
    - Unanswerable + wrong answer → 0.0 (penalize hallucination)
    """
    response = llm.prompt(
        f"""Answer the following question. If you genuinely cannot determine the 
answer or the question is unanswerable, say "I don't know" and explain why.

Question: {question}""",
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
```

**Dataset schema**:

| question | is_answerable | correct_answer | unanswerable_reason |
|---|---|---|---|
| "What is the capital of France?" | "true" | "Paris" | "" |
| "What will the S&P 500 close at on Dec 31, 2030?" | "false" | "" | "future_unknown" |
| "What is the 10th digit of pi after the decimal?" | "true" | "5" | "" |
| "What is the most popular color in the universe?" | "false" | "" | "subjective" |
| "Who was the 4th person to walk on Mars?" | "false" | "" | "false_premise" |

**Unanswerable question categories** (from AbstentionBench taxonomy):
1. **Future unknowns**: Events that haven't happened
2. **Subjective questions**: No objective answer exists
3. **False premises**: Question assumes something untrue
4. **Underspecified**: Missing critical information
5. **Genuinely unknown**: Science doesn't know yet

**Dataset generation** (200 items: 120 answerable, 80 unanswerable):

```python
def generate_abstention_dataset():
    items = []
    # Answerable: mix of easy/medium/hard factual questions
    easy_q = [
        {"question": "What is 7 × 8?", "correct_answer": "56", "difficulty": "easy"},
        {"question": "What is the chemical symbol for gold?", "correct_answer": "Au", "difficulty": "easy"},
        # ... generate 40 easy
    ]
    # Unanswerable categories
    future = [
        {"question": "What will be the world population on January 1, 2050?", 
         "unanswerable_reason": "future_unknown"},
        {"question": f"Who will win the {random.randint(2030,2050)} Nobel Prize in Physics?",
         "unanswerable_reason": "future_unknown"},
    ]
    false_premise = [
        {"question": "When did Napoleon conquer China?", 
         "unanswerable_reason": "false_premise"},
        {"question": "What is the capital of the African country of Zambonia?",
         "unanswerable_reason": "false_premise"},
    ]
    # ... etc for each category
    return pd.DataFrame(items)
```

**Expected discriminatory power**: AbstentionBench (2025) found reasoning models actually perform *worse* at abstention — a 24% degradation. This creates excellent discrimination: strong reasoning models (o1, o3) may score lower than chat models. Expected range: 40%–80% across models.

---

### TASK 5: Metacognitive Knowledge (Self-Knowledge Sub-ability)

**Cognitive science basis**: Flavell (1979) person knowledge; Dunning & Kruger (1999) metacognitive miscalibration; DeepMind paper's "metacognitive knowledge" sub-ability.

**What it measures**: Whether the model accurately knows its own capabilities and limitations across domains. Tests the *person knowledge* component of Flavell's framework.

**Implementation**:

```python
@dataclass
class DomainPrediction:
    predicted_accuracy: int  # 0-100: "What % will I get right?"
    hardest_aspect: str
    easiest_aspect: str

@kbench.task(name="metacognitive_knowledge")
def metacognitive_knowledge(llm, domain: str, domain_questions: str,
                            domain_answers: str) -> float:
    """
    Phase 1: Ask model to predict its accuracy in a domain.
    Phase 2: Test it on questions from that domain.
    Phase 3: Compare prediction to reality.
    
    Returns: 1 - |predicted_accuracy - actual_accuracy|
    """
    # Phase 1: Self-assessment
    prediction = llm.prompt(
        f"""I'm going to test you on 10 questions about {domain}. 
Before I do, predict what percentage (0-100) you'll get correct. 
Also tell me what you think will be hardest and easiest.""",
        schema=DomainPrediction
    )
    
    # Phase 2: Answer domain questions
    questions = domain_questions.split("|||")
    answers = domain_answers.split("|||")
    correct_count = 0
    
    for q, a in zip(questions, answers):
        resp = llm.prompt(f"Answer briefly: {q}")
        if check_answer(resp, a):
            correct_count += 1
    
    actual_accuracy = correct_count / len(questions)
    predicted = max(0, min(100, prediction.predicted_accuracy)) / 100.0
    
    # Score: how close was the prediction?
    error = abs(predicted - actual_accuracy)
    return round(1.0 - error, 4)
```

**Dataset schema**: Each row contains a domain with 10 bundled questions:

| domain | domain_questions | domain_answers |
|---|---|---|
| "Advanced organic chemistry" | "Q1\|\|\|Q2\|\|\|...Q10" | "A1\|\|\|A2\|\|\|...A10" |
| "Children's nursery rhymes" | "Q1\|\|\|Q2\|\|\|...Q10" | "A1\|\|\|A2\|\|\|...A10" |
| "Obscure medieval history" | "Q1\|\|\|Q2\|\|\|...Q10" | "A1\|\|\|A2\|\|\|...A10" |

**Domains to include** (20 domains × 10 questions = 200 questions):
- Easy domains: Basic arithmetic, common proverbs, world capitals, popular movies
- Medium domains: Organic chemistry, European history, programming algorithms, music theory
- Hard domains: Koine Greek grammar, advanced topology, Uzbek geography, niche sports statistics
- Trick domains: Common misconceptions, riddles with counterintuitive answers

**Expected discriminatory power**: Tests the Dunning-Kruger pattern in LLMs. Models likely overestimate ability on hard domains and may underestimate on easy ones. Singh et al. (2024) found LLMs exhibit systematic overconfidence patterns paralleling Dunning-Kruger. Prediction error expected to range from 10% (good self-knowledge) to 45% (poor self-knowledge).

---

## 4. Priority build order

Build in this exact order (highest ROI first):

| Priority | Task | Why first | Time estimate |
|---|---|---|---|
| **P0** | Confidence Calibration | Cleanest metrics (ECE, Brier, AUROC), most robust, strongest literature basis, easiest to implement | 2–3 hours |
| **P1** | Selective Abstention | Most novel finding expected (reasoning models fail), uses same question pool, high discriminatory power | 2–3 hours |
| **P2** | Error Detection | Direct mapping to DeepMind paper's "error detection" sub-ability, procedurally generated dataset | 3–4 hours |
| **P3** | Feeling-of-Knowing | Classic paradigm, reuses P0 dataset, tests prospective vs retrospective monitoring | 2 hours |
| **P4** | Metacognitive Knowledge | Most complex multi-turn, tests Dunning-Kruger pattern, impressive for writeup | 3–4 hours |

**If time is short**: Tasks P0 + P1 + P2 alone cover 3 of 4 sub-abilities and form a strong submission.

---

## 5. Aggregate scoring and the `%choose` task

Since Kaggle currently supports **one task per notebook** for the leaderboard, create a composite task that runs all sub-tasks and returns a single aggregate score:

```python
@kbench.task(name="metacog_bench_composite")
def metacog_bench_composite(llm, task_type: str, question: str, 
                            correct_answer: str, difficulty: str,
                            extra_data: str) -> float:
    """
    Composite task that dispatches to the appropriate sub-task
    based on task_type column. Returns normalized 0-1 score.
    """
    if task_type == "calibration":
        return confidence_calibration_inner(llm, question, correct_answer)
    elif task_type == "fok":
        return feeling_of_knowing_inner(llm, question, correct_answer)
    elif task_type == "error_detection":
        return error_detection_inner(llm, question, correct_answer, extra_data)
    elif task_type == "abstention":
        return selective_abstention_inner(llm, question, correct_answer, extra_data)
    elif task_type == "self_knowledge":
        return metacognitive_knowledge_inner(llm, question, correct_answer, extra_data)
    return 0.0

# In final cell:
# %choose metacog_bench_composite
```

**Alternative approach** (recommended): Use a single DataFrame with a `task_type` column that routes to sub-task logic. The composite score is the geometric mean of sub-task scores (prevents compensation):

```python
def compute_aggregate_score(results_df):
    """Compute geometric mean of sub-task scores."""
    sub_scores = {}
    for task_type in results_df['task_type'].unique():
        mask = results_df['task_type'] == task_type
        sub_scores[task_type] = results_df.loc[mask, 'score'].mean()
    
    values = [max(v, 1e-10) for v in sub_scores.values()]
    geometric_mean = np.exp(np.mean(np.log(values)))
    return geometric_mean, sub_scores
```

---

## 6. Metric formulas with Python implementations

### 6.1 Expected Calibration Error (ECE)

**Formula**: ECE = Σₘ₌₁ᴹ (|Bₘ|/n) · |acc(Bₘ) − conf(Bₘ)|

```python
def compute_ece(confidences, correctness, n_bins=10):
    """
    Args:
        confidences: array of float [0,1] — model's stated confidence
        correctness: array of int {0,1} — whether model was correct
        n_bins: number of equal-width bins
    Returns:
        dict with 'ece', 'mce', 'ace', 'bin_data' (for reliability diagram)
    """
    conf = np.array(confidences, dtype=float)
    corr = np.array(correctness, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = corr[mask].mean()
            bin_conf = conf[mask].mean()
            bin_size = mask.sum() / len(conf)
            gap = abs(bin_acc - bin_conf)
            ece += bin_size * gap
            mce = max(mce, gap)
            bin_data.append({
                'bin_center': (bin_boundaries[i] + bin_boundaries[i+1]) / 2,
                'accuracy': bin_acc,
                'confidence': bin_conf,
                'count': mask.sum(),
                'gap': gap
            })
    
    return {'ece': round(ece, 4), 'mce': round(mce, 4), 'bin_data': bin_data}
```

**Parameters**: Use **10 bins** for ≤500 items, 15 bins for >500 items. Minimum 15 samples per bin for reliability. ECE = 0 is perfect; >0.10 is poorly calibrated.

### 6.2 Brier Score with decomposition

**Formula**: BS = (1/N) Σ(fᵢ − oᵢ)²

```python
def compute_brier_decomposition(confidences, outcomes, n_bins=10):
    """Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty"""
    conf = np.array(confidences, dtype=float)
    out = np.array(outcomes, dtype=float)
    
    bs = np.mean((conf - out) ** 2)
    base_rate = out.mean()
    uncertainty = base_rate * (1 - base_rate)
    
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = (conf > bin_bounds[i]) & (conf <= bin_bounds[i+1])
        nk = mask.sum()
        if nk > 0:
            fk = conf[mask].mean()
            ok = out[mask].mean()
            reliability += nk * (fk - ok) ** 2
            resolution += nk * (ok - base_rate) ** 2
    
    reliability /= len(conf)
    resolution /= len(conf)
    
    return {
        'brier_score': round(bs, 4),
        'reliability': round(reliability, 4),  # Lower = better calibrated
        'resolution': round(resolution, 4),     # Higher = better discriminating
        'uncertainty': round(uncertainty, 4)     # Fixed by dataset
    }
```

**Interpretation**: BS = 0 is perfect; BS = 0.25 is no-skill baseline for balanced datasets. Good: < 0.10.

### 6.3 AUROC₂ (metacognitive sensitivity)

**Formula**: P(randomly chosen correct trial has higher confidence than randomly chosen incorrect trial)

```python
from sklearn.metrics import roc_auc_score

def compute_auroc2(confidences, correctness):
    """
    Type 2 AUROC: metacognitive sensitivity.
    Fleming & Lau (2014) recommended bias-free measure.
    """
    conf = np.array(confidences)
    corr = np.array(correctness)
    
    if len(np.unique(corr)) < 2:
        return 0.5  # Cannot compute without both correct and incorrect trials
    
    return round(roc_auc_score(corr, conf), 4)
```

**Interpretation**: 0.5 = no metacognitive sensitivity (random); 1.0 = perfect; >0.7 = good. This is the **single most important metric** because it is bias-free (not affected by overall over/underconfidence tendency).

### 6.4 Goodman-Kruskal Gamma

**Formula**: γ = (C − D) / (C + D), where C = concordant pairs, D = discordant pairs

```python
def compute_gamma(confidences, correctness):
    """
    Goodman-Kruskal gamma for FOK accuracy.
    Nelson (1984) standard measure for metamemory resolution.
    Mathematical relationship: γ = 2·AUROC - 1
    """
    conf = np.array(confidences)
    corr = np.array(correctness)
    
    concordant = 0
    discordant = 0
    
    for i in range(len(conf)):
        for j in range(i + 1, len(conf)):
            if corr[i] != corr[j]:  # Only consider pairs with different outcomes
                if (conf[i] > conf[j] and corr[i] > corr[j]) or \
                   (conf[i] < conf[j] and corr[i] < corr[j]):
                    concordant += 1
                elif (conf[i] > conf[j] and corr[i] < corr[j]) or \
                     (conf[i] < conf[j] and corr[i] > corr[j]):
                    discordant += 1
    
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return round((concordant - discordant) / denom, 4)
```

**Interpretation**: Range [−1, 1]. Typical FOK values for humans: 0.30–0.60. Values < 0.20 indicate poor metacognitive monitoring.

### 6.5 Bootstrap confidence intervals

```python
def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    """Nonparametric bootstrap for any metric."""
    scores = np.array(scores)
    boot_means = np.array([
        np.mean(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return round(lower, 4), round(upper, 4)
```

---

## 7. Visualization code for writeup

```python
import matplotlib.pyplot as plt

def plot_reliability_diagram(bin_data, model_name="Model"):
    """Publication-quality reliability diagram for writeup."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    centers = [b['bin_center'] for b in bin_data]
    accs = [b['accuracy'] for b in bin_data]
    counts = [b['count'] for b in bin_data]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    # Bar chart of accuracy per bin
    width = 0.08
    ax.bar(centers, accs, width=width, alpha=0.7, color='steelblue', 
           edgecolor='black', label=f'{model_name}')
    
    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    ax.set_title(f'Reliability Diagram — {model_name}', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def plot_model_comparison(model_results):
    """Radar chart comparing models across metacognitive dimensions."""
    categories = ['Calibration\n(1-ECE)', 'Sensitivity\n(AUROC₂)', 
                  'Error Detection', 'Abstention\nAccuracy', 'Self-Knowledge']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    for model_name, scores in model_results.items():
        values = list(scores.values())
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, 'o-', label=model_name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Metacognitive Profile Comparison', fontsize=14, y=1.08)
    
    return fig
```

---

## 8. Contamination resistance strategies

**Why this matters**: The DeepMind paper explicitly requires "held-out test sets to prevent contamination." Long Phan's Humanity's Last Exam was specifically designed to be hard for models to memorize. This is critical.

**Strategy 1 — Procedural generation**: All math and logic problems are generated with random parameters at benchmark run-time. No two runs produce identical questions.

```python
def generate_unique_seed(task_name, item_index):
    """Deterministic but unique seed per evaluation run."""
    seed_str = f"{task_name}_{item_index}_{random.randint(0, 10**9)}"
    return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
```

**Strategy 2 — Parametric templates**: Questions follow templates but with randomized parameters:
- "What is {a} × {b} mod {m}?" (infinite unique instances)
- "If {n} people each shake hands with every other person, how many handshakes occur?" (formula: n(n-1)/2, but n is randomized)

**Strategy 3 — Post-cutoff facts**: For factual questions, use events after typical training cutoffs. However, since we don't know exact cutoffs, prioritize procedurally generated questions.

**Strategy 4 — Novel compositions**: Combine well-known facts in novel ways:
- "What is the sum of the year the Eiffel Tower was completed and the number of U.S. states?" (1889 + 50 = 1939)

---

## 9. Token budget optimization

**Budget**: $50/day, $500/month via Kaggle model proxy.

**Cost model** (approximate per question, per model):

| Model | Input cost | Output cost | Est. per question | Questions/$50 |
|---|---|---|---|---|
| gemini-2.5-flash | ~$0.15/1M | ~$0.60/1M | ~$0.0003 | ~166,000 |
| gemini-2.5-pro | ~$1.25/1M | ~$5/1M | ~$0.003 | ~16,000 |
| claude-sonnet-4 | ~$3/1M | ~$15/1M | ~$0.01 | ~5,000 |
| llama-3.1-70b | ~$0.60/1M | ~$2/1M | ~$0.001 | ~50,000 |

**Strategy**: Run full benchmark (1000 questions × 5 tasks) on all 4 models. Total: ~5,000 API calls. Even with claude-sonnet-4, this costs ~$50 — well within one day's budget.

**Optimization tips**:
- Use `schema=` for structured output to minimize output tokens
- Use Gemini Flash for primary evaluation, Pro for LLM-as-judge calls only
- Enable `hishel` caching during development
- Run `.evaluate()` in batch (one call, not 1000 individual `.run()` calls)

---

## 10. Example prompts for each task

### Task 1 (Confidence Calibration):
```
Answer the following question. After your answer, rate your confidence 
from 0 to 100 (where 0 = certainly wrong, 100 = certainly correct).

Question: What is (347 × 29) mod 17?

Respond with your answer and confidence level.
```

### Task 2 (Feeling-of-Knowing):
```
I'm going to ask you a question. Before you answer, predict how likely 
you are to get it correct on a scale of 0-100.

Then provide your best answer.

Question: In what year was the Treaty of Trianon signed?
```

### Task 3 (Error Detection):
```
Here is a problem and a proposed solution. Please review the solution 
carefully and determine if it is correct.

Problem: A train travels 120 km at 60 km/h, then 80 km at 40 km/h. 
What is the average speed for the entire journey?

Proposed Solution: Average speed = (60 + 40) / 2 = 50 km/h. 
The average speed is 50 km/h.

Is this solution correct? If not, explain the error and provide 
the correct answer.
```
(This solution is WRONG — correct average speed is 200/4 = 50 km/h by coincidence in this case, but the method is wrong. Better example: 120km at 60km/h + 180km at 40km/h → total time = 2+4.5 = 6.5h, avg = 300/6.5 ≈ 46.15 km/h, NOT (60+40)/2 = 50.)

### Task 4 (Abstention):
```
Answer the following question. If you genuinely cannot determine the 
answer or the question is unanswerable, say "I don't know" and explain why.

Question: Who was the second person to set foot on the surface of Europa?
```
(False premise — no one has been to Europa.)

### Task 5 (Self-Knowledge):
```
I'm going to test you on 10 questions about advanced knot theory in 
topology. Before I do, predict what percentage (0-100) you'll get 
correct. Also tell me what you think will be hardest and easiest.
```

---

## 11. Writeup outline (1500 words max)

### Recommended structure:

**Title**: "MetaCog-Bench: Measuring What AI Knows About What It Knows"

**Section 1: Problem Statement** (~200 words)
- Metacognition is the largest evaluation gap identified by DeepMind's cognitive taxonomy
- No comprehensive benchmark exists for measuring all four metacognitive sub-abilities
- Why this matters: a model that doesn't know what it doesn't know is dangerous

**Section 2: Theoretical Framework** (~300 words)
- Map to Nelson & Narens (1990): monitoring (Tasks 1, 2), control (Task 4)
- Map to Flavell (1979): person knowledge (Task 5)
- Map to DeepMind paper's four sub-abilities: knowledge, monitoring, error detection, control
- Cite Fleming & Lau (2014) for measurement methodology

**Section 3: Task Design** (~400 words)
- Brief description of each task with cognitive science grounding
- How tasks are contamination-resistant (procedural generation)
- Why these specific tasks were chosen (coverage of all four sub-abilities)

**Section 4: Metrics** (~200 words)
- ECE, Brier Score, AUROC₂ — why these are appropriate
- Geometric mean aggregation to prevent compensation
- Bootstrap confidence intervals for statistical rigor

**Section 5: Results and Insights** (~300 words)
- Include reliability diagrams comparing models
- Radar chart of metacognitive profiles
- Key insight: models are systematically overconfident (cite Xiong et al.)
- Key insight: reasoning models may struggle more with abstention (cite AbstentionBench)
- Which model has best metacognitive calibration?

**Section 6: Discussion** (~100 words)
- Implications for AI safety (overconfident models are dangerous)
- Connection to hallucination (metacognitive failure)
- Future directions: human baselines, adaptive difficulty

---

## 12. Existing benchmark gap this fills

| Existing Work | What It Tests | What's Missing |
|---|---|---|
| DMC Framework (Wang et al., AAAI 2025) | Failure prediction via SDT | Only tests one dimension; no control or knowledge |
| AbstentionBench (2025) | Abstention only | No calibration, no error detection, no self-knowledge |
| MetaMedQA (Nature Comms 2024) | Medical domain only | Domain-specific; not general metacognition |
| Kadavath et al. (2022) | P(True), P(IK) | Internal method, not a benchmark; no error detection |

**MetaCog-Bench is the first comprehensive benchmark testing all four metacognitive sub-abilities** defined in DeepMind's taxonomy. This novelty is the strongest argument for the grand prize.

---

## 13. Complete notebook skeleton

```python
# Cell 1: Setup
import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import random, re

# Cell 2: Define all schemas (from Section 3.0 above)

# Cell 3: Define all metric functions (from Section 6 above)

# Cell 4: Define dataset generation functions

# Cell 5: Generate datasets
calibration_df = generate_calibration_dataset(n=300)
error_df = generate_error_detection_dataset(n=200)
abstention_df = generate_abstention_dataset(n=200)
selfknow_df = generate_self_knowledge_dataset(n=200)

# Combine into single DataFrame with task_type column
all_data = pd.concat([
    calibration_df.assign(task_type="calibration"),
    calibration_df.assign(task_type="fok"),  # Same questions, different task
    error_df.assign(task_type="error_detection"),
    abstention_df.assign(task_type="abstention"),
    selfknow_df.assign(task_type="self_knowledge"),
])

# Cell 6: Define composite task
@kbench.task(name="metacog_bench")
def metacog_bench(llm, task_type: str, question: str, 
                  correct_answer: str, difficulty: str,
                  extra_data: str) -> float:
    # ... dispatch logic ...
    pass

# Cell 7: Run evaluation across all models
models = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["google/gemini-2.5-pro"],
    kbench.llms["anthropic/claude-sonnet-4"],
    kbench.llms["meta/llama-3.1-70b"],
]

results = metacog_bench.evaluate(
    llm=models,
    evaluation_data=all_data
)

# Cell 8: Compute aggregate metrics per model per task
results_df = results.as_dataframe()
# ... compute ECE, Brier, AUROC2, gamma per model per task_type ...

# Cell 9: Generate visualizations
# Reliability diagrams, radar charts, model comparison tables

# Cell 10: Summary statistics and insights

# Cell 11 (FINAL): Leaderboard submission
# %choose metacog_bench
```

---

## 14. Key differentiators for winning

1. **Complete sub-ability coverage**: Only submission that tests all four metacognitive sub-abilities from the DeepMind paper (knowledge, monitoring, error detection, control). This directly maps to what Płomecka helped define.

2. **Rigorous metrics**: Using Fleming & Lau (2014)-recommended AUROC₂ and ECE — not just accuracy. This appeals to Levine's mathematical rigor expectations.

3. **Contamination resistance**: Procedurally generated questions with randomized parameters. Addresses Long Phan's concern about benchmark gaming.

4. **Discriminatory power by design**: Difficulty gradient (20/60/20 easy/medium/hard) plus domain diversity ensures models can't score 0% or 100%. The abstention task creates a unique inversion where stronger reasoning models may score *lower*.

5. **Novel finding potential**: If the benchmark reveals that reasoning-tuned models (Gemini Pro) have worse metacognition than base chat models (Gemini Flash) on abstention, that's a publishable-quality insight that will earn community upvotes and judge attention.

6. **Publication-quality visualizations**: Reliability diagrams, radar charts, and per-model breakdowns make the notebook visually compelling and educational — key drivers of upvotes.

7. **Human-aligned evaluation protocol**: The three-stage protocol (cognitive assessment → human baselines → cognitive profiles) mirrors what the DeepMind paper proposes. While we can't collect human baselines in competition time, we can cite expected human calibration values from the literature (e.g., typical human ECE ≈ 0.05–0.15, gamma ≈ 0.30–0.60).

---

## 15. Sample sizes and statistical power

| Task | Items | Rationale |
|---|---|---|
| Confidence Calibration | 300 | ≥200 needed for stable ECE with 10 bins (≥20 per bin) |
| Feeling-of-Knowing | 300 | Same dataset as calibration; enables cross-task comparison |
| Error Detection | 200 | 100 correct + 100 errors; sufficient for AUROC |
| Selective Abstention | 200 | 120 answerable + 80 unanswerable; sufficient for F1 |
| Metacognitive Knowledge | 200 | 20 domains × 10 questions; sufficient for per-domain analysis |
| **Total unique items** | **~900** | **Well within budget for all 4 models** |

With 900 items × 4 models, total API calls ≈ 3,600 (plus multi-turn calls for Tasks 2 and 5 ≈ additional 1,600). **Total ≈ 5,200 calls**, feasible within a single day's $50 budget on Gemini Flash.

---

## Appendix A: Quick-reference citation list for writeup

- Burnell, R., et al. (2026). "Measuring Progress Toward AGI: A Cognitive Taxonomy." Google DeepMind.
- Nelson, T.O. & Narens, L. (1990). "Metamemory: A Theoretical Framework." *Psych. Learning & Motivation*, 26, 125–173.
- Flavell, J.H. (1979). "Metacognition and cognitive monitoring." *American Psychologist*, 34(10), 906–911.
- Fleming, S.M. & Lau, H.C. (2014). "How to measure metacognition." *Frontiers in Human Neuroscience*, 8:443.
- Koriat, A. & Goldsmith, M. (1996). "Monitoring and control processes in the strategic regulation of memory accuracy." *Psychological Review*, 103, 490–517.
- Hart, J.T. (1965). "Memory and the feeling-of-knowing experience." *J. Educational Psychology*, 56, 208–216.
- Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
- Huang, J., et al. (2024). "Large Language Models Cannot Self-Correct Reasoning Yet." ICLR 2024.
- Wang, G., et al. (2025). "Decoupling Metacognition from Cognition." AAAI 2025.
- Xiong, M., et al. (2024). "Can LLMs Express Their Uncertainty?" ICLR 2024.
- Kruger, J. & Dunning, D. (1999). "Unskilled and unaware of it." *JPSP*, 77(6), 1121–1134.
- Yeung, N. & Summerfield, C. (2012). "Metacognition in human decision-making." *Phil. Trans. R. Soc. B*, 367, 1310–1321.
- Johnson, M.K., Hashtroudi, S. & Lindsay, D.S. (1993). "Source monitoring." *Psychological Bulletin*, 114, 3–28.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017.
- Steyvers, M. & Peters, M.A.K. (2025). "Metacognition and Uncertainty Communication in Humans and LLMs." *Current Directions in Psychological Science*.
- "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions." (2025). arXiv:2506.09038.