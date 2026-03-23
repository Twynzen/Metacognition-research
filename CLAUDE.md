# MetaCog-Bench — Build Instructions for Claude Code

## MISSION
Build a winning submission for Google DeepMind's "Measuring Progress Toward AGI: Cognitive Abilities" Kaggle hackathon ($25K grand prize). Target: **Metacognition track**.

## WHAT WE'RE BUILDING
A Python benchmark that tests AI models on metacognitive abilities using the `kaggle-benchmarks` SDK. The final deliverable is a **single consolidated Python file** that runs inside a Kaggle Notebook.

## CRITICAL CONSTRAINTS
- Final code runs in Kaggle Notebook with `import kaggle_benchmarks as kbench` pre-installed
- `kbench.llm` and `kbench.llms[...]` only work inside Kaggle — do NOT try to call them locally
- Build all logic (datasets, metrics, schemas, task logic) as pure Python that can be tested locally
- Then consolidate everything into `notebook/metacog_bench_final.py`
- Use `sklearn` and `numpy` (available in Kaggle). Do NOT use obscure packages.
- Target: ~900 test items across 5 tasks, runnable against 4 models within $50/day budget

---

## PROJECT STRUCTURE

```
metacog-bench/
├── CLAUDE.md                    ← THIS FILE (read first!)
├── src/
│   ├── __init__.py
│   ├── schemas.py               ← Pydantic/dataclass schemas for structured LLM output
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── calibration.py       ← Generate 300 questions (math/factual/logic, 3 difficulties)
│   │   ├── error_detection.py   ← Generate 200 items (100 correct + 100 with errors)
│   │   ├── abstention.py        ← Generate 200 items (120 answerable + 80 unanswerable)
│   │   └── self_knowledge.py    ← Generate 200 items (20 domains × 10 questions)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calibration.py       ← ECE, Brier Score, reliability diagram data
│   │   ├── discrimination.py    ← AUROC₂, Goodman-Kruskal gamma
│   │   └── aggregate.py         ← Geometric mean composite score, bootstrap CIs
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── confidence_calibration.py
│   │   ├── feeling_of_knowing.py
│   │   ├── error_detection.py
│   │   ├── selective_abstention.py
│   │   └── metacognitive_knowledge.py
│   └── visualizations.py        ← Reliability diagrams, radar charts (matplotlib)
├── tests/
│   ├── test_datasets.py         ← Verify datasets generate correctly, no duplicates
│   ├── test_metrics.py          ← Verify ECE, Brier, AUROC with known values
│   └── test_answer_checker.py   ← Verify answer matching logic
├── notebook/
│   └── metacog_bench_final.py   ← CONSOLIDATED single-file for Kaggle (auto-generated)
└── writeup/
    └── writeup.md               ← 1500-word writeup for competition submission
```

---

## SCHEMAS (src/schemas.py)

These dataclasses are used with `llm.prompt(..., schema=SchemaName)` in kaggle-benchmarks to force structured output from the model.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AnswerWithConfidence:
    """For Task 1: Confidence Calibration"""
    answer: str
    confidence: int  # 0-100

@dataclass
class FOKResponse:
    """For Task 2: Feeling-of-Knowing"""
    prediction: int  # 0-100 likelihood of answering correctly
    answer: str

@dataclass
class ErrorReview:
    """For Task 3: Error Detection"""
    has_error: bool
    error_explanation: str
    corrected_answer: str

@dataclass
class AbstentionResponse:
    """For Task 4: Selective Abstention"""
    can_answer: bool
    answer: Optional[str]
    confidence: int  # 0-100

@dataclass
class DomainPrediction:
    """For Task 5: Metacognitive Knowledge"""
    predicted_accuracy: int  # 0-100
    hardest_aspect: str
    easiest_aspect: str
```

---

## DATASET GENERATION SPECIFICATIONS

### General Rules
- All datasets return `pd.DataFrame`
- Use `random.seed(42)` for reproducibility during development
- Procedurally generate math/logic questions with random parameters (contamination resistance)
- Difficulty distribution: 20% easy / 60% medium / 20% hard
- Every item must have a **verifiably correct answer** (no ambiguity)

### Dataset 1: Calibration & FOK (300 items)
Shared dataset for Tasks 1 and 2. Columns: `question`, `correct_answer`, `difficulty`, `domain`

Three domains (100 each):
- **Math**: Arithmetic (easy) → modular arithmetic (medium) → prime factorization / combinatorics (hard)
- **Factual**: Common knowledge (easy) → specialized (medium) → obscure (hard)
- **Logic**: Simple syllogisms (easy) → multi-step deduction (medium) → constraint satisfaction (hard)

Math generation examples:
```python
# Easy: What is 47 + 83?  → "130"
# Medium: What is (347 × 29) mod 17?  → str((347*29) % 17)
# Hard: What is the largest prime factor of 7×31×67=14477?  → "67"
```

Factual generation: Use a curated list of ~150 questions with verified answers. Include:
- Easy: "What is the chemical symbol for gold?" → "Au"
- Medium: "In what year was the Treaty of Westphalia signed?" → "1648"
- Hard: "What is the atomic number of Rutherfordium?" → "104"

Logic generation examples:
```python
# Easy: "If all dogs are mammals and Rex is a dog, is Rex a mammal?" → "Yes"
# Medium: "A is taller than B. C is shorter than B. D is taller than A. Who is shortest?" → "C"
# Hard: Multi-step constraint satisfaction with 4+ variables
```

### Dataset 2: Error Detection (200 items)
Columns: `problem`, `presented_solution`, `solution_has_error` (bool as str), `error_type`

50% correct solutions, 50% with planted errors. Error types:
- `arithmetic`: Off-by-one, wrong operation (e.g., 17×23="381" instead of 391)
- `logical`: Affirming the consequent, false dichotomy
- `method`: Correct answer by wrong method (e.g., average speed ≠ average of speeds)
- `none`: Correct solution (no error)

CRITICAL: Errors must be plausible (not obviously wrong). Use "near-miss" errors.

### Dataset 3: Abstention (200 items)
Columns: `question`, `is_answerable` (bool as str), `correct_answer`, `unanswerable_reason`

120 answerable (mix of easy/medium/hard factual) + 80 unanswerable across 5 categories:
- `future_unknown`: "Who will win the 2035 Nobel Prize in Physics?"
- `subjective`: "What is the most beautiful city in the world?"
- `false_premise`: "When did Napoleon conquer China?"
- `underspecified`: "How far is it?" (missing context)
- `genuinely_unknown`: "What is the exact mechanism of consciousness?"

### Dataset 4: Self-Knowledge (20 domains × 10 questions = 200 items)
Columns: `domain`, `domain_questions` (pipe-separated), `domain_answers` (pipe-separated)

Include domains the model will definitely know well AND domains it'll struggle with:
- Known well: basic arithmetic, world capitals, popular movies, programming concepts
- Medium: organic chemistry, music theory, European history, astronomy
- Weak: Koine Greek, advanced topology, Uzbek geography, medieval numismatics
- Trick: common misconceptions, optical illusions described in text

---

## TASK IMPLEMENTATIONS

### Task 1: Confidence Calibration
```
Prompt template:
"Answer the following question. After your answer, rate your confidence 
from 0 to 100 (where 0 = certainly wrong, 100 = certainly correct).
Question: {question}"

Schema: AnswerWithConfidence
Scoring: Per-item → 1 - |confidence/100 - correctness|
Aggregate: ECE (10 bins), Brier Score, AUROC₂
```

### Task 2: Feeling-of-Knowing
```
Prompt template:
"I'm going to ask you a question. Before you answer, predict how likely 
you are to get it correct on a scale of 0-100. Then provide your best answer.
Question: {question}"

Schema: FOKResponse
Scoring: Per-item → 1 - |prediction/100 - correctness|
Aggregate: Gamma correlation, AUROC₂
```

### Task 3: Error Detection
```
Prompt template:
"Here is a problem and a proposed solution. Please review the solution 
carefully and determine if it is correct.
Problem: {problem}
Proposed Solution: {presented_solution}
Is this solution correct? If not, explain the error and provide the correct answer."

Schema: ErrorReview
Scoring: Binary (correct classification = 1.0, wrong = 0.0)
Aggregate: Balanced accuracy, F1, per-error-type breakdown
```

### Task 4: Selective Abstention
```
Prompt template:
"Answer the following question. If you genuinely cannot determine the 
answer or the question is unanswerable, say 'I don't know' and explain why.
Question: {question}"

Schema: AbstentionResponse
Scoring:
  - Answerable + correct → 1.0
  - Answerable + wrong → 0.0
  - Answerable + abstained → 0.25
  - Unanswerable + abstained → 1.0
  - Unanswerable + attempted → 0.0
Aggregate: Overall accuracy, abstention F1, over-caution rate
```

### Task 5: Metacognitive Knowledge
```
Phase 1 prompt:
"I'm going to test you on 10 questions about {domain}. 
Before I do, predict what percentage (0-100) you'll get correct."

Phase 2: Ask 10 domain questions sequentially

Schema: DomainPrediction (phase 1), str (phase 2)
Scoring: 1 - |predicted_accuracy/100 - actual_accuracy|
Aggregate: Mean prediction error, correlation, Dunning-Kruger pattern analysis
```

---

## METRIC IMPLEMENTATIONS (src/metrics/)

### ECE (Expected Calibration Error)
```python
def compute_ece(confidences, correctness, n_bins=10):
    """
    ECE = Σ (|B_m|/n) × |acc(B_m) - conf(B_m)|
    
    Args:
        confidences: list of float [0,1]
        correctness: list of int {0,1}
    Returns:
        dict with 'ece', 'mce', 'bin_data' (for reliability diagram)
    """
```
- Use 10 equal-width bins
- Return bin_data for visualization
- ECE = 0 is perfect; >0.10 is poorly calibrated

### Brier Score
```python
def compute_brier(confidences, outcomes):
    """BS = (1/N) × Σ(f_i - o_i)²"""
```
- BS = 0 is perfect; BS = 0.25 is no-skill baseline

### AUROC₂ (Type 2 AUROC)
```python
from sklearn.metrics import roc_auc_score
def compute_auroc2(confidences, correctness):
    """Metacognitive sensitivity: can model discriminate own right/wrong?"""
```
- 0.5 = random; >0.7 = good; 1.0 = perfect
- THIS IS THE MOST IMPORTANT METRIC (bias-free per Fleming & Lau 2014)

### Goodman-Kruskal Gamma
```python
def compute_gamma(confidences, correctness):
    """γ = (concordant - discordant) / (concordant + discordant)"""
```
- Range [-1, 1]; typical human FOK: 0.30-0.60

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Nonparametric bootstrap for any aggregate metric."""
```

### Composite Score
```python
def geometric_mean(sub_scores):
    """Geometric mean prevents compensation between tasks."""
    values = [max(v, 1e-10) for v in sub_scores]
    return np.exp(np.mean(np.log(values)))
```

---

## ANSWER CHECKING (shared utility)

```python
def check_answer(model_answer: str, correct_answer: str) -> bool:
    """
    Deterministic answer checker. Rules:
    1. Normalize: strip, lowercase, remove articles/punctuation
    2. Exact match after normalization
    3. Containment check (correct in model's answer)
    4. Numerical comparison with tolerance (±0.01)
    5. For yes/no questions: check for "yes"/"no" keywords
    """
```

IMPORTANT: This function must be thoroughly tested. False negatives (marking correct answers wrong) will corrupt ALL metrics.

---

## VISUALIZATION SPECIFICATIONS (src/visualizations.py)

### 1. Reliability Diagram
- X-axis: Mean predicted confidence per bin
- Y-axis: Fraction correct per bin
- Perfect calibration: diagonal line
- Show gap between bars and diagonal
- One diagram per model, or overlay multiple models

### 2. Radar Chart (Metacognitive Profile)
- 5 axes: Calibration (1-ECE), Sensitivity (AUROC₂), Error Detection, Abstention, Self-Knowledge
- One polygon per model
- Fill with alpha=0.1

### 3. Model Comparison Table
- Rows: models
- Columns: each metric per task + composite
- Highlight best/worst per column

---

## CONSOLIDATION STEP

After all modules are built and tested, create `notebook/metacog_bench_final.py`:

1. Inline all imports and utility functions
2. Inline all dataset generation
3. Inline all schemas
4. Define one `@kbench.task` per sub-task + one composite task
5. Generate datasets inside the notebook
6. Run `.evaluate()` on all models
7. Compute aggregate metrics
8. Generate visualizations
9. Final cell: `%choose metacog_bench`

Template:
```python
# ========================================
# MetaCog-Bench: Measuring What AI Knows About What It Knows
# Track: Metacognition
# ========================================

import kaggle_benchmarks as kbench
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import random, re

# --- SCHEMAS ---
# (paste from schemas.py)

# --- UTILITIES ---
# (paste check_answer and metrics)

# --- DATASET GENERATION ---
# (paste all generators, call them to create DataFrames)

# --- TASK DEFINITIONS ---
@kbench.task(name="metacog_confidence_calibration", 
             description="Measures confidence calibration via ECE and AUROC2")
def confidence_calibration(llm, question: str, correct_answer: str, difficulty: str) -> float:
    ...

@kbench.task(name="metacog_feeling_of_knowing",
             description="Tests prospective metacognitive monitoring via FOK paradigm")
def feeling_of_knowing(llm, question: str, correct_answer: str, difficulty: str) -> float:
    ...

# ... etc for all 5 tasks

# --- RUN EVALUATIONS ---
cal_results = confidence_calibration.evaluate(llm=[kbench.llm], evaluation_data=cal_df)
# ... etc

# --- COMPUTE AGGREGATE METRICS ---
# ... ECE, Brier, AUROC2, gamma per model

# --- VISUALIZATIONS ---
# ... reliability diagrams, radar chart

# --- FINAL CELL ---
# %choose metacog_confidence_calibration
```

---

## BUILD ORDER (Priority)

1. **P0**: `src/schemas.py` + `src/metrics/` + answer checker + tests → Foundation
2. **P1**: `src/datasets/calibration.py` + `src/tasks/confidence_calibration.py` → Core task
3. **P2**: `src/datasets/abstention.py` + `src/tasks/selective_abstention.py` → Novel finding
4. **P3**: `src/datasets/error_detection.py` + `src/tasks/error_detection.py` → Direct DeepMind mapping
5. **P4**: `src/tasks/feeling_of_knowing.py` → Reuses P1 dataset
6. **P5**: `src/datasets/self_knowledge.py` + `src/tasks/metacognitive_knowledge.py` → Complex multi-turn
7. **P6**: `src/visualizations.py` → Charts for writeup
8. **P7**: Consolidation into `notebook/metacog_bench_final.py`
9. **P8**: `writeup/writeup.md` → 1500-word competition writeup

---

## KEY CITATIONS FOR WRITEUP

- Burnell, R., et al. (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy. Google DeepMind.
- Nelson, T.O. & Narens, L. (1990). Metamemory: A Theoretical Framework. Psych. Learning & Motivation, 26, 125–173.
- Flavell, J.H. (1979). Metacognition and cognitive monitoring. American Psychologist, 34(10), 906–911.
- Fleming, S.M. & Lau, H.C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8:443.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Huang, J., et al. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
- Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? ICLR 2024.
- Wang, G., et al. (2025). Decoupling Metacognition from Cognition. AAAI 2025.
- Koriat, A. & Goldsmith, M. (1996). Monitoring and control processes. Psychological Review, 103, 490–517.
- Yeung, N. & Summerfield, C. (2012). Metacognition in human decision-making. Phil. Trans. R. Soc. B, 367, 1310–1321.

---

## TESTING REQUIREMENTS

Before consolidation, verify:
- [ ] All datasets generate without errors and have correct column names
- [ ] No duplicate questions in any dataset  
- [ ] All correct_answers are actually correct (spot-check 20+ items manually)
- [ ] ECE computation matches known test case: confidences=[0.9,0.9,0.1,0.1], correctness=[1,1,0,0] → ECE≈0.0
- [ ] ECE computation for miscalibrated: confidences=[0.9,0.9,0.9,0.9], correctness=[1,0,1,0] → ECE≈0.4
- [ ] AUROC₂ returns 0.5 for random confidences
- [ ] check_answer correctly handles: exact match, case insensitive, numerical, containment
- [ ] check_answer correctly rejects: partial matches that are wrong, off-by-one numbers
- [ ] All schemas can be instantiated with valid data
- [ ] Difficulty distribution is approximately 20/60/20 for easy/medium/hard
