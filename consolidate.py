"""
Consolidation script: builds notebook/metacog_bench_final.py
by inlining all src/ modules into a single Kaggle-ready file.
"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "notebook", "metacog_bench_final.py")


def read_src(rel_path):
    """Read a source file, stripping internal imports we handle at the top."""
    path = os.path.join(BASE, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    result = []

    for line in lines:
        # Skip lines importing from src.*
        if re.match(r"^\s*(from src|import src)", line):
            continue
        # Skip duplicate top-level imports (we add them in the header)
        if re.match(r"^\s*import (pandas|numpy|random|re|unicodedata|math)\b", line):
            continue
        if re.match(r"^\s*from (typing|dataclasses|itertools|math) import", line):
            continue
        if re.match(r"^\s*from (sklearn\.metrics|sklearn) import", line):
            continue
        if re.match(r"^\s*import matplotlib", line):
            continue
        if "matplotlib.rcParams" in line:
            continue
        # Stop at if __name__ blocks
        if line.strip().startswith("if __name__"):
            break
        result.append(line)

    return "\n".join(result)


HEADER = """# ========================================
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

"""

TASK_CODE = '''
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

print(f"\\nUnified evaluation DataFrame: {len(all_data)} rows")
print(f"Task distribution:\\n{all_data[\\\'task_type\\\'].value_counts().to_string()}")


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

print("\\n" + "=" * 60)
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
print(f"\\nResults collected: {len(results_df)} rows")

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
            print(f"\\n{task_type}: {mean_score:.4f} [{lower:.4f}, {upper:.4f}] (n={len(scores)})")

# Composite score (geometric mean)
sub_means = [task_scores[t]["mean"] for t in task_scores if task_scores[t]["mean"] > 0]
if sub_means:
    composite = geometric_mean(sub_means)
    print(f"\\n{'=' * 60}")
    print(f"COMPOSITE SCORE (geometric mean): {composite:.4f}")
    print(f"{'=' * 60}")

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\\nGenerating visualizations...")

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
print("\\n" + "=" * 60)
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
    print(f"\\nComposite: {composite:.4f}")


# %choose metacog_bench
'''


def build():
    sep = "=" * 60

    sections = [
        (f"# {sep}\n# SCHEMAS\n# {sep}", "src/schemas.py"),
        (f"# {sep}\n# ANSWER CHECKER\n# {sep}", "src/utils.py"),
        (f"# {sep}\n# METRICS: CALIBRATION\n# {sep}", "src/metrics/calibration.py"),
        (f"# {sep}\n# METRICS: DISCRIMINATION\n# {sep}", "src/metrics/discrimination.py"),
        (f"# {sep}\n# METRICS: AGGREGATE\n# {sep}", "src/metrics/aggregate.py"),
        (f"# {sep}\n# DATASET: CALIBRATION & FOK (300 items)\n# {sep}", "src/datasets/calibration.py"),
        (f"# {sep}\n# DATASET: ERROR DETECTION (200 items)\n# {sep}", "src/datasets/error_detection.py"),
        (f"# {sep}\n# DATASET: ABSTENTION (200 items)\n# {sep}", "src/datasets/abstention.py"),
        (f"# {sep}\n# DATASET: SELF-KNOWLEDGE (20 domains x 10 questions)\n# {sep}", "src/datasets/self_knowledge.py"),
        (f"# {sep}\n# VISUALIZATIONS\n# {sep}", "src/visualizations.py"),
    ]

    body_parts = []
    for label, src_file in sections:
        body_parts.append(label)
        body_parts.append(read_src(src_file))
        body_parts.append("")

    # Fix escaped quotes in TASK_CODE
    task_code_clean = TASK_CODE.replace("\\'", "'")

    final = HEADER + "\n".join(body_parts) + task_code_clean

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(final)

    line_count = final.count("\n")
    print(f"Consolidated file written to: {OUT}")
    print(f"Total lines: {line_count}")


if __name__ == "__main__":
    build()
