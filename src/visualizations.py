"""
Publication-quality visualizations for MetaCog-Bench.

1. Reliability Diagram — calibration per model
2. Radar Chart — metacognitive profile comparison
3. Model Comparison Table — formatted metrics summary
"""
import numpy as np
import matplotlib.pyplot as plt

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
    import pandas as pd

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
