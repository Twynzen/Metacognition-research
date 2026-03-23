# MetaCog-Bench

**Measuring What AI Knows About What It Knows**

A Python benchmark for evaluating metacognitive abilities in AI models. Built for the [Google DeepMind "Measuring Progress Toward AGI: Cognitive Abilities" Kaggle hackathon](https://www.kaggle.com/) — Metacognition track.

## What it tests

| Task | What it measures |
|------|-----------------|
| Confidence Calibration | Are the model's confidence scores aligned with actual accuracy? |
| Feeling-of-Knowing | Can the model predict its own performance before answering? |
| Error Detection | Can the model spot mistakes in presented solutions? |
| Selective Abstention | Does the model know when to say "I don't know"? |
| Metacognitive Knowledge | Can the model predict its strengths and weaknesses across domains? |

## Key metrics

- **ECE** (Expected Calibration Error)
- **AUROC2** (Type 2 metacognitive sensitivity)
- **Brier Score**
- **Goodman-Kruskal Gamma**
- Geometric mean composite score with bootstrap CIs

## Quick start

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run tests
pytest tests/

# The final benchmark runs inside a Kaggle Notebook
# using the kaggle-benchmarks SDK
```

## Project structure

```
src/           # Core modules (schemas, datasets, metrics, tasks)
tests/         # Unit tests
notebook/      # Consolidated single-file for Kaggle submission
writeup/       # Competition writeup
```

## License

MIT
