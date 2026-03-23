# EXECUTE.md — Autonomous Execution Plan for Claude Code

## YOUR MISSION

You are the lead engineer on MetaCog-Bench, a metacognition benchmark for Google DeepMind's Kaggle hackathon ($25K grand prize). You have two master documents:

- **CLAUDE.md**: The project architecture, schemas, dataset specs, task designs, and metrics
- **AUDIT.md**: A complete audit with 9 phases of bugs, optimizations, and innovations

Your job is to **execute the entire AUDIT.md autonomously from start to finish**. Work through every sprint in order. After completing each sprint, run tests to verify, then move to the next sprint. Do NOT stop between sprints — keep going until everything is done.

## EXECUTION RULES

1. **Read CLAUDE.md and AUDIT.md first** — understand the full picture before touching any code
2. **Work sprint by sprint** in the order defined in the AUDIT.md checklist (Sprint 1 → 2 → 3 → 4 → 5 → 6 → 7)
3. **After each sprint**: run all tests, verify nothing broke, print a brief status summary, then continue to next sprint
4. **If a change breaks something**: fix it before moving on. Never leave broken code behind.
5. **If an innovation from Phase 4 is too complex or risky**: implement the simpler version and move on. Don't get stuck.
6. **The final output must be a single consolidated Python file** at `notebook/metacog_bench_final.py` that can be pasted directly into a Kaggle Notebook
7. **Also produce** `writeup/writeup.md` — the 1500-word competition writeup

## SPRINT EXECUTION SEQUENCE

### SPRINT 1: Fix Critical Bugs + Add Robustness
Execute ALL items from AUDIT.md:
- BUG-001 through BUG-005 (Phase 0)
- ROBUST-001 through ROBUST-004 (Phase 1)

After completing: Run `python -m pytest tests/` or manually test dataset generation and metrics with mock data. Print: "Sprint 1 complete. [X] bugs fixed, [Y] robustness improvements added."

### SPRINT 2: Multi-Model + Discriminatory Power
Execute ALL items from AUDIT.md:
- DISC-001 through DISC-004 (Phase 2)

Change `llm=[kbench.llm]` to at minimum `llm=[kbench.llms["google/gemini-2.5-flash"], kbench.llms["google/gemini-2.5-pro"]]`. Add discrimination analysis code, difficulty gradient analysis, floor/ceiling checks.

Print: "Sprint 2 complete. Multi-model evaluation configured for [N] models."

### SPRINT 3: Optimize Prompts + Improve Datasets
Execute ALL items from AUDIT.md:
- PROMPT-001 through PROMPT-004 (Phase 5)
- DATA-001 through DATA-004 (Phase 6)

Improve calibration prompt with explicit scale, neutralize error detection prompt, expand abstention patterns, add novel composition questions, add subtle method errors.

Print: "Sprint 3 complete. [X] prompts improved, [Y] dataset items added/modified."

### SPRINT 4: Implement Innovations
Execute from AUDIT.md Phase 4. Prioritize by impact:
1. **INNOV-001** (Conditional ECE by difficulty) — MUST DO, no extra API cost
2. **INNOV-005** (Prospective vs retrospective comparison) — MUST DO, data already exists
3. **INNOV-003** (Adversarial confidence anchoring) — DO IF possible, adds novel paradigm
4. **INNOV-002** (Meta-metacognition) — SKIP unless trivial to add
5. **INNOV-004** (False feedback resilience) — SKIP unless trivial to add

Print: "Sprint 4 complete. [X] innovations implemented."

### SPRINT 5: Visualizations
Execute ALL visualization items from AUDIT.md Phase 7:
- VIZ-001: Reliability diagram (MANDATORY)
- VIZ-002: Radar chart (improved normalization)
- VIZ-003: Confidence distribution histograms
- VIZ-004: ECE heatmap by difficulty × domain (if data supports it)
- VIZ-005: Formatted comparison table

All visualizations must render inline with `plt.show()` AND save to files.

Print: "Sprint 5 complete. [X] visualizations added."

### SPRINT 6: Consolidation + Writeup
1. Consolidate everything into `notebook/metacog_bench_final.py` — one single file, all code inline, no external imports from our project
2. Verify the consolidated file has:
   - All schemas defined
   - All utility functions (check_answer, metrics, validators)
   - All dataset generators called
   - All tasks defined with @kbench.task
   - Multi-model evaluation
   - Aggregate metrics computation
   - All visualizations
   - `%choose metacog_bench` as LAST line (NOT commented out)
3. Write `writeup/writeup.md` following WRITEUP-001 template from AUDIT.md — exactly 1200-1500 words

Print: "Sprint 6 complete. Final notebook: [X] lines. Writeup: [Y] words."

### SPRINT 7: Polish
1. Add markdown-comment headers throughout the consolidated notebook for readability
2. Verify no syntax errors in the final file
3. Verify all dataset generation produces correct counts
4. Verify metrics work with synthetic test data
5. Add a brief intro comment at the top explaining what the benchmark does

Print: "Sprint 7 complete. MetaCog-Bench is ready for submission."

## FINAL DELIVERABLES

When all sprints are done, confirm these files exist and are complete:
1. `notebook/metacog_bench_final.py` — The complete, ready-to-paste Kaggle notebook
2. `writeup/writeup.md` — The 1500-word competition writeup
3. All files in `src/` updated with the improvements
4. All tests passing

Print a final summary:
```
============================================
MetaCog-Bench Build Complete
============================================
Notebook: notebook/metacog_bench_final.py ([X] lines)
Writeup: writeup/writeup.md ([Y] words)
Tasks: 5 metacognitive tasks
Items: ~[Z] evaluation items
Models: [N] frontier models configured
Metrics: ECE, Brier, AUROC2, Gamma, Composite
Visualizations: [V] charts
Innovations: [I] novel contributions
Status: READY FOR KAGGLE SUBMISSION
============================================
```

## BEGIN EXECUTION NOW

Start by reading CLAUDE.md and AUDIT.md, then execute Sprint 1.
