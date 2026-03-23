# MetaCog-Bench: Measuring What AI Knows About What It Knows

## 1. Problem Statement

Burnell et al. (2026) identify metacognition as a foundational cognitive ability for progress toward AGI, decomposing it into four sub-abilities: metacognitive knowledge, metacognitive monitoring, error detection and correction, and metacognitive control. Despite this clear theoretical mapping, no existing benchmark comprehensively evaluates all four. Current evaluation practice treats metacognition as an afterthought—calibration is occasionally measured as a side effect of other benchmarks, but never as the primary construct under study.

This gap is not merely academic. An AI system that cannot accurately assess the limits of its own knowledge is an unsafe system. Overconfident models produce authoritative-sounding hallucinations. Under-confident models refuse valid requests. Both failure modes stem from the same root cause: poor metacognitive monitoring. As language models are deployed in high-stakes domains—medicine, law, infrastructure planning—the ability to say "I don't know" appropriately becomes a safety-critical capability.

MetaCog-Bench addresses this gap with five tasks spanning all four sub-abilities, approximately 900 test items, and a metric suite grounded in the cognitive science literature on human metacognition. Every task has a direct antecedent in experimental psychology, every metric has a formal definition, and every dataset item has a verifiably correct answer. The benchmark is designed to discriminate meaningfully between current frontier models while remaining economical to run within a $50/day budget across four models.

## 2. Theoretical Framework

MetaCog-Bench is organized around two foundational frameworks from cognitive psychology, mapped explicitly onto the sub-abilities identified in DeepMind's cognitive taxonomy.

**Nelson and Narens (1990)** define metacognition as a two-level system: a meta-level that *monitors* an object-level (receiving information about cognitive states) and *controls* it (modifying processing in response). Our Tasks 1 and 2 (Confidence Calibration and Feeling-of-Knowing) target monitoring—the meta-level's ability to assess the accuracy of its own object-level outputs. Task 4 (Selective Abstention) targets control—the meta-level's ability to regulate behavior by withholding responses when monitoring signals indicate low reliability. This monitoring-control decomposition is the most important structural principle in the benchmark: a model that monitors well but controls poorly (high AUROC2 but low abstention accuracy) exhibits a qualitatively different failure mode than one that does neither.

**Flavell (1979)** distinguishes three types of metacognitive knowledge: person knowledge (what one knows about one's own cognitive strengths and weaknesses), task knowledge (understanding what makes a problem hard), and strategy knowledge (knowing which approaches work for which problems). Task 5 (Metacognitive Knowledge) directly operationalizes person knowledge by asking models to predict their own accuracy across 20 domains of varying difficulty before being tested. This design allows us to detect Dunning-Kruger patterns—systematic overestimation of competence in weak domains and underestimation in strong ones—previously documented in LLMs by Kadavath et al. (2022).

**Error detection** maps onto DeepMind's third sub-ability and draws on Yeung and Summerfield's (2012) framework of metacognition in decision-making. Task 3 presents models with worked solutions containing plausible errors (off-by-one arithmetic, affirming the consequent, averaging speeds incorrectly) and asks them to identify and correct these errors. This is distinct from simply solving problems—it requires evaluating another agent's reasoning, a second-order cognitive operation.

Our primary metric, Type 2 AUROC (AUROC2), follows Fleming and Lau (2014), who showed that traditional calibration measures confound metacognitive sensitivity with response bias. AUROC2 asks a cleaner question: can the model discriminate its own correct responses from incorrect ones, regardless of overall confidence level? This is the gold standard in human metacognition research and provides the fairest comparison across models with different base rates of accuracy. We complement AUROC2 with ECE for calibration slope, Brier Score for proper scoring, and Goodman-Kruskal gamma for the Feeling-of-Knowing paradigm, following Koriat and Goldsmith's (1996) analysis of the accuracy-informativeness tradeoff.

## 3. Task Design

MetaCog-Bench comprises five tasks totaling approximately 900 items, each grounded in a specific experimental paradigm from cognitive psychology.

**Task 1: Confidence Calibration (300 items).** The model answers questions spanning three domains—arithmetic, factual knowledge, and logical reasoning—and assigns a confidence rating from 0 to 100 for each response. Items are distributed across three difficulty levels (20% easy, 60% medium, 20% hard). Mathematical items are procedurally generated with randomized parameters (e.g., modular arithmetic with random moduli, prime factorization of products of random primes), providing strong contamination resistance. Factual items are drawn from a curated set with verified answers. Logic items range from simple syllogisms to multi-step constraint satisfaction problems.

**Task 2: Feeling-of-Knowing (300 items, shared dataset with Task 1).** Before answering each question, the model predicts the probability that it will answer correctly. This prospective judgment, studied extensively since Hart (1965), tests a different metacognitive process than retrospective confidence: the model must assess its knowledge *before* retrieval, not after. We measure performance with Goodman-Kruskal gamma correlation, the standard metric in FOK research, where typical human performance falls in the 0.30–0.60 range.

**Task 3: Error Detection (200 items).** Models review presented solutions to problems—half correct, half containing planted errors. Errors are designed to be plausible near-misses: arithmetic off-by-one errors, logical fallacies embedded in otherwise valid reasoning, and methodological errors (e.g., averaging speeds rather than computing harmonic mean). The 50/50 balance and four distinct error types (arithmetic, logical, methodological, none) allow us to compute balanced accuracy, per-type F1, and to identify systematic blind spots.

**Task 4: Selective Abstention (200 items).** Models are presented with a mix of answerable questions (120) and unanswerable ones (80). Unanswerable items span five categories: future unknowns, subjective questions, questions with false premises, underspecified queries, and genuinely unsolved problems. The scoring function creates a clear incentive structure: correct answers earn 1.0, incorrect answers earn 0.0, appropriate abstentions on unanswerable questions earn 1.0, but unnecessary abstentions on answerable questions earn only 0.25. This asymmetric payoff matrix, inspired by Koriat and Goldsmith's (1996) accuracy-informativeness framework, penalizes both overconfidence and excessive caution. Recent work on AbstentionBench suggests that reasoning-focused models may paradoxically perform *worse* on abstention—their stronger drive to produce answers may override appropriate uncertainty signals.

**Task 5: Metacognitive Knowledge (200 items across 20 domains).** In a two-phase design, models first predict their accuracy on 10 questions in a named domain, then answer those questions. Domains are deliberately chosen to span the full competence spectrum—from basic arithmetic and world capitals (expected high accuracy) to Koine Greek vocabulary and medieval numismatics (expected low accuracy). This design directly tests person knowledge and enables Dunning-Kruger analysis at the domain level.

All tasks use structured output schemas (dataclass-based) to ensure clean, parseable responses. Each schema constrains the model to return exactly the fields needed for scoring, eliminating post-hoc extraction heuristics that could introduce measurement noise.

## 4. Metrics

The metric suite is designed for mathematical rigor and interpretability. **AUROC2** serves as the primary metric across Tasks 1, 2, and 4. Following Fleming and Lau (2014), AUROC2 measures metacognitive *sensitivity*—the ability to discriminate correct from incorrect responses—independently of response bias. A model that assigns confidence 90 to all responses but is correct 90% of the time would have perfect calibration (ECE = 0) but AUROC2 near 0.5 (chance), revealing that its confidence carries no item-level discriminative information.

**Expected Calibration Error (ECE)** with 10 equal-width bins captures the slope and intercept of the calibration curve. **Brier Score** provides a proper scoring rule that jointly rewards calibration and discrimination. **Goodman-Kruskal gamma** measures ordinal association for the Feeling-of-Knowing task, where the relevant question is rank-order accuracy of predictions rather than absolute calibration.

For Task 3, we report balanced accuracy and per-error-type F1 to identify systematic blind spots. For Task 5, we compute mean absolute prediction error and the Pearson correlation between predicted and actual domain accuracy.

The composite score across all tasks uses the **geometric mean**, which prevents strong performance on one task from compensating for failure on another—a model must demonstrate broad metacognitive competence. All aggregate metrics include nonparametric **bootstrap confidence intervals** (n=1000, 95% CI) to quantify measurement uncertainty.

## 5. Expected Results and Insights

Based on the existing literature, we expect several systematic patterns. First, **models will be overconfident**, particularly on hard items. Xiong et al. (2024) demonstrated that LLMs systematically overestimate their accuracy when expressing uncertainty verbally. We expect ECE values in the 0.10–0.25 range, with the calibration curve lying consistently below the diagonal for high-confidence bins.

Second, **AUROC2 will vary more than ECE across models**. Because AUROC2 is bias-free, it will expose genuine differences in metacognitive sensitivity that ECE—which can be artificially low for a uniformly confident model—would obscure. We anticipate AUROC2 values between 0.55 and 0.75, with larger models performing better, consistent with findings from Kadavath et al. (2022) that scaling improves self-knowledge.

Third, **reasoning-optimized models may show a metacognitive paradox**. Models fine-tuned for chain-of-thought reasoning may generate more confident responses even when wrong, because their training objective rewards producing complete solutions rather than expressing uncertainty. This would manifest as lower AUROC2 and worse abstention accuracy compared to base models of similar scale—a finding with direct implications for deployment safety.

Fourth, **Task 5 will reveal Dunning-Kruger patterns**. We expect models to overestimate accuracy on domains where they perform poorly (Uzbek geography, medieval numismatics) and potentially underestimate accuracy on strong domains (basic arithmetic, programming concepts). The magnitude of this effect—if present—would be a novel empirical contribution, extending Wang et al.'s (2025) work on decoupling metacognition from cognition.

Fifth, the **error detection task will expose domain-dependent blind spots**. Models may reliably catch arithmetic errors but miss logical fallacies, or vice versa. Per-error-type F1 breakdowns will provide actionable information about which categories of metacognitive failure are most prevalent.

These expected findings, combined with the benchmark's formal metric suite, should provide the discriminatory power needed to rank models meaningfully on metacognitive ability—not just "how smart is it?" but "does it know the boundaries of its own competence?"

## 6. Discussion

MetaCog-Bench frames hallucination as fundamentally a metacognitive failure—a model that accurately monitored its own uncertainty would abstain rather than fabricate. This reframing suggests that improving metacognition, rather than expanding knowledge alone, may be the more tractable path to reliable AI systems.

The benchmark is designed for longitudinal tracking: as models improve, procedurally generated items can be regenerated with new parameters, maintaining contamination resistance. Future extensions could incorporate human baselines from the metacognition literature, adaptive difficulty based on model performance, and multi-turn deliberation protocols that test whether models can improve their own answers through self-reflection—or whether, as Huang et al. (2024) argue, self-correction remains an unsolved problem.

## References

- Burnell, R., et al. (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy. Google DeepMind.
- Flavell, J.H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911.
- Fleming, S.M. & Lau, H.C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8:443.
- Huang, J., et al. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *ICLR 2024*.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Koriat, A. & Goldsmith, M. (1996). Monitoring and control processes in the strategic regulation of memory accuracy. *Psychological Review*, 103, 490–517.
- Nelson, T.O. & Narens, L. (1990). Metamemory: A Theoretical Framework and New Findings. In *Psychology of Learning and Motivation*, 26, 125–173.
- Wang, G., et al. (2025). Decoupling Metacognition from Cognition. *AAAI 2025*.
- Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? *ICLR 2024*.
- Yeung, N. & Summerfield, C. (2012). Metacognition in human decision-making. *Philosophical Transactions of the Royal Society B*, 367, 1310–1321.
