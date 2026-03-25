# Revision Summary — Compact

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps._

Corpus: `data/results/` · Artifacts: `data/analysis_artifacts/revision_tracker/` · All commands are offline and deterministic.

---

## 1. Statistical Reporting

**Reviewer:** "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."

**What it means:** Beyond raw metric tables, the paper needed per-factor descriptive summaries and uncertainty estimates so reviewers can verify patterns without reverse-engineering ad hoc files.

**Implemented:** `lcm analyze summary` and `lcm analyze summary-bundle`. Compute grouped means, standard deviations, medians, mins, maxes, and 95% confidence intervals per factor level across all repetition rows.

**Finding:** Factor-level descriptive patterns — e.g., directionality effects for ALGO1, convergence effects for ALGO2 — are now explicit and auditable. Every number has provenance.

---

## 2. Formal Hypothesis Testing

**Reviewer:** "Add formal statistical testing with multiple comparison adjustments..."

**What it means:** The paper needed to distinguish genuine factor effects from noise. With hundreds of simultaneous tests (5 factors × 3 metrics × 18 source files), some will appear significant by chance. Multiple-comparison correction is required to keep the false-discovery rate below 5%.

**Implemented:** `lcm analyze hypothesis` and `lcm analyze hypothesis-bundle` with Benjamini-Hochberg FDR correction. For each factor, compares two levels (e.g., Convergence = −1 vs. Convergence = 1) on each metric. Reports raw p-values, BH-adjusted q-values, Cohen's d effect sizes, and 95% CIs around the mean difference.

**Finding:** Exhaustive 16-factor formal test confirms ALGO2 Convergence as dominant — high convergence reliably boosts all three metrics. ALGO3 has no robust factor effects: none of the tested factors produce a statistically significant difference after correction.

---

## 3. Output Validity

**Reviewer:** "It is important to know the reasons behind the failures as much as successful runs..."

**What it means:** The paper needed to distinguish malformed outputs, empty outputs, and valid-but-low-quality outputs — otherwise "variability" and "failure" are conflated.

**Implemented:** `lcm analyze output-validity-bundle`. Classifies every raw output row as `valid_output`, `empty_output`, or `malformed_output`, and records parsed edge count per row.

**Finding:** Imported raw outputs are overwhelmingly valid. The issue is variability, not parse failure. There are almost no malformed or empty outputs in the corpus.

---

## 4. Replication Stability

**Reviewer:** "A principled justification of the number of replications... the decision to use 5 replications does not seem to be justified by power analysis."

**What it means:** Five repetitions needed justification: are five runs enough to characterize run-to-run variability, or do metrics keep drifting with more runs?

**Implemented:** `lcm analyze stability`. Computes per-condition mean, standard deviation, min/max, range width, and coefficient of variation across the five recorded repetitions.

**Finding:** ALGO1 and ALGO2 are nearly repetition-stable — coefficient of variation is very low across all metrics. ALGO3 is not: its metrics drift noticeably across repetitions, especially recall. Five repetitions are sufficient to expose ALGO3's instability; they are more than sufficient to confirm ALGO1/2 stability.

---

## 5. Figure-Ready Exports

**Reviewer:** "Reporting confidence intervals or distributional plots alongside mean performance would further reinforce the paper's core message about variability."

**What it means:** The reviewer needed visual evidence — box plots, violin plots, faceted comparisons — that could be generated from structured data rather than manual assembly.

**Implemented:** `lcm analyze figures` and `lcm analyze figures-bundle`. Export tidy long-format rows (one row per repetition per metric) with preserved provenance columns. This is the input format for any external plotting tool.

**Finding:** The imported corpus can now be plotted directly with preserved provenance. The distributional summaries confirm the stability findings visually: ALGO1/2 distributions are tight; ALGO3 distributions are wide and bimodal.

---

## 6. Output Variability Mechanism

**Reviewer:** "The article shows that there is randomness but it is not really telling how and why this happens..."

**What it means:** The reviewer wanted a mechanism-level explanation: not just "which model is more variable," but what the variability actually looks like in terms of the raw outputs.

**Implemented:** `lcm analyze variability`. Computes three complementary drift measures: (1) pairwise Jaccard similarity of edge sets across repetitions, (2) exact-match rate, (3) breadth expansion ratio (union size / mean repetition size).

**Finding:** ALGO3 shows strong edge-set drift. Repetitions produce largely different edge sets — breadth expansion ratios are well above 1.0 and Jaccard similarities are well below 1.0. ALGO1/2 produce nearly identical edge sets across repetitions. This confirms ALGO3's instability is genuine, not a rounding artifact.

---

## 7. Non-LLM Baseline

**Reviewer:** "Include at least one non-LLM baseline method... to contextualize the value proposition of using LLMs despite their inherent variability."

**What it means:** The paper needed a fair reference point: does an LLM do better than a minimal reasonable strategy when both are allowed the same number of guesses?

**Implemented:** `lcm analyze baseline-bundle`. The random-k baseline: for each LLM output row, sample exactly k edges uniformly from the mother graph (k = LLM edge count per row, seed=42). No subgraph structure used. Both LLM and baseline evaluated against ground-truth cross edges.

**Finding:**

| | ALGO1 | ALGO2 | ALGO3 |
|---|---|---|---|
| Precision | LLM wins (6/6 models, avg +0.27) | LLM wins (6/6 models, avg +0.30) | LLM loses (0/6 models, avg −0.05) |
| Recall | LLM wins (6/6 models, avg +0.01) | LLM wins (6/6 models, avg +0.02) | LLM loses (0/6 models, avg −0.01) |
| Accuracy | LLM wins (6/6 models, avg +0.01) | LLM wins (6/6 models, avg +0.02) | LLM loses (0/6 models, avg −0.00) |

ALGO1/2: LLM is 6–8x more precise than random (26–40% vs 4–5%). The LLM has a genuine signal about which node pairs are connected — not just random guessing. ALGO3: the LLM loses to random on all metrics. ALGO3's noisy, inconsistent edge selection offers no advantage over picking edges at random.

---

## 8. Cross-Domain Generalization

**Reviewer:** "Add at least one more domain with other features to reinforce arguments that it is cross-domain."

**What it means:** Broader empirical scope, not just stronger analysis over the existing domain.

**Done:** Deferred. New domain data and likely new raw LLM runs would be required — this is outside the deterministic software tranche.

---

## Key Takeaways

1. Variability is not uniform: ALGO1 is very stable, ALGO2 is very stable, ALGO3 is the main source of instability.
2. Imported raw outputs almost never fail at the parser level. The issue is variability, not parse failure.
3. ALGO3's instability is genuine: repeated runs produce substantially different edge sets (low Jaccard, high breadth expansion).
4. ALGO1/2 beat random guessing by a large margin on precision (6–8x). The LLM is not just proposing edges at random — it has real signal.
5. ALGO3 adds no value over random guessing in this corpus.
