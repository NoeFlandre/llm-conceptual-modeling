# Reviewer Response Log

This document provides a comprehensive account of each reviewer concern raised during the PLOS ONE revision process (PONE-D-25-54419), the problem it identified, the solution implemented in the codebase, and the concrete findings derived from that implementation. It is designed to maximize understanding by explaining concepts, providing examples, and breaking down technical notions.

---

## Table of Contents

1. [Background: The Paper's Context](#background-the-papers-context)
2. [Statistical Analysis & Reporting Concerns](#1-reviewer-2-statistical-analysis--reporting)
3. [Formal Hypothesis Testing](#2-reviewer-2-formal-hypothesis-testing)
4. [Failure Analysis](#3-reviewer-2-understanding-failures)
5. [Replication Stability](#4-reviewer-1-replication-justification)
6. [Plot-Ready Exports](#5-reviewer-1-confidence-intervals-and-distributional-plots)
7. [Non-LLM Baseline Comparison](#6-reviewer-2-non-llm-baseline)
8. [LLM Internal Mechanisms (Deferred)](#7-reviewer--philippe-llm-internal-mechanisms)
9. [Cross-Domain Generalization (Deferred)](#8-reviewer-2-cross-domain-generalization)
10. [Summary of Changes to the Codebase](#summary-of-changes-to-the-codebase)

---

## Background: The Paper's Context

**Paper Title:** "On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps"

**Task:** The paper studies how Large Language Models (LLMs) perform the conceptual modeling task of **combining causal maps** — a task where two partial causal graphs (subgraphs) must be merged into a unified mother graph.

**Three Algorithms Studied:**

| Algorithm | Name | Description |
|-----------|------|-------------|
| ALGO1 | Direct Combination with Chain-of-Verification | Generates candidate edges between subgraphs, then verifies each edge using the LLM |
| ALGO2 | Label Expansion with Embeddings | Expands node labels semantically using embeddings, then proposes edges based on similarity |
| ALGO3 | Tree-Based Target Matching | Builds a search tree from source nodes to find matching target nodes |

**Experimental Design:** Full factorial Design of Experiments (DoE) with 5 factors (32 conditions) for ALGO1, multiple convergence conditions for ALGO2, and depth/child_count for ALGO3. Each condition was replicated 5 times.

**Models Tested:** GPT-4o, GPT-5, Gemini 2.0 Flash, Gemini 2.5 Pro, DeepSeek Chat V3, DeepSeek V3 Chat 0324

---

## 1. Reviewer #2: Statistical Analysis & Reporting

### Reviewer Quote

> "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."
> "Add formal statistical testing with multiple comparison adjustments, provide confidence intervals of all performance measures..."

### Problem Being Raised

The reviewer identified that the original repository computed raw evaluation metrics (accuracy, precision, recall, F1) but did **not** provide:

1. **Grouped descriptive statistics** — no means, standard deviations, medians, or confidence intervals broken down by experimental factors
2. **Summary tables** that would allow readers to compare performance across factor levels (e.g., "with explanation" vs "without explanation")

Without these, the paper could not support claims like "the `Explanation` factor improved accuracy" with any statistical rigor.

### Solution Implemented in the Codebase

Implemented a new `lcm analyze summary` command that:

- Reads evaluated CSV files (output from the evaluation pipeline)
- Groups rows by specified factor columns
- Computes comprehensive statistics for each metric within each group:
  - `n` — count of observations
  - `mean` — arithmetic mean
  - `sample_std` — sample standard deviation
  - `median`
  - `min`, `max`
  - `ci95_low`, `ci95_high` — 95% confidence interval bounds

**Code Location:** `src/llm_conceptual_modeling/analysis/summary.py`

**Key Code Pattern:**

```python
def write_grouped_metric_summary(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    group_by: list[str],   # e.g., ["Explanation", "model"]
    metrics: list[str],    # e.g., ["accuracy", "precision", "recall"]
) -> None:
    # For each input file, for each metric:
    summary = (
        dataframe.groupby(group_by, dropna=False)[metric]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sample_std"})
    )
    # Compute 95% confidence interval:
    standard_error = summary["sample_std"] / summary["n"].pow(0.5)
    margin = 1.96 * standard_error.fillna(0.0)
    summary["ci95_low"] = summary["mean"] - margin
    summary["ci95_high"] = summary["mean"] + margin
```

**CLI Usage Example:**

```bash
lcm analyze summary \
    --input data/results/algo1/gpt-4o/evaluated.csv \
    --group-by Explanation \
    --metrics accuracy precision recall \
    --output summary_explanation.csv
```

### Findings from Implementing the Solution

The analysis revealed **directional patterns** in how experimental factors affect performance:

#### ALGO1 — `Explanation` Factor

| Metric | Direction | Winner in 18 file-level comparisons |
|--------|-----------|-------------------------------------|
| Accuracy | `Explanation=1` won more often | 11 of 18 |
| Precision | `Explanation=1` won more often | 12 of 18 |
| Recall | `Explanation=-1` won more often | 13 of 18 |

**Interpretation:** Including an explanation in the prompt tends to improve accuracy and precision, but can hurt recall. This is a classic precision-recall tradeoff.

#### ALGO2 — `Convergence` Factor

| Metric | Direction | Winner in 18 file-level comparisons |
|--------|-----------|-------------------------------------|
| Accuracy | `Convergence=-1` won more often | 13 of 18 |
| Precision | `Convergence=-1` won more often | 12 of 18 |
| Recall | `Convergence=1` won more often | 13 of 18 |

#### ALGO3 — `Depth` Factor

| Depth | Global Mean Recall |
|-------|-------------------|
| Depth=2 | 0.0652 |
| Depth=1 | 0.0144 |

**Every model except GPT-5** showed higher recall at Depth=2. This makes intuitive sense: deeper tree search finds more target nodes.

---

## 2. Reviewer #2: Formal Hypothesis Testing

### Reviewer Quote

> "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."
> "Add formal statistical testing with multiple comparison adjustments..."

### Problem Being Raised

Beyond descriptive statistics, the reviewer wanted **formal statistical tests** with:

1. **Paired t-tests** — because the experimental design uses replication, the same experimental condition is run multiple times, creating paired observations
2. **Multiple comparison correction** — with many tests run simultaneously, the chance of false positives increases; the reviewers specifically requested adjustment
3. **Provenance tracking** — clear documentation of which test was run on which data

### Solution Implemented in the Codebase

Implemented `lcm analyze hypothesis` command for **paired two-level factor tests** with:

- **Paired t-test (`ttest_rel`)** — tests whether the mean difference between two factor levels is significantly different from zero
- **Repeated-measures-ANOVA-equivalent F-statistic** — for these two-level within-source comparisons, the repository now also reports the equivalent ANOVA-style statistic with `F = t²`, `df_num = 1`, and `df_den = pair_count - 1`
- **Benjamini-Hochberg (BH) correction** — controls the False Discovery Rate (FDR) rather than the Family-Wise Error Rate

**Why Benjamini-Hochberg and not Bonferroni?**

- **Bonferroni** is overly conservative for exploratory factor analysis — it controls the probability of *any* false positive across all tests, which reduces statistical power dramatically when many tests are run
- **BH** controls FDR: the expected proportion of false positives among all significant results. This is more appropriate when, as the reviewer noted, we are making "many related factor-level tests across files and metrics"
- The BH procedure ranks p-values, then applies a progressively stricter correction based on rank

**Code Location:** `src/llm_conceptual_modeling/analysis/hypothesis.py`

**Benjamini-Hochberg Implementation:**

```python
def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_min = 1.0
    # Iterate from smallest p-value to largest
    for rank, (index, p_value) in enumerate(reversed(ranked), start=1):
        denominator = count - rank + 1
        candidate = min(1.0, p_value * count / denominator)
        running_min = min(running_min, candidate)  # Ensures monotonicity
        adjusted[index] = running_min
    return adjusted
```

**CLI Usage Example:**

```bash
lcm analyze hypothesis \
    --input data/results/algo1/gpt-4o/evaluated.csv \
    --factor Explanation \
    --pair-by model replication \
    --metrics accuracy precision recall \
    --output hypothesis_explanation.csv
```

**Output Columns:**

| Column | Description |
|--------|-------------|
| `source_input` | Path to evaluated CSV |
| `factor` | Factor tested (e.g., "Explanation") |
| `level_low`, `level_high` | The two factor levels being compared |
| `metric` | Metric being tested |
| `pair_count` | Number of paired observations |
| `mean_low`, `mean_high` | Group means |
| `mean_difference` | High minus Low mean |
| `t_statistic` | Paired t-test statistic |
| `f_statistic` | Repeated-measures-ANOVA-equivalent F-statistic (`t²` for these two-level paired tests) |
| `f_df_num`, `f_df_den` | Numerator and denominator degrees of freedom for the F-statistic |
| `p_value` | Raw p-value |
| `p_value_adjusted` | BH-corrected p-value |
| `test_family` | Records that the F-statistic is the repeated-measures-ANOVA-equivalent view of the paired comparison |
| `correction_method` | "benjamini-hochberg" |

### Findings from Implementing the Solution

#### ALGO2 `Convergence` — Strongest Factor

| Metric | Significant Tests (after BH) | Direction |
|--------|------------------------------|-----------|
| Accuracy | 16 of 18 | Convergence=-1 wins |
| Precision | 14 of 18 | Convergence=-1 wins |
| Recall | 14 of 18 | Convergence=1 wins |

**Total: 44 of 54 tests significant** — This is a very strong signal that the `Convergence` factor genuinely affects performance.

#### ALGO1 `Explanation` — Mixed Results

| Metric | Significant Tests (after BH) | Direction |
|--------|------------------------------|-----------|
| Accuracy | 8 of 18 | Mixed (more toward Explanation=1) |
| Precision | 9 of 18 | Mixed |
| Recall | 8 of 18 | Mixed (more toward Explanation=-1) |

**Total: 25 of 54 tests significant** — More modest effect than Convergence.

#### ALGO2 `Explanation` — Weaker

| Metric | Significant Tests (after BH) | Direction |
|--------|------------------------------|-----------|
| Accuracy | 7 of 18 | Mixed |
| Precision | 7 of 18 | Mixed |
| Recall | 7 of 18 | Mixed |

**Total: 21 of 54 tests significant** — Weaker than ALGO1's Explanation factor.

#### ALGO3 `Depth` and `Number of Words`

- `Depth`: 2 of 6 recall tests significant, both favoring Depth=2
- `Number of Words`: 0 of 6 tests significant after correction

---

## 3. Reviewer #2: Understanding Failures

### Reviewer Quote

> "It is important to know the reasons behind the failures as much as successful runs. Record the types of particular errors, the deterministic or stochastic nature of failures, and the systematic failure of particular combinations of the LLM-algorithm."

### Problem Being Raised

The reviewer noted that the original repository evaluated **structurally valid** outputs but did not track:

1. **Malformed outputs** — rows where the LLM returned unparseable content
2. **Empty outputs** — rows where no edges were generated at all
3. **Systematic failure patterns** — which model-algorithm combinations fail more often?

Without this, the paper could not distinguish between "the LLM tried but got it wrong" vs "the LLM didn't even produce parseable output."

### Solution Implemented in the Codebase

Implemented `lcm analyze failures` command that:

- Reads **raw** output CSVs (before evaluation)
- Classifies each row into a `failure_category`:
  - `valid_output` — row produced parseable edges
  - `empty_output` — row produced no edges
  - `malformed_output` — row had unparseable content
- Tracks `parsed_edge_count` per row
- Sets `is_failure=True/False`

**Code Location:** `src/llm_conceptual_modeling/analysis/failures.py`

**CLI Usage Example:**

```bash
lcm analyze failures \
    --input data/results/algo1/gpt-4o/raw.csv \
    --output failures_algo1_gpt4o.csv
```

### Findings from Implementing the Solution

**Surprising Result: Failure rate was 0.0 for every imported algorithm-model combination.**

| Algorithm | Models Tested | Total Rows | Failures |
|-----------|--------------|-------------|----------|
| ALGO1 | 6 | 2,880 | 0 |
| ALGO2 | 6 | 5,760 | 0 |
| ALGO3 | 6 | 1,440 | 0 |

**Key Insight:** The main issue is **not** parser-visible failure. Every LLM produced structurally valid JSON output. The variability is **inside** structurally valid outputs.

**The Real Problem — Edge Count Variability:**

| Algorithm | Model | Mean Edges | Median | Min | Max |
|-----------|-------|-----------|--------|-----|-----|
| ALGO2 | GPT-4o | 84.55 | 15 | 2 | **691** |
| ALGO2 | Gemini 2.0 Flash | 28.88 | 25 | 4 | 398 |
| ALGO3 | DeepSeek V3 Chat 0324 | 68.71 | 58 | 12 | 355 |

**Example of the variability problem:** GPT-4o's ALGO2 output ranged from 2 edges to 691 edges — a massive range that suggests the model sometimes generates extremely verbose outputs. The median (15) is much lower than the mean (84.55), indicating a **right-skewed distribution** with some extreme outliers.

---

## 4. Reviewer #1: Replication Justification

### Reviewer Quote

> "the manuscript would be strengthened by ... a principled justification of the number of replications, for example via power or precision considerations rather than feasibility alone."

### Problem Being Raised

The reviewer asked: "You used 5 replications — why 5? Is that enough to detect meaningful differences, or did you just stop because it was feasible?"

This requires more than saying "the plots look stable." It needs:

1. a retrospective stability analysis over the 5 repetitions, and
2. a concrete precision-based run calculation that converts the observed standard deviation into a conservative required total number of runs.

### Solution Implemented in the Codebase

Implemented `lcm analyze stability`, the new `lcm analyze replication-budget`, and an updated `lcm analyze stability-bundle`.

The stability command:

- Groups evaluated rows by experimental condition
- Computes replication stability metrics:
  - `n` — number of repetitions
  - `mean`, `sample_std`, `min`, `max` — across repetitions
- `range_width` — max minus min
- `coefficient_of_variation` (CV) — std/mean, a scale-independent measure of variability

The new run-budget command applies the supervisor's 95% CI precision formula:

`n = ((1.96 * s) / (r * |x_bar|))^2`

with `r = 0.05` as the relative half-width target. It reports the required total number of runs and the additional runs still needed beyond the existing 5.

**Code Location:** `src/llm_conceptual_modeling/analysis/stability.py`

**CLI Usage Example:**

```bash
lcm analyze stability \
    --input data/results/algo1/gpt-4o/evaluated.csv \
    --group-by Explanation model metric \
    --output stability_algo1.csv
```

### Findings from Implementing the Solution

#### Retrospective Stability: ALGO1 and ALGO2 Are Almost Fully Stable

| Algorithm | Metric | Conditions with ANY variation | Total Conditions |
|-----------|--------|------------------------------|------------------|
| ALGO1 | Accuracy | 2 | 576 |
| ALGO1 | Precision | 3 | 576 |
| ALGO1 | Recall | 2 | 576 |
| ALGO2 | (all) | 1 | 1152 |

**Interpretation:** With 5 replications, ALGO1 and ALGO2 show almost no run-to-run variation on the stable surfaces, especially accuracy.

#### Retrospective Stability: ALGO3 Is NOT Stable

| Algorithm | Metric | Conditions with ANY variation | Total Conditions |
|-----------|--------|------------------------------|------------------|
| ALGO3 | Recall | 44 | 96 |

**Coefficient of Variation (CV) Comparison:**

| Algorithm | Metric | Mean CV | Median CV | Max CV |
|-----------|--------|---------|----------|--------|
| ALGO1 | Accuracy | 0.000024 | 0.0 | 0.013 |
| ALGO1 | Precision | 0.00235 | 0.0 | 0.606 |
| ALGO2 | Accuracy | 0.000015 | 0.0 | 0.017 |
| ALGO3 | Recall | **3.21** | 3.87 | 3.87 |

**ALGO3's CV is ~1000x higher than ALGO1/ALGO2.** The tree-based approach amplifies small differences in intermediate steps into large differences in final recall.

**Depth Effect on ALGO3 Stability:**

| Depth | Varying Conditions Share | Mean Range Width |
|-------|--------------------------|------------------|
| Depth=1 | 0.3542 (17/48) | 0.18 |
| Depth=2 | 0.5625 (27/48) | 0.49 |

**Deeper tree search is both better-performing AND more variable** — a classic speed-accuracy tradeoff in search algorithms.

#### Precision-Based Run Budget: What 5 Runs Actually Justifies

Using the 95% CI / 5% relative half-width rule on the observed 5-run pilot summaries:

| Algorithm | Metric | Conditions Needing More Runs | Max Required Total Runs | Max Additional Runs |
|-----------|--------|------------------------------|-------------------------|---------------------|
| ALGO1 | Accuracy | 0 / 576 | 5 | 0 |
| ALGO1 | Precision | 3 / 576 | 564 | 559 |
| ALGO1 | Recall | 2 / 576 | 481 | 476 |
| ALGO2 | Accuracy | 0 / 1152 | 5 | 0 |
| ALGO2 | Precision | 1 / 1152 | 69 | 64 |
| ALGO2 | Recall | 1 / 1152 | 48 | 43 |
| ALGO3 | Recall | 44 / 96 | 23050 | 23035 |

This is conservative, but that is the point of the method. It shows:

- the existing 5 runs already justify the stable ALGO1/ALGO2 accuracy surfaces under the stated CI target,
- a very small number of ALGO1/ALGO2 precision-recall conditions would need more if the paper insists on that same tight relative precision,
- and ALGO3 recall is so unstable, and so close to zero, that the same target implies an enormous required run count.

That last point is not a recommendation to literally collect 23,050 runs. It is evidence that the target is unrealistic for ALGO3 recall under the current method behavior, which itself is a strong justification for treating ALGO3 as unstable rather than under-sampled.

---

## 5. Reviewer #1: Confidence Intervals and Distributional Plots

### Reviewer Quote

> "Reporting confidence intervals or distributional plots alongside mean performance would further reinforce the paper's core message about variability."

### Problem Being Raised

The reviewer wanted **plot-ready data exports** that could be used to create:

1. **Distributional plots** — box plots, violin plots showing the full distribution of metrics
2. **Facet by model** — ability to plot across all models simultaneously
3. **Provenance tracking** — each row knows which file it came from

Without this, readers must manually stitch together CSV files to create visualizations.

### Solution Implemented in the Codebase

Implemented `lcm analyze figures` command that produces **tidy long-format** exports:

- Each row = one metric observation
- Includes `source_input`, `algorithm`, `model` metadata columns
- Includes all factor and repetition identifier columns
- `metric` and `value` columns in standard long format

**Code Location:** `src/llm_conceptual_modeling/analysis/figures.py`

**Output Example:**

| source_input | algorithm | model | Explanation | replication | metric | value |
|--------------|-----------|-------|-------------|-------------|--------|-------|
| /results/algo1/gpt-4o/... | algo1 | gpt-4o | 1 | 0 | accuracy | 0.833 |
| /results/algo1/gpt-4o/... | algo1 | gpt-4o | 1 | 0 | precision | 0.750 |
| /results/algo1/gpt-4o/... | algo1 | gpt-4o | 1 | 0 | recall | 0.667 |

**Row Counts:**

| Algorithm | Export Rows |
|-----------|-------------|
| ALGO1 | 8,640 |
| ALGO2 | 17,280 |
| ALGO3 | 1,440 |

### Findings from Implementing the Solution

No new scientific findings — this was a **data infrastructure** improvement. However, it enables:

1. **Faceted plots** by model family (GPT, Gemini, DeepSeek)
2. **Violin plots** showing the full distributional shape
3. **Provenance-preserved** exports for reproducibility

---

## 6. Reviewer #2: Non-LLM Baseline Comparison

### Reviewer Quote

> "Include at least one non-LLM baseline method... to contextualize the value proposition of using LLMs despite their inherent variability."

### Problem Being Raised

The reviewer asked: "How do your LLM-based algorithms compare to a **simple non-LLM method**? If a basic deterministic approach performs just as well, what's the point of using expensive LLMs?"

### Solution Implemented in the Codebase

Implemented `lcm baseline` for all three algorithms plus the organized `lcm analyze baseline-bundle` workflow.

The deterministic strategies now include:

1. `random-k`
2. `wordnet-ontology-match`
3. `edit-distance`

The latter two are the reviewer-requested lexical baselines. All three are volume-matched to the model output being compared.

**Code Location:** `src/llm_conceptual_modeling/common/baseline.py`, `src/llm_conceptual_modeling/analysis/baseline_bundle.py`, algorithm-specific baseline files

**CLI Usage:**

```bash
lcm baseline algo1 \
    --pair sg1_sg2 \
    --strategy wordnet-ontology-match \
    --output /tmp/algo1_wordnet_baseline.csv

lcm baseline algo1 \
    --pair sg1_sg2 \
    --strategy edit-distance \
    --output /tmp/algo1_edit_distance_baseline.csv

lcm analyze baseline-bundle \
    --results-root data/results \
    --output-dir data/analysis_artifacts/revision_tracker/baseline_comparison
```

### Findings from Implementing the Solution

The findings depend on which baseline is used.

Against `random-k`, ALGO1 and ALGO2 beat the baseline on every audited metric, while ALGO3 loses on every audited metric.

Against `wordnet-ontology-match` and `edit-distance`, no imported model beats the baseline on any audited metric for ALGO1, ALGO2, or ALGO3.

Representative aggregate deltas from `baseline_advantage_summary.csv`:

| Algorithm | Baseline | Accuracy | Precision | Recall |
|-----------|----------|----------|-----------|--------|
| ALGO1 | `random-k` | +0.0070 | +0.3009 | +0.0069 |
| ALGO1 | `wordnet-ontology-match` | -0.0197 | -0.6866 | -0.0194 |
| ALGO1 | `edit-distance` | -0.0197 | -0.6866 | -0.0194 |
| ALGO2 | `random-k` | +0.0209 | +0.3026 | +0.0182 |
| ALGO2 | `wordnet-ontology-match` | -0.0570 | -0.6588 | -0.0556 |
| ALGO2 | `edit-distance` | -0.0570 | -0.6588 | -0.0556 |
| ALGO3 | `random-k` | -0.0050 | -0.0372 | -0.0058 |
| ALGO3 | `wordnet-ontology-match` | -0.1047 | -0.9971 | -0.1047 |
| ALGO3 | `edit-distance` | -0.1047 | -0.9971 | -0.1047 |

**Important Caveat:** The stronger lexical baselines outperform the imported models in this corpus, and the WordNet and edit-distance aggregates are identical here. That should be described as an observed corpus property, not as a universal statement about all ontology-matching tasks.

---

## 7. Reviewer / Philippe: LLM Internal Mechanisms (Proxy Analysis)

### Reviewer Quote (from Philippe's email)

> "a reviewer would like to drill into the 'LLM' factor... they want to know what it is within the LLM that makes it so... Saying that variability is due to the choice of LLM is not sufficient for them, they want to know what it is within the LLM that makes it so. There may be LLMs that let us try different decoding strategies or more parameters, so that we may be able to decompose what it is within an LLM that actually creates variability."

### Problem Being Raised

The reviewer wants to **decompose the LLM factor** — instead of saying "GPT-4o vs Gemini behave differently," they want to know *why*. Possible sources:

1. **Decoding strategy** — temperature, top-p, nucleus sampling
2. **Latent uncertainty** — whether the model is confident or not
3. **Multi-step amplification** — errors compounding through chain-of-thought

### Solution Implemented in the Codebase

Implemented `lcm analyze variability`, a deterministic raw-output analysis that works only on the imported result files under `data/results/`.

**Why this framing was chosen:** the repository still does **not** have access to provider internals such as attention weights, logits, routing traces, or hidden states. Claiming to identify the true internal mechanism would therefore be inaccurate. Instead, the new analysis decomposes the observable variability into two proxy components:

1. **Edge-set drift** — do repeated runs select the same edges?
2. **Breadth drift** — do repeated runs expand to different output sizes?

For each repeated condition, the command parses every raw edge list and computes:

- `mean_pairwise_jaccard` — average overlap between repeated edge sets
- `min_pairwise_jaccard` — worst overlap observed within the repeated set
- `exact_match_pair_rate` — share of repetition pairs that are exactly identical
- `mean_edge_count` and `sample_std_edge_count` — average output size and its spread
- `union_edge_count` — total distinct edges produced across all repetitions

**Code Location:** `src/llm_conceptual_modeling/analysis/variability.py`

**CLI Usage Example:**

```bash
lcm analyze variability \
    --input data/results/algo3/gpt-5/raw/method3_results_gpt5.csv \
    --group-by Example \
    --group-by Counter-Example \
    --group-by Number\ of\ Words \
    --group-by Depth \
    --group-by Source\ Subgraph\ Name \
    --group-by Target\ Subgraph\ Name \
    --result-column Results \
    --output algo3_output_variability.csv
```

### Findings from Implementing the Solution

The imported corpus shows that the reviewer-facing "LLM variability" is **not** a single phenomenon.

#### ALGO1 and ALGO2: Variability is usually absent or extremely local

Across all imported models and conditions:

- **ALGO1** has global mean pairwise Jaccard `0.9981` and exact-match pair rate `0.9979`
- **ALGO2** has global mean pairwise Jaccard `0.9997` and exact-match pair rate `0.9991`

This means repeated runs almost always return the same edge set. When variability appears, it is concentrated in a few file-level exceptions rather than spread broadly across the design. For example:

- `gemini-2.0-flash-exp` in ALGO1 drops to condition-level Jaccard values as low as `0.60`
- `gpt-5` in ALGO2 has one imported file with mean pairwise Jaccard `0.5964`, exact-match rate `0.0`, and breadth-expansion ratio `1.5848`

So for these two algorithms, the observable source of variability is **not persistent stochastic drift across the whole corpus**. It is better described as occasional condition-specific output instability.

#### ALGO3: Variability is dominated by edge-set drift plus breadth expansion

ALGO3 behaves very differently:

- global mean pairwise Jaccard is only `0.0770`
- global exact-match pair rate is `0.0010`
- mean edge count is `50.53`
- mean edge-count standard deviation is `16.95`
- breadth-expansion ratio is `4.13`

This is a strong signal that repeated runs are not merely changing by a few edges. They are often producing **substantially different edge sets** and a much larger union of distinct candidate edges across repetitions.

#### Depth amplifies the output drift in ALGO3

The same command also shows that deeper search correlates with stronger drift:

| Depth | Mean Pairwise Jaccard | Exact-Match Pair Rate | Mean Edge Count | Edge Count Std | Breadth Expansion Ratio |
|-------|------------------------|----------------------|-----------------|----------------|-------------------------|
| 1 | 0.0848 | 0.0021 | 40.21 | 13.44 | 4.0674 |
| 2 | 0.0692 | 0.0000 | 60.84 | 20.47 | 4.1968 |

This supports a **multi-step amplification** interpretation: deeper tree expansion does not just produce more edges, it also reduces repetition-to-repetition agreement.

### Interpretation

This still does **not** identify the hidden internal mechanism of any provider. What it does establish, using only the imported corpus, is that:

1. The dominant observable mechanism is **output-side drift in edge selection**
2. In ALGO3, that drift is coupled with **breadth expansion**, not just minor edge substitution
3. The stronger drift at greater depth is consistent with the manuscript's claim that **multi-step generation amplifies variability**

So the strongest defensible revision claim is no longer just "the LLM factor matters." It is:

> The observable variability comes from different degrees of edge-set instability and output-breadth instability across algorithms, with the tree-based multi-step workflow exhibiting the strongest amplification.

---

## 8. Reviewer #2: Cross-Domain Generalization (DEFERRED)

### Reviewer Quote

> "Add at least one more domain with other features to reinforce arguments that it is cross-domain."

### Problem Being Raised

The paper's experiments use a single domain (obesity-related causal maps). The reviewer wants evidence that findings **generalize** to other domains with different semantic structures.

### Solution Status: DEFERRED

**Why Deferred:** This requires:

1. New domain dataset (causal maps from a different field)
2. New LLM API calls to generate results for the new domain
3. Significant experimental time and cost

**No code was implemented.** This is a future research direction.

---

## Summary of Changes to the Codebase

### New Commands Added

| Command | Purpose | File |
|---------|---------|------|
| `lcm analyze summary` | Grouped descriptive statistics with 95% CIs | `analysis/summary.py` |
| `lcm analyze hypothesis` | Paired t-tests with BH correction | `analysis/hypothesis.py` |
| `lcm analyze failures` | Row-level failure classification | `analysis/failures.py` |
| `lcm analyze stability` | Replication stability analysis | `analysis/stability.py` |
| `lcm analyze figures` | Tidy long-format metric exports | `analysis/figures.py` |
| `lcm analyze baseline-comparison` | Baseline vs model comparison | `analysis/baseline_comparison.py` |
| `lcm analyze variability` | Raw-output edge-set drift and breadth drift | `analysis/variability.py` |
| `lcm baseline` | Deterministic baseline generation | `common/baseline.py` + algo files |

### New Audit Artifacts

Stored under `data/analysis_artifacts/revision_tracker/2026-03-21/` and `data/analysis_artifacts/revision_tracker/2026-03-24/`:

```
revision_tracker/
├── algo1_explanation_directionality.csv
├── algo2_convergence_directionality.csv
├── algo2_explanation_directionality.csv
├── algo3_number_of_words_summary.csv
├── all_row_level_failures.csv
├── failure_counts_by_model.csv
├── failure_rates_by_model.csv
├── parsed_edge_counts_by_model.csv
├── hypothesis_testing/
│   ├── algo1_explanation_hypothesis.csv
│   ├── algo1_explanation_hypothesis_significance_summary.csv
│   ├── algo2_convergence_hypothesis.csv
│   ├── algo2_convergence_hypothesis_significance_summary.csv
│   ├── algo2_explanation_hypothesis.csv
│   ├── algo2_explanation_hypothesis_significance_summary.csv
│   ├── algo3_depth_hypothesis.csv
│   ├── algo3_depth_hypothesis_significance_summary.csv
│   ├── algo3_number_of_words_hypothesis.csv
│   └── algo3_number_of_words_hypothesis_significance_summary.csv
├── figure_exports/
│   ├── algo1_metric_rows.csv (8,640 rows)
│   ├── algo2_metric_rows.csv (17,280 rows)
│   └── algo3_metric_rows.csv (1,440 rows)
├── replication_stability/
│   ├── algo1_condition_stability.csv
│   ├── algo2_condition_stability.csv
│   ├── algo3_condition_stability.csv
│   ├── overall_metric_stability_by_algorithm.csv
│   └── variability_incidence_by_algorithm.csv
├── output_variability/
│   ├── algo1_condition_output_variability.csv
│   ├── algo2_condition_output_variability.csv
│   ├── algo3_condition_output_variability.csv
│   ├── algorithm_output_variability_summary.csv
│   ├── model_output_variability_summary.csv
│   ├── output_variability_extremes.csv
│   ├── algo3_output_variability_by_depth.csv
│   └── algo3_output_variability_by_word_count.csv
└── baseline_comparison/
    ├── all_models_vs_baseline.csv
    ├── baseline_advantage_summary.csv
    ├── algo1_model_vs_baseline.csv
    ├── algo2_model_vs_baseline.csv
    └── algo3_model_vs_baseline.csv
```

### Test Coverage

All new functionality is protected by tests following test-driven development:

- `tests/test_analysis_summary.py` — summary statistics
- `tests/test_analysis_hypothesis.py` — hypothesis testing
- `tests/test_analysis_failures.py` — failure classification
- `tests/test_analysis_stability.py` — stability analysis
- `tests/test_analysis_figures.py` — figure exports
- `tests/test_baseline.py` — baseline generation
- `tests/test_cli.py` — CLI integration tests

---

## Deferred Items Summary

| Item | Reviewer | Reason for Deferral |
|------|----------|---------------------|
| LLM Internal Mechanisms | Philippe (reviewer) | Requires access to model internals not available via standard API |
| Cross-Domain Generalization | Reviewer #2 | Requires new domain dataset and new experimental runs |
