# Revision Summary

This document is the self-contained software record of the revision work completed for the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

It is designed to answer five questions in one place:

1. What did the reviewers ask for, in their own words?
2. What code or command surface was added in response?
3. Which command actually produced the evidence?
4. What did the resulting artifacts show, in concrete numbers?
5. How should those findings be interpreted in relation to the paper's claims?

## Scope And Evidence Boundary

This document covers the deterministic, offline revision work that is reproducible from the current repository without rerunning live provider calls.

- Primary imported corpus: `data/results/`
- Audited revision artifacts:
  - `data/analysis_artifacts/revision_tracker/2026-03-21/`
  - `data/analysis_artifacts/revision_tracker/2026-03-24/`
- Main command surface:
  - `lcm analyze summary`
  - `lcm analyze hypothesis`
  - `lcm analyze failures`
  - `lcm analyze stability`
  - `lcm analyze figures`
  - `lcm analyze variability`
  - `lcm baseline`
  - `lcm analyze baseline-comparison`

Important limitation: this document does not claim direct observation of hidden states, logits, or attention weights inside provider-hosted LLMs. The "internal variability" response is therefore an output-side proxy analysis over repeated raw results.

## Executive Summary

The revision work substantially strengthened the repository's reviewer-facing evidence layer.

- It added grouped descriptive statistics and confidence intervals.
- It added paired hypothesis testing with multiple-comparison correction.
- It added deterministic failure classification over the full raw corpus.
- It added replication-stability analysis across the five recorded repetitions.
- It added plot-ready long-format exports with preserved provenance.
- It added a new raw-output variability analysis that directly quantifies edge-set drift.
- It added a deterministic non-LLM structural baseline.

The main empirical message sharpened by the revision work is that variability is not uniform across methods.

- ALGO1 is very stable.
- ALGO2 is also very stable, especially when `Convergence=1`.
- ALGO3 is qualitatively different: it is both less stable on evaluated recall and far less stable at the raw edge-set level.

The strongest new reviewer-facing conclusion is therefore not just "LLMs are variable," but rather:

> In this imported corpus, variability is method-dependent, concentrated in ALGO3, and expressed mainly as drift within structurally valid outputs rather than parser-visible failure.

## Revision Matrix

| Revision theme | Exact reviewer concern | Repository response | Main finding |
| --- | --- | --- | --- |
| Statistical reporting | Missing confidence intervals and grouped summaries | Added `lcm analyze summary` | Factor-level descriptive patterns are now explicit and auditable |
| Formal testing | Missing p-values and multiple-comparison adjustment | Added `lcm analyze hypothesis` with Benjamini-Hochberg correction | ALGO2 `Convergence` is the strongest tested factor in the imported corpus |
| Failure understanding | Need reasons behind failures, not only successes | Added `lcm analyze failures` | Imported raw outputs are overwhelmingly valid; the issue is variability, not parse failure |
| Replication justification | Five repetitions not justified | Added `lcm analyze stability` | ALGO1 and ALGO2 are nearly repetition-stable; ALGO3 is not |
| Plot-ready reporting | Need distributional and figure-ready outputs | Added `lcm analyze figures` | The imported corpus can now be plotted directly with preserved provenance |
| Internal variability explanation | Randomness shown, but mechanism unclear | Added `lcm analyze variability` | ALGO3 shows strong edge-set drift and breadth expansion across repetitions |
| Non-LLM comparator | Need at least one baseline | Added `lcm baseline` and `lcm analyze baseline-comparison` | No imported model beat the deterministic baseline on the audited metrics |
| Cross-domain reinforcement | Need at least one additional domain | Deferred | No new domain data was added in this software tranche |

## 1. Statistical Reporting And Confidence Intervals

### Exact Reviewer Quote

> "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."
>
> "Add formal statistical testing with multiple comparison adjustments, provide confidence intervals of all performance measures..."

### What The Reviewer Was Asking For

The repository originally produced raw evaluated metric files, but it did not produce reviewer-facing summaries that could answer questions such as:

- What is the mean effect of a factor level?
- How much spread is there around that mean?
- Are the apparent differences stable enough to report with confidence intervals?

### What Was Implemented

The revision added `lcm analyze summary`, which computes grouped descriptive statistics from evaluated CSVs.

For each metric and grouping, the command exports:

- `n`
- `mean`
- `sample_std`
- `median`
- `min`
- `max`
- `ci95_low`
- `ci95_high`

### Commands Used

Representative audited commands:

```bash
lcm analyze summary \
  --input data/results/algo1/*/evaluated/*.csv \
  --group-by Explanation \
  --metrics accuracy precision recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/algo1_explanation_directionality.csv

lcm analyze summary \
  --input data/results/algo2/*/evaluated/*.csv \
  --group-by Convergence \
  --metrics accuracy precision recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/algo2_convergence_directionality.csv

lcm analyze summary \
  --input data/results/algo3/*/evaluated/*.csv \
  --group-by Depth \
  --metrics Recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/algo3_depth_summary.csv
```

### Most Informative Output

ALGO1 `Explanation` directionality:

| Metric | Winner count across 18 file-level comparisons |
| --- | --- |
| Accuracy | `Explanation=1` won in 11 of 18 |
| Precision | `Explanation=1` won in 12 of 18 |
| Recall | `Explanation=-1` won in 13 of 18 |

ALGO2 `Convergence` directionality:

| Metric | Winner count across 18 file-level comparisons |
| --- | --- |
| Accuracy | `Convergence=-1` won in 13 of 18 |
| Precision | `Convergence=-1` won in 12 of 18 |
| Recall | `Convergence=1` won in 13 of 18 |

ALGO2 `Explanation` directionality:

| Metric | Winner count across 18 file-level comparisons |
| --- | --- |
| Accuracy | split 9 to 9 |
| Precision | split 9 to 9 |
| Recall | split 9 to 9 |

ALGO3 `Depth` summary:

| Depth | Global mean recall |
| --- | --- |
| `Depth=2` | `0.0652` |
| `Depth=1` | `0.0144` |

ALGO3 `Number of Words` summary:

| Number of Words | Global mean recall |
| --- | --- |
| `5` | `0.0478` |
| `3` | `0.0318` |

### Detailed Findings

The descriptive layer made several patterns much clearer than before.

- In ALGO1, adding explanation text was usually associated with better accuracy and precision, but not with better recall.
- In ALGO2, the `Convergence` factor behaved like a precision-recall tradeoff:
  - `Convergence=-1` more often improved accuracy and precision.
  - `Convergence=1` more often improved recall.
- In ALGO2, `Explanation` was much weaker than `Convergence`; there was no dominant direction across the imported files.
- In ALGO3, increasing depth from `1` to `2` improved mean recall for every model except GPT-5, where both levels remained at zero.
- For `Number of Words`, the global direction favored `5`, but it was not universal:
  - DeepSeek Chat v3.1 improved from `0.0284` to `0.0871`
  - Gemini 2.0 Flash improved from `0.0273` to `0.0860`
  - Google Gemini 2.5 Pro stayed flat at `0.0303`
  - GPT-5 stayed flat at `0.0000`
  - GPT-4o decreased slightly from `0.0898` to `0.0833`

### Interpretation

The main value of this revision item is not just that confidence intervals now exist. It is that the repository can now support more precise claims:

- some factors show directional effects,
- some factors primarily induce tradeoffs,
- and some factors are much weaker than they first appear from raw metric tables.

## 2. Formal Hypothesis Testing

### Exact Reviewer Quote

> "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."
>
> "Add formal statistical testing with multiple comparison adjustments..."

### What The Reviewer Was Asking For

The reviewers wanted the paper to move beyond descriptive patterns into formal tests with:

- explicit p-values,
- explicit pairing structure,
- and some control for multiple testing.

### What Was Implemented

The revision added `lcm analyze hypothesis`, which performs paired two-level tests and applies Benjamini-Hochberg correction within each output file.

The command emits:

- `pair_count`
- `mean_low`
- `mean_high`
- `mean_difference`
- `t_statistic`
- `p_value`
- `p_value_adjusted`
- `correction_method`

### Commands Used

Representative audited commands:

```bash
lcm analyze hypothesis \
  --input data/results/algo1/*/evaluated/*.csv \
  --factor Explanation \
  --pair-by Repetition Example Counterexample Array/List(1/-1) Tag/Adjacency(1/-1) \
  --metrics accuracy precision recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/algo1_explanation_hypothesis.csv

lcm analyze hypothesis \
  --input data/results/algo2/*/evaluated/*.csv \
  --factor Convergence \
  --pair-by Repetition Explanation Example Counterexample Array/List(1/-1) Tag/Adjacency(1/-1) \
  --metrics accuracy precision recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/algo2_convergence_hypothesis.csv
```

### Most Informative Output

ALGO2 `Convergence` significance summary:

| Metric | Significant tests after BH correction | Dominant direction |
| --- | --- | --- |
| Accuracy | 15 of 18 | mostly `high_lt_low`, meaning `Convergence=-1` wins |
| Precision | 14 of 18 | mostly `high_lt_low`, meaning `Convergence=-1` wins |
| Recall | 15 of 18 | mostly `high_gt_low`, meaning `Convergence=1` wins |

ALGO1 `Explanation` significance summary:

| Metric | Significant tests after BH correction | Directional pattern |
| --- | --- | --- |
| Accuracy | 8 of 18 | mostly toward `Explanation=1` |
| Precision | 9 of 18 | mixed, but more favorable to `Explanation=1` |
| Recall | 8 of 18 | mostly toward `Explanation=-1` |

ALGO3 factor testing:

| Factor | Significant tests after BH correction |
| --- | --- |
| `Depth` | 2 of 6, both favoring `Depth=2` |
| `Number of Words` | 0 of 6 |

### Detailed Findings

This revision item shows that the descriptive patterns were not all equally strong.

- ALGO2 `Convergence` was the strongest factor tested in the imported corpus:
  - `44` of `54` metric-level tests were significant after correction.
- ALGO1 `Explanation` was weaker:
  - `25` of `54` tests were significant after correction.
- ALGO2 `Explanation` was weaker again:
  - `21` of `54` tests were significant after correction.
- ALGO3's `Depth` effect existed, but it was much less uniform than the ALGO2 `Convergence` effect.
- ALGO3's `Number of Words` effect did not survive correction at all.

### Interpretation

The main contribution here is not simply the presence of p-values. It is that the codebase can now separate:

- strong factor effects from weak ones,
- stable directional factors from mixed ones,
- and effects that are visually plausible from effects that remain statistically fragile.

## 3. Failure Analysis

### Exact Reviewer Quote

> "It is important to know the reasons behind the failures as much as successful runs. Record the types of particular errors, the deterministic or stochastic nature of failures, and the systematic failure of particular combinations of the LLM-algorithm."

### What The Reviewer Was Asking For

The reviewer was explicitly asking the paper to distinguish between:

- malformed or unusable outputs,
- empty outputs,
- and valid but low-quality outputs.

Without that distinction, "variability" and "failure" would be conflated.

### What Was Implemented

The revision added `lcm analyze failures`, which classifies raw rows into reproducible categories and records parsed edge counts.

The main output columns are:

- `failure_category`
- `parsed_edge_count`
- `is_failure`

### Commands Used

Representative audited command:

```bash
lcm analyze failures \
  --input data/results/algo1/*/raw/*.csv \
  --input data/results/algo2/*/raw/*.csv \
  --input data/results/algo3/*/raw/*.csv \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/all_row_level_failures.csv
```

### Most Informative Output

Failure counts by algorithm and model:

| Algorithm | Models | Total rows | Malformed rows | Empty rows | Valid rows |
| --- | --- | --- | --- | --- | --- |
| ALGO1 | 6 | `2,880` | `0` | `0` | `2,880` |
| ALGO2 | 6 | `5,760` | `0` | `0` | `5,760` |
| ALGO3 | 6 | `1,440` | `0` | `0` | `1,440` |

Parsed edge-count examples:

| Algorithm | Model | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| ALGO1 | GPT-5 | `28.69` | `28` | `12` | `48` |
| ALGO2 | GPT-4o | `84.55` | `15` | `2` | `691` |
| ALGO2 | Gemini 2.0 Flash | `28.88` | `25` | `4` | `398` |
| ALGO3 | DeepSeek V3 Chat 0324 | `68.71` | `58` | `12` | `355` |
| ALGO3 | GPT-4o | `21.39` | `18` | `1` | `81` |

### Detailed Findings

This result is important because it is somewhat counterintuitive.

- The imported primary raw corpus contained `10,080` rows classified as `valid_output`.
- It contained `0` malformed rows.
- It contained `0` empty rows.
- Failure rate was therefore `0.0` for every imported algorithm-model combination.

What varied sharply was not parseability but output breadth.

- ALGO1 outputs were relatively compact, with means ranging from `8.26` to `28.69` edges depending on model.
- ALGO2 outputs were far more spread out:
  - Gemini 2.0 Flash mean: `28.88`
  - GPT-5 mean: `50.85`
  - GPT-4o mean: `84.55`
- The GPT-4o ALGO2 distribution is especially informative:
  - median `15`
  - mean `84.55`
  - maximum `691`

That combination strongly suggests a right-skewed distribution with a small number of very large outputs.

### Interpretation

The failure analysis narrows the real revision problem considerably.

The paper does not mainly face a "the models often break formatting" problem. It faces a "the models produce different valid outputs, sometimes with radically different breadth" problem.

## 4. Replication Stability

### Exact Reviewer Quote

> "the manuscript would be strengthened by ... a principled justification of the number of replications, for example via power or precision considerations rather than feasibility alone."
>
> "The decision to use 5 replications does not seem to be justified by power analysis."

### What The Reviewer Was Asking For

The key question was whether five repetitions are enough to characterize run-to-run variability.

The strongest software-side answer available from existing data is to measure how much the evaluated metrics actually move across the five recorded repetitions.

### What Was Implemented

The revision added `lcm analyze stability`, which computes repetition-level stability summaries over evaluated files.

The exported stability fields include:

- `n`
- `mean`
- `sample_std`
- `min`
- `max`
- `range_width`
- `coefficient_of_variation`

### Commands Used

Representative audited commands:

```bash
lcm analyze stability \
  --input data/results/algo1/*/evaluated/*.csv \
  --group-by model metric Explanation Example Counterexample Array/List(1/-1) Tag/Adjacency(1/-1) \
  --value-column value \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/replication_stability/algo1_condition_stability.csv

lcm analyze stability \
  --input data/results/algo3/*/evaluated/*.csv \
  --group-by model Recall Depth Number\ of\ Words Example Counter-Example Source\ Subgraph\ Name Target\ Subgraph\ Name \
  --value-column Recall \
  --output data/analysis_artifacts/revision_tracker/2026-03-21/replication_stability/algo3_condition_stability.csv
```

### Most Informative Output

Cross-algorithm variability incidence:

| Algorithm | Metric | Varying conditions | Total conditions | Share |
| --- | --- | --- | --- | --- |
| ALGO1 | Accuracy | `2` | `576` | `0.0035` |
| ALGO1 | Precision | `3` | `576` | `0.0052` |
| ALGO1 | Recall | `2` | `576` | `0.0035` |
| ALGO2 | Accuracy | `1` | `1152` | `0.0009` |
| ALGO2 | Precision | `1` | `1152` | `0.0009` |
| ALGO2 | Recall | `1` | `1152` | `0.0009` |
| ALGO3 | Recall | `44` | `96` | `0.4583` |

Cross-algorithm CV summary:

| Algorithm | Metric | Mean CV | Median CV | Mean range width |
| --- | --- | --- | --- | --- |
| ALGO1 | Accuracy | `0.000024` | `0.0` | `0.000048` |
| ALGO1 | Precision | `0.002346` | `0.0` | `0.001215` |
| ALGO2 | Accuracy | `0.000015` | `0.0` | `0.000037` |
| ALGO2 | Recall | `0.000180` | `0.0` | `0.000137` |
| ALGO3 | Recall | `3.211013` | `3.872983` | `0.336648` |

Level-specific stability:

| Condition | Varying-condition share |
| --- | --- |
| ALGO2 `Convergence=-1` | `0.001736` per metric |
| ALGO2 `Convergence=1` | `0.000000` per metric |
| ALGO3 `Depth=1` | `0.354167` |
| ALGO3 `Depth=2` | `0.562500` |

### Detailed Findings

The most important result is the separation between ALGO1/ALGO2 and ALGO3.

- ALGO1 and ALGO2 are almost completely stable on evaluated metrics across the five recorded repetitions.
- ALGO2 is especially stable when `Convergence=1`: no varying conditions at all in the audited incidence table.
- ALGO3 is not merely a little noisier; it is orders of magnitude noisier on evaluated recall.

The CV table is particularly informative.

- ALGO1 and ALGO2 mostly have median CVs of `0.0`.
- ALGO3's median CV is `3.872983`.

This means that the typical ALGO3 condition varies at a scale far larger than what is seen in ALGO1 or ALGO2.

### Interpretation

This revision item does not provide a formal power analysis. It does something more concrete for the existing corpus:

it shows that five repetitions are enough to reveal that ALGO1 and ALGO2 are already essentially stable, while ALGO3 is not.

## 5. Figure-Ready Exports

### Exact Reviewer Quote

> "Reporting confidence intervals or distributional plots alongside mean performance would further reinforce the paper's core message about variability."

### What The Reviewer Was Asking For

The reviewer needed the repository to support plotting and distributional analysis directly, instead of leaving figure assembly as a manual step outside the audited workflow.

### What Was Implemented

The revision added `lcm analyze figures`, which converts evaluated CSVs into long-format metric rows with explicit provenance.

Each row preserves:

- `source_input`
- `algorithm`
- `model`
- factor columns from the original evaluated rows
- metric name
- metric value

### Commands Used

Representative audited command:

```bash
lcm analyze figures \
  --input data/results/algo1/*/evaluated/*.csv \
  --input data/results/algo2/*/evaluated/*.csv \
  --input data/results/algo3/*/evaluated/*.csv \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/figure_exports/
```

### Most Informative Output

Export sizes:

| File | Rows |
| --- | --- |
| `algo1_metric_rows.csv` | `8,640` |
| `algo2_metric_rows.csv` | `17,280` |
| `algo3_metric_rows.csv` | `1,440` |

Per-model row counts are perfectly regular, which is itself useful:

- ALGO1: each model contributes `480` rows per metric.
- ALGO2: each model contributes `960` rows per metric.
- ALGO3: each model contributes `240` recall rows.

### Detailed Findings

This item is more about infrastructure than new scientific claims, but the resulting exports matter for interpretability.

- The exports can now support box plots, violin plots, faceted model comparisons, and confidence interval plots directly.
- The model provenance is already embedded in the rows, so downstream figures do not need ad hoc filename parsing.
- The regular per-model row counts also make the exported corpus easier to audit for omissions.

### Interpretation

The repository is no longer limited to summary tables. It now supports figure generation from a deterministic, provenance-preserving analysis layer.

## 6. Output Variability And Proxy Mechanisms

### Exact Reviewer Quote

> "the manuscript stops short of linking these differences to specific computational mechanisms."
>
> "The article shows that there is randomness but it is not really telling how and why this happens..."
>
> "a reviewer would like to drill into the 'LLM' factor... they want to know what it is within the LLM that makes it so... Saying that variability is due to the choice of LLM is not sufficient for them, they want to know what it is within the LLM that makes it so. There may be LLMs that let us try different decoding strategies or more parameters, so that we may be able to decompose what it is within an LLM that actually creates variability."

### What The Reviewer Was Asking For

The reviewer was asking for a mechanism-level explanation of variability, not just a model-name-level comparison.

Because this software tranche is restricted to imported outputs under `data/results/`, the strongest available response is a proxy mechanism analysis:

- Do repeated runs preserve the same edge set?
- If not, how much do they drift?
- Do they differ only slightly, or do they expand into very different unions of edges?

### What Was Implemented

The revision added `lcm analyze variability`, which works only from raw repeated outputs already stored in `data/results/`.

The command measures:

- `mean_pairwise_jaccard`
- `min_pairwise_jaccard`
- `exact_match_pair_rate`
- `mean_edge_count`
- `sample_std_edge_count`
- `union_edge_count`
- derived `breadth_expansion_ratio`

### Commands Used

Representative audited commands:

```bash
lcm analyze variability \
  --input data/results/algo1/*/raw/*.csv \
  --group-by model \
  --result-column result \
  --output data/analysis_artifacts/revision_tracker/2026-03-24/output_variability/algo1_condition_output_variability.csv

lcm analyze variability \
  --input data/results/algo2/*/raw/*.csv \
  --group-by model \
  --result-column result \
  --output data/analysis_artifacts/revision_tracker/2026-03-24/output_variability/algo2_condition_output_variability.csv

lcm analyze variability \
  --input data/results/algo3/*/raw/*.csv \
  --group-by model \
  --result-column Result \
  --output data/analysis_artifacts/revision_tracker/2026-03-24/output_variability/algo3_condition_output_variability.csv
```

### Most Informative Output

Cross-algorithm summary:

| Algorithm | Mean pairwise Jaccard | Exact-match pair rate | Mean edge count | Edge-count std | Breadth expansion ratio |
| --- | --- | --- | --- | --- | --- |
| ALGO1 | `0.998056` | `0.997917` | `14.046` | `0.005` | `1.0039` |
| ALGO2 | `0.999650` | `0.999132` | `34.885` | `0.006` | `1.0005` |
| ALGO3 | `0.076985` | `0.001042` | `50.525` | `16.954` | `4.1321` |

ALGO3 by depth:

| Depth | Mean pairwise Jaccard | Exact-match pair rate | Mean edge count | Breadth expansion ratio |
| --- | --- | --- | --- | --- |
| `1` | `0.084754` | `0.002083` | `40.208` | `4.0674` |
| `2` | `0.069216` | `0.000000` | `60.842` | `4.1968` |

ALGO3 by number of words:

| Number of Words | Mean pairwise Jaccard | Mean edge count |
| --- | --- | --- |
| `3` | `0.080223` | `36.142` |
| `5` | `0.073747` | `64.908` |

Model-level ALGO3 variability:

| Model | Mean pairwise Jaccard | Exact-match pair rate | Mean edge count |
| --- | --- | --- | --- |
| DeepSeek V3 Chat 0324 | `0.023565` | `0.000000` | `68.713` |
| DeepSeek Chat v3.1 | `0.053751` | `0.002083` | `43.196` |
| Gemini 2.0 Flash | `0.084758` | `0.000000` | `53.417` |
| GPT-5 | `0.092849` | `0.002083` | `49.896` |
| GPT-4o | `0.098106` | `0.002083` | `21.388` |
| Google Gemini 2.5 Pro | `0.108881` | `0.000000` | `66.542` |

### Detailed Findings

This was the most important new analysis added after the earlier revision tranche.

ALGO1 and ALGO2:

- are almost exact-output stable at the raw edge-set level,
- show near-perfect exact-match rates,
- and have breadth-expansion ratios extremely close to `1.0`.

That means repeated runs usually return essentially the same edge set.

ALGO3:

- has a global mean pairwise Jaccard of only `0.0770`,
- has an exact-match pair rate of only `0.0010`,
- and expands to unions more than four times as large as the mean single-run output.

That means repeated ALGO3 runs are not merely selecting slightly different subsets of a common answer. They are wandering across substantially different edge sets.

The extremes file reinforces this interpretation.

- Several ALGO3 source files have condition-level mean Jaccard `0.000000`.
- The same files show breadth-expansion ratios of `5.000000`.
- One deepseek-v3-chat-0324 ALGO3 raw file reaches union edge count `453` with mean edge count `90.6`.

By contrast:

- ALGO1 only shows meaningful raw drift in a few Gemini 2.0 Flash and GPT-4o files.
- ALGO2 is effectively deterministic except for one GPT-5 raw file with mean Jaccard `0.596439` and breadth expansion `1.584821`.

### Interpretation

This does not identify hidden-state mechanisms directly. It does, however, support a much sharper process-level explanation:

- ALGO1 and ALGO2 do not materially amplify run-to-run output differences.
- ALGO3 does.
- Greater depth and larger word budgets make ALGO3 outputs both broader and less overlapping.

So the strongest repository-backed answer to the reviewer is:

> In the imported corpus, the most plausible mechanism is multi-step output amplification, not parser failure and not uniform LLM instability across all methods.

## 7. Non-LLM Baseline Comparison

### Exact Reviewer Quote

> "Include at least one non-LLM baseline method... to contextualize the value proposition of using LLMs despite their inherent variability."

### What The Reviewer Was Asking For

The reviewer wanted a simple deterministic comparator so that the paper's LLM-based methods could be interpreted against a non-LLM reference point.

### What Was Implemented

The revision added:

- `lcm baseline`
- `lcm analyze baseline-comparison`

The baseline is the deterministic `direct-cross-graph` heuristic.

### Commands Used

Representative audited commands:

```bash
lcm baseline algo1 --strategy direct-cross-graph --output data/baselines/direct-cross-graph/algo1/
lcm baseline algo2 --strategy direct-cross-graph --output data/baselines/direct-cross-graph/algo2/
lcm baseline algo3 --strategy direct-cross-graph --output data/baselines/direct-cross-graph/algo3/

lcm analyze baseline-comparison \
  --baseline data/baselines/direct-cross-graph \
  --inputs data/results/algo1/*/evaluated/*.csv \
  --inputs data/results/algo2/*/evaluated/*.csv \
  --inputs data/results/algo3/*/evaluated/*.csv \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/baseline_comparison/
```

### Most Informative Output

Baseline advantage summary:

| Algorithm | Metric | Models beating baseline |
| --- | --- | --- |
| ALGO1 | Accuracy | `0 of 6` |
| ALGO1 | Precision | `0 of 6` |
| ALGO1 | Recall | `0 of 6` |
| ALGO2 | Accuracy | `0 of 6` |
| ALGO2 | Precision | `0 of 6` |
| ALGO2 | Recall | `0 of 6` |
| ALGO3 | Recall | `0 of 6` |

Closest models to the baseline:

| Algorithm | Metric | Best model | Delta to baseline |
| --- | --- | --- | --- |
| ALGO1 | Accuracy | Gemini 2.5 Pro | `-0.021099` |
| ALGO1 | Recall | GPT-5 | `-0.027382` |
| ALGO2 | Accuracy | Gemini 2.5 Pro | `-0.020612` |
| ALGO2 | Recall | GPT-5 | `-0.102181` |
| ALGO3 | Recall | GPT-4o | `-0.701326` |

Worst deltas:

| Algorithm | Metric | Worst model | Delta to baseline |
| --- | --- | --- | --- |
| ALGO1 | Precision | DeepSeek V3 Chat 0324 | `-0.739825` |
| ALGO2 | Accuracy | GPT-4o | `-0.321233` |
| ALGO2 | Precision | Gemini 2.0 Flash | `-0.707423` |
| ALGO3 | Recall | GPT-5 | `-0.787879` |

### Detailed Findings

This result is stronger than a simple "baseline included" checkbox.

- No imported model exceeded the baseline on any audited metric.
- The gap is small only in a few places:
  - ALGO1 accuracy with Gemini 2.5 Pro
  - ALGO2 accuracy with Gemini 2.5 Pro
  - ALGO1 recall with GPT-5
- In many places the gap is large, especially for precision and for ALGO3 recall.

The full comparison table also shows that the baseline mean is the same reference target across all compared models within each matched file group, which makes these deltas easy to interpret.

### Interpretation

This result must be read carefully.

The baseline is structurally privileged because it uses mother-graph structure directly. So it is not a fair replacement for generative modeling in all research settings.

What it does show is that the imported LLM workflows, as captured in this corpus, do not outperform a simple deterministic structural heuristic on the audited downstream metrics.

## 8. Cross-Domain Generalization

### Exact Reviewer Quote

> "Add at least one more domain with other features to reinforce arguments that it is cross-domain."

### What The Reviewer Was Asking For

The reviewer wanted broader empirical scope, not just stronger post-processing over the existing domain.

### What Was Done

This item was deferred in the current software tranche because it requires new domain data and likely new raw runs rather than deterministic post-processing over the imported corpus.

### Findings

No new finding is claimed here.

## Commands And Artifacts At A Glance

This section is intended to make the document operational as well as interpretive.

| Revision item | Main command | Main audited outputs |
| --- | --- | --- |
| Descriptive summaries | `lcm analyze summary` | `algo1_explanation_directionality.csv`, `algo2_convergence_directionality.csv`, `algo3_depth_summary.csv` |
| Hypothesis testing | `lcm analyze hypothesis` | `hypothesis_testing/*.csv` |
| Failure analysis | `lcm analyze failures` | `all_row_level_failures.csv`, `failure_counts_by_model.csv`, `parsed_edge_counts_by_model.csv` |
| Replication stability | `lcm analyze stability` | `replication_stability/*.csv` |
| Figure exports | `lcm analyze figures` | `figure_exports/*.csv` |
| Raw output variability | `lcm analyze variability` | `output_variability/*.csv` |
| Non-LLM baseline | `lcm baseline`, `lcm analyze baseline-comparison` | `baseline_comparison/*.csv` |

## Practical Reading Of The Revision Work

Taken together, the implemented revision work supports six concrete conclusions.

1. The repository now supports reviewer-facing descriptive and inferential reporting rather than only raw metric tables.
2. Variability in the imported corpus is not uniform across methods.
3. The imported raw outputs fail almost never at the parser-visible level.
4. ALGO1 and ALGO2 are close to repetition-stable under the audited design.
5. ALGO3 is the main source of instability, both on evaluated recall and on raw edge-set overlap.
6. The deterministic structural baseline remains stronger than every imported LLM condition on the audited comparisons.

## Recommended Companion Files

- `paper/revision/reviewer-response-log.md`
- `paper/revision/revision-tracker.md`
- `data/analysis_artifacts/revision_tracker/2026-03-21/`
- `data/analysis_artifacts/revision_tracker/2026-03-24/`
- `docs/architecture.md`
