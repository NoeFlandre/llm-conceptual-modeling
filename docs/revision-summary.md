# Revision Summary

This document is a self-contained software record of the revision work completed for the manuscript:

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
  - `lcm analyze summary-bundle`
  - `lcm analyze hypothesis`
  - `lcm analyze hypothesis-bundle`
  - `lcm analyze failures`
  - `lcm analyze output-validity-bundle`
  - `lcm analyze stability`
  - `lcm analyze figures`
  - `lcm analyze variability`
  - `lcm baseline`
  - `lcm analyze baseline-comparison`

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
| Formal testing | Missing p-values and multiple-comparison adjustment | Added `lcm analyze hypothesis` and `lcm analyze hypothesis-bundle` with Benjamini-Hochberg FDR correction | Exhaustive 16-factor formal test confirms ALGO2 `Convergence` as dominant; ALGO3 has no robust factor effects |
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

This request was broader than "add one confidence interval." The reviewer was effectively asking for a descriptive-analysis layer that could support all of the following:

- per-factor summaries rather than raw row dumps,
- uncertainty information rather than means alone,
- clearer factor-level comparisons for every algorithm,
- and a structure that a reviewer can actually inspect without reverse-engineering ad hoc files.

### What Was Implemented In Code

Two pieces now answer this request.

1. The generic descriptive-analysis command:

- `lcm analyze summary`

This computes grouped descriptive statistics from evaluated CSVs and exports:

- `n` — count of observations in the group
- `mean` — arithmetic average of the metric values
- `sample_std` — standard deviation of the metric values (uses n−1 denominator)
- `median` — middle value when observations are sorted
- `min` — smallest observed value
- `max` — largest observed value
- `ci95_low` — lower boundary of the 95% confidence interval around the mean
- `ci95_high` — upper boundary of the 95% confidence interval around the mean

2. A new organized bundle generator for this reviewer item:

- `lcm analyze summary-bundle`

This new bundle command was added specifically to make the evidence exhaustive and navigable. It now generates:

- one subdirectory per algorithm,
- one subdirectory per reviewed factor,
- one `grouped_metric_summary.csv` per factor,
- one `metric_overview.csv` per factor,
- a top-level `bundle_manifest.csv`,
- a top-level `bundle_overview.csv`,
- and a local `README.md` explaining the layout.

Code locations:

- `src/llm_conceptual_modeling/analysis/summary.py`
- `src/llm_conceptual_modeling/analysis/summary_bundle.py`

### Commands Used

The organized reviewer-facing bundle is now produced by:

```bash
lcm analyze summary-bundle \
  --results-root data/results \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/statistical_reporting
```

That bundle internally expands the descriptive analysis across all audited factors:

- ALGO1:
  - `Explanation`
  - `Example`
  - `Counterexample`
  - `Array/List(1/-1)`
  - `Tag/Adjacency(1/-1)`
- ALGO2:
  - `Explanation`
  - `Example`
  - `Counterexample`
  - `Array/List(1/-1)`
  - `Tag/Adjacency(1/-1)`
  - `Convergence`
- ALGO3:
  - `Depth`
  - `Number of Words`
  - `Example`
  - `Counter-Example`

### Evidence Organization

The evidence for this reviewer item is now organized under:

- `data/analysis_artifacts/revision_tracker/2026-03-21/statistical_reporting/README.md`
- `data/analysis_artifacts/revision_tracker/2026-03-21/statistical_reporting/bundle_manifest.csv`
- `data/analysis_artifacts/revision_tracker/2026-03-21/statistical_reporting/bundle_overview.csv`

Per-factor evidence now lives in nested directories such as:

- `.../statistical_reporting/algo1/explanation/`
- `.../statistical_reporting/algo2/convergence/`
- `.../statistical_reporting/algo3/depth/`

### Most Informative Output

The most compact reviewer-facing artifact is `bundle_overview.csv`. It shows, for every algorithm-factor-metric combination:

- the two levels being compared,
- the global mean at each level,
- the difference between those global means,
- how many source files favored each level,
- and how many source files tied.

Representative rows:

| Algorithm | Factor | Metric | Low level | High level | Global mean low | Global mean high | High-low difference | Winner count low | Winner count high | Ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALGO1 | `Explanation` | Accuracy | `-1` | `1` | `0.9062` | `0.9149` | `+0.0087` | `7` | `11` | `0` |
| ALGO1 | `Explanation` | Precision | `-1` | `1` | `0.3043` | `0.3250` | `+0.0207` | `6` | `12` | `0` |
| ALGO1 | `Explanation` | Recall | `-1` | `1` | `0.1153` | `0.1039` | `-0.0114` | `13` | `5` | `0` |
| ALGO1 | `Example` | Precision | `-1` | `1` | `0.2987` | `0.3306` | `+0.0319` | `3` | `15` | `0` |
| ALGO1 | `Example` | Recall | `-1` | `1` | `0.1234` | `0.0958` | `-0.0276` | `16` | `2` | `0` |
| ALGO2 | `Convergence` | Accuracy | `-1` | `1` | `0.8472` | `0.8796` | `+0.0324` | `13` | `5` | `0` |
| ALGO2 | `Convergence` | Precision | `-1` | `1` | `0.3508` | `0.3314` | `-0.0194` | `12` | `6` | `0` |
| ALGO2 | `Convergence` | Recall | `-1` | `1` | `0.1022` | `0.1144` | `+0.0122` | `5` | `13` | `0` |
| ALGO2 | `Tag/Adjacency(1/-1)` | Accuracy | `-1` | `1` | `0.8545` | `0.8723` | `+0.0178` | `4` | `14` | `0` |
| ALGO2 | `Explanation` | Accuracy | `-1` | `1` | `0.8633` | `0.8635` | `+0.0002` | `9` | `9` | `0` |
| ALGO3 | `Depth` | Recall | `1` | `2` | `0.0144` | `0.0652` | `+0.0508` | `0` | `5` | `1` |
| ALGO3 | `Number of Words` | Recall | `3` | `5` | `0.0318` | `0.0478` | `+0.0160` | `2` | `2` | `2` |

### Detailed Findings

The more exhaustive bundle changes the statistical-reporting interpretation in several useful ways.

#### ALGO1

- `Explanation`:
  - accuracy and precision favor `1`
  - recall favors `-1`
- `Example`:
  - precision strongly favors `1`
  - recall strongly favors `-1`
  - accuracy actually favors `-1` globally and only `3` source files prefer `-1` versus `15` preferring `1`, which shows the importance of reporting both global means and per-file winner counts
- `Counterexample`:
  - effects are comparatively small
  - recall is nearly neutral, with one tie and a global mean difference of only `-0.0006`
- `Array/List(1/-1)`:
  - accuracy and precision favor `-1`
  - recall is nearly balanced
- `Tag/Adjacency(1/-1)`:
  - accuracy favors `1`
  - precision favors `-1`
  - recall slightly favors `1`

This makes ALGO1 look less like "Explanation is the story" and more like a set of precision-recall and representation tradeoffs across several prompt dimensions.

#### ALGO2

- `Convergence` remains the strongest descriptive factor:
  - higher global mean accuracy at `1`
  - higher global mean recall at `1`
  - higher global mean precision at `-1`
  - winner counts reinforce the precision-recall tradeoff rather than a single uniformly better setting
- `Explanation` is descriptively weak:
  - accuracy split `9` to `9`
  - precision split `9` to `9`
  - recall split `9` to `9`
  - global mean differences are tiny
- `Example` and `Counterexample` both matter more descriptively than `Explanation`:
  - both tend to reduce accuracy and precision when moved from `-1` to `1`
  - both also slightly reduce recall on average
- `Tag/Adjacency(1/-1)` has a meaningful descriptive effect on accuracy:
  - global mean rises from `0.8545` to `0.8723`
  - `14` of `18` files favor level `1`

So ALGO2 is not just "Convergence plus everything else." It has a broader descriptive structure, but `Convergence` is still the clearest factor.

#### ALGO3

- `Depth` is the clearest descriptive signal:
  - global recall rises from `0.0144` to `0.0652`
  - `5` of `6` files favor `Depth=2`
  - `1` file ties
- `Number of Words` is mixed:
  - global mean favors `5`
  - winner counts are split `2` to `2` with `2` ties
- `Example` is mildly favorable:
  - recall rises from `0.0347` to `0.0449`
  - `4` of `6` files favor `1`
- `Counter-Example` is weak and mixed:
  - global mean slightly decreases from `0.0425` to `0.0371`
  - winner counts are `2` low, `3` high, `1` tie

This means that, even before formal hypothesis testing, ALGO3 already shows a distinction between one relatively strong descriptive factor (`Depth`) and several weaker ones.

### Interpretation

These numbers translate to three concise messages.

1. There is no single prompt setting that improves every metric at once. The descriptive summaries repeatedly show tradeoffs, especially between precision and recall.
2. Some factors matter much more than others. In particular, ALGO2 `Convergence` and ALGO3 `Depth` show clear directional structure, whereas ALGO2 `Explanation` is nearly neutral in the imported corpus.
3. The algorithms are not behaving in the same way. ALGO1 and ALGO2 are shaped by several prompt-design tradeoffs, while ALGO3 is driven much more strongly by search depth than by the example-related factors.

In plain terms, the descriptive layer says the observed variability is structured rather than random noise: certain design choices reliably push the methods toward broader recall, others toward cleaner precision, and some settings barely move results at all.

## 2. Formal Hypothesis Testing

### Exact Reviewer Quote

> "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..."
>
> "Add formal statistical testing with multiple comparison adjustments..."

### What The Reviewer Was Asking For

The reviewers wanted the paper to move beyond descriptive summaries and establish whether observed patterns are genuine or could arise by chance. Descriptive statistics show averages and winner counts, but they cannot tell us whether a difference between two factor levels is reliable. Formal hypothesis testing provides that answer in a probabilistically grounded way.

The reviewer also asked for multiple-comparison adjustment. When many tests are run at the same time, the odds of accidentally finding a "significant" result purely by chance increase. Imagine flipping a fair coin five times — getting all heads is unlikely, but if you repeat that experiment fifty times, getting five heads in a row becomes quite possible. Similarly, with 270 separate tests running (5 factors × 3 metrics × 18 source files), a few will appear significant even if nothing real is happening. Multiple-comparison adjustment is a statistical correction that makes it harder to call a result significant unless the evidence is strong enough to survive the multiplied testing burden — keeping the false-discovery rate (the expected share of results that look real but are not) below 5%.

### What Was Implemented

The revision added two commands. The first is a generic hypothesis-testing command for focused investigations:

```
lcm analyze hypothesis
```

The second is the organized bundle generator that produces the full reviewer-facing evidence:

```
lcm analyze hypothesis-bundle
```

Both apply the same core logic, described in detail below. Each row of the output corresponds to one hypothesis test — comparing two factor levels (e.g., Convergence = −1 vs. Convergence = 1) on one metric (e.g., accuracy) across all source files that have both levels. The columns produced for each test are:

- `pair_count` — how many source files contributed data to this comparison; a higher number means more reliable results
- `mean_low` / `mean_high` — the average metric value when the factor is set to its lower level (e.g., −1) or higher level (e.g., 1); these are the two numbers being compared
- `mean_difference` — the difference between them (high minus low); a positive value means the higher factor level tends to produce higher metric values
- `difference_ci95_low` / `difference_ci95_high` — the lower and upper ends of the 95% confidence interval around the mean difference; if this interval does not cross zero, the difference is considered reliable rather than noise
- `effect_size_paired_d` — Cohen's d for paired samples. This expresses the size of the difference between the two factor levels in units of standard deviation, rather than in the original metric units. Because accuracy, precision, and recall are on different scales, Cohen's d lets you compare effect sizes across metrics and factors fairly. A d of 1.0 means the two levels differ by one full standard deviation; the interpretation scale is given below.
- `t_statistic` / `p_value` — the raw paired t-test output. The t-statistic measures how many standard errors the observed mean difference is away from zero — larger values indicate a more extreme result. The p-value is the probability of observing a difference at least as large as the one found, if the true difference were actually zero. Smaller p-values mean stronger evidence against the null hypothesis. "Raw" here means before the multiple-comparison correction is applied, so raw p-values are optimistically biased.
- `p_value_adjusted` — the Benjamini-Hochberg corrected q-value. After running many tests, a raw p-value of 0.01 no longer means what it usually means — the multiple-testing burden has inflated the chance of a false positive. The BH q-value corrects for this by raising the threshold for significance in proportion to how many tests are in the family. The q-value can be interpreted as: "if I called this result significant, the expected proportion of false discoveries among all results I called significant is at most q." When comparing against 0.05, a q-value below 0.05 passes the adjusted threshold.
- `correction_method` — always `benjamini-hochberg`, recording that the Benjamini-Hochberg procedure was the specific correction applied. This field is included for full auditability of the decision rule.

Code locations:

- `src/llm_conceptual_modeling/analysis/hypothesis.py`
- `src/llm_conceptual_modeling/analysis/hypothesis_bundle.py`

### Command Used

The organized reviewer-facing bundle was produced by:

```bash
lcm analyze hypothesis-bundle \
  --results-root data/results \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing
```

This generates all factor-level paired tests and significance summaries internally. The generic single-factor command (`lcm analyze hypothesis`) is also available for focused investigations.

### Audited Factors

This bundle covers all valid two-level factors with explicit pairing across the three algorithms — 16 factor combinations in total, not a cherry-picked subset:

- ALGO1: `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO2: `Convergence`, `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO3: `Depth`, `Number of Words`, `Example`, `Counter-Example`

### Evidence Organization

- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/README.md`
- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/bundle_manifest.csv`
- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/bundle_overview.csv`

Per-factor evidence lives in nested directories such as:

- `.../hypothesis_testing/algo1/example/`
- `.../hypothesis_testing/algo2/convergence/`
- `.../hypothesis_testing/algo3/depth/`

Each directory contains `paired_tests.csv` (full test results per source file × metric), `significance_summary.csv` (counts by direction and significance), and `factor_overview.csv` (compact summary).

### The Statistical Design in Plain Terms

#### The problem: within-source pairing

Every source file in the corpus contains results at both levels of each two-level factor — for example, each source file was run once with Convergence = −1 and once with Convergence = 1, holding all other factors constant. This means each source file provides a natural paired comparison: the same underlying causal map processed at two factor levels.

Treating these as independent observations would ignore this structure and inflate the error term, making tests artificially conservative. The paired t-test is the appropriate tool here because it uses each source file as its own control.

#### The multiple-comparison problem

Running many tests — such as 5 factors × 3 metrics × 18 source files — means a fair number of p-values will fall below 0.05 purely by chance even if all null hypotheses are true. Benjamini-Hochberg (BH) correction controls the false-discovery rate: rather than asking "what is the probability of any false positive in this family?", it asks "of all the tests I called significant, what fraction are actually false positives?" BH correction keeps that expected fraction below 5%, which is the standard threshold for discovery-oriented research. The `correction_method` column in every output file makes this choice auditable.

#### Why report effect size alongside the p-value

A p-value tells only whether the observed data are unlikely under the null hypothesis of zero difference. It says nothing about how large the difference is. A tiny effect can be statistically significant with a large sample. Cohen's d addresses this by expressing the mean difference in standard deviation units, making it comparable across factors and algorithms even when sample sizes differ. The standard interpretation scale is:

| \|d\| range | Interpretation |
| --- | --- |
| < 0.2 | negligible |
| 0.2 – 0.5 | small |
| 0.5 – 0.8 | medium |
| > 0.8 | large |

### Detailed Findings — All 16 Factors

The exhaustive bundle reveals that the importance of each factor varies considerably across algorithms, and that some factors affect precision and recall in opposite directions.

#### ALGO1

**`Example`** is the dominant factor. When `Example = 1`, accuracy improves sharply (15 of 18 source files favor it, Cohen's d = 1.02, 83 % significant) and precision also improves (d = 0.77). However, recall consistently degrades with `Example = 1` (13 of 18 files favor `−1`, d = −1.26). This is a precision-recall tradeoff: adding examples produces cleaner but more conservative outputs.

**`Tag/Adjacency(1/-1)`** shows the opposite recall pattern to `Array/List(1/-1)`: `Tag/Adjacency = 1` favors recall (d = +0.88 on recall, 61 % significant) while `Array/List = 1` moves recall in the opposite direction (d = +0.88 but opposite sign, 61 % significant). Both have measurable accuracy effects of moderate size.

**`Explanation`** is metric-specific: it clearly improves precision (d = 1.05, 50 % significant) but has little consistent effect on accuracy or recall.

**`Counterexample`** is the weakest ALGO1 factor: no metric reaches above 6 significant tests out of 18, and effect sizes are consistently small to negligible.

#### ALGO2

**`Convergence`** is the strongest single factor in the entire corpus. Accuracy, recall, and precision all show strong, consistent patterns (44 of 54 metric-level tests significant, 81 %):

- Accuracy: 15 of 18 sources favor `Convergence = 1`, d = 1.37 — the largest single effect size observed
- Recall: 15 of 18 sources favor `Convergence = 1`, d = 0.86
- Precision: 14 of 18 sources favor `Convergence = −1`, d = −0.74

This is again a precision-recall tradeoff: `Convergence = 1` improves recall and accuracy but degrades precision.

**`Example` and `Counterexample`** both reliably reduce accuracy and precision when moved from `−1` to `1` (d = −1.33 and d = −0.46 respectively), but the effects are smaller than `Convergence`.

**`Explanation`** is weak in ALGO2 specifically: only 21 of 54 tests are significant, and effect sizes are consistently small. This contrasts with ALGO1 where `Explanation` is a meaningful factor for precision. This is an important algorithm-by-factor interaction.

**`Tag/Adjacency(1/-1)`** has a moderate accuracy effect (9 of 18 significant, d = 0.54), with precision and recall effects weaker.

#### ALGO3

Formal hypothesis testing adds little for ALGO3. Only `Depth` shows a directional signal, and it is fragile:

- `Depth` (Recall): 2 of 6 significant (both GPT-4o sources), d = 0.43 — notable given n = 6, but too few sources to be confident
- `Number of Words`, `Example`, and `Counter-Example`: 0 of 6 significant each; effect sizes small

This is consistent with the replication-stability finding: ALGO3's run-to-run noise is large enough to overwhelm factor-level signals.

### Full Significance Summary

| Algorithm | Factor | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- |
| ALGO1 | `Explanation` | 8 / 18 | 9 / 18 | 8 / 18 |
| ALGO1 | `Example` | 15 / 18 | 9 / 18 | 13 / 18 |
| ALGO1 | `Counterexample` | 4 / 18 | 6 / 18 | 8 / 18 |
| ALGO1 | `Array/List(1/-1)` | 8 / 18 | 10 / 18 | 11 / 18 |
| ALGO1 | `Tag/Adjacency(1/-1)` | 10 / 18 | 12 / 18 | 13 / 18 |
| ALGO2 | `Convergence` | 15 / 18 | 14 / 18 | 15 / 18 |
| ALGO2 | `Explanation` | 6 / 18 | 6 / 18 | 9 / 18 |
| ALGO2 | `Example` | 10 / 18 | 11 / 18 | 10 / 18 |
| ALGO2 | `Counterexample` | 9 / 18 | 7 / 18 | 7 / 18 |
| ALGO2 | `Array/List(1/-1)` | 8 / 18 | 5 / 18 | 1 / 18 |
| ALGO2 | `Tag/Adjacency(1/-1)` | 9 / 18 | 7 / 18 | 7 / 18 |
| ALGO3 | `Depth` | — | — | 2 / 6 |
| ALGO3 | `Number of Words` | — | — | 0 / 6 |
| ALGO3 | `Example` | — | — | 0 / 6 |
| ALGO3 | `Counter-Example` | — | — | 0 / 6 |

### What It Means

**No prompt setting improves all three metrics simultaneously.** Across all three algorithms, every strong factor shows a precision-recall tradeoff: setting a factor to improve accuracy or recall tends to hurt precision. This means there is no universally "correct" prompt design — researchers and practitioners must decide which metric matters most for their task and accept the corresponding tradeoff.

**Some factors genuinely move the output; others barely do.** For ALGO2, whether Convergence is set to 1 or −1 is the single most consequential design choice in the study — it produces the largest effect size in the entire corpus (d = 1.37 on accuracy). For ALGO3, even the strongest factor (Depth) produces an effect that is detectable in only 2 of 6 source files, meaning design choices have unreliable consequences for that algorithm.

**The algorithms behave differently.** ALGO1 is most sensitive to `Example`, ALGO2 is most sensitive to `Convergence`, and ALGO3 is not meaningfully sensitive to any factor in the formal sense. A factor that matters for one algorithm may not matter for another — generalizing prompt design guidance across algorithms without empirical validation is not supported by these data.

**A significant result alone is not enough to declare a factor important.** ALGO2 `Explanation` shows statistically significant results on recall (9 of 18 source files), but the effect size is small (d = 0.37) — meaning the practical difference between setting `Explanation = 1` and `Explanation = −1` is minimal even though it reliably avoids zero. Conversely, ALGO3 `Depth` has a borderline p-value but a medium effect size (d = 0.43), which is more meaningful in practical terms given the small number of sources. Both pieces of information are needed to avoid misleading claims.

## 3. Output Validity and Breadth

### Exact Reviewer Quote

> "It is important to know the reasons behind the failures as much as successful runs. Record the types of particular errors, the deterministic or stochastic nature of failures, and the systematic failure of particular combinations of the LLM-algorithm."

### What The Reviewer Was Asking For

The reviewer was asking the paper to distinguish between three things:

- malformed or unparseable outputs,
- empty outputs,
- and valid but low-quality outputs.

Without that distinction, "variability" and "failure" would be conflated. The reviewer also wanted to know whether failures, if they exist, are deterministic (same input always fails) or stochastic (inconsistent across repetitions).

### What Was Implemented

The revision added `lcm analyze output-validity-bundle`, which classifies every raw output row and produces an organized evidence bundle. The classification categories are:

- `valid_output`: the result string was parseable as an edge list with at least one edge
- `empty_output`: the result was missing, blank, or explicitly marked empty
- `malformed_output`: the result was present but could not be parsed as an edge list

For valid rows, `parsed_edge_count` records how many edges were extracted — a continuous measure of output breadth per row.

The bundle output columns are:

- `failure_category`: one of `valid_output`, `empty_output`, `malformed_output`
- `parsed_edge_count`: number of edges in the parsed output (0 for non-valid rows)
- `is_failure`: `True` if the row is not `valid_output`

Code locations:

- `src/llm_conceptual_modeling/analysis/failures.py`
- `src/llm_conceptual_modeling/analysis/output_validity_bundle.py`

### Command Used

```bash
lcm analyze output-validity-bundle \
  --results-root data/results \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/output_validity
```

This generates the full bundle with per-algorithm and per-model breakdowns. The single-input command (`lcm analyze failures`) remains available for targeted investigations.

### Evidence Organization

```
data/analysis_artifacts/revision_tracker/2026-03-21/output_validity/
├── README.md
├── bundle_manifest.csv
├── bundle_overview.csv
├── failure_rates.csv
├── parsed_edge_counts.csv
├── parsed_edge_quartiles.csv
├── algo1/
│   ├── row_level_validity.csv
│   ├── validity_summary.csv
│   └── breadth_distribution.csv
├── algo2/
│   ├── row_level_validity.csv
│   ├── validity_summary.csv
│   └── breadth_distribution.csv
└── algo3/
    ├── row_level_validity.csv
    ├── validity_summary.csv
    └── breadth_distribution.csv
```

- `bundle_manifest.csv`: index of all generated files with descriptions
- `bundle_overview.csv`: combined view of failure rates and parsed edge statistics across all 18 algorithm-model combinations
- `failure_rates.csv`: failure rate per algorithm-model (aggregate across all algorithms)
- `parsed_edge_counts.csv`: parsed edge count statistics (mean, median, min, max) per algorithm-model
- `parsed_edge_quartiles.csv`: quartile and percentile distributions (q1, q2, q3, p90, p95, p99, IQR) per algorithm-model

### Most Informative Output

Failure rates by algorithm and model:

| Algorithm | Models | Total rows | Failures (malformed or empty) | Failure rate |
| --- | --- | --- | --- | --- |
| ALGO1 | 6 | 2,880 | 0 | 0.0 % |
| ALGO2 | 6 | 5,760 | 0 | 0.0 % |
| ALGO3 | 6 | 1,440 | 0 | 0.0 % |

Every one of the 10,080 raw output rows is a valid, parseable edge list. There are zero malformed outputs and zero empty outputs across all three algorithms and all six models.

Parsed edge count distributions — output breadth per model:

| Algorithm | Model | Rows | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| ALGO1 | DeepSeek V3 Chat | 480 | 10.1 | 9 | 2 | 24 |
| ALGO1 | DeepSeek V3 Chat 0324 | 480 | 14.8 | 14 | 5 | 51 |
| ALGO1 | Gemini 2.0 Flash | 480 | 13.0 | 12 | 1 | 48 |
| ALGO1 | Gemini 2.5 Pro | 480 | 9.4 | 9 | 4 | 17 |
| ALGO1 | GPT-5 | 480 | 28.7 | 28 | 12 | 48 |
| ALGO1 | GPT-4o | 480 | 8.3 | 7 | 1 | 36 |
| ALGO2 | DeepSeek V3 Chat | 960 | 38.8 | 39 | 10 | 100 |
| ALGO2 | DeepSeek V3 Chat 0324 | 960 | 50.0 | 52 | 15 | 136 |
| ALGO2 | Gemini 2.0 Flash | 960 | 28.9 | 25 | 4 | 398 |
| ALGO2 | Gemini 2.5 Pro | 960 | 30.2 | 30 | 10 | 63 |
| ALGO2 | GPT-5 | 960 | 50.8 | 50 | 24 | 82 |
| ALGO2 | GPT-4o | 960 | 84.5 | 15 | 2 | 691 |
| ALGO3 | DeepSeek V3 Chat | 240 | 43.2 | 38 | 1 | 177 |
| ALGO3 | DeepSeek V3 Chat 0324 | 240 | 68.7 | 58 | 12 | 355 |
| ALGO3 | Gemini 2.0 Flash | 240 | 53.4 | 49 | 1 | 252 |
| ALGO3 | Google Gemini 2.5 Pro | 240 | 66.5 | 55.5 | 11 | 302 |
| ALGO3 | GPT-5 | 240 | 49.9 | 46.5 | 1 | 161 |
| ALGO3 | GPT-4o | 240 | 21.4 | 18 | 1 | 81 |

The quartile data reveals the severity of the skew. For ALGO2 GPT-4o: q1 = 9, q2 = 15, q3 = 25, p90 = 438, p95 = 590. This means the top 10 % of valid outputs contain 438 or more edges — the top quantile is an order of magnitude larger than the median. For ALGO1 GPT-4o, by contrast: q1 = 4, q2 = 7, q3 = 10, p90 = 15 — compact and consistent throughout.

### What It Means

**The parsing pipeline is sound for all three algorithms.** Every raw output — across all 10,080 rows, all 18 algorithm-model combinations — is a valid, parseable edge list. The paper does not face a "the models frequently produce malformed output" problem. The revision evidence confirms the infrastructure itself is not a source of data loss.

**Output breadth varies dramatically, and the pattern is algorithm-specific.** ALGO1 produces compact outputs: the median model generates between 7 and 28 edges per output, with a narrow IQR of 4–10. ALGO2 is far wider-spread: GPT-4o with ALGO2 produces a median of 15 edges but a mean of 84.5, with a maximum of 691. ALGO3 falls in between with the widest minimum-maximum range (1–355), suggesting the highest per-run unpredictability.

**The extreme outputs in ALGO2 GPT-4o are the most consequential finding.** The mean-median gap (84.5 vs 15) is not noise — the quartile data confirms it: p90 = 438, p95 = 590, and the IQR is only 16. A small subset of runs produces extremely large edge lists. This matters because a small number of very large outputs can heavily influence mean accuracy, precision, and recall scores when evaluated against a reference graph. The reviewer asked whether failures are deterministic or stochastic; the finding here is analogous but for output breadth: extreme breadth is intermittent rather than systematic, which is itself a form of instability worth noting alongside the formal hypothesis-test results.

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

The exported stability fields are computed over the five recorded repetitions for each unique experimental condition (holding model, subgraph pair, and all prompt factors constant):

- `n` — count of repetitions in this condition (always 5 in this corpus)
- `mean` — arithmetic average of the metric values across repetitions
- `sample_std` — standard deviation across repetitions (uses n−1 denominator); a low value means the metric is consistent across runs
- `min` — smallest metric value observed across repetitions
- `max` — largest metric value observed across repetitions
- `range_width` — max minus min; the total spread of observed values; smaller means more stable
- `coefficient_of_variation` — the standard deviation divided by the absolute mean (|mean|); this is a scale-free stability measure, so a CV of 0.01 means the noise is about 1% of the signal, making it comparable across metrics and conditions of very different magnitudes

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

The reviewer needed more than tables — they wanted visual evidence of variability. Specifically, confidence intervals around means and distributional plots that show how much the data scatters around each metric value across conditions.

### What Was Implemented

Two pieces address this request.

1. The generic long-format export command:

- `lcm analyze figures`

This melts evaluated CSVs into tidy long-format rows (one row per repetition per metric) with preserved provenance columns. It is the building block for box plots, violin plots, and faceted model comparisons in any external tool.

2. The organized bundle generator:

- `lcm analyze figures-bundle`

This new bundle command wraps the long-format export and adds per-model distributional summaries — producing a self-contained evidence package ready for a reviewer to open in a plotting tool without any manual assembly.

Code location:

- `src/llm_conceptual_modeling/analysis/figures_bundle.py`

### Command Used

```bash
lcm analyze figures-bundle \
  --results-root data/results \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/figure_exports
```

This generates long-format metric rows per algorithm plus per-model distributional summaries covering all 18 algorithm-model combinations.

### Evidence Organization

```
data/analysis_artifacts/revision_tracker/2026-03-21/figure_exports/
├── README.md
├── bundle_manifest.csv
├── bundle_overview.csv           ← distributional summary across all models
├── algo1_metric_rows.csv        ← long-format rows for all ALGO1 models
├── algo2_metric_rows.csv        ← long-format rows for all ALGO2 models
├── algo3_metric_rows.csv        ← long-format rows for all ALGO3 models
├── algo1/<model>/distributional_summary.csv
├── algo2/<model>/distributional_summary.csv
└── algo3/<model>/distributional_summary.csv
```

Each `distributional_summary.csv` contains: `n`, `mean`, `sample_std`, `ci95_low`, `ci95_high`, `median`, `q1`, `q3`, `min`, `max` — all per metric per model.

The `bundle_overview.csv` aggregates all model-level summaries into one reviewer-facing table.

### Most Informative Output

ALGO1 accuracy distributional summary across models:

| Model | n | Mean | 95% CI low | 95% CI high | Median | Q1 | Q3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V3 Chat | 160 | 0.915 | 0.913 | 0.917 | 0.919 | 0.909 | 0.922 |
| Gemini 2.0 Flash | 160 | 0.917 | 0.915 | 0.919 | 0.916 | 0.909 | 0.925 |
| Gemini 2.5 Pro | 160 | 0.933 | 0.932 | 0.934 | 0.935 | 0.929 | 0.938 |
| GPT-5 | 160 | 0.896 | 0.893 | 0.899 | 0.899 | 0.890 | 0.909 |
| GPT-4o | 160 | 0.931 | 0.930 | 0.933 | 0.932 | 0.925 | 0.938 |

ALGO3 recall distributional summary — showing extreme skew:

| Model | n | Mean | 95% CI low | 95% CI high | Median |
| --- | --- | --- | --- | --- | --- |
| DeepSeek V3 Chat | 240 | 0.058 | 0.030 | 0.086 | 0.0 |
| Gemini 2.0 Flash | 240 | 0.057 | 0.034 | 0.080 | 0.0 |
| GPT-4o | 240 | 0.087 | 0.051 | 0.122 | 0.0 |
| GPT-5 | 240 | 0.000 | 0.000 | 0.000 | 0.0 |

Long-format export row counts:

| File | Rows |
| --- | --- |
| `algo1_metric_rows.csv` | `8,640` |
| `algo2_metric_rows.csv` | `17,280` |
| `algo3_metric_rows.csv` | `1,440` |

### Detailed Findings

Two patterns are directly visible from the distributional summaries.

**ALGO1 accuracy is compact and reproducible.** All six models have 95% CIs that are tight and entirely above 0.89 — GPT-4o and Gemini 2.5 Pro are the strongest, GPT-5 the weakest. The IQR (Q1 to Q3) spans roughly 0.02 for every model, confirming that repetition-level scatter is small. This is consistent with the replication-stability findings: ALGO1 is not a source of meaningful visual spread in any figure drawn from this corpus.

**ALGO3 recall is extremely right-skewed.** Every model shows a median of exactly 0.0 with a positive mean and a wide CI that does not include zero — meaning the evaluated outputs are mostly empty but occasionally produce non-zero recall. GPT-4o has the highest mean recall (0.087) with the widest CI [0.051, 0.122], while GPT-5 has a mean of exactly 0.0. This bimodal pattern (near-zero median, positive tail) is exactly what the output-variability analysis (Section 6) attributes to ALGO3's multi-step amplification wandering.

### What It Means

The reviewer asked for distributional plots. The bundle now makes those plots deterministic and auditable: a reviewer can open `bundle_overview.csv` directly in a plotting tool and reproduce any figure without manual post-processing. The key takeaway is visual rather than tabular — the gap between the compact, high-accuracy ALGO1 distributions and the sparse, skewed ALGO3 recall distributions communicates the variability story more directly than any summary table.

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
| Hypothesis testing | `lcm analyze hypothesis`, `lcm analyze hypothesis-bundle` | `hypothesis_testing/bundle_overview.csv`, `hypothesis_testing/algo{1,2,3}/*/` |
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
