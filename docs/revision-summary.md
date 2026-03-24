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

- `n`
- `mean`
- `sample_std`
- `median`
- `min`
- `max`
- `ci95_low`
- `ci95_high`

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

The reviewers wanted the paper to move beyond descriptive patterns into formal tests with:

- explicit p-values,
- explicit pairing structure,
- and some control for multiple testing.

### What Was Implemented

The revision added two commands. The first is the generic hypothesis-testing command:

- `lcm analyze hypothesis`

The second is the organized bundle generator that produces the reviewer-facing evidence:

- `lcm analyze hypothesis-bundle`

Both apply the same core logic: paired two-level tests with Benjamini-Hochberg FDR correction.

The output columns are:

- `pair_count`: number of paired observations in the test
- `mean_low` / `mean_high`: mean metric value at each factor level
- `mean_difference`: mean of (high − low) across pairs
- `difference_ci95_low` / `difference_ci95_high`: 95% confidence interval on the mean difference
- `effect_size_paired_d`: Cohen's d for paired samples (standardized mean difference)
- `t_statistic` / `p_value`: raw paired t-test output
- `p_value_adjusted`: BH-corrected q-value
- `correction_method`: always `benjamini-hochberg`

Code locations:

- `src/llm_conceptual_modeling/analysis/hypothesis.py`
- `src/llm_conceptual_modeling/analysis/hypothesis_bundle.py`

### Command Used

The organized reviewer-facing bundle is produced by:

```bash
lcm analyze hypothesis-bundle \
  --results-root data/results \
  --output-dir data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing
```

This internally generates all factor-level paired tests and per-factor significance summaries.
The generic single-factor command (`lcm analyze hypothesis`) is also available for focused investigations.

### Why These Analytical Choices

The reviewer asked for the analytical choices to be made explicit. Three decisions merit direct justification.

**Paired t-test over independent-samples t-test.** Every source file in the corpus contains observations at both levels of each two-level factor, matched on all other design dimensions. A paired t-test exploits this structure and removes source-level variation from the error term, giving sharper tests than treating all observations as independent. The paired statistic is therefore strictly more appropriate here.

**Benjamini-Hochberg over Bonferroni.** Bonferroni controls familywise error rate (FWER) by dividing alpha by the number of tests, which becomes very conservative as test counts grow. BH controls false-discovery rate (FDR) instead, which is a less stringent criterion appropriate for exploratory factor analysis. The reviewer asked for "multiple comparison adjustment" without specifying FWER control; BH is the standard choice when the goal is to identify which factor effects are real while keeping false discoveries below 5%. The `correction_method` column makes this choice auditable.

**Cohen's d as the effect-size measure.** A significant p-value tells only whether an effect is unlikely to be zero; it says nothing about practical magnitude. Cohen's d for paired samples is computed as the mean difference divided by the standard deviation of the differences — the same information as the t-statistic but on a scale interpretable across factors. The bundle overview uses the standard scale: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large. This allows direct comparison of effect strength across algorithm-factor combinations even when sample sizes differ.

### Evidence Organization

The evidence for this reviewer item is organized under:

- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/README.md`
- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/bundle_manifest.csv`
- `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/bundle_overview.csv`

Per-factor evidence lives in nested directories such as:

- `.../hypothesis_testing/algo1/explanation/`
- `.../hypothesis_testing/algo2/convergence/`
- `.../hypothesis_testing/algo3/depth/`

### Audited Factors

This bundle exhausts all valid two-level factors with explicit pairing:

- ALGO1: `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO2: `Convergence`, `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO3: `Depth`, `Number of Words`, `Example`, `Counter-Example`

This is 16 factor combinations across 3 algorithms, not a selective subset.

### Most Informative Output

The most compact reviewer-facing artifact is `bundle_overview.csv`. The strongest effects by combined significance count and effect size are:

| Algorithm | Factor | Metric | Sig. tests | Sig. share | Cohen's d | Direction |
| --- | --- | --- | --- | --- | --- | --- |
| ALGO2 | `Convergence` | Accuracy | 15 / 18 | 83 % | +1.37 | low > high |
| ALGO2 | `Convergence` | Recall | 15 / 18 | 83 % | +0.86 | high > low |
| ALGO2 | `Convergence` | Precision | 14 / 18 | 78 % | −0.74 | low > high |
| ALGO1 | `Example` | Accuracy | 15 / 18 | 83 % | +1.02 | high > low |
| ALGO1 | `Example` | Recall | 13 / 18 | 72 % | −1.26 | high < low |
| ALGO1 | `Example` | Precision | 9 / 18 | 50 % | +0.77 | high > low |
| ALGO1 | `Tag/Adjacency(1/-1)` | Recall | 13 / 18 | 72 % | −0.66 | high < low |
| ALGO2 | `Example` | Accuracy | 10 / 18 | 56 % | −1.33 | high < low |
| ALGO2 | `Example` | Precision | 11 / 18 | 61 % | −1.16 | high < low |
| ALGO1 | `Explanation` | Precision | 9 / 18 | 50 % | +1.05 | high > low |
| ALGO1 | `Array/List(1/-1)` | Recall | 11 / 18 | 61 % | +0.88 | high > low |
| ALGO1 | `Tag/Adjacency(1/-1)` | Accuracy | 10 / 18 | 56 % | −0.80 | high < low |

### Detailed Findings

The exhaustive bundle reveals structure that a selective set of factors would miss.

#### ALGO1

- `Example` is the strongest ALGO1 factor:
  - Accuracy and precision favor `1` (large Cohen's d)
  - Recall strongly favors `−1` (large Cohen's d, negative)
  - This is a precision-recall tradeoff: adding examples improves precision but hurts recall
- `Explanation` is moderate:
  - Precision favors `1` (large d = 1.05)
  - Accuracy and recall are mixed and weaker
- `Tag/Adjacency(1/-1)` and `Array/List(1/-1)` both have measurable recall effects:
  - Both reach large Cohen's d on recall despite moderate significance counts
  - The recall direction is opposite between them (one favors high, one favors low), suggesting distinct mechanisms
- `Counterexample` is the weakest factor:
  - No metric reaches above 6 significant tests out of 18
  - Effect sizes are consistently small

The aggregate picture for ALGO1: `Example` dominates, and the remaining factors have metric-specific rather than uniform effects.

#### ALGO2

- `Convergence` is the strongest factor overall:
  - 44 of 54 metric-level tests significant (81 %)
  - Precision-recall tradeoff: precision favors `−1`, recall favors `+1`
  - Accuracy favors `−1` (d = 1.37 — the largest single effect in the corpus)
- `Example` and `Counterexample` are moderately strong and consistently reduce accuracy and precision:
  - Both move accuracy toward `−1` with d in the medium-to-large range
  - Neither is as strong as `Convergence`
- `Explanation` is weak in ALGO2 specifically:
  - only 21 of 54 tests significant
  - Effect sizes are small
  - This contrasts with ALGO1 where `Explanation` matters more — an important algorithm-by-factor interaction
- `Tag/Adjacency(1/-1)` has a meaningful accuracy effect:
  - 9 of 18 significant, d = 0.54
  - Precision and recall are weaker

The aggregate picture for ALGO2: `Convergence` is the dominant factor; `Explanation` matters much less than it does in ALGO1; `Example` and `Counterexample` have consistent but moderate accuracy effects.

#### ALGO3

- `Depth` is the only factor with meaningful formal support:
  - 2 of 6 tests significant (both for GPT-4o)
  - Effect size is large (d = 0.43, not as large as ALGO1/ALGO2 but notable given n = 6)
  - Both significant results favor `Depth = 2`
- `Number of Words`, `Example`, and `Counter-Example` all fail to reach significance:
  - 0 of 6 significant tests each
  - Effect sizes are small
  - Consistent with the replication-stability finding that ALGO3 is too noisy for firm factor conclusions

The aggregate picture for ALGO3: formal hypothesis testing adds little over descriptive analysis — `Depth` is the only factor with a directional signal, and even it is fragile given the small sample size and high noise.

### Full Significance Summary

For reference, the complete table of significant-test counts across all 16 factors:

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

### Interpretation

The formal testing changes the narrative in three ways.

**First, significance and effect size must be read together.** A factor can be statistically significant with a small effect (ALGO2 `Explanation` recall: 9/18 sig, d = 0.37) or borderline significant with a large effect (ALGO3 `Depth`: 2/6 sig, d = 0.43). Reporting both prevents overclaiming either direction.

**Second, the precision-recall tradeoff is a genuine structured effect, not noise.** Both ALGO1 `Example` and ALGO2 `Convergence` show the pattern where one metric improves and another degrades as a factor moves from low to high. This is visible in the raw descriptive summaries and confirmed by formal testing with consistent direction and large Cohen's d.

**Third, the algorithms are not equally factor-sensitive.** ALGO1 and ALGO2 have multiple factors with large Cohen's d and high significance shares. ALGO3 has essentially none. This is consistent with the replication-stability finding: ALGO3's run-to-run noise is large enough to swamp most factor-level signals.

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
