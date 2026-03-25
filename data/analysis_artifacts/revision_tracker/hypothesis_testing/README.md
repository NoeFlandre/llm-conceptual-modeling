# Hypothesis-Testing Audit Bundle

This directory contains the organized artifacts for the formal hypothesis-testing revision item.

## Purpose

The reviewer asked for p-values, multiple-comparison adjustment, and stronger inferential support.
This bundle captures those paired tests in a structure that can be read by factor rather than as a
flat directory of unrelated CSV files.

## Statistical Design

### Pairing logic

Each source file provides a paired observation by matching on all other design factors
(`Repetition`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`,
`Convergence`, `Depth`, `Number of Words`, `Source Subgraph Name`, `Target Subgraph Name`).
Within a source file, the two levels of the tested factor define the pair.
This means every comparison is within-source, controlling for source-level confounds.

### Why paired t-test

A paired t-test is appropriate here because every source file contains observations at both
levels of each two-level factor. This controls for source-level variation that would otherwise
inflate the error term. The test reports:

- `mean_difference`: mean of (high − low) across pairs
- `difference_ci95_low/high`: 95% confidence interval on the mean difference
- `effect_size_paired_d`: standardized mean difference (Cohen's d for paired samples)
- `t_statistic` and `p_value`: standard paired t-test output

### Multiple-comparison correction

Benjamini-Hochberg (BH) FDR correction is applied within each source file × metric group.
BH is less conservative than Bonferroni and is appropriate when the goal is discovery across
many tests rather than strict familywise error control. The FDR target is 5%.
`p_value_adjusted` is the BH-corrected q-value; it is compared to 0.05 to determine significance.

### Effect size interpretation (Cohen's d for paired samples)

| \|d\| range | Interpretation |
| --- | --- |
| < 0.2 | negligible |
| 0.2 – 0.5 | small |
| 0.5 – 0.8 | medium |
| > 0.8 | large |

The bundle overview uses this scale to distinguish strong factor effects from fragile ones.

## Layout

- `bundle_manifest.csv`
  Index of every generated paired-test file, significance-summary file, and factor overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with significant-test counts, direction counts, and
  the strongest adjusted result.
- `<algorithm>/<factor_slug>/paired_tests.csv`
  Full paired t-test output for that factor. One row per source file × metric.
- `<algorithm>/<factor_slug>/significance_summary.csv`
  Count summary by metric, direction, and adjusted-significance status.
- `<algorithm>/<factor_slug>/factor_overview.csv`
  Compact reviewer-facing overview for that factor.

## Audited Factors

This bundle exhausts all valid two-level factors with explicit pairing:

- ALGO1: `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO2: `Convergence`, `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO3: `Depth`, `Number of Words`, `Example`, `Counter-Example`
