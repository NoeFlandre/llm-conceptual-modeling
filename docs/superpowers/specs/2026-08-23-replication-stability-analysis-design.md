# Replication Stability Analysis Design

## Goal

Create a reproducible, model-aware stability analysis for Algorithm 3 before any
paper revision. The analysis must show that a useful replication budget depends
on the model and condition rather than being a universal number.

## Evidence and scope

The six Algorithm 3 evaluated files from Git revision `b82ec062^` are the
approved source because the current working tree has the historical `data/`
tree deleted. Their compacted metric columns reproduce the paper's existing
53-of-288 varying-condition count. Only the metric columns needed for this
analysis will be reconstructed; generated edge text and other unrelated data
will not be restored.

The unit of analysis is an exact model, prompt-factor setting, directed
source--target graph pair, and metric. Graph pairs remain separate so graph
heterogeneity is not incorrectly treated as repeated calls to the same
condition. The primary scope is Algorithm 3 and Recall, the paper's stated
high-variability case.

## Analysis

For every exact condition, the tool will report the five observed runs, mean,
sample standard deviation, range, coefficient of variation, and cumulative
means for prefixes of one through five runs. It will also estimate the total
number of runs needed for a 95% normal-approximation confidence interval with a
5% relative half-width using

\[
n^* = \left\lceil\left(\frac{1.96s}{0.05|\bar{x}|}\right)^2\right\rceil.
\]

The estimate is labelled as a precision estimate, not as an observed plateau.
Zero-mean cells are reported as not estimable under a relative-error target;
they are never counted as evidence that five runs are sufficient. Model-level
summaries include the number of varying cells, non-zero cells, median and
maximum estimated run counts, and the share requiring more than five runs.

## Components and outputs

- `src/llm_conceptual_modeling/analysis/replication_stability.py` owns input
  validation, exact-condition statistics, cumulative means, budget estimates,
  and model summaries.
- `tests/analysis/test_replication_stability.py` verifies these behaviors and
  the CLI through small deterministic fixtures.
- The `lcm analyze replication-stability` command reads a manifest and writes
  condition-level, cumulative-mean, and model-summary CSV files.
- `data/analysis_artifacts/reviewer_replication_stability/` contains compact
  source inputs, the source manifest, generated tables, and a README with
  provenance and interpretation.

## Verification

The implementation follows red-green TDD. The focused analysis tests, Ruff,
the repository test suite, a coverage/CRAP audit, and a mutation run limited to
the new module must pass. The paper source is not modified, and no commit is
created.
