# Revision Summary

This document consolidates the software-facing revision work completed for the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

It is designed to answer three questions clearly:

1. What concern did the reviewers raise?
2. What was implemented in the repository in response?
3. What did the implemented analysis actually find?

The emphasis is on reproducible, file-backed findings from the imported result corpus under `data/results/`.

## Scope And Evidence Boundary

This summary covers the deterministic offline revision work that is reproducible from the current repository.

- Primary imported corpus: `data/results/`
- Audited revision artifacts: `data/analysis_artifacts/revision_tracker/2026-03-21/` and `data/analysis_artifacts/revision_tracker/2026-03-24/`
- Verification surface: `lcm analyze ...`, `lcm baseline ...`, and `lcm verify ...`

The document does not claim direct measurement of hidden states or internal provider-side mechanisms. Where a reviewer asked for an explanation of "what inside the LLM" causes variability, the implemented response is an output-side proxy analysis over repeated raw results.

## Executive Summary

The revision work added a full deterministic post-processing layer around the imported experiment corpus.

- Statistical summaries and confidence intervals were added.
- Formal paired hypothesis testing with multiple-comparison correction was added.
- Failure classification was added and showed that the imported corpus is dominated by structurally valid outputs rather than parser-visible failures.
- Replication-stability analysis was added and showed that ALGO1 and ALGO2 are highly stable across repetitions, while ALGO3 is not.
- Output-variability analysis was added and showed that ALGO3 drift is qualitatively different from ALGO1 and ALGO2 at the raw edge-set level.
- A deterministic non-LLM structural baseline was added and outperformed all imported model outputs on the audited comparisons.

The main empirical message sharpened by the revision work is that variability in this corpus is not uniform across methods. ALGO1 and ALGO2 are close to repetition-stable on the imported runs, whereas ALGO3 exhibits strong multi-step output drift and breadth expansion.

## Revision Matrix

| Revision theme | Reviewer concern | Repository response | Main finding |
| --- | --- | --- | --- |
| Statistical reporting | Missing confidence intervals and grouped summaries | Added `lcm analyze summary` | Factor-level descriptive patterns are now explicit and auditable |
| Formal testing | Missing p-values and adjustment for multiple comparisons | Added `lcm analyze hypothesis` with Benjamini-Hochberg correction | `Convergence` in ALGO2 is the strongest tested factor in the imported corpus |
| Failure understanding | Need reasons behind failures, not only successes | Added `lcm analyze failures` | Imported raw outputs are overwhelmingly valid; the issue is variability, not parse failure |
| Replication justification | Five repetitions not justified | Added `lcm analyze stability` | ALGO1 and ALGO2 are nearly repetition-stable; ALGO3 is not |
| Plot-ready reporting | Need distributional and figure-ready outputs | Added `lcm analyze figures` | The imported corpus can now be plotted directly with preserved provenance |
| Internal variability explanation | Randomness shown, but mechanism unclear | Added `lcm analyze variability` | ALGO3 shows strong edge-set drift and breadth expansion across repetitions |
| Non-LLM comparator | Need at least one baseline | Added `lcm baseline` and `lcm analyze baseline-comparison` | No imported model beat the deterministic baseline on the audited metrics |
| Cross-domain reinforcement | Need at least one additional domain | Deferred | No new domain data was added in this software tranche |

## 1. Statistical Reporting And Confidence Intervals

### Reviewer Concern

Reviewer #2 asked for formal statistical reporting, including confidence intervals and summary statistics rather than only raw metric tables.

### What Was Implemented

The repository now includes `lcm analyze summary`, which computes grouped descriptive statistics from evaluated CSVs.

For each requested metric and factor grouping, the command exports:

- `n`
- `mean`
- `sample_std`
- `median`
- `min`
- `max`
- `ci95_low`
- `ci95_high`

This makes factor-level comparisons auditable across the imported corpus without rerunning any provider calls.

### Findings

The grouped summaries revealed consistent directional patterns that were not previously documented clearly.

- In imported ALGO1 evaluated files, `Explanation=1` won more often on mean accuracy and precision, while `Explanation=-1` won more often on recall.
- In imported ALGO2 evaluated files, `Convergence=-1` won more often on mean accuracy and precision, while `Convergence=1` won more often on recall.
- In imported ALGO3 evaluated files, `Depth=2` produced higher global mean recall than `Depth=1` (`0.0652` vs `0.0144`).

Interpretation: the revision work clarified that some prompt factors are associated with a precision-recall tradeoff rather than a uniform improvement across all metrics.

## 2. Formal Hypothesis Testing

### Reviewer Concern

Reviewer #2 explicitly requested hypothesis testing with p-values and multiple-comparison adjustment.

### What Was Implemented

The repository now includes `lcm analyze hypothesis`, which runs paired two-level tests over evaluated outputs and applies Benjamini-Hochberg correction to the resulting p-values.

This choice was made because the repository emits related families of tests across metrics and files, making false-discovery-rate control more appropriate than a strongly conservative familywise correction.

### Findings

The audited tests identify ALGO2 `Convergence` as the strongest factor in the imported corpus.

- ALGO2 `Convergence`: 44 of 54 metric-level tests significant after correction
- ALGO1 `Explanation`: 25 of 54 metric-level tests significant after correction
- ALGO2 `Explanation`: 21 of 54 metric-level tests significant after correction
- ALGO3 `Depth`: 2 of 6 recall tests significant, both favoring `Depth=2`
- ALGO3 `Number of Words`: 0 of 6 tests significant after correction

Interpretation: the revision did not just add statistical machinery; it showed that some factors have clear, repeated effects in the imported corpus, while others are weaker or mixed.

## 3. Failure Analysis

### Reviewer Concern

Reviewer #2 asked for a clearer understanding of failures, including their type and whether certain algorithm-model combinations fail systematically.

### What Was Implemented

The repository now includes `lcm analyze failures`, which classifies raw outputs into reproducible categories and records parsed edge counts.

This allows the imported raw corpus to be inspected for malformed outputs, empty outputs, and structurally valid outputs.

### Findings

The imported primary raw corpus contained no parser-visible failures.

- `10,080` rows were classified as `valid_output`
- `0` rows were classified as malformed or empty
- Failure rate was `0.0` for every imported algorithm-model combination

At the same time, output size still varied sharply across models. For example, ALGO2 mean parsed edge count ranged from `28.88` for Gemini 2.0 Flash to `84.55` for GPT-4o, and GPT-4o produced an extreme maximum of `691` parsed edges in one raw output.

Interpretation: in the imported corpus, the main issue is not format failure. It is variability within structurally valid outputs.

## 4. Replication Stability

### Reviewer Concern

Reviewer #1 and Reviewer #2 questioned the use of five repetitions without a stronger empirical justification.

### What Was Implemented

The repository now includes `lcm analyze stability`, which quantifies run-to-run variation across repeated evaluated outputs.

The command exports grouped stability summaries including:

- `n`
- `mean`
- `sample_std`
- `min`
- `max`
- `range_width`
- `coefficient_of_variation`

### Findings

The stability results separate the three algorithms clearly.

- ALGO1 and ALGO2 are almost fully stable across the five imported repetitions.
- Only 2 of 576 ALGO1 accuracy conditions, 3 of 576 ALGO1 precision conditions, and 2 of 576 ALGO1 recall conditions showed any run-to-run variation.
- For ALGO2, only 1 of 1152 conditions per metric showed variation, and when `Convergence=1`, zero of 576 conditions varied for any metric.
- ALGO3 is different: 44 of 96 recall conditions varied across repetitions.
- ALGO3 variability is stronger at `Depth=2` than at `Depth=1`, with varying-condition shares of `0.5625` vs `0.3542` and a larger mean range width (`0.4915` vs `0.1818`).

Interpretation: the imported corpus supports the five-repetition design as largely sufficient for ALGO1 and ALGO2, but it also shows that ALGO3 remains intrinsically less stable.

## 5. Figure-Ready Exports

### Reviewer Concern

Reviewer #1 asked for confidence intervals and distributional reporting that could support clearer visualization.

### What Was Implemented

The repository now includes `lcm analyze figures`, which converts evaluated CSVs into tidy long-format metric rows with explicit provenance.

Each exported row preserves:

- source file
- algorithm
- model
- metric
- factor metadata from the input row

### Findings

The imported corpus is now directly usable for plotting without manual file stitching.

- ALGO1 figure export: `8,640` rows
- ALGO2 figure export: `17,280` rows
- ALGO3 figure export: `1,440` rows

Interpretation: the revision work turned the imported results into a figure-ready analysis surface rather than leaving visualization as an ad hoc manual step.

## 6. Output Variability And Proxy Mechanisms

### Reviewer Concern

Reviewer #2 and Philippe asked for a stronger explanation of what drives LLM variability internally.

### What Was Implemented

The repository now includes `lcm analyze variability`, which works only from raw repeated outputs already stored in `data/results/`.

The command measures output-side drift using:

- `mean_pairwise_jaccard`
- `min_pairwise_jaccard`
- `exact_match_pair_rate`
- `mean_edge_count`
- `sample_std_edge_count`
- `union_edge_count`

This is intentionally a proxy analysis. It does not observe hidden states, logits, or attention heads. Instead, it asks whether repeated runs preserve the same edge set and output breadth.

### Findings

ALGO1 and ALGO2 are almost exact-output stable at the raw edge-set level. ALGO3 is not.

Global cross-corpus summary:

- ALGO1: mean pairwise Jaccard `0.9981`, exact-match pair rate `0.9979`
- ALGO2: mean pairwise Jaccard `0.9997`, exact-match pair rate `0.9991`
- ALGO3: mean pairwise Jaccard `0.0770`, exact-match pair rate `0.0010`, breadth-expansion ratio `4.13`

Within ALGO3:

- `Depth=2` has lower mean pairwise Jaccard than `Depth=1` (`0.0692` vs `0.0848`)
- `Depth=2` produces more edges on average (`60.84` vs `40.21`)
- `Depth=2` shows a larger edge-count spread (`20.47` vs `13.44`)

Interpretation: the implemented evidence supports a process-level explanation of variability. In the imported corpus, ALGO3 appears to amplify small run-to-run differences through a multi-step expansion process, while ALGO1 and ALGO2 do not exhibit that same output-side drift.

## 7. Non-LLM Baseline Comparison

### Reviewer Concern

Reviewer #2 asked for at least one non-LLM baseline to contextualize the value proposition of the LLM workflows.

### What Was Implemented

The repository now includes a deterministic `direct-cross-graph` baseline through `lcm baseline`, together with `lcm analyze baseline-comparison`.

This baseline uses mother-graph structure directly and can be generated, evaluated, and compared without any provider call.

### Findings

No imported model beat the deterministic baseline on the audited comparisons.

- `models_beating_baseline = 0` for ALGO1 accuracy, precision, and recall
- `models_beating_baseline = 0` for ALGO2 accuracy, precision, and recall
- `models_beating_baseline = 0` for ALGO3 recall

The closest imported models were:

- Gemini 2.5 Pro for ALGO1 and ALGO2 accuracy and precision
- GPT-5 for ALGO1 and ALGO2 recall
- GPT-4o for ALGO3 recall

Interpretation: this finding is important but must be read carefully. The baseline is structurally privileged because it uses mother-graph information directly, so it is a conservative heuristic comparator rather than a learned replacement for LLM-based generation.

## 8. Cross-Domain Generalization

### Reviewer Concern

Reviewer #2 asked for reinforcement of the claims through at least one additional domain.

### What Was Done

This item was deferred in the current software tranche because it requires new domain data and likely new raw runs rather than deterministic post-processing over the imported corpus.

### Findings

No new finding is claimed here.

## Repository Outputs Added For The Revision

The following reproducible command surfaces were added as part of the revision work:

- `lcm analyze summary`
- `lcm analyze hypothesis`
- `lcm analyze failures`
- `lcm analyze stability`
- `lcm analyze figures`
- `lcm analyze variability`
- `lcm baseline`
- `lcm analyze baseline-comparison`

These outputs are backed by tests and audited artifacts stored in the repository.

## Practical Reading Of The Revision Work

Taken together, the revision implementation supports four strong conclusions.

1. The repository now supports reviewer-facing statistical reporting rather than only raw metrics.
2. Variability in the imported corpus is method-dependent rather than uniformly distributed across all algorithms.
3. The main reproducible issue is not parser-visible failure; it is output drift within valid results, especially for ALGO3.
4. The baseline comparison and stability analyses provide a more demanding context for interpreting the LLM workflows than the original software surface did.

## Recommended Companion Files

- `docs/architecture.md`
- `data/analysis_artifacts/revision_tracker/2026-03-21/`
- `data/analysis_artifacts/revision_tracker/2026-03-24/`
- `paper/revision-tracker.md`
