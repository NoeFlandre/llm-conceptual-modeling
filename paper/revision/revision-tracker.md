# Revision Tracker

This document tracks reviewer requests that may require software work, the strategy chosen to address them, and the current implementation status.

Current evidence boundary: the local workspace now contains imported primary result files under `data/results/` for Algorithms 1, 2, and 3 across the available model families in the legacy project. Findings below are based on those imported result files, excluding archival playground copies and caches.

Audit artifacts for the implemented findings are stored under `data/analysis_artifacts/revision_tracker/2026-03-21/` and `data/analysis_artifacts/revision_tracker/2026-03-24/`.

## Status Key

- `planned`: request understood, implementation not started
- `in_progress`: implementation or validation ongoing
- `done`: software work implemented and locally verified
- `deferred`: not rejected, but intentionally postponed pending scope or data decisions

## Review Requests

| Reviewer citation | Code-facing interpretation | Strategy | Status | Result | Findings |
| --- | --- | --- | --- | --- | --- |
| Reviewer #2: "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..." and "Add formal statistical testing with multiple comparison adjustments, provide confidence intervals of all performance measures..." | The current repo computes evaluation metrics and factorial summaries, but not reviewer-facing grouped descriptive statistics | Add a deterministic `lcm analyze summary` command that works on evaluated CSVs, computes grouped summary statistics, and writes figure-ready CSV output | `done` | Implemented `analyze summary` with grouped `n`, `mean`, `sample_std`, `median`, `min`, `max`, `ci95_low`, and `ci95_high` outputs. Protected by CLI tests. | Across the imported ALGO1 evaluated files, `Explanation=1` won on mean accuracy in 11 of 18 file-level comparisons and on precision in 12 of 18, while `Explanation=-1` won on recall in 13 of 18. Across the imported ALGO2 evaluated files, `Convergence=-1` won on mean accuracy in 13 of 18 file-level comparisons and on precision in 12 of 18, while `Convergence=1` won on recall in 13 of 18. ALGO2 `Explanation` showed no dominant direction overall: winner counts split 9 to 9 for accuracy, precision, and recall. Across imported ALGO3 evaluated files, mean recall at `Depth=2` exceeded `Depth=1` for every model except GPT-5, with global means `0.0652` versus `0.0144`. For `Number of Words`, the global mean recall was higher at `5` than at `3` (`0.0478` versus `0.0318`), although this direction was not consistent for every model. |
| Reviewer #2: "The article does not have basic statistical hypothesis testing. No ANOVA F-tests, p-values, and confidence intervals are provided..." and "Add formal statistical testing with multiple comparison adjustments..." | The repository still lacks actual p-values and adjusted p-values over the imported evaluated corpus | Add a deterministic `lcm analyze hypothesis` command for paired two-level factor tests with explicit pairing columns and Benjamini-Hochberg correction, then generate audited outputs for the main imported factors. Benjamini-Hochberg was chosen because these outputs contain families of related factor-level tests across files and metrics; controlling the false discovery rate is a better fit here than a more conservative familywise correction such as Bonferroni. | `done` | Implemented `analyze hypothesis` with paired `ttest_rel` outputs, adjusted p-values, and full provenance. Protected by CLI tests. Audit artifacts stored under `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/`. | Across the imported ALGO2 files, `Convergence` is the strongest factor in the current formal tests: 44 of 54 metric-level tests were significant after Benjamini-Hochberg correction, with most accuracy and precision tests favoring `Convergence=-1` and most recall tests favoring `Convergence=1`. ALGO1 `Explanation` was significant in 25 of 54 tests and showed the same mixed direction already seen descriptively: more significant accuracy shifts toward `Explanation=1`, more significant recall shifts toward `Explanation=-1`, and mixed precision behavior. ALGO2 `Explanation` was weaker and more mixed, with 21 of 54 significant tests. For ALGO3, `Depth` produced 2 significant recall tests out of 6, both favoring `Depth=2`, while `Number of Words` produced 0 significant tests out of 6 after correction. |
| Reviewer #2: "It is important to know the reasons behind the failures as much as successful runs. Record the types of particular errors, the deterministic or stochastic nature of failures, and the systematic failure of particular combinations of the LLM-algorithm." | The current repo evaluates outputs but does not classify malformed or unusable raw outputs | Add a deterministic failure-analysis command that classifies raw outputs into a small set of reproducible categories and exports counts for inspection | `done` | Implemented `analyze failures` with row-level `failure_category`, `parsed_edge_count`, and `is_failure` outputs. Protected by CLI tests. | Across all imported primary raw result files, the current classifier found 10,080 `valid_output` rows and zero malformed or empty rows. Failure rate was `0.0` for every imported algorithm-model combination. On the imported data, the main issue is therefore not parser-visible failure but variability inside structurally valid outputs. The parsed edge-count distribution still varies sharply by model, for example ALGO2 ranges from a mean of `28.88` edges for Gemini 2.0 Flash to `84.55` for GPT-4o, with GPT-4o also showing an extreme maximum of `691` edges in one raw output. |
| Reviewer #1: "Reporting confidence intervals or distributional plots alongside mean performance would further reinforce the paper's core message about variability." | The repository should make it easy to export plot-ready summaries from existing evaluated results | Add a deterministic `lcm analyze figures` command that converts evaluated CSVs into tidy long-format metric rows with parsed algorithm/model metadata and full provenance | `done` | Implemented `analyze figures` for tidy row-level metric exports. Protected by CLI tests. Audit artifacts stored under `data/analysis_artifacts/revision_tracker/2026-03-21/figure_exports/`. | The imported corpus is now available in three plot-ready long tables: `8640` rows for ALGO1, `17280` rows for ALGO2, and `1440` rows for ALGO3. Each export carries `source_input`, `algorithm`, and `model` metadata, so distributional plots can be built across all imported models without manual file stitching or loss of provenance. |
| Reviewer #1: "the manuscript would be strengthened by ... a principled justification of the number of replications, for example via power or precision considerations rather than feasibility alone." Reviewer #2: "The decision to use 5 replications does not seem to be justified by power analysis." | Existing evaluated CSVs contain repeated runs but there is no explicit convergence or precision-based run calculation | Add deterministic `lcm analyze stability` and `lcm analyze replication-budget` commands over evaluated/stability CSVs, then compute audited cross-model stability and CI-based run-budget artifacts from the imported corpus | `done` | Implemented `analyze stability` plus `analyze replication-budget`, with grouped `n`, `mean`, `sample_std`, `min`, `max`, `range_width`, `coefficient_of_variation`, `required_total_runs`, and `additional_runs_needed` outputs. Protected by CLI and bundle tests. Audit artifacts stored under `data/analysis_artifacts/revision_tracker/replication_stability/`. | Across the imported corpus, ALGO1 and ALGO2 are almost fully stable across the five recorded repetitions, while ALGO3 is not. Only 2 of 576 ALGO1 accuracy conditions, 3 of 576 ALGO1 precision conditions, 2 of 576 ALGO1 recall conditions, and 1 of 1152 ALGO2 conditions per metric showed any run-to-run variation. By contrast, 44 of 96 ALGO3 recall conditions varied across repetitions. The 95% CI / 5% relative half-width run-budget calculation is more conservative: ALGO1 and ALGO2 accuracy already satisfy the target with the existing runs, a very small number of ALGO1/2 precision-recall rows require more, and ALGO3 recall does not. The worst-case required totals are `564` for ALGO1 precision, `481` for ALGO1 recall, `69` for ALGO2 precision, `48` for ALGO2 recall, and `23050` for ALGO3 recall, reflecting near-zero means combined with substantial instability. |
| Reviewer #1: "the manuscript stops short of linking these differences to specific computational mechanisms." Reviewer #2: "The article shows that there is randomness but it is not really telling how and why this happens..." Philippe: "a reviewer would like to drill into the 'LLM' factor... they want to know what it is within the LLM that makes it so..." | Existing offline code cannot identify provider-internal mechanisms directly, but it can distinguish output-side drift patterns in the imported raw results | Add a deterministic `lcm analyze variability` command over raw CSVs that quantifies edge-set drift and breadth drift across repeated runs, then generate audited cross-model summaries from the full imported corpus | `done` | Implemented `analyze variability` with `mean_pairwise_jaccard`, `min_pairwise_jaccard`, `exact_match_pair_rate`, `mean_edge_count`, `sample_std_edge_count`, and `union_edge_count`. Protected by unit and CLI tests. Audit artifacts stored under `data/analysis_artifacts/revision_tracker/2026-03-24/output_variability/`. | Across the imported corpus, ALGO1 and ALGO2 are almost fully repetition-stable at the raw edge-set level, with global mean pairwise Jaccard values of `0.9981` and `0.9997` and exact-match pair rates of `0.9979` and `0.9991`. ALGO3 is qualitatively different: its global mean pairwise Jaccard is only `0.0770`, its exact-match pair rate is `0.0010`, and its breadth-expansion ratio is `4.13`, showing that repeated runs both pick different edges and expand to much larger unions of distinct edges. Within ALGO3, deeper search amplifies this drift: `Depth=2` has lower mean pairwise Jaccard than `Depth=1` (`0.0692` versus `0.0848`) while also producing more edges on average (`60.84` versus `40.21`) and a larger edge-count spread (`20.47` versus `13.44`). These findings support a proxy mechanism interpretation based on multi-step amplification rather than parser-visible failure. |
| Reviewer #2: "Include at least one non-LLM baseline method... to contextualize the value proposition of using LLMs despite their inherent variability." | The repository needs a deterministic comparator that can be generated, evaluated, and audited without any new provider call | Implement one explicit structural baseline, `direct-cross-graph`, that proposes direct mother-graph cross-subgraph edges, then run it through the same evaluation surface and store baseline-vs-model comparison artifacts under `data/analysis_artifacts/revision_tracker/2026-03-21/baseline_comparison/`. | `done` | Implemented `lcm baseline` for Algorithms 1, 2, and 3 plus `lcm analyze baseline-comparison` for audited baseline-vs-model summaries. Protected by baseline and CLI tests. | Across the imported corpus, no imported model beats the deterministic baseline on any audited metric. In `baseline_advantage_summary.csv`, `models_beating_baseline` is `0` for ALGO1 accuracy, precision, and recall, `0` for ALGO2 accuracy, precision, and recall, and `0` for ALGO3 recall. The closest imported models are Gemini 2.5 Pro for ALGO1 and ALGO2 accuracy and precision, GPT-5 for ALGO1 and ALGO2 recall, and GPT-4o for ALGO3 recall, but all still remain below the baseline. This result should be interpreted carefully: the baseline uses mother-graph structure directly, so it is a conservative structural heuristic comparator rather than a learned replacement for the LLM workflows. |
| Reviewer #2: "Add at least one more domain with other features to reinforce arguments that it is cross-domain." | Requires new domain data and likely new raw runs, not just code | Keep out of the first software-response tranche | `deferred` | No code change yet. | No finding yet. |

## Verification Log

### 2026-03-21

- Added tests for `lcm analyze summary` before implementation.
- Confirmed red state with `uv run pytest tests/test_analysis_summary.py`.
- Implemented grouped summary analysis over evaluated CSVs.
- Confirmed green state with `uv run pytest tests/test_analysis_summary.py`.
- Added tests for `lcm analyze failures` before implementation.
- Confirmed red state with `uv run pytest tests/test_analysis_failures.py`.
- Implemented row-level failure classification for raw outputs.
- Confirmed green state with `uv run pytest tests/test_analysis_failures.py`.
- Extended `lcm analyze summary` and `lcm analyze failures` to support repeated `--input` values and preserve `source_input` provenance.
- Imported the primary non-archive legacy result files into `data/results/`.
- Aggregated findings across the imported algorithm-model result set before updating the findings column.
- Added tests for `lcm analyze stability` before implementation.
- Confirmed red state with `uv run pytest tests/test_analysis_stability.py`.
- Implemented grouped replication-stability analysis over evaluated CSVs.
- Confirmed green state with `uv run pytest tests/test_analysis_stability.py`.
- Generated audited replication-stability artifacts under `data/analysis_artifacts/revision_tracker/2026-03-21/replication_stability/`.
- Added tests for `lcm analyze hypothesis` before implementation.
- Confirmed red state with `uv run pytest tests/test_analysis_hypothesis.py`.
- Implemented paired factor-level hypothesis analysis with adjusted p-values.
- Confirmed green state with `uv run pytest tests/test_analysis_hypothesis.py`.
- Generated audited hypothesis-testing artifacts under `data/analysis_artifacts/revision_tracker/2026-03-21/hypothesis_testing/`.
- Added tests for `lcm analyze figures` before implementation.
- Confirmed red state with `uv run pytest tests/test_analysis_figures.py`.
- Implemented tidy figure-ready metric exports over evaluated CSVs.
- Confirmed green state with `uv run pytest tests/test_analysis_figures.py`.
- Generated audited figure-export artifacts under `data/analysis_artifacts/revision_tracker/2026-03-21/figure_exports/`.
- Added tests for `lcm baseline` and `lcm analyze baseline-comparison` before implementation.
- Confirmed red states with targeted `uv run pytest` runs for the new baseline and comparison behaviors.
- Implemented deterministic `direct-cross-graph` baseline generation for Algorithms 1, 2, and 3.
- Implemented audited baseline-vs-model comparison outputs over the imported evaluated corpus.
- Confirmed green state with targeted `uv run pytest` runs for baseline and baseline-comparison behaviors.
- Generated audited baseline artifacts under `data/baselines/direct-cross-graph/`.
- Generated audited baseline-comparison artifacts under `data/analysis_artifacts/revision_tracker/2026-03-21/baseline_comparison/`.
- Ran a small audited live Mistral pilot using `mistral-small-2603` and the live-debug probe matrix runner with a single model.
- Confirmed the provider path completed end to end and wrote prompts, responses, run logs, checkpoints, and summary CSVs under `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/pilot_20260321/`.
- Updated the live-debug finding note to reflect the pilot results:
  - ALGO1 improved accuracy and precision on the sampled rows, but recall dropped sharply.
  - ALGO2 improved accuracy slightly, but precision and recall both dropped on the sampled rows.
  - ALGO3 stayed at zero recall on both sampled rows, while producing more parsed edges than the historical rows.
- Ran a broader audited live Mistral pilot using both `mistral-small-2603` and `mistral-medium-2508` on the same representative probe set.
- Confirmed the broader provider path completed end to end and wrote prompts, responses, run logs, checkpoints, and summary CSVs under `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/pilot_20260321_sm_md/`.
- Updated the live-debug finding note again to reflect the two-model pilot:
  - ALGO1 accuracy improved for both models, with `mistral-small-2603` giving slightly higher precision and `mistral-medium-2508` producing fewer edges.
  - ALGO2 accuracy improved for both models, but recall remained much lower than the historical rows; `mistral-medium-2508` improved precision relative to `mistral-small-2603`, but neither model recovered the historical recall level.
  - ALGO3 remained at zero recall for both models on both sampled rows, even though the medium model produced slightly more parsed edges than the small model.
- Ran a wider audited live Mistral pilot with four sampled rows per algorithm and both `mistral-small-2603` and `mistral-medium-2508`.
- Confirmed the wider provider path completed end to end and wrote prompts, responses, run logs, checkpoints, and summary CSVs under `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/pilot_20260321_wide/`.
- Updated the live-debug finding note again to reflect the wider pilot:
  - ALGO1 remained higher-precision than the historical GPT-5 rows on average, while recall stayed lower; the medium model was slightly more precise, while the small model returned more edges.
  - ALGO2 remained a precision-recall tradeoff: both models improved accuracy, the medium model improved precision relative to the small model, and neither model recovered the historical recall level.
  - ALGO3 still stayed at zero recall across all sampled rows, but the number of parsed edges became much larger for the small model than for the medium model, suggesting the prompt is still not controlling output breadth tightly enough.
- Resumed the larger audited matrix attempt `big_20260321` after adding retry/resume/failure-isolation logic to the matrix runner.
- Confirmed the runner now reuses cached responses for completed rows, retries transient `HTTPError 429` and `URLError` failures, and records per-model failures without aborting the whole run.
- The larger attempt completed only partially because repeated DNS-level `URLError` failures continued during live Mistral calls.
- The completed portion wrote `75` scored result rows, including all `24` ALGO1 rows, `3` ALGO2 rows, and `0` ALGO3 rows.
- The run also recorded `47` model failures, all with `URLError` and the same DNS-level transport message.
- The partial run artifacts were kept under `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/big_20260321/` for auditability.
- Added a shared `call_with_retry` helper so the reusable Mistral chat and embedding clients retry transient `URLError` and retryable `HTTPError` conditions before surfacing a hard failure.
- Confirmed with targeted unit tests that transient transport failures are retried and that nonretryable HTTP errors are not retried.
- Ran single-probe live smoke tests for ALGO1, ALGO2, and ALGO3 using `mistral-small-2603`; each probe logged retry attempts, then failed cleanly when DNS resolution remained unavailable after the retry budget.
- The smoke runs wrote structured `manifest.json`, `state.json`, `run.log`, and `error.json` artifacts under `/tmp/lcm-smoke-algo1`, `/tmp/lcm-smoke-algo2`, and `/tmp/lcm-smoke-algo3`, confirming the failure path is auditable rather than silent.

### 2026-03-24

- Added tests for `lcm analyze variability` before implementation.
- Confirmed red state with a targeted `pytest` run failing because `analysis/variability.py` did not yet exist.
- Implemented deterministic raw-output variability analysis over repeated edge lists.
- Confirmed green state with targeted `pytest` runs for `tests/test_analysis_variability.py` and `tests/test_cli_variability.py`.
- Generated audited output-variability artifacts under `data/analysis_artifacts/revision_tracker/2026-03-24/output_variability/` across all imported ALGO1, ALGO2, and ALGO3 raw result files.

### 2026-03-28

- Added tests for `lcm analyze replication-budget` before implementation.
- Confirmed red state with targeted `pytest` runs failing because the CLI target and bundle outputs did not yet exist.
- Implemented deterministic CI-based run-budget analysis using the 95% CI / 5% relative half-width rule over the existing stability summaries.
- Extended `lcm analyze stability-bundle` to emit per-condition replication budgets and a cross-algorithm `replication_budget_overview.csv`.
- Confirmed green state with targeted `pytest` runs for `tests/analysis/test_analysis_replication_budget.py`, `tests/analysis/test_analysis_replication_budget_bundle.py`, `tests/analysis/test_analysis_stability.py`, `tests/analysis/test_analysis_stability_bundle.py`, and CLI-adjacent regression tests.
- Refreshed audited replication-stability artifacts under `data/analysis_artifacts/revision_tracker/replication_stability/`.
