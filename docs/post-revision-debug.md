# Post-Revision Debugging

This document describes the version split and the current live-debug workflow used after the pre-revision repository state was frozen.

## Repository Split

- Pre-revision frozen state:
  `pre-revision-freeze-2026-03-21`
- Post-revision live-debug branch:
  `post-revision-debug`

The purpose of this split is to keep the imported historical corpus and its original audit trail usable without contamination from later experiments.

## Artifact Policy

- Historical imported results stay under `data/results/`.
- Deterministic post-processing and reviewer-facing analyses stay under `data/analysis_artifacts/revision_tracker/`.
- New live-provider debugging outputs go under `data/analysis_artifacts/post_revision_debug/`.

## Mistral Probe Runner

The first reproducible live-debug tool is:

```bash
export MISTRAL_API_KEY=...
PYTHONPATH=src python scripts/post_revision_debug/run_mistral_probe_matrix.py \
  --run-name representative_matrix_v1
```

By default the runner probes:

- `mistral-small-2506`
- `mistral-small-2603`
- `mistral-medium-2508`

It uses JSON schema mode on `/v1/chat/completions` so output-format compliance is enforced at the API level rather than left to prompt-only instructions.

## Logging

Each canonical run writes:

- `run.log`
  Human-readable execution trace.
- `state.json`
  Resume state and completed stage list.
- `execution_checkpoint.json`
  Serializable checkpoint that can be reused after a partial interruption.
- `events.jsonl`
  Structured machine-readable event log.
- `manifest.json`
  Exact model and probe configuration.
- per-row `prompt.txt`
  Prompt snapshots.
- per-model `*_response.json`
  Raw provider responses.

This is intended to make every run replayable and auditable.

The `lcm generate ...` and `lcm probe ...` entry points also accept `--resume` for rerunning a partially completed run without reissuing provider calls for completed stages. The live Mistral matrix runner also retries transient HTTP 429 responses and reuses cached per-row response files when `--resume` is set, so a partial matrix can be resumed without restarting completed model calls.

For a compact audit of the paper-facing contract, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from llm_conceptual_modeling.cli import main; raise SystemExit(main(['audit', 'paper-alignment', '--json']))"
```

The audit report includes the Method 2 `0.01` threshold, the Mistral embedding model, resume support, fixture-backed metric schemas, and probe checkpointing evidence in a single JSON payload.

## Current Findings

The first canonical run is:

- `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/representative_matrix_v1/`

The smaller live pilot used to validate the provider plumbing is:

- `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/pilot_20260321/`

The next pilot added a second model for the same representative probe set:

- `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/pilot_20260321_sm_md/`

From `probe_summary_by_algorithm_and_model.csv`:

- For the representative ALGO1 rows, `mistral-small-2603` and `mistral-medium-2508` increased mean accuracy and precision relative to the imported GPT-5 historical rows, but reduced recall sharply. This suggests that a strict, high-precision prompt can trade recall for precision rather than reproducing the historical GPT-5 balance.
- For the representative ALGO2 rows, `mistral-small-2603` also improved mean accuracy and precision relative to the imported GPT-5 historical rows, while recall still dropped sharply. The same precision-recall tradeoff appears again.
- For the representative ALGO3 rows, all three tested Mistral models remained at zero recall under the current structured prompt. This differs from earlier ad hoc ALGO3 probes, which means ALGO3 is highly sensitive to prompt framing and edge-budget choices.

The smaller one-model pilot on `mistral-small-2603` points in the same direction:

- ALGO1 improved accuracy and precision on the sampled rows, but recall dropped sharply.
- ALGO2 improved accuracy slightly, but precision and recall both dropped on the sampled rows.
- ALGO3 stayed at zero recall on both sampled rows, while producing more parsed edges than the historical rows.

The two-model pilot shows the same broad pattern:

- ALGO1 accuracy improves for both models, with `mistral-small-2603` giving slightly higher precision and `mistral-medium-2508` producing fewer edges.
- ALGO2 accuracy improves for both models, but recall remains much lower than the historical rows; `mistral-medium-2508` improves precision relative to `mistral-small-2603`, but neither model recovers the historical recall level.
- ALGO3 remains at zero recall for both models on both sampled rows, even though the medium model produces slightly more parsed edges than the small model.

The wider four-row pilot keeps the same qualitative picture:

- ALGO1 remains higher-precision than the historical GPT-5 rows on average, while recall is still lower; the medium model is slightly more precise, while the small model returns more edges.
- ALGO2 remains a precision-recall tradeoff: both models improve accuracy, the medium model improves precision relative to the small model, and neither model recovers the historical recall level.
- ALGO3 still stays at zero recall across all sampled rows, but the number of parsed edges becomes much larger for the small model than for the medium model, which suggests the prompt is still not controlling output breadth tightly enough.

The larger resumed matrix attempt `big_20260321` is now also an audited artifact. It did not complete all probe combinations because the live transport layer repeatedly failed with `URLError` on several model calls, but the runner kept the completed outputs and isolated the failures per model instead of aborting the whole run. The completed portion is still informative:

- 75 scored result rows were written before the run stopped.
- All 24 ALGO1 rows completed for the three models.
- Only 1 ALGO2 row completed for one model; the remaining ALGO2 cells were recorded as transport failures.
- No ALGO3 result rows completed in this attempt.
- 47 model failures were recorded, all with `URLError` and the same DNS-level transport message.

This means the runner is now usable for partial reruns and failure auditing, but the environment still does not support a clean full big run with the current Mistral endpoints and local network conditions.

## Interpretation

The current post-revision evidence does not support one simple claim such as "Mistral is better" or "Mistral is worse."

Instead, it supports three narrower points:

- output-format control matters
- prompt framing strongly affects which metric improves
- ALGO3 remains the most fragile workflow and likely needs the most redesign attention
