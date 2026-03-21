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
- `events.jsonl`
  Structured machine-readable event log.
- `manifest.json`
  Exact model and probe configuration.
- per-row `prompt.txt`
  Prompt snapshots.
- per-model `*_response.json`
  Raw provider responses.

This is intended to make every run replayable and auditable.

## Current Findings

The first canonical run is:

- `data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/representative_matrix_v1/`

From `probe_summary_by_algorithm_and_model.csv`:

- For the representative ALGO1 rows, `mistral-small-2603` and `mistral-medium-2508` increased mean accuracy and precision relative to the imported GPT-5 historical rows, but reduced recall sharply. This suggests that a strict, high-precision prompt can trade recall for precision rather than reproducing the historical GPT-5 balance.
- For the representative ALGO2 rows, `mistral-small-2603` also improved mean accuracy and precision relative to the imported GPT-5 historical rows, while recall still dropped sharply. The same precision-recall tradeoff appears again.
- For the representative ALGO3 rows, all three tested Mistral models remained at zero recall under the current structured prompt. This differs from earlier ad hoc ALGO3 probes, which means ALGO3 is highly sensitive to prompt framing and edge-budget choices.

## Interpretation

The current post-revision evidence does not support one simple claim such as "Mistral is better" or "Mistral is worse."

Instead, it supports three narrower points:

- output-format control matters
- prompt framing strongly affects which metric improves
- ALGO3 remains the most fragile workflow and likely needs the most redesign attention
