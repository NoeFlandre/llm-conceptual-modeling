# Post Revision Debug Module Guide

This module contains debug, replay, and probe-context utilities that support revision work.

Purpose:

- Keep live-provider debugging artifacts separate from the deterministic research pipeline.
- Provide reusable helpers for recording probe runs, parsing debug payloads, and post-processing replay outputs.

Key files:

- `artifacts.py`: JSONL event logging for debug artifacts.
- `run_context.py`: probe-run context management and manifest bookkeeping.
- `mistral_probe.py`: parsing and evaluation helpers for Mistral probe outputs.
- `replay_postprocess.py`: replay summary aggregation and exported post-processing tables.

Working rules:

- Keep debug-only logic isolated from production analysis modules.
- Prefer adding small parsing or logging helpers here rather than scattering them across command handlers.
- If `run_context.py`, `mistral_probe.py`, or `replay_postprocess.py` grows further, split by artifact type or workflow stage.
- Do not let this module become a second home for algorithm logic; keep it as support tooling for revision work only.
