# Commands Module Guide

This module owns the `lcm` CLI command handlers.

Purpose:

- Translate parsed CLI arguments into calls into the algorithm, analysis, and verification layers.
- Keep the command layer thin so most business logic lives in the underlying modules.
- Let `cli.py` stay import-light by loading handlers only when a command is actually dispatched.

Key files:

- `analyze.py`: reviewer-facing analysis commands.
- `audit.py`: repository audit commands.
- `baseline.py`: structural baseline command dispatch.
- `doctor.py`: lightweight health-report command.
- `eval.py`: evaluation command dispatch for algorithm outputs.
- `factorial.py`: factorial-analysis command dispatch.
- `generate.py`: generation-manifest and live-provider orchestration.
- `probe.py`: probing workflows for algorithm families.
- `verify.py`: verification command dispatch.

Working rules:

- Keep handlers thin and deterministic.
- Keep imports local in the larger orchestration handlers when that avoids pulling heavy provider stacks into module import time.
- Move reusable logic into `analysis`, `common`, or algorithm modules instead of adding complexity here.
- If a handler file grows beyond the size target, split it by subcommand or algorithm family.
- `generate.py` and `probe.py` are the first candidates for decomposition because they already hold the most orchestration logic.
