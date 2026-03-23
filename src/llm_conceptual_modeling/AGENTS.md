# Package Root Guide

This package is the top-level entry point for the project.

Purpose:

- Connect the CLI, generation manifest helpers, experiment manifest parsing, and verification workflows.
- Keep the top-level package thin and use the subpackages for real implementation work.

Key files:

- `cli.py`: CLI parser and dispatch.
- `generation.py`: algorithm-agnostic generation stub payload routing.
- `verification.py`: repository-wide verification orchestration.
- `experiment_manifest.py`: experiment-manifest parsing and validation.

Working rules:

- Keep top-level orchestration small and defer domain logic to the algorithm, analysis, common, or command subpackages.
- If `cli.py` or `verification.py` grows further, split them by command or verification concern.
- Treat this directory as a coordination layer, not a place for new feature logic.
