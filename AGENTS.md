# Repository Agent Guide

This repository is being modularized around small, self-contained directories and files.

Rules for future work:

- Keep directories under 20 files when possible.
- Split a directory into subdirectories once it grows beyond that limit.
- Keep files under 300 lines.
- Split large files into focused modules instead of extending them further.
- Add or update an `AGENTS.md` in each module so future work has local context.

Repository structure:

- `src/llm_conceptual_modeling/common`: shared primitives used by algorithms and commands.
- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`: algorithm-specific pipelines and adapters.
- `src/llm_conceptual_modeling/commands`: CLI command handlers.
- `src/llm_conceptual_modeling/analysis`: deterministic analysis and reporting helpers.
- `src/llm_conceptual_modeling/post_revision_debug`: debug and replay utilities for revision work.

When editing, prefer the smallest viable change that improves separation of concerns.
