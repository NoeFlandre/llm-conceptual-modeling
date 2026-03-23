# Scripts Guide

This directory is reserved for repo-level automation and reproducibility scripts.

Purpose:

- Host small operational scripts that sit outside the Python package.
- Keep ad hoc workflows separate from library code and tests.

Working rules:

- Keep scripts short and single-purpose.
- If a script grows beyond the size target, split it into helper modules or subdirectories.
- Add an `AGENTS.md` in any new script subdirectory so the intent stays local.
- Prefer reusing package functions instead of duplicating logic here.
