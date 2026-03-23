# Tests Module Guide

This directory contains the regression suite for the repository.

Purpose:

- Validate algorithm behavior, CLI dispatch, analysis outputs, verification flows, and replay/debug utilities.
- Keep coverage close to the code boundaries described by the package-level `AGENTS.md` files.
- Protect fixture layout and snapshot contracts.

Working rules:

- Prefer focused tests that exercise one boundary at a time.
- Keep fixture updates deliberate and minimal.
- When adding a new module or submodule, add matching tests near the relevant feature area.
- If this directory grows further, split by concern into subdirectories such as `cli/`, `analysis/`, `algo1/`, `algo2/`, `algo3/`, and `debug/`.
- Treat snapshot and fixture tests as contract tests, not as a place for broad integration behavior changes.
