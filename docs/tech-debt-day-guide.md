# Tech Debt Day Guide

This document is a step‑by‑step operating guide for a quality‑focused cleanup
pass. The goal is to elevate maintainability without regressions by applying a
disciplined TDD loop, aggressive but safe modularization, and YAGNI‑driven
minimalism.

Use this guide as a checklist. Do not skip steps.

## Goals

- Reduce cognitive load: fewer giant files, clearer module boundaries.
- Remove dead code, duplicate logic, and stale docs.
- Keep behavior stable. No regressions.
- Add guardrails (tests, invariants, CI checks) that make wrongness obvious.
- Leave the repo easier to reason about for the next contributor.

## Non‑Negotiable Principles

- **TDD red/green** for every meaningful behavior change.
- **Minimal diff** that fixes the specific issue or reduces specific debt.
- **YAGNI**: do not add features or abstractions without a current need.
- **One responsibility per file** whenever practical.
- **Small verified loops** first; only then run broader checks.
- **No guessing**: verify assumptions against code, fixtures, and outputs.
- **Document truthfully**: align docs with what the code does now.

## Preparation (10–20 minutes)

1. **Pull and inspect branch state**
   - `git status`
   - `git log -n 5 --oneline`
2. **Identify baseline commands**
   - Preferred: `uv run pytest`, `uv run ruff check .`, `uv run ty check`
   - If repo defines `make` targets, note them (`make verify`, `make test`).
3. **Run the smallest baseline**
   - Example: `uv run pytest tests/verification/test_repo_tree_hygiene.py -q`
   - Record the command and result in your working notes.

## Choose the Focus Area (5 minutes)

Pick exactly one slice for a micro‑pass. Examples:

- One oversized module (e.g., `hf_experiments.py`).
- One flat directory with too many files.
- One subpackage missing a local README.
- One pile of duplicate helpers across modules.

The scope should fit in a focused change and a focused test.

## Micro‑Pass Workflow (Repeatable Loop)

### Step 1: Inspect

- Read the local package README.
- Read the smallest relevant module(s).
- Locate the existing tests around the behavior.
- Search for duplicate or dead paths with `rg`.

### Step 2: Write a Targeted Red Test

Create or update the narrowest test that captures the intended change.

Use tests to enforce:

- API boundary behavior
- stable import locations
- deterministic outputs
- layout/hygiene constraints
- invariants that should never break

Confirm it fails before you implement the change.

### Step 3: Implement the Minimal Change

Guidelines:

- Split files by responsibility, not by arbitrary size.
- Move logic into a package, then make the old file a thin compatibility shim.
- Remove dead wrappers only after callers are updated.
- Avoid speculative refactors. Only move what you can test.

### Step 4: Green Test

Re‑run the exact red test and confirm it passes.

If it does not pass, fix immediately. Do not proceed until green.

### Step 5: Update Local Docs

Update:

- package `README.md`
- `docs/onboarding.md` if a workflow or boundary changed
- `docs/README.md` if the docs structure changed

### Step 6: Run the Small Gate

Run the smallest safe verification:

- `uv run pytest path/to/your/new_test.py -q`
- `uv run ruff check path/to/changed/files`

### Step 7: Commit the Micro‑Pass

Commit messages should be explicit:

- “refactor: package X helpers”
- “tests: add guardrail for Y”
- “docs: clarify Z boundary”

Push if the repo policy expects it.

## Refactor Patterns (Preferred)

### Pattern: Package Split with Compatibility Shim

1. Move implementation into `package/` folder.
2. Keep old root file as a thin import shim.
3. Update callers to use the package directly.
4. Add a test that ensures the shim still delegates correctly.

Why: minimizes breakage while letting the new structure take over.

### Pattern: Consolidate Duplicate Helpers

1. Pick the best implementation (or the simplest).
2. Move it into a single package module.
3. Update all callers.
4. Delete the duplicates.
5. Add tests to enforce the canonical import path.

### Pattern: Data‑Path Canonicalization

1. Use the canonical `data/` layout.
2. Add or update path helpers.
3. Add a test that enforces the canonical path.
4. Update docs with the correct path.

## Aggressive Modularity Without Fragmentation

Aim for:

- < ~300–500 lines per file for core logic modules.
- < ~10–12 files per folder when possible.

If a folder grows too large:

- introduce a subfolder with a README.
- avoid a “misc” folder unless it has clear rules.

If a file grows too large:

- split by responsibility.
- keep entry points thin.
- push heavy logic down into focused modules.

## TDD Red/Green Discipline (Checklist)

For each change:

- [ ] test exists and fails before the change
- [ ] change implemented minimally
- [ ] test passes after the change
- [ ] small gate run completed
- [ ] docs updated if behavior or layout changed

If any box is not checked, do not move on.

## Documentation Rules

- Every major package has a README.
- Onboarding is the source of truth for repo‑wide rules.
- No stale “revision summary” docs.
- No undocumented new subpackages.
- Keep docs factual and minimal: describe behavior, not intention.

## Common Debt Targets

### Oversized Files

Look for:

- multiple unrelated helper clusters
- large blocks of orchestration + data parsing + logging in one file

Split into:

- `runtime.py` (orchestration)
- `helpers.py` (pure helpers)
- `types.py` (dataclasses, enums)

### Orphaned Scripts

If a script is no longer part of an active workflow:

- move it under `scripts/` with a README
- or delete it if it is obsolete

### Duplicate Parsing / Validation Logic

Consolidate to a single module, then add:

- a regression test for known tricky input
- a contract test for canonical output format

### Stale Docs

Delete any doc that:

- contradicts the current structure
- repeats onboarding information
- describes a retired workflow

## Verification Gates (Recommended)

Minimal gate for each micro‑pass:

```bash
uv run pytest path/to/your_test.py -q
uv run ruff check path/to/changed/files
```

End‑of‑day gate:

```bash
uv run pytest
uv run ruff check .
uv run ty check
uv run lcm verify all --json
```

## Quality Bar Examples

Good changes:

- Split a 1,000‑line file into 3 files by concern, added a package README, and
  added an alias import test. All tests green.
- Removed duplicate config helpers and replaced call sites with one canonical
  module. Added a guardrail test.
- Updated onboarding and package docs after a module split.

Bad changes:

- Large refactor without tests.
- New helper with no callers and no tests.
- File split without updating import paths or docs.

## How To Decide You Are Done

A tech‑debt day is complete when:

- the planned slices are done
- the diff is minimal and reviewable
- every change has a test or a clear verification command
- docs match the new structure
- the end‑of‑day gate is green

If any of those are missing, the pass is not done.

## Suggested Micro‑Pass Roadmap

If you need a prioritized sequence, use this:

1. Flatten oversized files into package modules + shims.
2. Consolidate duplicate helpers into canonical modules.
3. Add package READMEs where missing.
4. Add guardrail tests for structure and path contracts.
5. Remove obsolete scripts/docs.

Repeat until the repo is stable, then stop.
