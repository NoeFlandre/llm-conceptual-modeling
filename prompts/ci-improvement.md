# CI Improvement Prompt

## Goal

Improve CI for `[repository, branch, or workflow]` so the checks match the
repository's local verification workflow, catch important regressions, and stay
fast enough for routine development.

## Context to inspect

- Existing CI workflows, scripts, Makefile targets, README verification
  commands, and contributor docs.
- Project configuration, dependency files, lock files, supported language
  versions, and test configuration.
- Existing local quality gates such as tests, linting, formatting, type checks,
  CLI smoke checks, artifact checks, and reproducibility checks.
- Recent failures or slow jobs if logs are provided.
- Secrets or external services documented as optional or required. Do not
  assume they exist.

## Constraints

- Prefer the repository's existing local commands over new CI-only behavior.
- Do not add secret-dependent, network-dependent, GPU-dependent, or paid-service
  jobs unless the user explicitly asks and the repository documents the
  requirement.
- Keep the first improvement minimal and reviewable.
- Do not mask failures with broad `continue-on-error` settings unless the job is
  explicitly informational.
- Avoid duplicating checks that already run in another required job without a
  reason.
- Do not change production code unless a CI failure exposes a confirmed defect
  and the user asks for a fix.

## Expected deliverables

- Updated CI workflow, script, Makefile target, or documentation as needed.
- Alignment between local commands and CI commands.
- Clear job names and failure surfaces.
- Caching only when it is supported by existing dependency management and does
  not hide stale state.
- A summary of what each job protects and what remains intentionally out of CI.

## Verification requirements

- Run the local commands that CI will execute before editing when feasible.
- Validate workflow syntax with available local tooling if the repository
  provides it.
- After edits, run the same local checks that the CI job will run.
- If a full CI run cannot be triggered locally, state that limitation and
  provide the exact expected CI command sequence.
- Record exact commands and observed outcomes.

## Final response format

Use this structure:

```markdown
## CI Changes

- `path`: ...

## Local/CI Alignment

- Local command: `...`
- CI command: `...`

## Verification

- Baseline: `command` -> observed result
- After edits: `command` -> observed result
- Not run: `command` -> reason

## Remaining CI Gaps

- ...
```
