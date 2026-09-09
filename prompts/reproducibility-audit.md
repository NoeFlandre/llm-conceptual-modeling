# Reproducibility Audit Prompt

## Goal

Audit `[repository root, branch, release candidate, or artifact bundle]` for
reproducibility. Determine whether a fresh agent or reviewer can recreate the
documented environment, run the verification commands, reproduce key outputs,
and understand any external data requirements.

## Context to inspect

- `README.md`, onboarding docs, package-local README files, runbooks, and
  method guides.
- Dependency and environment files such as `pyproject.toml`, lock files,
  Dockerfiles, Makefile, CI workflows, and scripts.
- CLI commands, help text, examples, and verification entry points.
- Data layout docs, artifact manifests, schema files, fixtures, golden outputs,
  and external data references.
- Tests that encode reproducibility, parity, schema, or hygiene contracts.

## Constraints

- Run commands before claiming they work.
- Do not assume external buckets, credentials, GPUs, APIs, or private datasets
  are available.
- Separate offline reproducibility from live-provider or remote-infrastructure
  reproducibility.
- Do not modify code unless explicitly asked. If fixes are requested, make the
  smallest change that improves reproducibility and verify it.
- Do not include private data, tokens, or unpublished file contents in the
  report.

## Expected deliverables

- A reproduction path from clean checkout to local verification.
- A list of commands that work, commands that fail, and commands that are
  documented but unverified.
- A dependency/environment assessment, including missing pins or implicit
  system assumptions.
- A data/artifact availability assessment, including which paths are committed,
  generated, external, or optional.
- Concrete recommended fixes, ordered by impact and effort.

## Verification requirements

- Run the documented setup command if feasible.
- Run the narrowest useful baseline check before deeper validation.
- Run the main verification command if it is local and practical.
- Run representative CLI help or smoke commands.
- Compare documented paths and commands against the actual repository state.
- Record exact command outputs or concise failure summaries.

## Final response format

Use this structure:

```markdown
## Reproducibility Verdict

- Status: reproducible / partially reproducible / not reproducible / unclear
- Scope verified: ...

## Working Path

1. `command`
2. `command`

## Findings

- Severity: high/medium/low
  Evidence: confirmed/plausible/unclear
  Issue: ...
  Fix: ...

## Data And Artifact Notes

- ...

## Verification Log

- `command` -> observed result
- Not run: `command` -> reason

## Recommended Next Changes

1. ...
```
