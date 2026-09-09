# No-Behavior-Change Refactor Prompt

## Goal

Refactor `[module, package, files, or branch]` without changing runtime
behavior, public interfaces, output formats, data semantics, or documented
commands. Improve clarity, locality, maintainability, and testability with the
smallest reviewable diff.

## Context to inspect

- Repository instructions, architecture docs, onboarding docs, and local
  package README files.
- The target files, their direct callers, public import surfaces, tests, and
  CLI commands.
- Fixtures, golden outputs, schema checks, and snapshot tests that define
  existing behavior.
- Open issues or review comments that describe the refactor motivation.

## Constraints

- No behavior changes unless the user explicitly approves them.
- Do not rename public commands, flags, files, columns, schemas, config keys,
  or import paths unless a compatibility path is preserved and tested.
- Prefer moving code into clearer boundaries over introducing speculative
  abstractions.
- Keep edits mechanical and small enough to review.
- Delete dead code only after confirming it has no callers and no documented
  role.
- If a behavior change appears necessary, stop and report the evidence before
  implementing it.

## Expected deliverables

- Refactored code with equivalent behavior.
- Characterization tests or strengthened existing tests when behavior is not
  already pinned down.
- A moved-symbol or changed-boundary summary when files or functions move.
- Removed dead code, if confirmed safe.
- Updated local README or docs only when ownership boundaries change.

## Verification requirements

- Run the existing baseline before editing and record the result.
- Add or run characterization tests before changing code when behavior is not
  already covered.
- Run focused tests after each meaningful move.
- Run the broader relevant quality gate before handoff.
- Compare representative outputs before and after when the refactor touches
  serialization, analysis, CLI output, or artifacts.
- Record commands and observed outcomes.

## Final response format

Use this structure:

```markdown
## Summary

- ...

## Behavior Preservation

- Public interfaces checked: ...
- Outputs compared: ...

## Files Changed

- `path`: ...

## Verification

- Baseline before edits: `command` -> observed result
- After edits: `command` -> observed result
- Not run: `command` -> reason

## Residual Risk

- ...
```
