# Test-Suite Expansion Prompt

## Goal

Expand the test suite for `[module, workflow, bug, feature, or risk area]` so
important behavior is externally checkable. Prefer focused tests that protect
contracts, failure modes, fixtures, schemas, and deterministic outputs.

## Context to inspect

- Existing tests near the target behavior.
- Source code, package README files, CLI docs, schemas, fixtures, and golden
  outputs.
- Bug reports, review comments, incidents, or TODOs motivating the coverage.
- Test configuration and established test style.
- Existing utilities, builders, fixtures, snapshot conventions, and CLI helpers.

## Constraints

- Do not add brittle tests that only mirror implementation details.
- Do not change production behavior unless the task explicitly includes a fix
  or a new test exposes a confirmed bug that the user asks you to fix.
- Keep tests deterministic and fast enough for local iteration.
- Prefer one narrow regression test over a broad integration test unless the
  behavior spans multiple boundaries.
- Avoid live network, credentials, hidden system dependencies, and timing-based
  assertions unless they are the actual contract under test.
- Use repository conventions instead of inventing a new test harness.

## Expected deliverables

- New or updated tests with clear names and focused assertions.
- Fixtures or sample data only when they make behavior easier to verify.
- Any minimal production fix explicitly requested or required by a confirmed
  failing test.
- A short explanation of the contract each test protects.
- Notes on remaining coverage gaps.

## Verification requirements

- Run the baseline or nearest existing test before editing.
- For bugfix work, add a failing test first and show that it fails for the
  expected reason.
- Run the focused new tests until they pass.
- Run the relevant broader test group after focused tests pass.
- Run lint/type checks if production code changed.
- Record exact commands and observed outcomes.

## Final response format

Use this structure:

```markdown
## Tests Added

- `test_name` in `path`: contract protected

## Production Changes

- None, or `path`: reason

## Verification

- Baseline: `command` -> observed result
- Red test: `command` -> expected failure
- Green test: `command` -> observed result
- Broader check: `command` -> observed result

## Remaining Gaps

- ...
```
