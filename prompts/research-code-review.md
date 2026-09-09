# Research Code Review Prompt

## Goal

Review `[branch, PR, commit range, or changed files]` as a rigorous research
software code review. Prioritize correctness, reproducibility, method fidelity,
data integrity, failure modes, and missing tests over style preferences.

## Context to inspect

- Repository instructions: `AGENTS.md`, `README.md`, contributor docs, and any
  package-local README files.
- Project configuration: dependency files, test configuration, lint/type
  configuration, Makefile, CI workflows, and CLI entry points.
- The changed files and their nearest tests.
- Method documentation, paper notes, experiment configs, schemas, fixtures,
  golden outputs, and artifact manifests that define expected behavior.
- Existing issue, PR, or task description if one is provided.

## Constraints

- Take a code-review stance. Findings must lead the response.
- Do not rewrite the code unless explicitly asked.
- Do not report speculative concerns as confirmed defects.
- Classify evidence as confirmed, plausible, unclear, or false.
- Focus on issues that can affect correctness, reproducibility, maintainability,
  or reviewability.
- Avoid broad architectural advice unless it is directly tied to the reviewed
  change.
- Do not expose private data or secrets from logs, configs, or artifacts.

## Expected deliverables

- Ordered findings with severity, file path, line reference, observed behavior,
  expected behavior, and why it matters.
- Missing-test and missing-verification risks.
- Any paper/method/data-contract mismatch found during review.
- Open questions only when the answer changes the review conclusion.
- A brief change summary only after findings.

## Verification requirements

- Run the repository's existing baseline or the narrowest relevant tests before
  drawing conclusions. If the baseline is already failing, record the failure
  and separate it from reviewed-change risks.
- Run focused tests or CLI checks that exercise the changed behavior when
  feasible.
- Inspect generated outputs or fixtures directly when the change affects data,
  analysis, or artifacts.
- Record exact commands and observed outcomes.
- If a check cannot be run, state why and what risk remains.

## Final response format

Use this structure:

```markdown
## Findings

- [P1/P2/P3] Title
  File: `path:line`
  Evidence: confirmed/plausible/unclear/false
  Issue: ...
  Impact: ...
  Suggested fix or test: ...

## Open Questions

- ...

## Verification

- `command` -> observed result
- Not run: `command` -> reason

## Change Summary

- ...

## Residual Risk

- ...
```
