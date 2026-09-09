# Paper-To-Code Consistency Audit Prompt

## Goal

Audit consistency between `[paper, manuscript section, method description, or
reviewer claim]` and the repository implementation, prompts, configs, tests,
and artifacts. Identify confirmed mismatches, ambiguous claims, undocumented
implementation choices, and missing verification.

## Context to inspect

- The paper, manuscript excerpt, reviewer response, or method notes supplied by
  the user.
- Repository README, architecture docs, method guides, and package-local
  README files.
- Algorithm or pipeline implementation files.
- Prompt templates, config files, default parameters, factorial designs,
  seeds, stopping rules, parsing rules, and output schemas.
- Tests, fixtures, snapshots, parity checks, and generated artifacts that
  demonstrate implemented behavior.

## Constraints

- If the paper or manuscript is not available in the workspace, ask for it or
  limit the audit to repository documentation. Do not invent paper claims.
- Quote only short excerpts needed to identify the claim being checked.
- Distinguish paper wording, code behavior, and inference.
- Do not treat a mismatch as a defect until the expected source of truth is
  established.
- Do not modify code or text unless explicitly asked.
- Do not disclose private reviewer material beyond what is needed for the
  local task.

## Expected deliverables

- A claim-by-claim consistency table.
- Confirmed mismatches with code/doc/artifact evidence.
- Ambiguous claims that need author judgment.
- Implementation details that should be documented in the paper or docs.
- Missing tests or fixtures that would make consistency easier to verify.
- Recommended edits, ordered by severity and review risk.

## Verification requirements

- Run focused tests or CLI commands that demonstrate the checked behavior when
  feasible.
- Inspect artifacts or generated outputs directly when claims concern output
  format, counts, metrics, prompts, or experiment design.
- Compare configured defaults against paper-described defaults.
- Record exact files, commands, and observations.
- If behavior cannot be executed locally, state the limitation and rely only on
  inspected source evidence.

## Final response format

Use this structure:

```markdown
## Consistency Verdict

- Overall status: consistent / partially consistent / inconsistent / unclear

## Claim Table

| Claim | Source | Code or artifact evidence | Status | Action |
| --- | --- | --- | --- | --- |
| ... | ... | ... | confirmed/plausible/unclear/false | ... |

## Highest-Risk Mismatches

- ...

## Verification

- `command` -> observed result
- Artifact inspected: `path` -> observation

## Recommended Next Edits

1. ...
```
