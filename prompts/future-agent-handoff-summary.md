# Future-Agent Handoff Summary Prompt

## Goal

Create a concise handoff summary for the next coding agent working on
`[repository, branch, task, or incident]`. The summary should let the next agent
continue without re-discovering context, while being explicit about what is
verified, what is uncertain, and what remains.

## Context to inspect

- Current user request and any later corrections or scope changes.
- Repository instructions, README, onboarding docs, and relevant package-local
  README files.
- Git status or changed-file list.
- Recent commands run, outputs observed, failures, and generated artifacts.
- Modified files, tests added, decisions made, blocked items, and open
  questions.
- Any external data, credentials, services, or local-only state involved in the
  task.

## Constraints

- Do not invent progress. Clearly separate done, in progress, blocked, and not
  started.
- Do not include secrets, tokens, private raw data, or unnecessary personal
  paths.
- Do not say tests pass unless they were run and passed.
- Keep the handoff useful but compact.
- Include enough exact file paths and commands for continuation.
- Flag dirty worktree state that predates the current task when known.

## Expected deliverables

- Objective and current status.
- Files changed and why.
- Commands run with observed outcomes.
- Important decisions and reasoning.
- Known failures, blockers, and uncertainty.
- Next recommended steps in order.
- Any cleanup or verification still required before handoff is complete.

## Verification requirements

- Inspect the current changed-file state before writing the summary.
- Confirm whether long-running commands are still active.
- Re-check the newest user request so the handoff is not anchored to an older
  task.
- Include only verification results that were actually observed.
- If no verification was run, state that plainly.

## Final response format

Use this structure:

```markdown
## Objective

...

## Current Status

- Done: ...
- In progress: ...
- Blocked: ...

## Files Changed

- `path`: ...

## Verification

- `command` -> observed result
- Not run: `command` -> reason

## Decisions And Constraints

- ...

## Next Steps

1. ...

## Risks Or Unknowns

- ...
```
