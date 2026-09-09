# Debugging A Failed Pipeline Run Prompt

## Goal

Debug `[failed run directory, log file, CI job, pipeline command, or batch ID]`
systematically. Identify the smallest confirmed cause of failure, preserve
useful artifacts, implement a minimal fix only if requested, and verify the
fixed path.

## Context to inspect

- Failure logs, stderr/stdout, status files, ledgers, manifests, checkpoints,
  run summaries, and partial output artifacts.
- The exact command, config, environment variables, input paths, output paths,
  and runtime context used for the failed run.
- Pipeline source code, orchestration scripts, CLI entry points, retry logic,
  parser code, artifact writers, and state/resume code.
- Tests covering the failed stage and nearby failure modes.
- Recent code or data changes if a regression window is available.

## Constraints

- Use systematic debugging: reproduce, localize, explain, fix, and verify.
- Do not delete failed-run artifacts unless the user explicitly asks and the
  evidence has been captured.
- Do not guess from one artifact in isolation when ledgers, logs, configs, and
  run directories can be compared.
- Distinguish infrastructure failure, data-contract failure, parser failure,
  model/provider failure, and code regression.
- Do not add broad retries or ignore errors until the failure mode is understood.
- Do not expose secrets or private raw data from logs.

## Expected deliverables

- Failure timeline and smallest confirmed failing stage.
- Root cause with evidence, or a bounded statement of what remains unclear.
- Minimal reproduction command or fixture when feasible.
- Fix and regression test if implementation is requested.
- Verification evidence for the fixed path or for the diagnosis.
- Notes on preserved artifacts and any cleanup that remains.

## Verification requirements

- Inspect the failed-run artifacts before changing code.
- Reproduce the failure with the smallest local command or test when feasible.
- Add a regression test before fixing code when the failure is a confirmed code
  defect.
- Run the focused test or CLI command after the fix.
- Run the broader relevant pipeline or verification command when practical.
- Record exact commands, paths, and observed outcomes.

## Final response format

Use this structure:

```markdown
## Diagnosis

- Failure class: infrastructure/data-contract/parser/provider/code/unclear
- Root cause: ...
- Evidence: ...

## Timeline

1. ...

## Changes Made

- None, or `path`: ...

## Verification

- Reproduction: `command` -> observed result
- Fix check: `command` -> observed result
- Broader check: `command` -> observed result
- Not run: `command` -> reason

## Preserved Artifacts

- `path`: ...

## Remaining Risk

- ...
```
