# Prompt Style Guide

Future prompts in this library should match the repository's engineering
preferences: minimal scope, explicit evidence, reproducible commands, and
verification before claims.

## Required Structure

Every prompt should include these sections:

- `Goal`
- `Context to inspect`
- `Constraints`
- `Expected deliverables`
- `Verification requirements`
- `Final response format`

Keep the prompt self-contained. It should still work when copied into a fresh
agent session without the surrounding README.

## Writing Rules

- Use a neutral technical tone. Do not praise, sell, or overstate.
- Prefer concrete instructions over vague quality words.
- Ask the agent to inspect before editing and to run the baseline before making
  changes when the task involves repository work.
- Require evidence: file paths, commands run, outputs observed, and artifacts
  produced.
- Distinguish confirmed facts from plausible inferences and unresolved
  uncertainty.
- State when the agent should stop and ask for input, especially for ambiguous
  requirements, external data, private data, credentials, destructive actions,
  or broad semantic changes.
- Keep prompts reusable across repositories by using placeholders and
  "if present" wording.
- Avoid private data, local secrets, personal tokens, machine-specific paths,
  and unpublished identifiers.
- Do not claim that a tool, connector, service, dataset, or credential exists.
  Tell the agent to discover available tools from the repository and local
  environment.
- Prefer the repository's existing commands over invented commands. If no
  command exists, ask the agent to say that plainly.

## Verification Bias

Prompts should push agents toward short verification loops:

- narrow tests before broad suites
- CLI smoke checks before full pipelines
- schema or fixture checks before manual inspection
- exact output comparison where deterministic outputs exist
- final broad checks only after focused checks pass

For Python repositories, it is reasonable to suggest `uv`, `pytest`, `ruff`,
and `ty` when the repository already uses them. Phrase this as an inspection
instruction, not as a universal guarantee.

## Final Answer Bias

Final response formats should make review easy:

- findings before summary for review prompts
- commands and observed outcomes for verification prompts
- changed files and behavioral impact for implementation prompts
- blockers and uncertainty called out explicitly
- no unsupported claims that something "works" without execution evidence
