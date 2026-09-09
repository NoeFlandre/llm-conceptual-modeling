# Prompt Library

This directory contains reusable prompts for Codex and other coding agents. The
prompts are written for research software repositories that value small
verified changes, reproducible workflows, explicit provenance, and direct
technical reporting.

Each prompt is copy-pasteable as a standalone instruction. Replace bracketed
placeholders such as `[repository root]`, `[branch]`, `[paper path]`, or
`[failed run directory]` before sending the prompt to an agent.

## Prompt Selection Guide

Use the narrowest prompt that matches the job.

- Start with `research-code-review.md` when you need a critical review of a
  code change, PR, implementation branch, or method implementation.
- Start with `reproducibility-audit.md` when the main question is whether a
  fresh agent or reviewer can reproduce setup, tests, outputs, and claims.
- Start with `artifact-provenance-validation.md` when the concern is whether
  outputs can be traced back to inputs, configs, code, manifests, ledgers, or
  run metadata.
- Start with `debug-failed-pipeline-run.md` when there is a concrete failed
  run, broken batch, missing artifact, or failing workflow to diagnose.
- Start with `paper-to-code-consistency-audit.md` when the risk is mismatch
  between manuscript claims, method descriptions, prompts, configs, and code.
- Start with `no-behavior-change-refactor.md` only after the behavior to
  preserve is clear and testable.
- Start with `test-suite-expansion.md` when the desired change is stronger
  executable coverage rather than production behavior changes.
- Start with `ci-improvement.md` when local checks exist but automation is
  missing, incomplete, slow, or not aligned with the repository toolchain.
- Start with `geospatial-data-pipeline-audit.md` for repositories with
  geospatial semantics, graph/map data, ETL flows, tabular data contracts, or
  multi-stage data pipelines.
- Use `future-agent-handoff-summary.md` at the end of a long session, before
  changing agents, or before pausing unfinished work.

## Available Prompts

- [Research code review](research-code-review.md)
- [Reproducibility audit](reproducibility-audit.md)
- [Geospatial/data pipeline audit](geospatial-data-pipeline-audit.md)
- [No-behavior-change refactor](no-behavior-change-refactor.md)
- [Test-suite expansion](test-suite-expansion.md)
- [Paper-to-code consistency audit](paper-to-code-consistency-audit.md)
- [Artifact/provenance validation](artifact-provenance-validation.md)
- [CI improvement](ci-improvement.md)
- [Future-agent handoff summary](future-agent-handoff-summary.md)
- [Debugging a failed pipeline run](debug-failed-pipeline-run.md)

## Repository Fit

For this repository, the first prompts to use are likely:

1. `artifact-provenance-validation.md`
2. `reproducibility-audit.md`
3. `paper-to-code-consistency-audit.md`

Those prompts match the repository's publication-facing surface: deterministic
offline workflows, external canonical data payloads, run artifacts, method
guides, and paper/code consistency.

## Future Prompt Style

Use [STYLE_GUIDE.md](STYLE_GUIDE.md) when adding or revising prompts in this
library.
