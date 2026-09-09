# Artifact/Provenance Validation Prompt

## Goal

Validate that `[artifact directory, result bundle, release candidate, pipeline
output, or dataset]` has enough provenance to be trusted, reproduced, audited,
and connected back to its inputs, code, configuration, and execution context.

## Context to inspect

- Artifact directories, manifests, ledgers, status files, run summaries, logs,
  configs, checkpoints, and generated tables.
- Data layout docs, artifact READMEs, runbooks, and repository onboarding docs.
- Code that writes, reads, packages, validates, or syncs artifacts.
- Tests for schemas, path resolution, parity, packaging, cleanup, and output
  validity.
- External-data references, bucket paths, version notes, and documented
  environment variables.

## Constraints

- Preserve provenance-bearing files. Do not delete or rewrite artifacts unless
  explicitly asked and backed up by verification.
- Do not expose private data, secrets, credentials, raw sensitive records, or
  unpublished payloads in the report.
- Do not infer lineage from directory names alone; prefer manifests, ledgers,
  configs, hashes, timestamps, and code paths.
- Distinguish canonical artifacts, archived artifacts, scratch artifacts, and
  generated previews.
- Treat missing provenance as a risk even when output values look plausible.

## Expected deliverables

- Provenance map from inputs and configs to generated outputs.
- Inventory of required, optional, missing, duplicate, stale, and unexplained
  artifacts.
- Validation of schemas, counts, identifiers, path references, config values,
  and artifact consistency.
- Risks that affect reproducibility, publication claims, or downstream reuse.
- Recommended machine-checkable guardrails such as schema tests, manifest
  checks, hash checks, or CLI validation commands.

## Verification requirements

- Run the repository's artifact, schema, packaging, or verification tests when
  available and relevant.
- Run a CLI validation or smoke command if one exists.
- Inspect representative artifacts directly.
- Cross-check counts and identifiers across at least two independent sources
  when available, such as ledger versus directory tree or summary versus raw
  rows.
- Record exact commands, files inspected, and observed results.

## Final response format

Use this structure:

```markdown
## Provenance Verdict

- Status: valid / partially valid / invalid / unclear
- Artifact scope: ...

## Provenance Map

- Input/config -> process -> output

## Findings

- Severity: high/medium/low
  Evidence: confirmed/plausible/unclear
  Artifact: `path`
  Issue: ...
  Recommended guardrail: ...

## Verification

- `command` -> observed result
- Artifact inspected: `path` -> observation

## Next Checks

1. ...
```
