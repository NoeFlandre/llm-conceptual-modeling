# Geospatial/Data Pipeline Audit Prompt

## Goal

Audit `[repository root, pipeline, dataset, or changed files]` for data-pipeline
correctness. If the repository has geospatial semantics, include coordinate,
geometry, CRS, spatial-join, tiling, and unit checks. If it does not, do not
invent geospatial requirements; focus on tabular, graph, schema, lineage, and
artifact contracts.

## Context to inspect

- Data layout docs, manifests, schemas, sample inputs, fixtures, and golden
  outputs.
- Pipeline entry points, CLI commands, orchestration scripts, notebooks, and CI
  jobs.
- Transform code for parsing, filtering, joins, aggregation, normalization,
  deduplication, and output writing.
- Tests for data contracts, schemas, round trips, edge cases, and provenance.
- For geospatial repositories, inspect CRS declarations, geometry columns,
  bounds, topology rules, spatial index usage, projections, unit conversions,
  and geocoding assumptions.
- For graph or conceptual-map repositories, inspect node/edge identity,
  directionality, label normalization, duplicate handling, and source/target
  semantics.

## Constraints

- Do not treat a term like "map" as geospatial without evidence.
- Do not load or print private datasets unless the user explicitly authorizes
  inspection and the data is safe to disclose.
- Preserve source data and provenance-bearing artifacts.
- Do not rewrite the pipeline during the audit unless explicitly asked.
- Prefer schema checks, row-count checks, and deterministic fixtures over
  manual spot checks alone.
- Distinguish data-quality defects from undocumented but intentional domain
  choices.

## Expected deliverables

- Pipeline inventory: inputs, transforms, outputs, contracts, and owners.
- Confirmed defects or risks in schemas, joins, identifiers, geometry, graph
  semantics, units, missing values, duplicates, ordering, determinism, and
  output paths.
- Geospatial-specific findings only when geospatial evidence exists.
- Recommended regression tests, schema checks, or CLI guardrails.
- A clear statement of what was verified manually versus by automation.

## Verification requirements

- Run the narrowest pipeline test, schema test, or fixture replay that exists.
- Run a representative CLI smoke command if available and safe.
- Validate a small input/output pair by inspecting row counts, columns, types,
  identifiers, and expected invariants.
- For geospatial data, verify CRS, bounds, geometry validity, units, and any
  spatial join assumptions with available local tools.
- For graph data, verify node/edge identity, edge direction, duplicate policy,
  and serialization round trips.
- Record commands run, artifacts inspected, and observed results.

## Final response format

Use this structure:

```markdown
## Scope

- Pipeline audited: ...
- Geospatial semantics present: yes/no/unclear

## Dataflow

- Input -> transform -> output

## Findings

- Severity: high/medium/low
  Evidence: confirmed/plausible/unclear
  Contract affected: ...
  Issue: ...
  Recommended check or fix: ...

## Verification

- `command` -> observed result
- Artifact inspected: `path` -> observation

## Residual Risk

- ...
```
