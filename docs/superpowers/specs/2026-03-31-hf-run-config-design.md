# HF Transformers Run Config Design

## Goal

Replace the hardcoded local-`transformers` DOE batch settings with a single executable YAML configuration that is both:

- the source of truth for execution
- the pre-run overview a human can inspect before renting GPU time

Changing the YAML must change the actual run. It is not a narrative document.

## Requirements

- One checked-in YAML describes the run.
- The YAML includes:
  - provider, models, embedding model, replications
  - decoding algorithms and parameter levels
  - temperature and generation/context settings
  - reusable prompt fragments
  - per-algorithm prompt assembly rules
  - DOE factor definitions and factor-to-fragment mappings
  - graph/pair selection
  - output and preview settings
- No duplicated prompt text when the same text is reused across algorithms or conditions.
- The runner must load the YAML and stop relying on hardcoded batch defaults.
- The code must generate a resolved preview artifact showing what will actually run:
  - selected models
  - decoding grid
  - replications
  - factor counts
  - exact assembled prompts per algorithm/condition template
  - total planned run count
  - runtime invariants and any unsupported settings
- Validation must fail loudly on inconsistent or unsupported settings.

## Approach Options

### Option 1: Single compositional YAML plus resolved preview

Store shared and algorithm-specific prompt fragments in one YAML, compose prompts at runtime, and write a resolved preview file before execution.

Pros:
- one-glance audit surface
- no duplicated prompt text
- exact execution contract is inspectable
- easiest to keep consistent with the real run

Cons:
- config schema is larger

### Option 2: Separate runtime YAML and prompt YAMLs

Split runtime/model settings from prompt assembly.

Pros:
- smaller files

Cons:
- weaker pre-run overview
- more moving parts to cross-check

## Decision

Use Option 1.

## Config Shape

File path:

- `configs/hf_transformers_paper_batch.yaml`

Top-level sections:

- `run`
  - provider
  - output_root
  - replications
  - resume
- `runtime`
  - temperature
  - device_policy
  - quantization
  - context_policy
  - max_new_tokens_by_schema
- `models`
  - chat_models
  - embedding_model
- `decoding`
  - greedy
  - beam
  - contrastive
- `inputs`
  - graph_source
  - algorithm_pairs
- `shared_fragments`
  - reusable prompt text blocks
- `algorithms`
  - `algo1`
  - `algo2`
  - `algo3`

Per-algorithm config:

- `base_fragments`
  Ordered fragment names for the always-on prompt skeleton.
- `factors`
  Mapping from DOE factor names to:
  - encoded output column name
  - low/high level values
  - fragment name(s) toggled by the factor
  - any algorithm-specific semantic mapping
- `prompt_templates`
  Template lines that inject map content, label lists, and other runtime payloads.
- `pair_names`
  Which graph pairs/subgraphs this algorithm runs on.
- `evaluation`
  Result column and metric columns used downstream.

## Source Of Truth Semantics

- CLI `run` commands take `--config`.
- The runner resolves all parameters from YAML.
- CLI overrides are limited to safe execution controls such as:
  - `--dry-run`
  - optionally `--output-root` if explicitly allowed later
- Model lists, decoding levels, temperature, factor fragments, and prompts come from YAML only.

## Prompt Assembly

Prompt text is built from:

1. shared fragments
2. algorithm base fragments
3. active factor fragments
4. algorithm runtime template blocks

The resolved preview stores both:

- the fragment names used
- the final assembled prompt text

This keeps authoring DRY while still making the true prompt visible before execution.

## Preview Artifacts

Validation or dry-run planning writes:

- `resolved_run_config.yaml`
  Normalized config after defaults and validation.
- `resolved_run_plan.json`
  Expanded run counts, model grid, decoding grid, factor counts, and paths.
- `prompt_preview/`
  One resolved prompt artifact per algorithm and factor condition template.

The preview must be readable without starting the run.

## Validation Rules

- provider must be `hf-transformers`
- temperature must match the runtime invariants currently supported
- quantization must be `none`
- device policy must require CUDA
- Qwen thinking disable may be enabled only on the supported model path
- decoding parameter sets must match decoding family
- every referenced fragment name must exist
- every factor referenced by an algorithm must declare low/high mapping
- every algorithm must declare enough information to generate raw rows compatible with existing evaluation code

## CLI Changes

Add:

- `lcm run validate-config --config ... --output-dir ...`
- `lcm run paper-batch --config ...`
- `lcm run algo1|algo2|algo3 --config ...`

Behavior:

- `validate-config` performs schema/invariant validation and writes the resolved preview artifacts.
- `paper-batch` and per-algorithm `run` commands validate first, then execute using the resolved config.

## Testing

Red/green coverage:

- config loader reads YAML and resolves fragments
- missing fragment references fail validation
- changing temperature in YAML changes the runtime plan
- resolved prompt preview contains the exact assembled prompt text
- run commands consume config rather than hardcoded values
- per-algorithm run commands still filter correctly under config-driven execution

## Notes

- Existing legacy offline analysis surfaces stay unchanged.
- The new config-driven layer should be additive and should not break the revision-analysis workflows already in the repo.
