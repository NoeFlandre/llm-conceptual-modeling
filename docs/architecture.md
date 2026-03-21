# Architecture

This document is implementation-oriented. For project context, reproducibility commands, and reviewer-facing usage, see [README.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/README.md).

## Design Goals

The codebase is organized around three requirements:

- deterministic research workflows should be reproducible from the command line
- shared logic should be centralized where behavior is actually identical
- correctness should be externally checkable through fixtures, schemas, and verification commands

## Package Layout

- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`
  Algorithm-specific entry points and behavior that remains distinct across the three study workflows.
- `src/llm_conceptual_modeling/common`
  Shared graph loading, connection evaluation, factorial-analysis helpers, CSV schema checks, literal parsing, and typed data structures.
- `src/llm_conceptual_modeling/commands`
  CLI handlers. These modules keep argument wiring separate from the domain logic so workflows remain directly testable.
- `data/inputs`
  Input graph files referenced by the generation-manifest layer.
- `data/results`
  Imported primary experiment outputs organized by algorithm and model.
- `data/analysis_artifacts`
  Reproducible audit artifacts derived from `data/results` for revision support.
- `tests/fixtures/legacy`
  Committed oracle artifacts used for parity verification.

## Workflow Model

The repository implements three command families:

- `lcm eval ...`
  Convert raw algorithm outputs into evaluated CSVs.
- `lcm factorial ...`
  Compute factorial-analysis summaries from evaluated CSVs.
- `lcm analyze ...`
  Produce reviewer-facing post-processing artifacts such as grouped descriptive summaries, replication-stability summaries, paired hypothesis tests, and raw-output failure classifications.

For the paired hypothesis-test workflow, adjusted p-values use Benjamini-Hochberg correction. That choice matches the repository's use case better than a familywise-error correction: the tests are emitted in related families across metrics and files, and the purpose is to control false discoveries while retaining enough sensitivity to inspect potentially real effects in the imported corpus.
- `lcm verify ...`
  Run repository health checks and deterministic parity checks.

The `lcm generate ...` commands are intentionally narrower. They expose the experiment contract for each algorithm without executing live provider calls.

## Verification Strategy

The verification model is layered:

- unit and workflow tests protect parsing, graph logic, schemas, failure modes, and output contracts
- golden and parity fixtures check that deterministic outputs remain stable
- `lcm doctor` checks basic repository prerequisites
- `lcm verify legacy-parity` reruns the offline workflows against committed oracle artifacts
- `lcm verify all` provides a single machine-readable gate for local work and CI

This structure is intended to make regressions visible quickly and to keep future manuscript-driven changes inside short verification loops.

## Explicit Boundary

The repository does not currently validate historical live LLM behavior by reissuing provider calls. Instead, the generation layer is preserved as offline manifests describing dataset paths, experimental conditions, replications, subgraph coverage, and prompt previews.

This boundary is deliberate: offline outputs are reproducible and regression-tested, whereas live provider behavior may drift across model versions, serving infrastructure, and time.
