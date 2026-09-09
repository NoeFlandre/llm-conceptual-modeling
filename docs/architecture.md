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
  The canonical copy is published in the Hugging Face bucket.
- `data/results/frontier`
  Imported frontier-model experiment outputs grouped by algorithm.
- `data/results/open_weights`
  Canonical paper-facing Qwen/Mistral outputs and variance-decomposition artifacts.
- `data/results/archives`
  Preserved OLMO artifacts and operational workdirs retained for provenance.
- `data/analysis_artifacts`
  Reproducible audit artifacts derived from `data/results` for revision support.
  The canonical copy is published in the Hugging Face bucket.
- `tests/fixtures/legacy`
  Committed oracle artifacts used for parity verification.

## Workflow Model

The repository implements three command families:

- `lcm eval ...`
  Convert raw algorithm outputs into evaluated CSVs.
- `lcm factorial ...`
  Compute factorial-analysis summaries from evaluated CSVs.
- `lcm baseline ...`
  Generate deterministic structural baseline raw outputs for each algorithm.
- `lcm analyze ...`
  Produce reviewer-facing post-processing artifacts such as grouped descriptive summaries, replication-stability summaries, paired hypothesis tests, tidy figure exports, baseline comparisons, and raw-output failure classifications.

For the paired hypothesis-test workflow, adjusted p-values use Benjamini-Hochberg correction. That choice matches the repository's use case better than a familywise-error correction: the tests are emitted in related families across metrics and files, and the purpose is to control false discoveries while retaining enough sensitivity to inspect potentially real effects in the imported corpus.
- `lcm verify ...`
  Run repository health checks and deterministic parity checks.

The `lcm generate ...` commands expose the experiment contract for each algorithm in their default form. When given explicit model, pair, and output-root arguments, they also execute the corresponding live-backed Mistral experiment path for that algorithm.

The `lcm baseline ...` commands are also intentionally narrow. They expose deterministic graph and lexical heuristics, including WordNet-based ontology matching and edit-distance ranking. These baselines are auditable comparators, not substitutes for provider-backed generation.

## Repository Diagrams

These diagrams are intentionally coarse. They show the maintained code paths and
artifact boundaries without listing every helper module, test file, or generated
CSV.

### High-Level Architecture

```mermaid
flowchart TD
    CLI["lcm CLI<br/>commands/*"]
    Algo["algo1 / algo2 / algo3<br/>method, eval, baseline, generation"]
    Analysis["analysis<br/>summaries, plots, bundles, variance"]
    HFOps["hf_* packages<br/>config, batch, pipeline, state, resume, drain, worker"]
    Common["common<br/>graphs, schemas, parsing, clients, metrics"]
    Data["data/<br/>inputs, results, baselines, analysis_artifacts"]
    Scripts["scripts/vast<br/>sync, bootstrap, preview, launch, fetch"]
    Docs["docs/ and package READMEs<br/>methods, runbooks, architecture"]
    Tests["tests/<br/>unit, workflow, parity, hygiene"]

    Scripts --> CLI
    CLI --> Algo
    CLI --> Analysis
    CLI --> HFOps
    Algo --> Common
    Analysis --> Common
    HFOps --> Common
    CLI <--> Data
    Docs --> CLI
    Tests --> CLI
    Tests --> Common
```

### Data-Flow Pipeline

```mermaid
flowchart LR
    Inputs["data/inputs<br/>graph CSVs and resources"]
    Frontier["data/results/frontier<br/>imported raw outputs"]
    OpenWeights["data/results/open_weights<br/>Qwen/Mistral run outputs"]
    Eval["lcm eval algo1|algo2|algo3"]
    Evaluated["evaluated CSVs"]
    Factorial["lcm factorial algo1|algo2|algo3"]
    AnalysisCLI["lcm analyze ..."]
    Baseline["lcm baseline ..."]
    Artifacts["data/analysis_artifacts<br/>tables, summaries, plots"]
    Baselines["data/baselines<br/>deterministic comparators"]
    Verify["lcm verify all<br/>legacy parity and health checks"]
    Fixtures["tests/fixtures and snapshots"]

    Inputs --> Baseline --> Baselines
    Frontier --> Eval
    OpenWeights --> Eval
    Eval --> Evaluated
    Evaluated --> Factorial
    Evaluated --> AnalysisCLI
    Factorial --> AnalysisCLI
    Baselines --> AnalysisCLI
    AnalysisCLI --> Artifacts
    Fixtures --> Verify
    Artifacts --> Verify
```

### Experiment Lifecycle

```mermaid
flowchart TD
    Configs["configs/*.yaml<br/>checked-in HF run configs"]
    Load["hf_config.load_hf_run_config"]
    Preview["lcm run validate-config<br/>resolved config, plan, prompt previews"]
    Preflight["lcm run resume-preflight<br/>or resume-sweep"]
    Runtime["lcm run prefetch-runtime<br/>model/cache warmup"]
    Smoke["lcm run smoke<br/>single selected spec"]
    Batch["lcm run paper-batch<br/>or lcm run algo1|algo2|algo3"]
    Pipeline["hf_pipeline + algo packages<br/>execute method and metrics"]
    RunDirs["run directories<br/>manifest, prompts, stages, summary/error"]
    State["batch_status.json and ledger.json"]
    Remote["scripts/vast wrappers<br/>sync, bootstrap, doctor, launch, fetch"]

    Configs --> Load --> Preview --> Preflight
    Preflight --> Runtime --> Smoke --> Batch
    Batch --> Pipeline --> RunDirs --> State
    Remote --> Preview
    Remote --> Batch
    Remote --> State
```

### Artifact And Cache Lifecycle

```mermaid
flowchart LR
    Config["runtime_config.yaml<br/>or resolved_run_config.yaml"]
    Plan["resolved_run_plan.json<br/>condition matrix and prompt_preview/"]
    RunSpec["selected run spec"]
    Active["state/checkpoint/stage artifacts"]
    Finished["summary.json<br/>raw/evaluated outputs"]
    Failed["error.json<br/>failure classification"]
    BatchStatus["batch_status.json"]
    Ledger["ledger.json<br/>canonical completion view"]
    Resume["resume reports<br/>unfinished manifests and drain state"]
    Sync["results-sync-*<br/>watcher status and logs"]
    Review["analysis bundles<br/>variance, stability, figures"]

    Config --> Plan --> RunSpec --> Active
    Active --> Finished --> Ledger
    Active --> Failed --> Ledger
    Ledger --> BatchStatus
    Ledger --> Resume
    Ledger --> Review
    Sync --> Ledger
```

### CLI And Module Map

```mermaid
flowchart TD
    Entry["pyproject script: lcm<br/>llm_conceptual_modeling.cli"]
    Parser["commands.cli<br/>argparse and dispatch"]
    EvalCmd["eval / baseline / factorial"]
    AnalyzeCmd["analyze"]
    RunCmd["run"]
    GenerateCmd["generate"]
    VerifyCmd["doctor / verify"]
    AlgoPkgs["algo1, algo2, algo3"]
    AnalysisPkg["analysis"]
    HFPkgs["hf_config, hf_batch, hf_pipeline,<br/>hf_state, hf_resume, hf_drain,<br/>hf_execution, hf_worker, hf_tail"]
    VerificationPkg["verification"]
    GenerationPkg["generation"]

    Entry --> Parser
    Parser --> EvalCmd --> AlgoPkgs
    Parser --> AnalyzeCmd --> AnalysisPkg
    Parser --> RunCmd --> HFPkgs
    Parser --> GenerateCmd --> GenerationPkg
    Parser --> VerifyCmd --> VerificationPkg
```

### Agent Workflow

```mermaid
flowchart LR
    Request["Task request"]
    Instructions["AGENTS.md<br/>prompts/ and docs/onboarding.md"]
    Inspect["Inspect code, docs,<br/>configs, scripts, tests"]
    Baseline["Run narrow baseline<br/>before edits"]
    Change["Small scoped change<br/>docs, tests, or code"]
    Verify["Focused checks<br/>then broader gate"]
    Report["Final report<br/>evidence and residual risk"]

    Request --> Instructions --> Inspect --> Baseline --> Change --> Verify --> Report
    Verify -->|failure| Inspect
```

## Verification Strategy

The verification model is layered:

- unit and workflow tests protect parsing, graph logic, schemas, failure modes, and output contracts
- golden and parity fixtures check that deterministic outputs remain stable
- `lcm doctor` checks basic repository prerequisites
- `lcm verify legacy-parity` reruns the offline workflows against committed oracle artifacts
- `lcm verify all` provides a single machine-readable gate for local work and CI

This structure is intended to make regressions visible quickly and to keep future manuscript-driven changes inside short verification loops.

## Explicit Boundary

The repository does not currently validate historical live LLM behavior by reissuing the exact legacy provider calls. Instead, the generation layer now exposes both offline manifests and live-backed Mistral execution paths that reproduce the paper's method structure against the imported data and tracked inputs.

This boundary is deliberate: the code and verification surface remain in GitHub, while the canonical experimental data payload is externalized to the Hugging Face bucket. Offline outputs are reproducible and regression-tested, whereas live provider behavior may drift across model versions, serving infrastructure, and time.
