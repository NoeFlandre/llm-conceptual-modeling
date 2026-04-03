# Agent Onboarding Notes

This document is for a fresh coding agent taking over work in `llm-conceptual-modeling`.
It is an operator handoff, not a paper summary. Its purpose is to let a new agent quickly:

- understand what this project is doing
- find the correct files and commands
- operate the local HF batch workflow
- manage rented SSH GPU hosts safely
- monitor and sync long-running experiments
- diagnose the common failures we have already seen
- continue the paper-support work we have been doing together

This file is intentionally detailed. Read it before touching the remote experiment workflow.

## 1. Project Purpose

This repository studies variability in LLM-based conceptual model / causal map combination.

There are three active algorithm families:

- `algo1`
- `algo2`
- `algo3`

There are two broad operational tracks:

- main experiments used in the paper, including frontier closed models and analysis artifacts
- open-weight HF Transformers experiments run on rented NVIDIA GPUs over SSH

The active coding/operations work in this project is mostly around the second track:

- open-weight local `transformers` inference
- DOE-style batch execution
- resumable artifacts
- remote SSH orchestration
- result syncing back to the local machine
- paper updates and revision-support analysis based on those results

## 2. Repository Layout

High-signal top-level directories:

- `configs/`
  - checked-in run configs for single-model and batch HF experiments
- `src/llm_conceptual_modeling/`
  - core runtime, planning, parsing, algorithms, CLI
- `tests/`
  - regression tests, algorithm tests, CLI tests, analysis tests
- `results/`
  - local synced results from active SSH runs
- `data/results/`
  - checked-in evaluated outputs from the main experiments
- `data/analysis_artifacts/revision_tracker/`
  - plots and revision-support artifacts
- `docs/`
  - remote bootstrap and operational docs
- `scripts/vast/`
  - SSH/Vast helper scripts
- `paper/`
  - manuscript and revision support notes

Important local roots:

- repo root:
  - `/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling`
- paper operator note you are reading:
  - `/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/paper/agent-onboarding.md`

## 3. Core Runtime Files

If you need to understand or patch the HF execution path, read these first:

- `src/llm_conceptual_modeling/common/hf_transformers.py`
  - local HF chat and embedding clients
  - decoding compatibility
  - structured-output parsing and bounded recovery
- `src/llm_conceptual_modeling/hf_worker_state.py`
  - shared worker lifecycle state helpers
  - canonical worker phases used by all HF algorithms
- `src/llm_conceptual_modeling/hf_worker_policy.py`
  - shared timeout and retry policy helpers
  - canonical source for startup timeout, stage timeout, and retry-attempt semantics
- `src/llm_conceptual_modeling/hf_worker_result.py`
  - shared worker result-artifact loader
  - canonical source for `worker_result.json` success/failure decoding
- `src/llm_conceptual_modeling/hf_experiments.py`
  - main HF batch orchestration
  - failure handling
  - run loop semantics
- `src/llm_conceptual_modeling/hf_batch_planning.py`
  - generation grid planning
  - compatibility filtering
- `src/llm_conceptual_modeling/hf_resume_state.py`
  - resume semantics and failure-class policy
- `src/llm_conceptual_modeling/hf_persistent_worker.py`
  - persistent worker behavior for long-lived model processes
- `src/llm_conceptual_modeling/common/structured_output.py`
  - schema normalization
- `src/llm_conceptual_modeling/analysis/plots.py`
  - plot generation used in paper/revision support

Algorithm-specific prompt/build paths:

- `src/llm_conceptual_modeling/algo1/`
- `src/llm_conceptual_modeling/algo2/`
- `src/llm_conceptual_modeling/algo3/`

## 4. Important Checked-In Configs

These are the main single-model remote configs we have been using:

- `configs/hf_transformers_algo2_olmo.yaml`
- `configs/hf_transformers_algo2_qwen.yaml`
- `configs/hf_transformers_algo2_mistral.yaml`
- `configs/hf_transformers_algo3_olmo.yaml`
- `configs/hf_transformers_algo3_qwen.yaml`
- `configs/hf_transformers_algo3_mistral.yaml`

There are also `algo1` result trees locally, but most of the recent live operations were around:

- `algo2` open-weight runs
- `algo3` open-weight runs

## 5. Models and Decoding

Open-weight chat models:

- `allenai/Olmo-3-7B-Instruct`
- `Qwen/Qwen3.5-9B`
- `mistralai/Ministral-3-8B-Instruct-2512`

Embedding model used for open-weight `algo2`:

- `Qwen/Qwen3-Embedding-0.6B`

Decoding families:

- greedy
- beam with `num_beams in {2, 6}`
- contrastive with `penalty_alpha in {0.2, 0.8}` and `top_k = 4`

Critical compatibility fact:

- `Qwen/Qwen3.5-9B` does not support our current contrastive path in plain `transformers`
- this is now handled generally in planning/runtime compatibility code
- unsupported Qwen contrastive combinations should not be scheduled anymore

## 5.1 Worker Lifecycle Phases

HF runs now use explicit worker lifecycle phases in `worker_state.json`.

- `loading_model`
  - the worker process is alive
  - model/tokenizer/runtime initialization may still be running
  - this should be treated as startup time, not generation-stage time
- `executing_algorithm`
  - model resources are loaded
  - the algorithm has started and stage heartbeats are now meaningful

Practical implication:

- do not call a run “live” just because a worker file exists
- do not treat a slow `loading_model` worker as a generation-stage timeout
- when normalizing stale runs, check `worker_pid` as well as legacy `pid`

## 6. Experiment Status Model

Each batch is resumable. Runs write a run directory with artifacts like:

- `manifest.json`
- `state.json`
- `worker_state.json`
- `runtime.json`
- `raw_response.json`
- `summary.json`
- `error.json` on failure

For multi-stage algorithms, stage caches can also exist.

Important conceptual distinction:

1. operational success
- the process runs
- artifacts are produced
- resume works safely

2. structural output validity
- response can be normalized into the expected schema

3. scientific quality
- response is actually useful or plausible

We often work first on `(1)` and `(2)`, then analyze `(3)` separately.
Do not silently convert garbage into finished runs. We only add bounded recoveries for malformed
outputs when the recovered structure is still legitimately the intended schema.

## 7. Local Results Trees

Current local result directories include:

- `results/hf-paper-batch-algo1-mistral-current`
- `results/hf-paper-batch-algo1-olmo-current`
- `results/hf-paper-batch-algo1-qwen`
- `results/hf-paper-batch-algo2-mistral-current`
- `results/hf-paper-batch-algo2-olmo-current`
- `results/hf-paper-batch-algo2-qwen-current`
- `results/hf-paper-batch-algo3-mistral-current`
- `results/hf-paper-batch-algo3-olmo-current`
- `results/hf-paper-batch-algo3-qwen-current`

These local trees matter because:

- they are the safety copy before killing a bad host
- they are the source for resume on a fresh host
- they are the basis for local analysis and paper-support summaries

## 8. Main Remote Operational Pattern

Treat rented hosts as disposable execution surfaces.

Do not depend on remote git state.
Do not assume a host is trustworthy just because it worked earlier.
Do not assume a dead host can be salvaged faster than a clean host replacement.

The usual remote workflow is:

1. prepare config locally
2. verify parser/runtime behavior locally with small tests
3. sync repo or minimal project bundle to host
4. bootstrap the host environment
5. sync the local results tree to the host if resuming
6. launch the batch under `nohup ... &`
7. babysit until it is actually generating
8. set up or verify periodic local result sync

Fast-path wrapper when the user pastes a raw SSH command:

- `scripts/vast/quick_resume_from_ssh.sh`
  - accepts a pasted SSH command such as:
    - `ssh -p 35895 root@1.193.139.231 -L 8080:localhost:8080`
  - extracts target + port automatically
  - delegates to `scripts/vast/prepare_and_resume_hf_batch.sh`
  - use this when the goal is “resume from local results on a fresh host as fast as possible”

## 9. Fresh-Host Bootstrap

Primary bootstrap script:

- `scripts/vast/bootstrap_gpu_host.sh`

That script:

- installs `uv` if missing
- reuses a healthy `.venv` when possible
- deletes a broken `.venv` if it detects the known CUDA/NCCL symbol failure
- runs `uv sync --no-install-package torch --no-install-package triton`
- pins CUDA `torch`
- pins `triton`
- disables the HF/Xet transfer path that caused stalls
- verifies CUDA availability
- writes `.bootstrap-runtime.json`

Important pinned versions in that script:

- `TORCH_VERSION=2.5.1+cu121`
- `TRITON_VERSION=3.1.0`
- `TRANSFORMERS_VERSION=5.4.0`

If a host is empty, the bootstrap can take a long time due to:

- `torch`
- `triton`
- CUDA libs
- `transformers`

If a host is polluted or partially broken, a clean fresh host is usually faster than trying to
rescue a badly mutated environment.

## 10. Common Local Verification Commands

Use small verified loops.

Baseline tests for the parser/runtime path:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/common/test_hf_transformers.py -q
uv run ruff check src/llm_conceptual_modeling/common/hf_transformers.py tests/common/test_hf_transformers.py
```

Configuration validation:

```bash
uv run lcm run validate-config \
  --config configs/hf_transformers_algo2_olmo.yaml \
  --output-dir /tmp/hf-preview-olmo
```

Other high-signal tests:

- `tests/core/test_hf_experiments.py`
- `tests/core/test_hf_batch_planning.py`
- `tests/core/test_hf_resume_state.py`
- `tests/core/test_hf_persistent_worker.py`
- `tests/core/test_hf_run_config.py`
- `tests/analysis/test_analysis_plots.py`
- `tests/cli/test_cli.py`

## 11. Common Commands We Use Together

### 11.1 Local status aggregation

To see what local result trees contain:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
python3 - <<'PY'
import json
from pathlib import Path
for name in sorted(Path("results").glob("hf-paper-batch-*")):
    status_path = name / "batch_status.json"
    if not status_path.exists():
        continue
    status = json.loads(status_path.read_text())
    print(name.name, status.get("finished_count"), status.get("failed_count"), status.get("running_count"), status.get("pending_count"))
PY
```

### 11.2 One-shot result pull from host

Typical pattern:

```bash
rsync -az --delete -e 'ssh -i ~/.ssh/id_rsa -p <PORT>' \
  root@<HOST>:/workspace/results/<REMOTE_RESULT_DIR>/ \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/<LOCAL_RESULT_DIR>/
```

### 11.3 Live remote monitors

We often use custom SSH monitor one-liners that show:

- algorithm name
- model name
- finished / failed / running / pending
- current pair / condition bits
- decoding / replication
- timeout
- heartbeat timer
- GPU utilization / memory

The exact monitor command is often generated ad hoc for a specific host/run.
If the user asks for it, give a copy-paste-ready one-liner.

### 11.4 Relaunch on a host

Typical pattern:

```bash
cd /workspace/llm-conceptual-modeling
nohup .venv/bin/python -m llm_conceptual_modeling.cli run hf-experiments \
  --config configs/hf_transformers_algo2_olmo.yaml \
  --resume \
  > /workspace/results/hf-paper-batch-algo2-olmo-current/restart.log 2>&1 < /dev/null &
```

### 11.5 Fresh-host seeding from local results

When moving a batch to a new host:

1. sync the project
2. create `/workspace/results/...`
3. sync the local results tree to the remote results tree
4. resume there

This prevents regeneration of already-finished runs.

## 12. Local Result Sync Automations

We use local `launchd` agents to keep pulling remote results.

Known labels:

- `com.noeflandre.lcm-results-sync-qwen`
- `com.noeflandre.lcm-results-sync-mistral`
- `com.noeflandre.lcm-results-sync-olmo`
- `com.noeflandre.lcm-results-sync-qwen-algo3`
- `com.noeflandre.lcm-results-sync-mistral-algo3`
- `com.noeflandre.lcm-results-sync-olmo-algo3`

Inspect them with:

```bash
launchctl list | grep 'com.noeflandre.lcm-results-sync'
```

Be careful:

- a label existing does not mean the sync is healthy
- status `24` / `255` has appeared
- stale plists can point at dead hosts

Always inspect the plist and/or the log if sync health matters.

Typical plist location:

- `~/Library/LaunchAgents/com.noeflandre.lcm-results-sync-*.plist`

Typical local log destination:

- inside the local result tree as `results-sync.log`

## 13. Common Failure Modes We Have Already Seen

### 13.1 Qwen contrastive unsupported

Observed repeatedly in `algo1` and `algo2`:

- `ValueError: contrastive search is not supported with stateful models, such as Qwen3_5ForCausalLM`

Current fix:

- compatibility gate in planning/runtime
- unsupported Qwen contrastive combinations should be filtered out generally

### 13.2 Broken persistent worker transport

Observed on Mistral:

- transient transport errors such as `BrokenPipeError`

Current fix:

- persistent worker path now retries transport-class failures instead of treating them as terminal

### 13.3 Structured-output parser failures

Observed across OLMo and Mistral:

- empty `assistant` artifacts
- fenced Python outputs
- malformed `children_by_label`
- malformed `label_list`
- odd flat `edge_list` payloads
- trailing prose after otherwise recoverable payloads

Current parser hardening in `hf_transformers.py` covers bounded versions of:

- fenced payload extraction
- bare `children_by_label` mappings
- duplicate child keys with empty later values
- truncated list closing bracket recovery
- packed single-string label lists
- malformed comma-separated label lists with inconsistent quoting
- single bare list-item recovery for the exact OLMo shape we observed
- one-shot retry for retryable malformed outputs instead of instant failure

Do not broaden this casually.
Every recovery should stay bounded and test-backed.

### 13.4 Remote host CUDA / env corruption

Observed:

- bad `.venv`
- missing `torch`
- CUDA unavailable at runtime
- polluted environment that took longer to rescue than replacing the host

Practical lesson:

- if bootstrap is clearly plateauing on a bad host, move to a fresh host sooner

### 13.5 Infrastructure failures should not burn the whole DOE

Current behavior in resume/orchestration:

- infrastructure-class failures should abort the batch instead of wasting thousands of runs on a dead host
- those failures should be retryable on a healthy replacement host

## 14. Known User-Facing Tasks We Commonly Do

There are three recurring work categories.

### 14.1 Remote run operations

Typical requests:

- set up a new SSH host for one algorithm/model
- babysit the host until it is genuinely generating
- provide a live monitor command
- sync results back locally
- diagnose whether failures are host issues or code issues
- move a stalled batch to a fresh host without losing completed work

### 14.2 Paper/revision support

Typical requests:

- write or revise sections in the paper using the repository as ground truth
- produce provisional tables for partially completed runs
- summarize results so far without overclaiming
- create reviewer-facing plots from main experiment data

Important paper-related assets:

- manuscript under `paper/`
- main evaluated data under `data/results/`
- revision plot outputs under `data/analysis_artifacts/revision_tracker/plots/`

### 14.3 Result analysis

Typical requests:

- aggregate metrics across synced `summary.json` artifacts
- compare models by mean/median precision/recall/F1
- analyze failure distributions
- determine whether a model is uniformly bad or only bad in certain decoding pockets

When doing analysis:

- separate operational failures from output-quality failures
- do not confuse unsupported decoding with bad model quality
- be explicit whether numbers are from finished runs only

## 15. Current Important Local Findings

These are the most important operational lessons from recent work:

- `algo2 | OLMo` became the highest-priority blocked run because it had a huge pending backlog and no live runner
- `algo2 | Qwen` needed the contrastive compatibility filter to stop wasting impossible runs
- `algo2 | Mistral` and `algo3 | Mistral` both needed robustness work in the persistent worker / parser path
- `algo3` finished runs for Qwen and Mistral were often structurally fine but scientifically poor
- `algo3 | OLMo` needed parser recovery for malformed label-list outputs to avoid wasting recoverable runs

## 16. Current Status Snapshot Pattern

Do not hardcode this section into logic, but know the pattern:

- some batches are healthy and running remotely even when coarse process probes look misleading
- `batch_status.json` plus current heartbeat is usually the best truth source
- GPU memory alone is not sufficient evidence of a healthy run
- a monitor that shows:
  - `running = 0`
  - `current = None`
  - stale/no timer
  - `gpu = 0%`
  means the run is stalled even if memory is still reserved

## 17. How To Decide What To Resume First

When multiple batches exist, prioritize:

1. batches with `running = 0` and large pending backlog
2. batches where failures are caused by code/runtime issues we can actually fix
3. batches closest to generating publishable paper evidence
4. only then the healthy batches that are already progressing

In other words:

- blocked and important beats merely incomplete

## 18. SSH Hygiene

Typical SSH form:

```bash
ssh -i ~/.ssh/id_rsa -p <PORT> root@<HOST>
```

Common issues:

- host key mismatch when a provider reuses the same IP on a new instance

Fix:

```bash
ssh-keygen -R '[<HOST>]:<PORT>'
ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_rsa -p <PORT> root@<HOST>
```

## 19. If A Host Is Broken

Use this order:

1. sync results back locally
2. confirm local copy exists
3. kill/abandon the host
4. start a fresh host
5. seed it from the local results tree
6. resume there

Do not spend large amounts of paid GPU time trying to salvage a clearly bad host when:

- bootstrap is plateauing
- CUDA is broken
- process exits immediately with no batch progress

## 20. Definition Of A Good Takeover

A fresh agent should be able to answer:

- What are the active algorithms and models?
- Where are the configs?
- Where are the local synced result trees?
- How do I resume safely on a fresh host?
- How do I monitor a remote run?
- How do I sync results back locally?
- Which failures are already understood?
- Which parser/runtime fixes already exist?
- What work is paper-writing versus run-operations versus analysis?

If this file stops letting a new agent do that quickly, update it.
