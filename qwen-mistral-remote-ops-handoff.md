# Qwen/Mistral Remote Reliability Handoff

## Scope

This run is only for the unfinished `Qwen/Qwen3.5-9B` and `mistralai/Ministral-3-8B-Instruct-2512` work in the canonical batch at:

- local repo: `/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling`
- local results root: `/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical`
- remote host: `ssh -p 44598 root@14.186.40.25`
- remote repo: `/workspace/llm-conceptual-modeling`
- remote results root: `/workspace/results/hf-paper-batch-canonical`

Do not reintroduce OLMo. Do not broaden the manifest. Do not recompute already finished runs.

## Onboarding Summary

If you are taking over this codebase cold, start with these rules:

1. Read the canonical `ledger.json` before trusting any old counts or shell notes.
2. Treat remote `batch_status.json`, local watcher status, and local canonical ledger as three different surfaces with different meanings.
3. Do not interpret `current_run` as proof of useful work.
4. When a fresh host looks idle, inspect model download state before touching parser/runtime code.
5. When a dominant failure family is real, capture the exact payload, write the smallest failing regression, make it fail, then patch narrowly.
6. Do not destroy a remote host until results are saved locally and the canonical local run dirs or ledger state you care about are verified.

## Final Tail Outcome

The final dedicated Qwen `algo1` tail was completed successfully with:

- remote isolated tail root: `10 finished / 0 pending / 0 failed`
- all 10 target Qwen `algo1` identities merged into the canonical local root as `finished`
- canonical `ledger.json.records` updated so those 10 identities are marked `finished`

The dedicated tail identities were exactly:

- model: `Qwen/Qwen3.5-9B`
- algorithm: `algo1`
- decoding: `contrastive_penalty_alpha_0.8`
- pair: `sg1_sg2`
- bits: `00101` and `10100`
- replications: `0..4`

## Current Operating Goal

Keep the compact unfinished remote surface running smoothly until the remaining Qwen/Mistral runs finish, with:

- finished runs increasing remotely
- local watcher staying healthy
- local canonical ledger updating forward and never rolling back
- retryable failures staying retryable, not becoming permanent loss

## What Has Been Learned

### 1. Remote status surfaces mean different things

- `batch_status.json` on the remote is the live execution view.
- `results-sync-status.json` locally is the watcher health view.
- `ledger.json` locally is the canonical full-study aggregation across the full `25,200` run study.

Do not confuse `failure_count` with `failed_count`.

- `failed_count` is the true hard-failed run count in the live remote batch.
- `failure_count` can include deferred retryable failures kept in the retry surface.

### 2. The current dominant runtime issue is parser churn, not infrastructure

The current failure family is malformed `children_by_label` output from Qwen contrastive algo3 runs. Typical examples:

- `{Weight bias: ['Body shyness', 'Muscle dysmorphia']`
- `{nutritional awareness: ['dietary knowledge', [1, 2, 3], [4, 5, 6]]`
- `{Physical well-being: ['Mental health', '[email protected]_social_well-being']`
- `{nutritional awareness": ["dietary knowledge", "nutri-literacymythbusting"`

These are retryable structural failures. The parser has been progressively hardened to recover them instead of failing the run.

### 3. Remote deployment must be verified, not assumed

`cat localfile | ssh ... "cat > remotefile"` produced an empty-file deploy once. The correct rule is:

1. copy the file
2. verify remote SHA-256 hash
3. run `python3 -m py_compile` on the remote file when applicable
4. only then relaunch

Use `scp` or `rsync` and always hash-check after deployment.

### 4. Watcher must not overwrite the local canonical ledger

The watcher/fetch path previously rolled the ledger backward because remote `ledger.json` overwrote the local rebuilt ledger.

That has already been fixed by excluding local-derived files from the fetch script. If you see ledger rollback again, audit the fetch path first.

### 5. The remaining tail is dominated by model-switch overhead

Once the batch was reduced to the last unfinished Qwen/Mistral tail, some "stuck" runs were not stalled generations at all. They were workers spending long periods in:

- `phase: "prefetching_model"`
- `model_loaded: false`

The practical consequence is that a run can look idle for minutes even though the worker is only reloading the next model.

Two mitigations are now deployed:

- resume ordering prefers the currently active model from `batch_status.json.current_run.model`, and if the batch is idle it prefers `batch_status.json.last_completed_run.model`
- `runtime_config.yaml` now uses `max_requests_per_worker_process: 64` for this tail run so the worker survives more requests before recycling

These do not change experiment semantics. They only reduce repeated model reloads.

### 6. Cold download can look exactly like a dead run unless you inspect the right files

The final fresh-host Qwen tail exposed a concrete distinction:

- `worker_state.phase = "prefetching_model"` and `model_loaded = false` does not mean the batch is broken by itself
- if `/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/blobs/*.incomplete` files are still growing, the host is still materializing model weights
- if those `.incomplete` files stop growing and the Xet logs stop advancing, the host is stalled in the model transfer layer, not in inference

Files that matter:

- `/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/blobs/*.incomplete`
- `/workspace/.hf_home/xet/logs/*.log`
- run-local `worker_state.json`
- run-local `active_stage.json`
- remote `nvidia-smi`

Interpretation:

- growing `.incomplete` blobs + no GPU compute process: cold download in progress
- frozen `.incomplete` blobs + frozen Xet log timestamps: dead Xet download path
- `model_loaded = true` + GPU attached + `active_stage.json` or `raw_response.json` updating: live inference

### 7. The most reliable recovery for fresh-host Qwen download stalls was disabling Xet

On the final dedicated Qwen tail, the rented host stalled with:

- frozen `.incomplete` blob sizes
- frozen Xet log timestamps
- no GPU compute process
- worker stuck in `prefetching_model`

The working recovery was:

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME=/workspace/.hf_home
```

Then relaunch the isolated Qwen tail without deleting the partially downloaded blobs.

That resumed the same cold-download payload on the non-Xet transfer path, finished model materialization, and let the persistent worker move into live Qwen inference.

### 8. The final Qwen parser fix was not “more retries”; it was a narrow exhausted-retry recovery

The dominant last-mile Qwen failure was a truncated `edge_list` payload such as:

- `[( "Prevalen`
- with trailing `</think>`

Important lesson:

- do not normalize that malformed output immediately
- let the existing Qwen malformed-output retries run first
- only after the retry budget is exhausted should the runtime collapse that specific malformed-single-endpoint case to an empty `edge_list`

That preserved the existing retry behavior while removing the final hard structural failure family from the tail.

### 6. Algo3 frontier expansion is now batched per depth level

`algo3` used to call the model once per expanded parent label. That made even shallow depth-1 or depth-2 runs expensive.

The live code in `src/llm_conceptual_modeling/algo3/method.py` now batches the entire frontier for each depth level into a single proposer call. This preserves the same experiment inputs and outputs while dramatically reducing the number of model calls.

If performance regresses again on `algo3`, verify that the remote hash for `algo3/method.py` matches local before diagnosing anything else.

## Required TDD Loop

For every new failure family:

1. capture the exact remote failure payload or exception text
2. reproduce it locally with the narrowest possible test
3. make the test fail first
4. implement the smallest targeted fix
5. run the focused green tests
6. deploy only the necessary file(s)
7. verify remote hashes
8. relaunch the remote batch
9. watch remote status, watcher health, and ledger movement
10. repeat if a new dominant failure family appears

Do not skip the red step. Do not deploy unverified code.

## Commands

### Dedicated final Qwen tail

Prepare the isolated 10-run tail locally:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling

uv run lcm run prepare-qwen-algo1-tail \
  --canonical-results-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  --tail-results-root /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  --remote-output-root /workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail \
  --json

uv run lcm run qwen-algo1-tail-preflight \
  --repo-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  --canonical-results-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  --tail-results-root /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  --json
```

Launch the isolated tail on a fresh SSH:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling

bash scripts/vast/prepare_and_resume_qwen_algo1_tail.sh \
  root@HOST \
  PORT \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  /workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail
```

If preview itself is the stall, launch the isolated batch directly after bootstrap and doctor with:

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME=/workspace/.hf_home
```

and run `paper-batch --resume` against `/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail/runtime_config.yaml`.

Finalize locally after remote completion:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling

bash scripts/vast/finalize_qwen_algo1_tail.sh \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical
```

### Remote status

```bash
ssh -p 44598 root@14.186.40.25 'cd /workspace/results/hf-paper-batch-canonical && python3 - <<'"'"'PY'"'"'
import json, pathlib
status = json.loads(pathlib.Path("batch_status.json").read_text())
print(json.dumps({
    "updated_at": status.get("updated_at"),
    "total_runs": status.get("total_runs"),
    "finished_count": status.get("finished_count"),
    "pending_count": status.get("pending_count"),
    "failed_count": status.get("failed_count"),
    "failure_count": status.get("failure_count"),
    "running_count": status.get("running_count"),
    "current_run": status.get("current_run"),
    "last_completed_run": status.get("last_completed_run"),
    "failures": status.get("failures", [])[:10],
}, indent=2))
PY'
```

### Local watcher and ledger

```bash
python - <<'PY'
import json, pathlib
base = pathlib.Path("/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical")
status = json.loads((base / "results-sync-status.json").read_text())
ledger = json.loads((base / "ledger.json").read_text())
print(json.dumps({
    "watcher": {
        "status": status.get("status"),
        "updated_at": status.get("updated_at"),
        "last_success_at": status.get("last_success_at"),
        "consecutive_failures": status.get("consecutive_failures"),
        "message": status.get("message"),
    },
    "ledger": {
        "generated_at": ledger.get("generated_at"),
        "finished_count": ledger.get("finished_count"),
        "pending_count": ledger.get("pending_count"),
        "retryable_failed_count": ledger.get("retryable_failed_count"),
        "terminal_failed_count": ledger.get("terminal_failed_count"),
    },
}, indent=2))
PY
```

### Remote process and GPU inspection

```bash
ssh -p 44598 root@14.186.40.25 "ps -ef | grep -E 'hf_worker|lcm run paper-batch' | grep -v grep || true"
ssh -p 44598 root@14.186.40.25 "nvidia-smi || true"
```

### Remote relaunch

```bash
ssh -p 44598 root@14.186.40.25 "bash /workspace/llm-conceptual-modeling/scripts/vast/remote_resume_launch.sh /workspace/llm-conceptual-modeling /workspace/results/hf-paper-batch-canonical /workspace/results/hf-paper-batch-canonical/runtime_config.yaml /workspace/results/hf-paper-batch-canonical/run.log /workspace/results/hf-paper-batch-canonical/batch.pid"
```

### Safe deployment pattern

```bash
scp -P 44598 /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/common/hf_transformers.py root@14.186.40.25:/workspace/llm-conceptual-modeling/src/llm_conceptual_modeling/common/hf_transformers.py
shasum -a 256 /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/common/hf_transformers.py
ssh -p 44598 root@14.186.40.25 "shasum -a 256 /workspace/llm-conceptual-modeling/src/llm_conceptual_modeling/common/hf_transformers.py"
ssh -p 44598 root@14.186.40.25 "python3 -m py_compile /workspace/llm-conceptual-modeling/src/llm_conceptual_modeling/common/hf_transformers.py"
```

## Tests That Have Been Useful

Run focused parser regressions first:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/common/test_hf_transformers.py -k 'parse_generated_json_recovers' -q
```

Run the exact remote-regression slice:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/common/test_hf_transformers.py -k 'remote_unquoted_children_key or remote_unquoted_children_key_with_nonstrings or remote_malformed_terminal_key_quote' -q
```

Run the resume-order regressions for the tail anti-churn behavior:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/core/test_hf_resume_state.py -k 'prefers_current_model_from_batch_status or prefers_last_completed_model_when_idle or prioritizes_cheaper_algorithm_when_risk_matches or prioritizes_cheaper_algorithm_before_history_rates or prioritizes_cheaper_algorithm_before_pending_bucket' -q
uv run pytest tests/core/test_hf_experiments.py -k 'run_paper_batch_resume_prioritizes_low_timeout_risk_pairs or run_paper_batch_resume_retry_timeouts_mode_prioritizes_timeout_failures' -q
```

Run the algo3 batching regressions:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/algo3/test_algo3_method.py -q
uv run pytest tests/algo3/test_algo3_mistral.py -q
uv run pytest tests/core/test_hf_experiments.py -k '_run_algo3 or configured_prompt_accepts_literal_dictionary_braces' -q
```

## Current Parser Fixes Already Added

`src/llm_conceptual_modeling/common/hf_transformers.py` now has additional recovery for:

- unquoted `children_by_label` keys
- truncated dictionary blocks
- nested non-string list elements inside malformed child lists
- malformed terminal key quotes in truncated mappings

Those exact cases are covered by regression tests in:

- `tests/common/test_hf_transformers.py`

### Last-mile parser lessons

The remaining stable rules that kept the parser from regressing were:

- reject obviously partial one-key parses instead of accepting the first broken recovery
- strip orphan parenthetical note lines before line-based recovery
- prefer the quote-closing candidate for nested-list markdown failures
- use the double-quoted list extractor only on one-brace candidates so it does not steal multi-block inputs
- do not trust a single helper that returns a partial result if a later sanitizer can recover the full mapping
- malformed-output retries must not reuse the exact same seed; otherwise Qwen contrastive can replay the same bad completion forever

If a future failure looks like a “mostly correct” one-key mapping with one malformed value, treat that as a red flag for partial recovery rather than success.

## What “Smooth” Means Here

Call the run smooth only when all of the following are true over time, not just one sample:

- remote `finished_count` increases
- remote `failed_count` stays at `0` or near-zero
- remote `current_run` changes over time
- local watcher remains `healthy` or transiently `syncing` with `consecutive_failures=0`
- local ledger `finished_count` advances and does not roll back

Do not call it smooth just because `running_count=1`.

Also do not call it smooth just because `current_run` changes. `current_run` is a claim/start signal, not proof of completion. Confirm either:

- `finished_count` moves, or
- the live run dir is writing fresh `active_stage.json` / `raw_response.json`

## Handoff Rule

If you touch the remote again:

- capture the exact failure
- add the failing test first
- keep the patch narrow
- verify locally
- verify the remote file hash
- relaunch
- watch for real improvement in `finished_count`

## New Lesson: OOM Can Be Batch-Local Session Residency, Not Just a Bad Prompt

One of the main remote failure bursts on `14.186.40.25:44598` was not caused by malformed structured output. It was caused by multiple persistent model workers staying resident on a single GPU at the same time.

What this looked like:

- `batch_status.json` showed large `failure_count` growth dominated by `OutOfMemoryError`
- `nvidia-smi` showed two Python workers resident at once
- the two workers were for different queue directories:
  - `worker-queues/Qwen__Qwen3.5-9B`
  - `worker-queues/mistralai__Ministral-3-8B-Instruct-2512`
- combined GPU memory pressure was enough to starve the current run even when each worker looked individually plausible

This matters because the failure can look like generic retry churn unless you inspect both:

- remote failure histogram
- live GPU process list

### Exact diagnosis pattern

Use these together:

```bash
ssh -p 44598 root@14.186.40.25 "python3 - <<'PY'
import json, pathlib, collections
status = json.loads(pathlib.Path('/workspace/results/hf-paper-batch-canonical/batch_status.json').read_text())
counter = collections.Counter()
for failure in status.get('failures', []):
    key = (failure.get('type'), failure.get('message', '').splitlines()[0][:160])
    counter[key] += 1
for (failure_type, first_line), count in counter.most_common(12):
    print(count, failure_type, first_line)
PY"

ssh -p 44598 root@14.186.40.25 "ps -ef | grep -E 'hf_worker|paper-batch --config' | grep -v grep; echo '==='; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true"
```

If you see multiple resident queue workers for different models on a single GPU, treat that as the dominant problem first.

### Code paths that matter

The relevant runtime/session code lives in:

- `src/llm_conceptual_modeling/hf_execution_runtime.py`
- `src/llm_conceptual_modeling/hf_experiments.py`

The critical behavior now enforced is:

- before a contrastive subprocess run, close and clear all persistent sessions
- before starting a persistent worker for model `X`, close any persistent sessions for models other than `X`

This prevents a mixed-model single-GPU batch from carrying both Qwen and Mistral resident at once.

### Regression tests for this specific issue

Run:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/core/test_hf_execution_runtime.py tests/core/test_hf_experiments.py -k 'closes_incompatible_persistent_sessions or closes_other_model_sessions or closes_all_persistent_sessions' -q
```

Broader focused verification:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/core/test_hf_execution_runtime.py tests/core/test_hf_experiments.py -k 'persistent_mode or contrastive_runs_through_subprocess or closes_incompatible_persistent_sessions or closes_other_model_sessions or closes_all_persistent_sessions' -q
```

### What improvement should look like after deploy

After a correct redeploy and relaunch:

- remote `failure_count` should reset or stop climbing rapidly
- remote `failed_count` should stay at `0` for retryable churn
- `ps` + `nvidia-smi` should show one active worker resident on GPU, not both model workers at once
- `finished_count` should resume moving

Do not call the fix successful just because the launcher restarted. Check GPU residency and the failure histogram.

## New Lesson: A Bare Scalar Can Still Be a Valid `label_list` Recovery Case

After the OOM fix, the next dominant live failure on the same host was:

- `ValueError: Model did not return valid structured output: Hypertension`

This came from:

- `schema_name = "label_list"`
- Qwen contrastive
- `algo2`

Important nuance:

- the model sometimes returns a single plain scalar label instead of a JSON list
- this is still salvageable as a one-item `label_list`
- before the patch, `_recover_label_list_from_lines()` only handled bracketed or segmented list-like forms
- a raw scalar such as `Hypertension` fell through and raised

The narrow fix was to add a scalar-label recovery path in:

- `src/llm_conceptual_modeling/common/hf_transformers.py`

Regression:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/common/test_hf_transformers.py -k 'plain_scalar_label_for_label_list or single_bare_label_list_item' -q
```

Broader parser slice:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/common/test_hf_transformers.py -k 'plain_scalar_label_for_label_list or single_bare_label_list_item or remote_unquoted_children_key or malformed_terminal_key_quote or unicode_punctuation_children_mapping_string or comment_suffix' -q
```

Observed live effect after deploy and relaunch:

- remote `finished_count` advanced
- remote `failure_count` returned to `0`
- local watcher stayed healthy
- canonical ledger advanced again

This is a good pattern to remember:

- if a failure message is a single human-readable concept and the schema is `label_list`, test scalar-label recovery before treating it as an irrecoverable structural failure

## New Lesson: Remote Repo Shadow Packages Can Completely Misroute the Batch

One relaunch failed in a misleading way:

- traceback showed imports from `/workspace/llm-conceptual-modeling/llm_conceptual_modeling/...`
- `REPO_ROOT` became `/workspace`
- `default_inputs_root()` became `/workspace/data/inputs`
- the batch then died with missing-input errors even though the repo copy under `/workspace/llm-conceptual-modeling/data/inputs` existed

Root cause:

- the remote host contained a stale top-level package directory:
  - `/workspace/llm-conceptual-modeling/llm_conceptual_modeling`
- Python imported that shadow copy instead of the `src/` tree
- the editable-install `.pth` was also pointing at the repo root instead of `src`
- because `paths.py` computes `REPO_ROOT` from `__file__`, the shadow import changed every derived path

### Operational fix

Two things were required:

1. make repo sync use `rsync --delete` so stale remote files do not survive future deploys
2. make remote scripts export:

```bash
PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
```

That guard now exists in:

- `scripts/vast/remote_resume_preview.sh`
- `scripts/vast/remote_resume_launch.sh`
- `scripts/vast/remote_runtime_doctor.sh`

### Regression coverage

Run:

```bash
cd /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling
uv run pytest tests/core/test_prepare_and_resume_hf_batch_script.py tests/core/test_sync_repo_to_vast_script.py -q
```

### Remote verification procedure

Check the remote package resolution directly:

```bash
ssh -p 44598 root@14.186.40.25 "cd /workspace/llm-conceptual-modeling && PYTHONPATH=/workspace/llm-conceptual-modeling/src .venv/bin/python - <<'PY'
import inspect
import llm_conceptual_modeling.paths as p
print(inspect.getsourcefile(p))
print('REPO_ROOT', p.REPO_ROOT)
print('default_inputs_root', p.default_inputs_root())
PY"
```

Healthy output characteristics:

- source file should resolve under `/workspace/llm-conceptual-modeling/src/...`
- `REPO_ROOT` should be `/workspace/llm-conceptual-modeling`
- `default_inputs_root` should be `/workspace/llm-conceptual-modeling/data/inputs`

If it resolves under `/workspace/llm-conceptual-modeling/llm_conceptual_modeling/...`, the host is still importing the wrong tree.

## Practical Operating Loop

When you take over the live remote, use this order. Do not skip steps.

1. Inspect current remote state.

```bash
ssh -p 44598 root@14.186.40.25 "python3 - <<'PY'
import json, pathlib, collections
p = pathlib.Path('/workspace/results/hf-paper-batch-canonical/batch_status.json')
d = json.loads(p.read_text())
print(json.dumps({k: d.get(k) for k in ['updated_at','finished_count','pending_count','failed_count','failure_count','running_count','current_run','last_completed_run']}, indent=2))
print('FAILURE_HISTOGRAM')
ctr = collections.Counter()
for f in d.get('failures', []):
    key = (f.get('failure_kind'), f.get('type'), f.get('message', '').splitlines()[0][:180])
    ctr[key] += 1
for (kind, typ, msg), count in ctr.most_common(12):
    print(count, kind, typ, msg)
PY"
```

2. Inspect live processes and GPU residency.

```bash
ssh -p 44598 root@14.186.40.25 "ps -ef | grep -E 'hf_worker|paper-batch --config' | grep -v grep; echo '==='; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true"
```

3. Inspect local watcher and canonical ledger.

```bash
python - <<'PY'
import json, pathlib
base = pathlib.Path('/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical')
status = json.loads((base/'results-sync-status.json').read_text())
ledger = json.loads((base/'ledger.json').read_text())
print(json.dumps({
  'watcher': {k: status.get(k) for k in ['status','updated_at','last_success_at','consecutive_failures','message']},
  'ledger': {k: ledger.get(k) for k in ['generated_at','finished_count','pending_count','retryable_failed_count','terminal_failed_count']}
}, indent=2))
PY
```

4. Pick the dominant failure family and write the failing test first.

- parser / structured output:
  - `tests/common/test_hf_transformers.py`
- persistent-session / worker lifetime:
  - `tests/core/test_hf_execution_runtime.py`
  - `tests/core/test_hf_experiments.py`
- deploy / bootstrap / remote launch:
  - `tests/core/test_prepare_and_resume_hf_batch_script.py`
  - `tests/core/test_sync_repo_to_vast_script.py`
  - `tests/core/test_bootstrap_gpu_host.py`

5. Run the narrow red step.

6. Patch the smallest relevant code or script.

7. Run the narrow green step, then a slightly broader focused slice.

8. Deploy only the files you changed.

- For Python source files, `scp` the exact file(s).
- For remote launch/runtime scripts, `scp` the exact script(s).
- If a stale remote file is part of the bug, remove it explicitly or use repo sync with `--delete`.

9. Verify the remote file content or behavior, then relaunch.

10. Babysit until all three are true:

- remote `finished_count` is moving
- local watcher is healthy or briefly syncing with `consecutive_failures=0`
- canonical ledger is updating and not rolling back

## Additional Lessons Learned

### Do not use `generate(max_time=...)` for contrastive runs

We confirmed a real failure mode on the live `Qwen + Mistral` contrastive batch:

- the worker-level timeout was set to `45s`
- the same `generation_timeout_seconds` was also being injected directly into
  `transformers.generate(..., max_time=45)`
- for contrastive decoding, that produced syntactically truncated outputs such as:
  - `[\n  ("Prevalene`
  - `{ <think>\n"Okay, let's tackle this problem...`

These are not parser bugs in the primary sense. They are execution cutoffs: valid
contrastive generations are being interrupted mid-structure.

The fix was:

- keep the outer monitored worker timeout in place
- stop passing `max_time` into `generate()` when `decoding.algorithm == "contrastive"`

Files:

- `src/llm_conceptual_modeling/common/hf_transformers.py`
- `tests/common/test_hf_transformers.py`

Relevant regression tests:

```bash
uv run pytest tests/common/test_hf_transformers.py -k 'passes_generation_timeout_as_max_time or does_not_pass_generation_timeout_as_max_time_for_contrastive' -q
```

Interpretation rule:

- if a structural output starts correctly and then cuts off almost immediately,
  inspect timeout policy before adding parser recovery
- do not "recover" missing edges or labels by inventing content
- for experiment integrity, truncated outputs should be retried, not hallucinated into validity

### Recover the first valid entry when Mistral emits one good mapping plus markdown garbage

Another recurring contrastive failure family on Mistral was a fenced `children_by_label`
payload where:

- the first quoted `key: [values]` entry was structurally valid
- later lines degenerated into markdown-heavy commentary, bold markers, numbering,
  parenthetical notes, or malformed alternative suggestions

Example shape:

```text
```python {
  'Good key': ['valid value 1', 'valid value 2', ...],
  **Original concept: ...** :
  [ **1. ...** **2. ...** ... ]
}
```
```

The correct recovery is:

- sanitize fenced artifacts and comments
- extract the first valid quoted `key: [values]` entry from the whole block
- parse its list leniently
- ignore the later markdown garbage

Do not reject the entire payload just because later commentary is malformed.
Do not invent missing later entries either. If the first valid mapping exists, use it.

Files:

- `src/llm_conceptual_modeling/common/hf_transformers.py`
- `tests/common/test_hf_transformers.py`

### Distinguish stale deferred failures from live failures

When `batch_status.json` shows old OOM entries mentioning a dead PID, do not assume
the current worker is still OOMing.

Check:

```bash
ssh -p 44598 root@14.186.40.25 "ps -ef | grep -E 'hf_worker|paper-batch --config' | grep -v grep; echo '==='; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true"
```

If the referenced PID is gone and the live GPU has one active worker, the OOM entry
is stale deferred history, not the current dominant blocker.

### Do not trust a naive global scan of `worker_state.json`

This results tree contains many stale historical artifacts from prior hosts and prior
passes. A naive scan like:

```bash
find ... -name worker_state.json
```

or a blanket `rglob('worker_state.json')` can show thousands of entries with
`"status": "running"` from old dates and dead PIDs.

That does **not** mean the current host is actually running all of them.

When validating live activity, prefer this order:

1. `batch_status.json`
2. current `paper-batch` PID + live `hf_worker` PID from `ps`
3. current GPU attachment from `nvidia-smi`
4. the specific `active_stage.json` / `worker_state.json` for the current run only

For the specific current run, inspect the run directory that matches:

- `current_run.algorithm`
- `current_run.model`
- `current_run.decoding_algorithm`
- `current_run.condition_bits`
- `current_run.pair_name`
- `current_run.replication`

Then read only that run’s:

- `active_stage.json`
- `worker_state.json`
- `raw_response.json`
- `error.json`
- `summary.json`

Do not infer liveness from unrelated stale run directories.

## Exact Work Pattern For Future Agents

Use this loop every time, in this order:

1. Read live remote status.
2. Read the local watcher and canonical ledger.
3. Pull the exact raw failing artifacts for the dominant failure family.
4. Decide whether it is:
   - a real parser gap
   - a runtime/timeout policy issue
   - a deployment drift issue
   - a stale-history accounting issue
5. Add or tighten the narrowest failing regression test.
6. Run that red test and confirm it fails for the intended reason.
7. Patch the smallest relevant file.
8. Run the narrow green test.
9. Run one slightly broader focused slice.
10. Deploy only the touched files to the remote host.
11. Verify the remote file hash or behavior.
12. Relaunch the remote batch.
13. Babysit until:
    - `failed_count == 0`
    - `failure_count` is zero or clearly dropping
    - `running_count == 1` while active
    - `finished_count` resumes moving
    - watcher stays healthy
    - ledger updates forward

If any of those fail, repeat the loop. Do not stop at diagnosis.

## What To Run When The Remote Looks Dead

If the batch died and you need the shortest trustworthy recovery sequence:

```bash
ssh -p 44598 root@14.186.40.25 "cd /workspace/llm-conceptual-modeling && PYTHONPATH=/workspace/llm-conceptual-modeling/src .venv/bin/python - <<'PY'
import inspect
import llm_conceptual_modeling.paths as p
print(inspect.getsourcefile(p))
print(p.REPO_ROOT)
print(p.default_inputs_root())
PY"

ssh -p 44598 root@14.186.40.25 "cd /workspace/llm-conceptual-modeling && bash scripts/vast/remote_runtime_doctor.sh /workspace/llm-conceptual-modeling"

ssh -p 44598 root@14.186.40.25 "cd /workspace/llm-conceptual-modeling && bash scripts/vast/remote_resume_preview.sh /workspace/llm-conceptual-modeling /workspace/results/hf-paper-batch-canonical /workspace/results/hf-paper-batch-canonical/runtime_config.yaml /workspace/results/hf-paper-batch-canonical/runtime_config.yaml /workspace/results/hf-paper-batch-canonical/preview"

ssh -p 44598 root@14.186.40.25 "bash /workspace/llm-conceptual-modeling/scripts/vast/remote_resume_launch.sh /workspace/llm-conceptual-modeling /workspace/results/hf-paper-batch-canonical /workspace/results/hf-paper-batch-canonical/runtime_config.yaml /workspace/results/hf-paper-batch-canonical/run.log /workspace/results/hf-paper-batch-canonical/batch.pid"
```

Then re-check remote status, GPU processes, local watcher, and ledger before deciding whether another code fix is needed.

## Latest Accounting Lesson

Retryable failures should not remain in the live failure bucket after resume.
For the current batch, the operational contract is:

- `finished_count + pending_count` should describe the unfinished shard surface the remote is actually processing.
- retryable structural / timeout / OOM / infrastructure failures should be reclassified into pending on resume
- `failed_count` and `failure_count` should stay at zero unless a truly terminal failure is observed
- the canonical ledger should keep advancing on the local side without rolling back to stale remote artifacts

If the remote total looks lower than expected, check the shard manifest and the ledger snapshot before changing worker logic.
The count mismatch can come from the seeded identity set, not just from status classification.

Current verified state:

- regenerating the manifest from the current ledger yields `125` unfinished active-model identities
- the earlier `519` shard was stale relative to the current ledger snapshot
- if a future agent sees a different total, it should verify the manifest file in the results root first, then the remote `batch_status.json`

Files that currently encode this behavior:

- `src/llm_conceptual_modeling/hf_experiments.py`
- `src/llm_conceptual_modeling/hf_batch/monitoring.py`
- `src/llm_conceptual_modeling/hf_shard_manifest.py`
- `src/llm_conceptual_modeling/hf_ledger.py`

Relevant regression tests:

- `tests/core/test_hf_experiments.py`
- `tests/common/test_hf_transformers.py`

## Destroy-SSH Rule

It is safe to destroy a rented host only when all of these are true:

- the relevant remote results root has been synced back locally
- the local canonical run dirs you care about show `state.json = finished`
- the corresponding local canonical `ledger.json.records` entries are `finished`

For the dedicated final Qwen tail, the exact local confirmation surface is:

- isolated local `batch_status.json` shows `10 finished / 0 pending / 0 failed`
- canonical local run dirs for:
  - `algo1 / Qwen__Qwen3.5-9B / contrastive_penalty_alpha_0.8 / sg1_sg2 / 00101 / rep_00..04`
  - `algo1 / Qwen__Qwen3.5-9B / contrastive_penalty_alpha_0.8 / sg1_sg2 / 10100 / rep_00..04`
  all show `finished`
- those same 10 identities are marked `finished` in the canonical local `ledger.json.records`

Do not rely on watcher logs alone for the destroy decision.
