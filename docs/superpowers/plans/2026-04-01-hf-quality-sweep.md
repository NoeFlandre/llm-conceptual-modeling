# HF Quality Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce technical debt in the HF/Vast orchestration path without changing experiment behavior.

**Architecture:** Keep `hf_experiments.py` as the public entrypoint, but extract narrow helpers into focused modules for execution dispatch and resume logic. Clean up Vast script glue only after the Python seams are stable.

**Tech Stack:** Python, pytest, ruff, ty, bash, uv

---

### Task 1: Record The Sweep Design

**Files:**
- Create: `docs/superpowers/specs/2026-04-01-hf-quality-sweep-design.md`
- Create: `docs/superpowers/plans/2026-04-01-hf-quality-sweep.md`

- [ ] **Step 1: Add the design and plan docs**

- [ ] **Step 2: Verify docs are present**

Run: `test -f docs/superpowers/specs/2026-04-01-hf-quality-sweep-design.md && test -f docs/superpowers/plans/2026-04-01-hf-quality-sweep.md`
Expected: exit code `0`

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-04-01-hf-quality-sweep-design.md docs/superpowers/plans/2026-04-01-hf-quality-sweep.md
git commit -m "Document HF quality sweep plan"
```

### Task 2: Extract HF Execution Dispatch

**Files:**
- Create: `src/llm_conceptual_modeling/hf_execution_runtime.py`
- Modify: `src/llm_conceptual_modeling/hf_experiments.py`
- Test: `tests/core/test_hf_experiments.py`

- [ ] **Step 1: Write a failing test for the extracted execution dispatcher**

Target behavior:
- `run_paper_batch` and `run_single_spec` still route ephemeral vs persistent execution correctly
- policy resolution remains unchanged

- [ ] **Step 2: Run the narrow test to verify it fails**

Run: `uv run pytest tests/core/test_hf_experiments.py -q`
Expected: FAIL in the new extraction-specific test

- [ ] **Step 3: Implement the minimal extraction**

Move the worker-process-mode execution helpers into `hf_execution_runtime.py` and update imports.

- [ ] **Step 4: Run narrow verification**

Run: `uv run pytest tests/core/test_hf_experiments.py tests/core/test_hf_persistent_worker.py tests/core/test_hf_worker.py -q`
Expected: PASS

- [ ] **Step 5: Run broader verification**

Run: `uv run pytest tests/common/test_hf_transformers.py tests/core/test_hf_experiments.py tests/core/test_hf_worker.py tests/core/test_hf_persistent_worker.py tests/core/test_hf_subprocess.py tests/core/test_prepare_and_resume_hf_batch_script.py -q`
Expected: PASS

- [ ] **Step 6: Run quality gates**

Run:

```bash
uv run ruff check src/llm_conceptual_modeling/hf_execution_runtime.py src/llm_conceptual_modeling/hf_experiments.py tests/core/test_hf_experiments.py
uv run ty check src/llm_conceptual_modeling/hf_execution_runtime.py src/llm_conceptual_modeling/hf_experiments.py
```

Expected: PASS

- [ ] **Step 7: Commit**

Run:

```bash
git add src/llm_conceptual_modeling/hf_execution_runtime.py src/llm_conceptual_modeling/hf_experiments.py tests/core/test_hf_experiments.py
git commit -m "Extract HF execution runtime helpers"
```

### Task 3: Extract Resume Ordering And Seeded Run Accounting

**Files:**
- Create: `src/llm_conceptual_modeling/hf_resume_policy.py`
- Modify: `src/llm_conceptual_modeling/hf_experiments.py`
- Test: `tests/core/test_hf_experiments.py`

- [ ] **Step 1: Write a failing test for extracted resume ordering helpers**

- [ ] **Step 2: Run the narrow test to verify it fails**

Run: `uv run pytest tests/core/test_hf_experiments.py -q`
Expected: FAIL in the new resume-policy extraction test

- [ ] **Step 3: Implement the minimal extraction**

- [ ] **Step 4: Run narrow verification**

Run: `uv run pytest tests/core/test_hf_experiments.py -q`
Expected: PASS

- [ ] **Step 5: Run broader verification**

Run: `uv run pytest tests/common/test_hf_transformers.py tests/core/test_hf_experiments.py tests/core/test_hf_worker.py tests/core/test_hf_persistent_worker.py tests/core/test_hf_subprocess.py tests/core/test_prepare_and_resume_hf_batch_script.py -q`
Expected: PASS

- [ ] **Step 6: Run quality gates**

Run:

```bash
uv run ruff check src/llm_conceptual_modeling/hf_resume_policy.py src/llm_conceptual_modeling/hf_experiments.py tests/core/test_hf_experiments.py
uv run ty check src/llm_conceptual_modeling/hf_resume_policy.py src/llm_conceptual_modeling/hf_experiments.py
```

Expected: PASS

- [ ] **Step 7: Commit**

Run:

```bash
git add src/llm_conceptual_modeling/hf_resume_policy.py src/llm_conceptual_modeling/hf_experiments.py tests/core/test_hf_experiments.py
git commit -m "Extract HF resume policy helpers"
```

### Task 4: Clean Vast Sync/Launch Glue

**Files:**
- Modify: `scripts/vast/prepare_and_resume_hf_batch.sh`
- Modify: `scripts/vast/fetch_results_from_vast.sh`
- Modify: `scripts/vast/watch_results_from_vast.sh`
- Modify: `docs/vast-ai-transformers.md`
- Test: `tests/core/test_prepare_and_resume_hf_batch_script.py`
- Test: `tests/core/test_fetch_results_from_vast_script.py`

- [ ] **Step 1: Write or tighten failing tests for any duplicated or unclear script behavior**

- [ ] **Step 2: Run the narrow script test slice**

Run: `uv run pytest tests/core/test_fetch_results_from_vast_script.py tests/core/test_prepare_and_resume_hf_batch_script.py -q`
Expected: FAIL if the cleanup changes behavior assumptions

- [ ] **Step 3: Implement the minimal cleanup**

- [ ] **Step 4: Verify scripts**

Run:

```bash
uv run pytest tests/core/test_fetch_results_from_vast_script.py tests/core/test_prepare_and_resume_hf_batch_script.py -q
bash -n scripts/vast/fetch_results_from_vast.sh
bash -n scripts/vast/watch_results_from_vast.sh
bash -n scripts/vast/prepare_and_resume_hf_batch.sh
```

Expected: PASS

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/vast/fetch_results_from_vast.sh scripts/vast/watch_results_from_vast.sh scripts/vast/prepare_and_resume_hf_batch.sh docs/vast-ai-transformers.md tests/core/test_fetch_results_from_vast_script.py tests/core/test_prepare_and_resume_hf_batch_script.py
git commit -m "Clean Vast sync and launch helpers"
```

### Task 5: Final Sweep

**Files:**
- Modify: any touched docs/tests/modules from earlier tasks only if needed

- [ ] **Step 1: Remove dead code and stale references revealed by the extractions**

- [ ] **Step 2: Run the full HF-focused quality sweep**

Run:

```bash
uv run pytest tests/common/test_hf_transformers.py tests/core/test_hf_experiments.py tests/core/test_hf_worker.py tests/core/test_hf_persistent_worker.py tests/core/test_hf_subprocess.py tests/core/test_fetch_results_from_vast_script.py tests/core/test_prepare_and_resume_hf_batch_script.py -q
uv run ruff check src tests
uv run ty check src
```

Expected: PASS

- [ ] **Step 3: Commit**

Run:

```bash
git add docs src tests scripts
git commit -m "Finish HF quality sweep"
```
