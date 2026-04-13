# Orchestration Helpers Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce clutter in the `src/` orchestration layer by grouping ledger, shard-manifest, and active-model helpers into a small cohesive package without changing behavior.

**Architecture:** Keep the public API stable through thin compatibility shims, but move the actual implementation into a focused package with one responsibility: result-batch state discovery and serialization. Preserve existing CLI and test imports during the transition so the refactor is reviewable and low risk. Use tests to pin the current behavior before moving code, then verify that both the new package and the legacy import paths continue to work.

**Tech Stack:** Python, pytest, uv, pathlib, JSON/YAML IO helpers.

---

### Task 1: Pin the current orchestration helper behavior with tests

**Files:**
- Modify: `tests/core/test_hf_ledger.py`
- Modify: `tests/core/test_hf_shard_manifest.py`
- Modify: `tests/core/test_hf_active_models.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
from llm_conceptual_modeling.hf_ledger import refresh_ledger
from llm_conceptual_modeling.hf_shard_manifest import write_unfinished_shard_manifest
from llm_conceptual_modeling.hf_active_models import resolve_active_chat_models

def test_ledger_and_manifest_helpers_still_operate_from_current_paths(tmp_path):
    # Build a minimal fixture tree that exercises ledger refresh, shard manifest
    # generation, and active-model discovery without needing the full batch tree.
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_hf_ledger.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_active_models.py -q`
Expected: at least one failure or missing-test failure until the fixture exists.

- [ ] **Step 3: Write minimal implementation**

```python
# No code change yet; this step is only to lock the current API and expected outputs.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_hf_ledger.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_active_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/core/test_hf_ledger.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_active_models.py
git commit -m "test: pin orchestration helper behavior"
```

### Task 2: Move orchestration helper implementations into a dedicated package

**Files:**
- Create: `src/llm_conceptual_modeling/hf_state/__init__.py`
- Create: `src/llm_conceptual_modeling/hf_state/ledger.py`
- Create: `src/llm_conceptual_modeling/hf_state/shard_manifest.py`
- Create: `src/llm_conceptual_modeling/hf_state/active_models.py`
- Modify: `src/llm_conceptual_modeling/hf_ledger.py`
- Modify: `src/llm_conceptual_modeling/hf_shard_manifest.py`
- Modify: `src/llm_conceptual_modeling/hf_active_models.py`
- Modify: imports in `src/llm_conceptual_modeling/hf_qwen_algo1_tail.py`, `src/llm_conceptual_modeling/commands/run.py`, and any other direct import sites that benefit from the new package

- [ ] **Step 1: Write the failing test**

```python
from llm_conceptual_modeling.hf_state.ledger import refresh_ledger
from llm_conceptual_modeling.hf_state.shard_manifest import write_unfinished_shard_manifest
from llm_conceptual_modeling.hf_state.active_models import resolve_active_chat_models

def test_new_package_exposes_same_behaviour():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_hf_ledger.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_active_models.py -q`
Expected: import failure until package modules exist.

- [ ] **Step 3: Write minimal implementation**

```python
# Move the concrete implementations into the new package modules.
# Leave legacy modules as thin re-export shims.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_hf_ledger.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_active_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_conceptual_modeling/hf_state src/llm_conceptual_modeling/hf_ledger.py src/llm_conceptual_modeling/hf_shard_manifest.py src/llm_conceptual_modeling/hf_active_models.py
git commit -m "refactor: group orchestration helpers into hf_state package"
```

### Task 3: Document the new orchestration helper boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/vast-ai-transformers.md`
- Modify: `docs/variance-decomposition.md` only if import paths changed there

- [ ] **Step 1: Write the failing test**

```python
# Documentation changes are verified by path search in a repo hygiene test,
# so this task uses the existing docs links as the check surface.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/verification/test_repo_tree_hygiene.py -q`
Expected: PASS after the doc links are updated and no stale public references remain.

- [ ] **Step 3: Write minimal implementation**

```markdown
# Update the docs to point at the new hf_state package and explain the boundary.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/verification/test_repo_tree_hygiene.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/vast-ai-transformers.md docs/variance-decomposition.md
git commit -m "docs: describe orchestration helper package boundary"
```
