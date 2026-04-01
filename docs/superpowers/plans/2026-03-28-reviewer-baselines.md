# Reviewer Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one WordNet-based ontology-matching baseline and one edit-distance baseline to the existing deterministic baseline and baseline-comparison workflow.

**Architecture:** Extend the shared baseline utilities with deterministic lexical scorers, expose them through the existing `lcm baseline` CLI strategies, and update the baseline comparison bundle to evaluate LLM outputs against the new ranked baselines in the same volume-matched way as the current random-k comparison.

**Tech Stack:** Python 3.14, pandas, pytest, uv, repository-local deterministic lexical resource derived from WordNet.

---

### Task 1: Red Tests For Shared Lexical Baselines

**Files:**
- Modify: `tests/core/test_baseline.py`
- Test: `tests/core/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_propose_wordnet_cross_subgraph_edges_prefers_semantic_matches() -> None:
    ...


def test_propose_edit_distance_cross_subgraph_edges_prefers_lexically_similar_labels() -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_baseline.py -k "wordnet or edit_distance" -v`
Expected: FAIL because the new baseline functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def propose_wordnet_cross_subgraph_edges(...):
    ...


def propose_edit_distance_cross_subgraph_edges(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_baseline.py -k "wordnet or edit_distance" -v`
Expected: PASS

### Task 2: Red Tests For CLI Strategy Surface

**Files:**
- Modify: `tests/cli/test_cli.py`
- Modify: `src/llm_conceptual_modeling/cli.py`
- Test: `tests/cli/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

```python
def test_cli_baseline_accepts_wordnet_and_edit_distance_strategies(tmp_path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_cli.py -k "wordnet and edit_distance" -v`
Expected: FAIL because the parser choices do not yet include the new strategies.

- [ ] **Step 3: Write minimal implementation**

```python
choices=[
    "direct-cross-graph",
    "random-uniform-subset",
    "wordnet-ontology-match",
    "edit-distance",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_cli.py -k "wordnet and edit_distance" -v`
Expected: PASS

### Task 3: Baseline Bundle Comparison Support

**Files:**
- Modify: `src/llm_conceptual_modeling/analysis/baseline_bundle.py`
- Modify: `tests/analysis/test_analysis_baseline_bundle.py`

- [ ] **Step 1: Write the failing analysis test**

```python
def test_baseline_bundle_writes_strategy_specific_outputs(tmp_path: Path) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/analysis/test_analysis_baseline_bundle.py -k strategy_specific -v`
Expected: FAIL because the bundle only writes random-k outputs.

- [ ] **Step 3: Write minimal implementation**

```python
def write_baseline_comparison_bundle(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/analysis/test_analysis_baseline_bundle.py -k strategy_specific -v`
Expected: PASS

### Task 4: Documentation And Verified Results

**Files:**
- Modify: `README.md`
- Modify: `docs/revision-summary.md`
- Create or Modify: `data/analysis_artifacts/revision_tracker/baseline_comparison/...`

- [ ] **Step 1: Run the updated bundle command**

Run: `uv run lcm analyze baseline-bundle --results-root data/results --output-dir data/analysis_artifacts/revision_tracker/baseline_comparison`
Expected: PASS and refreshed comparison artifacts for all baseline strategies.

- [ ] **Step 2: Update docs with truthful commands and observed results**

Document:
- which strategies were added,
- which commands were run,
- where the generated artifacts live,
- and what the measured comparison showed.

- [ ] **Step 3: Run final verification**

Run:
- `uv run pytest tests/core/test_baseline.py tests/cli/test_cli.py tests/analysis/test_analysis_baseline_bundle.py -v`
- `uv run ruff check .`
- `uv run ty check`

Expected: PASS
