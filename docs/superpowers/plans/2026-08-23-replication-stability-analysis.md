# Replication Stability Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested CLI analysis that estimates model-specific replication requirements for the high-variability Algorithm 3 case without editing the paper.

**Architecture:** A focused analysis module will read a manifest of model-labelled CSV inputs, validate the required Algorithm 3 columns, compute exact condition-level statistics and cumulative means, and write three CSV outputs. A thin CLI branch will expose the module through the existing `lcm analyze` command. Compact inputs and provenance will live beside the generated reviewer-analysis artifacts.

**Tech Stack:** Python 3.12+, pandas, argparse-based existing CLI, pytest, Ruff, coverage.

---

### Task 1: Add the red tests

**Files:**
- Create: `tests/analysis/test_replication_stability.py`

- [ ] **Step 1: Write tests for exact grouping and the precision estimate**

  Use a two-model fixture with two graph pairs and three repetitions. Assert
  that graph pairs remain separate, zero means are marked
  `zero_mean_not_estimable`, a varying non-zero cell receives the formula-based
  estimate, and a constant cell is stable at its observed run count.

- [ ] **Step 2: Write tests for cumulative means and model summaries**

  Assert that cumulative means are emitted for prefixes 1, 2, and 3 and that
  the model summary counts exact cells, varying cells, zero-mean cells, and
  cells requiring more than the observed five-run budget.

- [ ] **Step 3: Run the new tests and verify they fail**

  Run:

  ```bash
  pytest -q tests/analysis/test_replication_stability.py
  ```

  Expected: collection or import failure because the new analysis module and
  CLI target do not yet exist.

### Task 2: Implement the analysis module

**Files:**
- Create: `src/llm_conceptual_modeling/analysis/replication_stability.py`

- [ ] **Step 1: Implement pure statistics helpers**

  Add small functions for the required-run estimate, exact-condition rows,
  cumulative means, and model summaries. Use explicit status values for
  `zero_mean_not_estimable`, `stable_at_observed_runs`, and
  `requires_more_runs`.

- [ ] **Step 2: Implement manifest loading and CSV writing**

  Require `model` and `input_path` manifest columns. Resolve relative paths
  relative to the manifest, require the Algorithm 3 factor, pair, repetition,
  and Recall columns, reject duplicate repetitions within an exact condition,
  and write deterministic column order.

- [ ] **Step 3: Run the focused tests and verify they pass**

  Run:

  ```bash
  pytest -q tests/analysis/test_replication_stability.py
  ```

  Expected: all new tests pass.

### Task 3: Expose the command

**Files:**
- Modify: `src/llm_conceptual_modeling/commands/cli.py`
- Modify: `src/llm_conceptual_modeling/commands/analyze.py`
- Modify: `tests/analysis/test_replication_stability.py`

- [ ] **Step 1: Add the `replication-stability` parser**

  Accept `--manifest`, `--output-dir`, and optional confidence/precision
  parameters with defaults matching the paper analysis.

- [ ] **Step 2: Dispatch to the module**

  Return zero on success and preserve the existing `ValueError` to stderr
  behavior for invalid manifests or data.

- [ ] **Step 3: Verify the CLI fixture**

  Run:

  ```bash
  pytest -q tests/analysis/test_replication_stability.py
  ```

  Expected: the direct API and CLI tests pass.

### Task 4: Build the compact provenance-controlled analysis bundle

**Files:**
- Create: `data/analysis_artifacts/reviewer_replication_stability/input/*.csv`
- Create: `data/analysis_artifacts/reviewer_replication_stability/source_manifest.csv`
- Create: `data/analysis_artifacts/reviewer_replication_stability/condition_budget.csv`
- Create: `data/analysis_artifacts/reviewer_replication_stability/cumulative_means.csv`
- Create: `data/analysis_artifacts/reviewer_replication_stability/model_summary.csv`
- Create: `data/analysis_artifacts/reviewer_replication_stability/README.md`

- [ ] **Step 1: Extract only needed columns from `b82ec062^`**

  Reconstruct the six historical evaluated files through `git show`, retain
  model, repetition, four prompt factors, directed graph-pair names, and
  Recall, and write compact CSV inputs without restoring the deleted raw
  result files.

- [ ] **Step 2: Run the new CLI**

  Use the manifest to generate the three output tables.

- [ ] **Step 3: Record interpretation and provenance**

  Document the source revision, exact grouping, formula, zero-mean treatment,
  observed-prefix limitation, and the two contrasting model rows useful for
  the paper response.

### Task 5: Verify quality and handoff

**Files:**
- No paper changes; inspect `paper/V2/paper.tex` as a guardrail.

- [ ] **Step 1: Run focused and full repository tests**

  Run the focused analysis tests, then the full pytest suite and Ruff on the
  changed Python files.

- [ ] **Step 2: Run coverage/CRAP and mutation checks**

  Measure the new module with branch coverage, verify CRAP is below 6, and
  run the repository's available mutation tool limited to the new module. If a
  required tool is unavailable in the current environment, report that fact
  rather than claiming the gate passed.

- [ ] **Step 3: Inspect the final diff and outputs**

  Confirm paper/V2/paper.tex is unchanged, unrelated dirty paths are
  untouched, outputs are internally consistent, and no commit or push occurs.
