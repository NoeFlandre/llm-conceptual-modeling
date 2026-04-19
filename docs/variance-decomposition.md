# Variance Decomposition

This repository exposes one authoritative variance-decomposition workflow for the
open-weight ablation study. It covers only:

- `Qwen/Qwen3.5-9B`
- `mistralai/Ministral-3-8B-Instruct-2512`
- `algo1`, `algo2`, and `algo3`

The workflow is deterministic and recomputes the decomposition directly from the
canonical finished-run ledger.

## Source Of Truth

Inputs come from:

- `data/results/open_weights/hf-paper-batch-canonical/ledger.json`

The generator ignores non-finished runs and uses only the decoding conditions
that belong to the open-weight ablation study:

- `greedy`
- `beam_num_beams_2`
- `beam_num_beams_6`
- `contrastive_penalty_alpha_0.2`
- `contrastive_penalty_alpha_0.8`

## Method

The implementation uses a balanced-design sum-of-squares decomposition instead of
the older regression-style drop-one approximation.

- Binary prompt factors are decoded in the planner order used by each algorithm.
- Decoding family is represented with orthogonal contrasts:
  - `Greedy vs Beam/Contrastive`
  - `Beam vs Contrastive`
  - `Beam Width`
  - `Contrastive Penalty`
- This is deliberate. The decoding family has three levels (`greedy`, `beam`,
  `contrastive`), so it contributes two orthogonal degrees of freedom, not three
  independent one-hot factors.
- Pairwise interactions are included between prompt factors and decoding factors.
- Nested decoding-internal interactions are excluded because they are not a
  meaningful variance-decomposition surface for this design.
- The residual `Error` term is computed as:
  - `SS_total - sum(SS_effects)`

Two percentages are written:

- `pct_with_error`
  Share of total variance, including the error term.
- `pct_without_error`
  Share after renormalizing only the non-error terms.

For the `Error` row itself:

- `pct_with_error` is the residual share of total variance.
- `pct_without_error` is always `0.0`.

This is intentional. The error term is not part of the renormalized
without-error decomposition.

## Outputs

All generated variance-decomposition artifacts live under:

- `data/results/open_weights/hf-paper-batch-canonical/variance_decomposition/`

Files:

- `variance_decomposition.csv`
  Combined machine-readable decomposition for all algorithms and both models.
- `variance_decomposition_algo1.csv`
- `variance_decomposition_algo2.csv`
- `variance_decomposition_algo3.csv`
  Per-algorithm CSV exports.
- `variance_decomposition.tex`
  Combined LaTeX tables.
- `variance_decomposition_algo1.tex`
- `variance_decomposition_algo2.tex`
- `variance_decomposition_algo3.tex`
  Per-algorithm LaTeX tables.

## Command

Regenerate the full variance-decomposition bundle from the canonical ledger:

```bash
uv run lcm analyze variance-decomposition-bundle \
  --results-root data/results/open_weights/hf-paper-batch-canonical \
  --output-dir data/results/open_weights/hf-paper-batch-canonical/variance_decomposition
```

This command reads the canonical ledger and rewrites the full contents of:

- `data/results/open_weights/hf-paper-batch-canonical/variance_decomposition/`

The legacy helper script remains available for convenience:

```bash
uv run python generate_variance_decomposition.py
```

## Tests

Primary regression coverage:

- `tests/analysis/test_variance_decomposition.py`

What is enforced:

- factor extraction follows the real planner bit ordering
- decomposition closes to `100.0` for every `(algorithm, model, metric)`
- non-error terms close to `100.0` in `pct_without_error`
- `Error.pct_without_error == 0.0`
- deterministic regeneration yields byte-identical CSV output
- `algo3` renders recall only

Run the focused tests:

```bash
uv run pytest tests/analysis/test_variance_decomposition.py -q
```

Run the broader analysis suite:

```bash
uv run pytest tests/analysis -q
```

## Obsolete Artifacts

The older root-level variance artifacts and intermediate `factorial_algo*.csv`
files are not part of the maintained variance-decomposition workflow anymore.
They should not be used as references for paper tables or downstream analysis.
