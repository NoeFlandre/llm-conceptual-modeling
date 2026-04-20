# Open-Weight Map-Extension Canonical Results

This directory is the canonical local mirror of the scoped open-weight
map-extension batch.

Batch design:

- 3 causal maps: `babs_johnson`, `clarice_starling`, `philip_marlowe`
- 2 models: Qwen 3.5 9B and Ministral 3 8B Instruct
- Algo 3 only
- decoding fixed to `beam_num_beams_6`
- counterexamples fixed off
- prompt factors varied: `example`, `number_of_words`, `depth`
- 5 replications
- total planned runs: `720`

Layout:

- `runs/`
  - per-run artifacts, prompts, raw outputs, summaries, and evaluation payloads
- `aggregated/`
  - grouped raw/evaluated/factorial exports derived from the finished run tree
- `variance_decomposition/`
  - compact summary tables, combined variance decomposition, and one CSV per
    causal map
- `ledger.json`
  - finished/pending run identity state used by resume and audit tooling
- `batch_status.json`
  - latest batch progress snapshot
- `batch_summary.csv`
  - per-run consolidated summary rows

This is the maintained local publication tree. Do not keep a second maintained
copy under the repository root `results/`.
