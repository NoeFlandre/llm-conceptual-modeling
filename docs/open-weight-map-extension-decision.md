# Open-Weight Map-Extension Batch Decision

This note records the planned scoped extension of the open-weight experiments to
additional causal maps. It is a decision record only; it does not change the run
configuration or execution code.

## Background

The earlier frontier-model experiments tested three algorithms on one causal map
with different prompting strategies. Those results showed meaningful output
variability, so the follow-up open-weight batch repeated the core experiments
with Qwen and Mistral under several decoding strategies.

The next batch extends the open-weight work across additional causal maps. The
goal is not to rerun the full sweep. Instead, the batch keeps the most relevant
sources of variation and fixes low-contribution factors to reduce the number of
runs while still addressing the need for more case studies.

## Map Selection

The selected maps are:

- Babs Johnson
- Clarice Starling
- Philip Marlowe

This selection was independently checked against the 15 candidate maps in
`temporary/causal_maps` using a deterministic procedure:

1. Compute the best 3-way clustering for each map across Girvan-Newman,
   spectral clustering, and recursive Kernighan-Lin.
2. Keep only maps in the top quartile of balance entropy.
3. Embed node labels with TF-IDF character n-grams and compute exact Earth
   Mover's Distance between maps using cosine distance as the ground cost.
4. Score every triplet by the sum of its three pairwise EMD distances.

The top balance-quartile candidates were Babs Johnson, Connie Marble, Clarice
Starling, and Philip Marlowe. The highest-diversity triplet was Babs Johnson,
Clarice Starling, and Philip Marlowe.

Babs Johnson is the most balanced structurally, with clusters of 33, 32, and 25
nodes and high modularity. Clarice Starling remains structurally strong and
balanced enough. Philip Marlowe adds topical diversity because it comes from the
suicide maps rather than the ACEs maps.

## Experimental Design

The batch uses Algo 3 only. In the prior open-weight results, Algo 3 remained the
largest source of error/variance, so it is the most informative algorithm to
stress-test across new maps.

The batch keeps both open-weight LLMs:

- `Qwen/Qwen3.5-9B`
- `mistralai/Ministral-3-8B-Instruct-2512`

The following factors vary:

- causal map: Babs Johnson, Clarice Starling, Philip Marlowe
- LLM: Qwen, Mistral
- source-target subgraph pair: three directed Algo 3 pairs per map
- example: off/on
- number of words: 3/5
- depth: 1/2
- replication: 5 repetitions

The following factors are fixed:

- algorithm: Algo 3
- decoding: `beam_num_beams_6`
- counterexample: off

Counterexamples are fixed off because their variance contribution was below the
5% threshold across the relevant open-weight factor tables.

Decoding is fixed to `beam_num_beams_6`. In the prior Algo 3 open-weight ledger,
`beam_num_beams_6` produced the best mean recall for Qwen. Mistral was slightly
better under greedy decoding, but the absolute differences were small. Since the
factor analysis suggested that greedy-vs-non-greedy and beam width were both
below the 5% threshold, while beam-vs-contrastive mattered more for Qwen,
`beam_num_beams_6` is the most defensible single fixed setting. It also avoids
contrastive decoding, which performed poorly for Qwen in Algo 3.

## Run Count

The run count is:

```text
3 maps
× 2 LLMs
× 1 algorithm
× 1 decoding setting
× 3 source-target subgraph pairs
× 8 prompt-factor conditions
× 5 replications
= 720 runs
```

The eight prompt-factor conditions are:

```text
Example off/on × Number of Words 3/5 × Depth 1/2
```

This gives:

- 240 runs per map
- 720 runs total
- 30% of the previous Algo 3 open-weight sweep
- about 4.3% of the full 16,800-run prior open-weight sweep

## CI Sufficiency Policy

The batch keeps five replications upfront for comparability with the previous
open-weight experiments. The confidence-interval sufficiency method remains a
post-hoc audit rather than a rule for deciding replication counts before the
batch is run.

This keeps the design balanced and avoids selectively adding replications only
where the CI requirement is cheap. After the batch completes, the same 90% and
95% CI sufficiency analysis can be used to report whether five replications were
enough for the new maps.
