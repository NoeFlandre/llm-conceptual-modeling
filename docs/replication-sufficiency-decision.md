# Replication Sufficiency Decision

This note documents the decision to keep the open-weight ablation study at the
original balanced design of five replications per experimental condition.

## Context

The Qwen and Mistral open-weight experiments produced 16,800 finished runs across
the three algorithms and decoding strategies. We audited whether five
replications were sufficient using the existing relative confidence-interval
rule:

```text
n = ((z * s) / (0.05 * |mean|))^2
```

where `z` is the confidence-level critical value, `s` is the sample standard
deviation across replications, and `mean` is the observed metric mean. The audit
was run for both 90% and 95% confidence intervals.

The compact CSV report is:

```text
data/results/open_weights/hf-paper-batch-canonical/replication_budget_sufficiency_compact.csv
```

The detailed report with exact underpowered condition-metric rows is:

```text
data/results/open_weights/hf-paper-batch-canonical/replication_budget_sufficiency_summary.csv
```

## How to Read the Compact CSV

Each row is one `Algorithm x Decoding` group, with Qwen and Mistral columns shown
side by side.

For example, `qwen_ci90_needing_more = 12` means that 12 Qwen
condition-metric cells in that row require more than the existing five
replications under the 90% CI rule.

`qwen_ci90_max_required_runs = 81` is the maximum required total replication
count among those cells. It does not mean the whole row can be fully resolved
with 81 total runs. It means the worst cell in that row requires 81 total
replications, or 76 additional replications beyond the five already completed.

The compact CSV is therefore useful for locating where replication insufficiency
is concentrated. The detailed CSV is the correct source for computing exact
additional run counts.

## Findings

The five-replication design is sufficient for almost all condition-metric cells.

| CI threshold | Cells checked | Cells needing more than five replications | Share sufficiently powered |
| --- | ---: | ---: | ---: |
| 90% CI | 12,000 | 65 | 99.46% |
| 95% CI | 12,000 | 69 | 99.43% |

When metric-level failures are collapsed to source runs, one additional
replication gives all metrics for that condition. Under that source-run
interpretation:

| CI threshold | Unique underpowered conditions | Additional source runs needed to fully power them |
| --- | ---: | ---: |
| 90% CI | 21 | 5,699 |
| 95% CI | 22 | 8,128 |

These failures are not evenly distributed. They are concentrated in a small
number of high-variability settings:

- Algorithm 1 / Qwen / `beam_num_beams_2`
- Algorithm 1 / Qwen / `beam_num_beams_6`
- Algorithm 2 / Qwen / `contrastive_penalty_alpha_0.2`
- Algorithm 2 / Mistral / `contrastive_penalty_alpha_0.2`
- Algorithm 2 / Mistral / `beam_num_beams_6`

Algorithm 3 has no underpowered cells under either threshold.

## Decision

We choose Option A: keep the balanced five-replication design and report the
replication sufficiency audit transparently.

We do not selectively add replications for only the feasible cells. Selective
top-ups would make the design uneven after seeing the results, while leaving the
highest-variability cells untouched. That would complicate interpretation and
could look post hoc unless governed by a predefined cutoff.

The high required replication counts are themselves informative. They occur
where the relative-CI formula becomes extremely demanding: low or unstable means
combined with non-trivial variance make the denominator small and cause required
`n` to grow into the hundreds or thousands. In this study, those cases indicate
localized high variability rather than a broad failure of the replication
design.

## Suggested Framing

We used a balanced five-replication design for all open-weight experimental
conditions. A post-hoc CI sufficiency audit showed that this was sufficient for
99.46% of condition-metric cells at 90% CI and 99.43% at 95% CI. The remaining
underpowered cells were highly localized and corresponded to low-mean,
high-variance conditions where the relative-CI formula becomes extremely
demanding. Fully powering those cells would require approximately 5,699
additional source runs at 90% CI, or 8,128 at 95% CI. We therefore report the
sufficiency audit and treat these localized failures as part of the variability
findings rather than selectively increasing replications post hoc.
