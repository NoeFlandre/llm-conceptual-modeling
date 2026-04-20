# Open-Weight Map-Extension Inputs

This directory contains the three causal maps used by the scoped open-weight
map-extension batch.

- `manifest.yaml`
  - canonical registry for the three graph sources and their display labels
- `*_categories.csv`
  - node-to-cluster assignments used to build `subgraph_1`, `subgraph_2`, and
    `subgraph_3`
- `*_edges.csv`
  - raw directed edge lists for each selected map

Selected graph sources:

- `babs_johnson`
- `clarice_starling`
- `philip_marlowe`

The runtime loader reads these files through
`llm_conceptual_modeling.common.graph_data`, validates that each map exposes
exactly three non-empty clusters, and then materializes the intra-cluster
subgraphs used by Algo 3.
