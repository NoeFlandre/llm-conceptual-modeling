# Open-Weight Map-Extension Baseline Comparison Bundle

This directory contains the corrected non-LLM baseline comparison for the ALGO3
open-weight map-extension runs. Because map extension is interpreted through recall,
this bundle reports recall only. `k` is the scored LLM cross-subgraph connection count,
random-k is sampled from all admissible source-target pairs with five deterministic
replications, and WordNet is interpreted as a direct lexical matching baseline.
