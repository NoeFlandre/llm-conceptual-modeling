# Output Validity and Breadth Audit Bundle

This directory contains the organized artifacts for the output-validity-and-breadth revision item.

## Purpose

The reviewer asked to distinguish malformed outputs, empty outputs, and
valid-but-low-quality outputs, and to understand whether failures are
deterministic or stochastic. This bundle addresses both questions by
classifying every raw output row and measuring output breadth (parsed
edge count) as a continuous signal of output size variability.

## Output Classification

Every raw output row is classified into one of three categories:

- `valid_output`: the result string was parseable as an edge list with at least one edge
- `empty_output`: the result was missing, blank, or explicitly marked empty
- `malformed_output`: the result was present but could not be parsed as an edge list

A `parsed_edge_count` is recorded for each `valid_output` row — the number of edges extracted from
the output. This provides a continuous measure of output breadth per row.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Combined view of failure rates and parsed edge statistics across all algorithm-model combinations.
- `failure_rates.csv`
  Failure rate (empty + malformed) per algorithm-model, across all algorithms.
- `parsed_edge_counts.csv`
  Parsed edge count statistics (mean, median, min, max) per algorithm-model.
- `parsed_edge_quartiles.csv`
  Quartile and percentile distributions of parsed edge counts per algorithm-model.
- `<algorithm>/row_level_validity.csv`
  One row per raw output, classified. Useful for tracing individual outputs.
- `<algorithm>/validity_summary.csv`
  Validity counts per model for this algorithm (valid, empty, malformed breakdown).
- `<algorithm>/breadth_distribution.csv`
  Parsed edge count statistics per model for this algorithm.

## Interpretation

The primary finding is that all 10,080 raw output rows across all three
algorithms and all six models are valid, parseable edge lists — there are
zero failures in the imported corpus. The revision-relevant variation is
therefore not parseability but output breadth. Parsed edge counts vary
dramatically: ALGO2 GPT-4o has a median of 15 edges but a mean of 84.5 and
a maximum of 691, indicating a small subset of extremely large outputs. This
right-skewed pattern is relevant to interpreting accuracy, precision, and
recall scores for ALGO2, as extreme output breadth can heavily influence
evaluation metrics.
