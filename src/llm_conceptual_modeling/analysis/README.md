# `analysis`

Deterministic offline analysis helpers live here.

## Contents

- grouped summaries and confidence intervals
- hypothesis-testing and bundle generation
- stability and replication-budget analysis
- figure-ready exports
- output-validity, variability, baseline, and variance-decomposition reports

## Notes

These modules should stay deterministic and file-driven. If a new analysis needs
shared logic, prefer adding a focused helper in `common/` rather than expanding a
single large file.
