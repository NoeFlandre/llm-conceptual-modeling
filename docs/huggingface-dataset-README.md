# llm-variability-conceptual-modeling Bucket

This Hugging Face bucket stores the large offline artifacts for the `llm-conceptual-modeling` project:

- `inputs/`
  Original causal-map source files and auxiliary lexical resources used by the offline pipeline.
- `results/`
  Imported experiment outputs organized by algorithm and model. In the local
  code checkout, these are partitioned into `data/results/frontier`,
  `data/results/open_weights`, and `data/results/archives`.
- `analysis_artifacts/`
  Reviewer-facing derived artifacts generated from the imported corpus.

The source code, tests, fixtures, and contributor documentation live in the GitHub repository:

- GitHub: [NoeFlandre/llm-conceptual-modeling](https://github.com/NoeFlandre/llm-conceptual-modeling)

Recommended navigation:

1. Read the GitHub repository for setup, CLI usage, verification, and implementation details.
2. Use this bucket when you need the canonical inputs, imported corpus, or generated revision artifacts.

If you clone or download this dataset locally, point the code repository at it with:

```bash
export LCM_INPUTS_ROOT="/path/to/llm-variability-conceptual-modeling/inputs"
export LCM_RESULTS_ROOT="/path/to/llm-variability-conceptual-modeling/results"
export LCM_ANALYSIS_ARTIFACTS_ROOT="/path/to/llm-variability-conceptual-modeling/analysis_artifacts"
```
