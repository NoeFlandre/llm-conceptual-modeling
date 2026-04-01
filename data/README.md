# External Data Layout

This repository keeps the `data/` directory only as a local mount point. The
canonical experiment data lives in the Hugging Face bucket:

- Bucket: [NoeFlandre/llm-variability-conceptual-modeling](https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling)
- Source code: [NoeFlandre/llm-conceptual-modeling](https://github.com/NoeFlandre/llm-conceptual-modeling)

The bucket contains:

- `inputs/`
- `results/`
- `analysis_artifacts/`

## Expected External Directories

Set these environment variables if the bucket contents are stored outside this
checkout:

```bash
export LCM_INPUTS_ROOT=/path/to/llm-variability-conceptual-modeling/inputs
export LCM_RESULTS_ROOT=/path/to/llm-variability-conceptual-modeling/results
export LCM_ANALYSIS_ARTIFACTS_ROOT=/path/to/llm-variability-conceptual-modeling/analysis_artifacts
```

The CLI defaults read those variables automatically. Without them, the code
falls back to `data/inputs`, `data/results`, and `data/analysis_artifacts`
inside the local checkout.

## Navigation

- Start from GitHub when you want code, tests, and the CLI.
- Start from the Hugging Face bucket when you want the canonical inputs, raw and
  evaluated results, or revision-tracker artifacts.
