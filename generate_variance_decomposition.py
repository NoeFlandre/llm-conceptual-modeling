"""Generate deterministic variance decomposition tables for Qwen and Mistral."""

from __future__ import annotations

from pathlib import Path

from llm_conceptual_modeling.analysis.variance_decomposition import (
    DEFAULT_OUTPUT_DIRNAME,
    generate_variance_decomposition_bundle,
)


RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "open_weights" / "hf-paper-batch-canonical"


def main() -> None:
    output_root = RESULTS_ROOT / DEFAULT_OUTPUT_DIRNAME
    bundle = generate_variance_decomposition_bundle(RESULTS_ROOT, output_root)
    print(bundle["combined_table"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
