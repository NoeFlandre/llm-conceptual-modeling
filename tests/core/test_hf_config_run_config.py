from pathlib import Path
from textwrap import dedent

from llm_conceptual_modeling.hf_config.run_config import load_hf_run_config


def test_hf_config_run_config_package_imports_and_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "runtime_config.yaml"
    path.write_text(
        dedent(
            """
            run:
              provider: hf-transformers
              output_root: /tmp/results
              replications: 1
            runtime:
              seed: 7
              temperature: 0.0
              quantization: none
              device_policy: cuda-only
              context_policy:
                prompt_truncation: forbid
              max_new_tokens_by_schema:
                edge_list: 128
              thinking_mode_by_model:
                Qwen/Qwen3.5-9B: disabled
            models:
              chat_models:
                - Qwen/Qwen3.5-9B
              embedding_model: Qwen/Qwen3-Embedding-0.6B
            decoding:
              - algorithm: greedy
                temperature: 0.0
            inputs:
              graph_source: default
            shared_fragments: {}
            algorithms: {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_hf_run_config(path)

    assert config.run.provider == "hf-transformers"
    assert config.models.chat_models == ["Qwen/Qwen3.5-9B"]
