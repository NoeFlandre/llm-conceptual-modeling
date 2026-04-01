from pathlib import Path


def test_vast_gpu_dockerfile_reuses_validated_bootstrap() -> None:
    dockerfile_path = Path("docker/vast-gpu.Dockerfile")
    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime" in dockerfile_text
    assert "HF_HUB_DISABLE_XET=1" in dockerfile_text
    assert "HF_HUB_ENABLE_HF_TRANSFER=0" in dockerfile_text
    assert (
        "RUN bash scripts/vast/bootstrap_gpu_host.sh /workspace/llm-conceptual-modeling"
        in dockerfile_text
    )
