from pathlib import Path


def test_bootstrap_skips_locked_torch_and_triton_packages() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "--no-install-package torch" in script_text
    assert "--no-install-package triton" in script_text


def test_bootstrap_repairs_broken_torch_virtualenv() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "undefined symbol: ncclCommWindowDeregister" in script_text
    assert "rm -rf .venv" in script_text


def test_bootstrap_uses_safe_heredoc_health_probe() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "<<'PY' 2>&1 || true" not in script_text
    assert "{ .venv/bin/python - <<'PY' 2>&1; } || true" in script_text


def test_bootstrap_disables_xet_download_path() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert 'export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"' in script_text
    assert 'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"' in script_text
