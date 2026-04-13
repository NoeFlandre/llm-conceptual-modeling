from pathlib import Path


def test_bootstrap_skips_locked_torch_and_triton_packages() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "--no-install-package torch" in script_text
    assert "--no-install-package triton" in script_text


def test_bootstrap_retries_transient_package_download_failures() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "vast_retry() {" in script_text
    assert 'vast_retry 3 10 uv sync --no-install-package torch --no-install-package triton' in (
        script_text
    )
    assert 'vast_retry 3 10 uv pip install --python .venv/bin/python wheel' in script_text
    assert 'vast_retry 3 10 uv pip install \\' in script_text
    assert '  --python .venv/bin/python \\' in script_text
    assert '  --index-url "$TORCH_INDEX_URL" \\' in script_text
    assert '  --upgrade \\' in script_text
    assert '  "torch==$TORCH_VERSION"' in script_text
    assert '  --no-deps \\' in script_text
    assert '  "triton==$TRITON_VERSION"' in script_text


def test_bootstrap_repairs_broken_torch_virtualenv() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "undefined symbol: ncclCommWindowDeregister" in script_text
    assert "rm -rf .venv" in script_text


def test_bootstrap_uses_safe_heredoc_health_probe() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "<<'PY' 2>&1 || true" not in script_text
    assert "vast_torch_runtime_probe()" in script_text


def test_bootstrap_enables_xet_download_path_by_default() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert 'export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"' in script_text
    assert 'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"' in script_text


def test_bootstrap_installs_triton_without_re_resolving_torch() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert '--no-deps \\' in script_text
    assert '"triton==$TRITON_VERSION"' in script_text


def test_bootstrap_checks_health_with_existing_venv_python() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "uv run python - <<'PY'" not in script_text
    assert ".venv/bin/python - <<'PY'" in script_text


def test_bootstrap_short_circuits_when_existing_env_is_healthy() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()
    snapshot_assignment = (
        'BOOTSTRAP_SNAPSHOT_PATH="${BOOTSTRAP_SNAPSHOT_PATH:-$REPO_DIR/.bootstrap-runtime.json}"'
    )
    healthy_probe = 'if TORCH_HEALTH_OUTPUT="$(vast_torch_runtime_probe 2>&1)"; then'

    assert snapshot_assignment in script_text
    assert healthy_probe in script_text
    assert 'cat "$BOOTSTRAP_SNAPSHOT_PATH"' in script_text
    assert "exit 0" in script_text


def test_bootstrap_exports_runtime_version_guards_for_health_probe() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert (
        'export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"'
        in script_text
    )
    assert 'export TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu128}"' in script_text
    assert 'export TRITON_VERSION="${TRITON_VERSION:-3.6.0}"' in script_text
    assert 'export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"' in script_text


def test_bootstrap_smokes_cuda_kernels_before_accepting_the_environment() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()

    assert "torch.cuda.is_available()" in script_text
    assert 'torch.ones(1, device="cuda")' in script_text
    assert "torch.cuda.synchronize()" in script_text
    assert '"cuda_kernel_smoke": true' in script_text or '"cuda_kernel_smoke": True' in script_text


def test_bootstrap_writes_runtime_snapshot_after_repair() -> None:
    script_path = Path("scripts/vast/bootstrap_gpu_host.sh")
    script_text = script_path.read_text()
    snapshot_redirects = (
        '> "$BOOTSTRAP_SNAPSHOT_PATH"',
        '>"$BOOTSTRAP_SNAPSHOT_PATH"',
    )

    assert '"python_version": sys.version.split()[0]' in script_text
    assert '"timestamp": datetime.now(timezone.utc).isoformat()' in script_text
    assert any(redirect in script_text for redirect in snapshot_redirects)
