import json
from argparse import Namespace
from collections.abc import Callable, Mapping

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    build_runtime_factory,
)
from llm_conceptual_modeling.hf_batch.monitoring import collect_batch_status
from llm_conceptual_modeling.hf_config.run_config import (
    HFRunConfig,
    load_hf_run_config,
    write_resolved_run_preview,
)
from llm_conceptual_modeling.hf_drain import (
    build_drain_plan,
    read_drain_state_report,
    run_drain_supervisor,
)
from llm_conceptual_modeling.hf_experiments import (
    run_paper_batch,
    run_single_spec,
    select_run_spec,
)
from llm_conceptual_modeling.hf_resume.preflight import build_resume_preflight_report
from llm_conceptual_modeling.hf_resume.sweep import build_resume_sweep_report
from llm_conceptual_modeling.hf_state.ledger import refresh_ledger
from llm_conceptual_modeling.hf_state.shard_manifest import write_unfinished_shard_manifest
from llm_conceptual_modeling.hf_tail.qwen_algo1 import (
    build_qwen_algo1_tail_preflight_report,
    prepare_qwen_algo1_tail_bundle,
)


def handle_run(args: Namespace) -> int:
    handler = _RUN_TARGET_HANDLERS.get(args.run_target)
    if handler is not None:
        return handler(args)
    return _handle_experiment_run(args)


def _handle_validate_config(args: Namespace) -> int:
    config = load_hf_run_config(args.config)
    write_resolved_run_preview(config=config, output_dir=args.output_dir)
    return 0


def _handle_resume_preflight(args: Namespace) -> int:
    config = load_hf_run_config(args.config)
    report = build_resume_preflight_report(
        config=config,
        repo_root=args.repo_root,
        results_root=args.results_root,
        allow_empty=args.allow_empty,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in (
            "results_root",
            "total_runs",
            "finished_count",
            "failed_count",
            "pending_count",
            "can_resume",
            "resume_mode",
        ):
            print(f"{_resume_preflight_output_key(key)}={report[key]}")
    return 0


def _resume_preflight_output_key(key: str) -> str:
    return {
        "finished_count": "finished",
        "failed_count": "failed",
        "pending_count": "pending",
    }.get(key, key)


def _handle_resume_sweep(args: Namespace) -> int:
    report = build_resume_sweep_report(repo_root=args.repo_root, results_root=args.results_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in (
            "repo_root",
            "results_root",
            "root_count",
            "ready_count",
            "needs_config_fix_count",
            "invalid_config_count",
            "active_count",
            "finished_count",
        ):
            print(f"{_resume_sweep_output_key(key)}={report[key]}")
    return 0


def _resume_sweep_output_key(key: str) -> str:
    return {
        "root_count": "roots",
        "ready_count": "ready",
        "needs_config_fix_count": "needs_config_fix",
        "invalid_config_count": "invalid_config",
        "active_count": "active",
        "finished_count": "finished",
    }.get(key, key)


def _handle_prefetch_runtime(args: Namespace) -> int:
    config = load_hf_run_config(args.config)
    report = prefetch_runtime_for_config(config=config)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for line in _prefetch_runtime_report_lines(report):
            print(line)
    return 0


def _handle_status(args: Namespace) -> int:
    status = collect_batch_status(args.results_root)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        for key in (
            "total_runs",
            "finished_count",
            "failed_count",
            "running_count",
            "pending_count",
        ):
            print(f"{_status_output_key(key)}={status[key]}")
        print(f"complete={status['percent_complete']}%")
        _print_optional_status_fields(status)
    return 0


def _status_output_key(key: str) -> str:
    return {
        "total_runs": "total",
        "finished_count": "finished",
        "failed_count": "failed",
        "running_count": "running",
        "pending_count": "pending",
    }.get(key, key)


def _print_optional_status_fields(status: Mapping[str, object]) -> None:
    for key in ("worker_pid", "worker_status", "active_stage_age_seconds"):
        value = status.get(key)
        if value is not None:
            print(f"{key}={value}")


def _handle_refresh_ledger(args: Namespace) -> int:
    ledger = refresh_ledger(results_root=args.results_root, ledger_root=args.ledger_root)
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    else:
        for key in (
            "ledger_root",
            "expected_total_runs",
            "finished_count",
            "retryable_failed_count",
            "terminal_failed_count",
            "pending_count",
        ):
            value = args.ledger_root if key == "ledger_root" else ledger[key]
            print(f"{key}={value}")
    return 0


def _handle_write_unfinished_manifest(args: Namespace) -> int:
    manifest = write_unfinished_shard_manifest(
        results_root=args.results_root,
        ledger_root=args.ledger_root,
        manifest_path=args.manifest_path,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"manifest_path={args.manifest_path}")
        print(f"identity_count={len(manifest['identities'])}")
        print(f"active_chat_models={','.join(manifest['active_chat_models'])}")
    return 0


def _handle_prepare_qwen_algo1_tail(args: Namespace) -> int:
    report = prepare_qwen_algo1_tail_bundle(
        canonical_results_root=args.canonical_results_root,
        tail_results_root=args.tail_results_root,
        remote_output_root=args.remote_output_root,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in ("tail_results_root", "identity_count", "config_path", "manifest_path"):
            print(f"{key}={report[key]}")
    return 0


def _handle_qwen_algo1_tail_preflight(args: Namespace) -> int:
    report = build_qwen_algo1_tail_preflight_report(
        repo_root=args.repo_root,
        canonical_results_root=args.canonical_results_root,
        tail_results_root=args.tail_results_root,
        watcher_status_path=args.watcher_status_path,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        resume_preflight = report["resume_preflight"]
        print(f"tail_results_root={report['tail_results_root']}")
        print(f"tail_pending_count={resume_preflight['pending_count']}")
        print(f"tail_can_resume={resume_preflight['can_resume']}")
    return 0


def _handle_drain_remaining(args: Namespace) -> int:
    report = _drain_remaining_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"state_file={report['state_file']}")
        print(f"safe_queue_count={report.get('safe_queue_count', 0)}")
        print(f"risky_queue_count={report.get('risky_queue_count', 0)}")
        if report.get("adopted_results_root"):
            print(f"adopted_results_root={report['adopted_results_root']}")
    return 0


def _drain_remaining_report(args: Namespace) -> dict[str, object]:
    common = {
        "repo_root": args.repo_root,
        "results_root": args.results_root,
        "ssh_command": args.ssh_command,
        "state_file": args.state_file,
        "phase": args.phase,
        "full_coverage": args.full_coverage,
        "root_name_contains": args.root_name_contains,
    }
    if args.plan_only:
        return build_drain_plan(**common)
    return run_drain_supervisor(
        **common,
        poll_seconds=args.poll_seconds,
        stale_after_seconds=args.stale_after_seconds,
        quick_resume_script=args.quick_resume_script,
    )


def _handle_drain_status(args: Namespace) -> int:
    report = read_drain_state_report(args.state_file)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"state_file={args.state_file}")
        for key in ("health", "current_phase", "current_results_root"):
            print(f"{key}={report.get(key, 'unknown')}")
    return 0


def _handle_smoke(args: Namespace) -> int:
    config = load_hf_run_config(args.config)
    spec = select_run_spec(
        config=config,
        algorithm=args.algorithm,
        model=args.model,
        graph_source=args.graph_source,
        pair_name=args.pair_name,
        condition_bits=args.condition_bits,
        decoding=_decoding_from_args(args),
        replication=args.replication,
    )
    summary = run_single_spec(
        spec=spec,
        output_root=args.output_root,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _handle_experiment_run(args: Namespace) -> int:
    config = _load_optional_run_config(args)
    _require_hf_transformers_provider(args, config)
    algorithms = _run_algorithms(args.run_target)
    run_paper_batch(
        output_root=args.output_root,
        models=args.model or [],
        embedding_model=args.embedding_model or "",
        replications=args.replications,
        algorithms=algorithms,
        config=config,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    return 0


def _load_optional_run_config(args: Namespace) -> HFRunConfig | None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return None
    return load_hf_run_config(config_path)


def _require_hf_transformers_provider(
    args: Namespace,
    config: HFRunConfig | None,
) -> None:
    provider = config.run.provider if config is not None else args.provider
    if provider != "hf-transformers":
        raise ValueError("The run command currently supports only --provider hf-transformers.")


def _run_algorithms(run_target: str) -> tuple[str, ...] | None:
    if run_target == "paper-batch":
        return None
    if run_target in {"algo1", "algo2", "algo3"}:
        return (run_target,)
    raise ValueError(f"Unsupported run target: {run_target}")


_RUN_TARGET_HANDLERS: dict[str, Callable[[Namespace], int]] = {
    "validate-config": _handle_validate_config,
    "resume-preflight": _handle_resume_preflight,
    "resume-sweep": _handle_resume_sweep,
    "prefetch-runtime": _handle_prefetch_runtime,
    "status": _handle_status,
    "refresh-ledger": _handle_refresh_ledger,
    "write-unfinished-manifest": _handle_write_unfinished_manifest,
    "prepare-qwen-algo1-tail": _handle_prepare_qwen_algo1_tail,
    "qwen-algo1-tail-preflight": _handle_qwen_algo1_tail_preflight,
    "drain-remaining": _handle_drain_remaining,
    "drain-status": _handle_drain_status,
    "smoke": _handle_smoke,
}


def _decoding_from_args(args: Namespace) -> DecodingConfig:
    if args.decoding == "greedy":
        return DecodingConfig(algorithm="greedy")
    if args.decoding == "beam":
        return DecodingConfig(
            algorithm="beam",
            num_beams=args.num_beams,
        )
    return DecodingConfig(
        algorithm="contrastive",
        penalty_alpha=args.penalty_alpha,
        top_k=args.top_k,
    )


def prefetch_runtime_for_config(*, config: HFRunConfig) -> dict[str, object]:
    runtime_factory = build_runtime_factory()
    return runtime_factory.prefetch_models(
        chat_models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
    )


def _prefetch_runtime_report_lines(report: Mapping[str, object]) -> list[str]:
    chat_model_names = _prefetch_chat_model_names(report)
    embedding_model = _prefetch_embedding_model(report)
    return [
        f"chat_models={','.join(chat_model_names)}",
        f"embedding_model={embedding_model}",
    ]


def _prefetch_chat_model_names(report: Mapping[str, object]) -> list[str]:
    chat_models = report.get("chat_models")
    if not _is_string_list(chat_models):
        raise ValueError("Prefetch runtime report chat_models must be a list of strings")
    return [item for item in chat_models if isinstance(item, str)]


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _prefetch_embedding_model(report: Mapping[str, object]) -> str:
    embedding_model = report.get("embedding_model")
    if not isinstance(embedding_model, str):
        raise ValueError("Prefetch runtime report embedding_model must be a string")
    return embedding_model
