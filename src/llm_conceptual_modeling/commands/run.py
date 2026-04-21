import json
from argparse import Namespace
from collections.abc import Mapping
from typing import cast

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
from llm_conceptual_modeling.hf_resume.preflight import (
    ResumePreflightReport,
    build_resume_preflight_report,
)
from llm_conceptual_modeling.hf_resume.sweep import build_resume_sweep_report
from llm_conceptual_modeling.hf_state.ledger import refresh_ledger
from llm_conceptual_modeling.hf_state.shard_manifest import write_unfinished_shard_manifest
from llm_conceptual_modeling.hf_tail.qwen_algo1 import (
    build_qwen_algo1_tail_preflight_report,
    prepare_qwen_algo1_tail_bundle,
)


def handle_run(args: Namespace) -> int:
    if args.run_target == "validate-config":
        config = load_hf_run_config(args.config)
        write_resolved_run_preview(config=config, output_dir=args.output_dir)
        return 0
    if args.run_target == "resume-preflight":
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
            print(f"results_root={report['results_root']}")
            print(f"total_runs={report['total_runs']}")
            print(f"finished={report['finished_count']}")
            print(f"failed={report['failed_count']}")
            print(f"pending={report['pending_count']}")
            print(f"can_resume={report['can_resume']}")
            print(f"resume_mode={report['resume_mode']}")
        return 0
    if args.run_target == "resume-sweep":
        report = build_resume_sweep_report(
            repo_root=args.repo_root,
            results_root=args.results_root,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"repo_root={report['repo_root']}")
            print(f"results_root={report['results_root']}")
            print(f"roots={report['root_count']}")
            print(f"ready={report['ready_count']}")
            print(f"needs_config_fix={report['needs_config_fix_count']}")
            print(f"invalid_config={report['invalid_config_count']}")
            print(f"active={report['active_count']}")
            print(f"finished={report['finished_count']}")
        return 0
    if args.run_target == "prefetch-runtime":
        config = load_hf_run_config(args.config)
        report = prefetch_runtime_for_config(config=config)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for line in _prefetch_runtime_report_lines(report):
                print(line)
        return 0
    if args.run_target == "status":
        status = collect_batch_status(args.results_root)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(f"total={status['total_runs']}")
            print(f"finished={status['finished_count']}")
            print(f"failed={status['failed_count']}")
            print(f"running={status['running_count']}")
            print(f"pending={status['pending_count']}")
            print(f"complete={status['percent_complete']}%")
            if status.get("worker_pid") is not None:
                print(f"worker_pid={status['worker_pid']}")
            if status.get("worker_status") is not None:
                print(f"worker_status={status['worker_status']}")
            if status.get("active_stage_age_seconds") is not None:
                print(f"active_stage_age_seconds={status['active_stage_age_seconds']}")
        return 0
    if args.run_target == "refresh-ledger":
        ledger = refresh_ledger(results_root=args.results_root, ledger_root=args.ledger_root)
        if args.json:
            print(json.dumps(ledger, indent=2, sort_keys=True))
        else:
            print(f"ledger_root={args.ledger_root}")
            print(f"expected_total_runs={ledger['expected_total_runs']}")
            print(f"finished_count={ledger['finished_count']}")
            print(f"retryable_failed_count={ledger['retryable_failed_count']}")
            print(f"terminal_failed_count={ledger['terminal_failed_count']}")
            print(f"pending_count={ledger['pending_count']}")
        return 0
    if args.run_target == "write-unfinished-manifest":
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
    if args.run_target == "prepare-qwen-algo1-tail":
        report = prepare_qwen_algo1_tail_bundle(
            canonical_results_root=args.canonical_results_root,
            tail_results_root=args.tail_results_root,
            remote_output_root=args.remote_output_root,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"tail_results_root={report['tail_results_root']}")
            print(f"identity_count={report['identity_count']}")
            print(f"config_path={report['config_path']}")
            print(f"manifest_path={report['manifest_path']}")
        return 0
    if args.run_target == "qwen-algo1-tail-preflight":
        report = build_qwen_algo1_tail_preflight_report(
            repo_root=args.repo_root,
            canonical_results_root=args.canonical_results_root,
            tail_results_root=args.tail_results_root,
            watcher_status_path=args.watcher_status_path,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            resume_preflight = cast(ResumePreflightReport, report["resume_preflight"])
            print(f"tail_results_root={report['tail_results_root']}")
            print(f"tail_pending_count={resume_preflight['pending_count']}")
            print(f"tail_can_resume={resume_preflight['can_resume']}")
        return 0
    if args.run_target == "drain-remaining":
        if args.plan_only:
            report = build_drain_plan(
                repo_root=args.repo_root,
                results_root=args.results_root,
                ssh_command=args.ssh_command,
                state_file=args.state_file,
                phase=args.phase,
                full_coverage=args.full_coverage,
                root_name_contains=args.root_name_contains,
            )
        else:
            report = run_drain_supervisor(
                repo_root=args.repo_root,
                results_root=args.results_root,
                ssh_command=args.ssh_command,
                state_file=args.state_file,
                phase=args.phase,
                full_coverage=args.full_coverage,
                root_name_contains=args.root_name_contains,
                poll_seconds=args.poll_seconds,
                stale_after_seconds=args.stale_after_seconds,
                quick_resume_script=args.quick_resume_script,
            )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"state_file={report['state_file']}")
            print(f"safe_queue_count={report.get('safe_queue_count', 0)}")
            print(f"risky_queue_count={report.get('risky_queue_count', 0)}")
            if report.get("adopted_results_root"):
                print(f"adopted_results_root={report['adopted_results_root']}")
        return 0
    if args.run_target == "drain-status":
        report = read_drain_state_report(args.state_file)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"state_file={args.state_file}")
            print(f"health={report.get('health', 'unknown')}")
            print(f"current_phase={report.get('current_phase', 'unknown')}")
            print(f"current_results_root={report.get('current_results_root', 'unknown')}")
        return 0
    if args.run_target == "smoke":
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

    config = load_hf_run_config(args.config) if getattr(args, "config", None) else None
    provider = config.run.provider if config is not None else args.provider
    if provider != "hf-transformers":
        raise ValueError("The run command currently supports only --provider hf-transformers.")

    if args.run_target == "paper-batch":
        algorithms = None
    elif args.run_target in {"algo1", "algo2", "algo3"}:
        algorithms = (args.run_target,)
    else:
        raise ValueError(f"Unsupported run target: {args.run_target}")

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


def _decoding_from_args(args: Namespace) -> DecodingConfig:
    if args.decoding == "greedy":
        return DecodingConfig(algorithm="greedy", temperature=0.0)
    if args.decoding == "beam":
        return DecodingConfig(
            algorithm="beam",
            num_beams=args.num_beams,
            temperature=0.0,
        )
    return DecodingConfig(
        algorithm="contrastive",
        penalty_alpha=args.penalty_alpha,
        top_k=args.top_k,
        temperature=0.0,
    )


def prefetch_runtime_for_config(*, config: HFRunConfig) -> dict[str, object]:
    runtime_factory = build_runtime_factory()
    return runtime_factory.prefetch_models(
        chat_models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
    )


def _prefetch_runtime_report_lines(report: Mapping[str, object]) -> list[str]:
    chat_models = report.get("chat_models")
    if not isinstance(chat_models, list) or not all(isinstance(item, str) for item in chat_models):
        raise ValueError("Prefetch runtime report chat_models must be a list of strings")
    chat_model_names = cast(list[str], chat_models)
    embedding_model = report.get("embedding_model")
    if not isinstance(embedding_model, str):
        raise ValueError("Prefetch runtime report embedding_model must be a string")
    return [
        f"chat_models={','.join(chat_model_names)}",
        f"embedding_model={embedding_model}",
    ]
