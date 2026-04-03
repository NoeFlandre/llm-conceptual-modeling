import json
from argparse import Namespace

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig
from llm_conceptual_modeling.hf_batch.monitoring import collect_batch_status
from llm_conceptual_modeling.hf_experiments import (
    run_paper_batch,
    run_single_spec,
    select_run_spec,
)
from llm_conceptual_modeling.hf_resume_preflight import build_resume_preflight_report
from llm_conceptual_modeling.hf_resume_sweep import build_resume_sweep_report
from llm_conceptual_modeling.hf_run_config import (
    load_hf_run_config,
    write_resolved_run_preview,
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
    if args.run_target == "smoke":
        config = load_hf_run_config(args.config)
        spec = select_run_spec(
            config=config,
            algorithm=args.algorithm,
            model=args.model,
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
