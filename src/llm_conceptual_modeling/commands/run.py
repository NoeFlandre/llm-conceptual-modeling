from argparse import Namespace

from llm_conceptual_modeling.hf_experiments import run_paper_batch
from llm_conceptual_modeling.hf_run_config import (
    load_hf_run_config,
    write_resolved_run_preview,
)


def handle_run(args: Namespace) -> int:
    if args.run_target == "validate-config":
        config = load_hf_run_config(args.config)
        write_resolved_run_preview(config=config, output_dir=args.output_dir)
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
