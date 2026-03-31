from argparse import Namespace

from llm_conceptual_modeling.hf_experiments import run_paper_batch


def handle_run(args: Namespace) -> int:
    if args.provider != "hf-transformers":
        raise ValueError(
            "The run command currently supports only --provider hf-transformers."
        )

    if args.run_target == "paper-batch":
        algorithms = None
    elif args.run_target in {"algo1", "algo2", "algo3"}:
        algorithms = (args.run_target,)
    else:
        raise ValueError(f"Unsupported run target: {args.run_target}")

    run_paper_batch(
        output_root=args.output_root,
        models=args.model,
        embedding_model=args.embedding_model,
        replications=args.replications,
        algorithms=algorithms,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    return 0
