"""Compatibility shim for the main CLI entrypoint."""

from llm_conceptual_modeling.commands.cli import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
