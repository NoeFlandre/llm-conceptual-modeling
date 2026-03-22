"""Experiment manifest YAML parsing and validation.

This module provides :func:`parse_manifest`, which loads an experiment manifest
YAML file and validates it against the expected schema before returning a
fully-constructed :class:`~llm_conceptual_modeling.common.types.ExperimentManifest`
object.

Schema fields
-------------
experiment_id : str
    Unique identifier for the run. Format: ``{algorithm}_{pair_name}_rep{index}_cond{bits}``.
algorithm : str
    One of ``algo1``, ``algo2``, ``algo3``.
model : str
    The LLM model name (e.g. ``mistral-small-2603``).
provider : str
    One of ``mistral``, ``anthropic``.
temperature : float
    LLM sampling temperature (0.0 – 2.0).
top_p : float | None
    Nucleus sampling parameter.
max_tokens : int | None
    Maximum tokens in LLM response.
prompt_factors : dict[str, bool | int]
    Prompt configuration factors and their levels.
full_prompt : str
    The complete prompt text sent to the LLM.
input_subgraph_pairs : list[dict[str, Any]]
    The input subgraph pairs. Algo1/2 format: ``{subgraph_name, edges}``.
    Algo3 format: ``{subgraph_name, labels}``.
output_dir : str
    Absolute path to the experiment output directory.
timestamp : str
    ISO 8601 timestamp of when the manifest was created.
repetitions : int
    Total number of repetitions in the full experiment.
condition_bits : str
    Bitstring encoding the experimental condition.
pair_name : str
    Human-readable name for the subgraph pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_conceptual_modeling.common.types import ExperimentManifest

#: Valid algorithm identifiers.
_VALID_ALGORITHMS = frozenset(("algo1", "algo2", "algo3"))

#: Required top-level fields in the manifest YAML.
_REQUIRED_FIELDS = (
    "experiment_id",
    "algorithm",
    "model",
    "provider",
    "temperature",
    "prompt_factors",
    "full_prompt",
    "input_subgraph_pairs",
    "output_dir",
    "timestamp",
    "repetitions",
    "condition_bits",
    "pair_name",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest YAML fails schema validation."""


def _validate(data: dict[str, Any]) -> None:
    """Validate a parsed manifest dictionary against the expected schema.

    Parameters
    ----------
    data
        A dictionary parsed from a manifest YAML file.

    Raises
    ------
    ManifestValidationError
        If a required field is missing or a field has an invalid value.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ManifestValidationError(
            f"Manifest is missing required fields: {missing}"
        )

    algorithm = data.get("algorithm", "")
    if algorithm not in _VALID_ALGORITHMS:
        raise ManifestValidationError(
            f"Invalid algorithm {algorithm!r}; must be one of {sorted(_VALID_ALGORITHMS)}"
        )

    # Validate repetitions
    repetitions = data.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ManifestValidationError(
            f"repetitions must be a positive integer, got {repetitions!r}"
        )

    # Validate temperature
    temperature = data.get("temperature")
    if not isinstance(temperature, (int, float)):
        raise ManifestValidationError(
            f"temperature must be a number, got {temperature!r}"
        )

    # Validate input_subgraph_pairs is a list
    isp = data.get("input_subgraph_pairs")
    if not isinstance(isp, list):
        raise ManifestValidationError(
            f"input_subgraph_pairs must be a list, got {type(isp).__name__}"
        )


def parse_manifest(yaml_path: str | Path) -> ExperimentManifest:
    """Load and validate an experiment manifest YAML file.

    Parameters
    ----------
    yaml_path
        Path to the ``manifest.yaml`` file produced by an experiment run.

    Returns
    -------
    ExperimentManifest
        A fully-validated manifest object capturing all experiment configuration.

    Raises
    ------
    ManifestValidationError
        If the YAML is present but fails schema validation.
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    path = Path(yaml_path)
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)  # type: ignore[assignment]

    if data is None:
        raise ManifestValidationError("Manifest YAML file is empty")

    _validate(data)

    return ExperimentManifest.from_dict(data)
