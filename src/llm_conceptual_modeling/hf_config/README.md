# `hf_config`

Configuration helpers for Hugging Face-backed workflows live here.

## Contents

- run-config loading and normalization
- config objects used by batch and resume workflows

## Maintenance Rule

Keep this package limited to configuration parsing and validation. Runtime and
execution code should not grow here.
