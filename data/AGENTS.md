# Data Directory Guide

This directory contains committed input data, benchmark results, and analysis artifacts.

Purpose:

- Preserve canonical inputs and imported result corpora used by the repository.
- Separate immutable source data from derived artifacts.

Working rules:

- Treat committed data as versioned reference material.
- Prefer adding a subdirectory guide when a data family has its own contract.
- Keep derived outputs grouped away from raw inputs.
- Do not repurpose these directories for ad hoc scratch data.
