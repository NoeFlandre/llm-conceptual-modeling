# Legacy Fixtures Guide

This directory contains the committed regression fixtures for legacy algorithm outputs.

Purpose:

- Preserve the raw, evaluated, and factorial CSV inputs used by the tests.
- Keep the legacy directory structure stable so path-based contracts remain reproducible.

Layout:

- `algo1/gpt-5/raw`: raw Algorithm 1 outputs.
- `algo1/gpt-5/evaluated`: evaluated Algorithm 1 metrics.
- `algo1/gpt-5/factorial`: factorial outputs for Algorithm 1.
- `algo2/gpt-5/raw`: raw Algorithm 2 outputs.
- `algo2/gpt-5/evaluated`: evaluated Algorithm 2 metrics.
- `algo2/gpt-5/factorial`: factorial outputs for Algorithm 2.
- `algo3/gpt-5/raw`: raw Algorithm 3 outputs.
- `algo3/gpt-5/evaluated`: evaluated Algorithm 3 metrics.
- `algo3/gpt-5/factorial`: factorial outputs for Algorithm 3.

Working rules:

- Treat these files as immutable regression artifacts unless a test contract changes.
- Keep naming consistent with the existing CLI and verification paths.
- If a new fixture family is added, mirror the same `raw`/`evaluated`/`factorial` split under the relevant algorithm.
