# Legacy Algo2 Fixtures Guide

This directory contains the committed legacy fixtures for Algorithm 2.

Purpose:

- Keep the raw, evaluated, and factorial CSVs for Algorithm 2 together.
- Preserve the `gpt-5` layout expected by the regression tests.

Working rules:

- Do not rename or move these files unless the associated tests are updated.
- Keep the `raw` / `evaluated` / `factorial` split stable.
- Add a deeper guide if new model-specific fixture families are introduced under this subtree.
