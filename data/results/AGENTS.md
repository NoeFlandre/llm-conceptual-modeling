# Results Directory Guide

This directory contains imported result corpora organized by algorithm and model.

Purpose:

- Preserve historical result layouts used by the analysis and verification workflows.
- Keep imported outputs separate from source inputs and debug artifacts.

Working rules:

- Treat the directory structure as part of the contract.
- Keep algorithm/model subtrees stable unless the regression tests are updated.
- Add an `AGENTS.md` in a deeper subtree when that subtree has its own naming or content rules.
- Do not mix imported corpora with generated scratch outputs here.
