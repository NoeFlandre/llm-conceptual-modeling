# Snapshot Fixtures Guide

This directory contains JSON snapshots for CLI and verification contract tests.

Purpose:

- Preserve stable output expectations for selected commands.
- Keep snapshot-based tests easy to inspect and update deliberately.

Working rules:

- Update these files only when the expected CLI or verification contract changes.
- Keep each snapshot focused on one command or one top-level workflow.
- Preserve compact JSON formatting so diffs stay readable.
- If new snapshot families are added, split them into subdirectories by command or workflow.
