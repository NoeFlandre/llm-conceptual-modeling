# Fixtures Guide

This directory groups committed test fixtures and snapshots.

Purpose:

- Keep regression data organized under clear subdirectories.
- Separate legacy CSV fixtures from snapshot JSON fixtures.

Working rules:

- Treat fixture updates as contract changes and keep them narrow.
- Add a subdirectory-level `AGENTS.md` when a fixture family has its own rules.
- Preserve stable naming so tests can reference paths directly.
- If a fixture family grows substantially, split it by algorithm, command, or output type.
