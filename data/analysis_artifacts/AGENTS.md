# Analysis Artifacts Guide

This directory contains derived analysis outputs and debug artifacts.

Purpose:

- Keep revision-tracker outputs and post-revision debug runs separated.
- Store artifacts that support auditability without mixing them with imported result corpora.

Working rules:

- Treat these files as derived evidence, not source inputs.
- Add a deeper guide under any subdirectory that has its own naming or retention rules.
- Keep debug and revision-tracking outputs isolated from one another.
- Preserve the directory structure because other tooling may walk it directly.
