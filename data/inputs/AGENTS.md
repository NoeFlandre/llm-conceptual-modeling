# Inputs Guide

This directory contains the canonical input datasets used by the package.

Purpose:

- Store the committed graph CSVs and thesaurus JSON used by the algorithms.
- Keep source inputs separate from derived outputs and analysis artifacts.

Working rules:

- Treat these files as canonical references.
- Keep filenames stable because several modules resolve them directly.
- If a new input family is added, group it by dataset rather than mixing formats in this directory.
