# `hf_tail`

Specialized helpers for the final Qwen `algo1` tail sweep live here.

## Contents

- dedicated tail manifest and config generation
- isolated preflight checks
- fresh-host launch helpers for the tail run

## Maintenance Rule

Keep this package narrow. It should only contain the special-purpose tail path
needed to finish the final Qwen/Mistral residual work.
