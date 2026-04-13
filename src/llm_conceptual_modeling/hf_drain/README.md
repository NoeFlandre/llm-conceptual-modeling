# `hf_drain`

Drain-planning and remaining-results supervisor logic live here.

## Contents

- queue shaping for the remaining unfinished runs
- drain planning and scheduling
- runtime supervisor helpers used by the tail sweep

## Maintenance Rule

Keep the drain logic separate from the canonical paper-batch runner. The drain
path is operational glue, not experiment logic.
