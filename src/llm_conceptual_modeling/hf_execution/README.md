# `hf_execution`

Local worker execution helpers live here.

## Contents

- monitored subprocess execution
- download-environment setup
- persistent worker/session orchestration

## Maintenance Rule

Put execution mechanics here, not experiment semantics. If a helper only exists
to launch, supervise, or retry a worker, it belongs in this package.
