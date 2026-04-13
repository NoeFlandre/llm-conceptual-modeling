# `hf_worker`

Worker-side request, result, state, and policy helpers live here.

## Contents

- worker entrypoint and main loop
- request/result serialization helpers
- worker state helpers
- runtime policy and timeout helpers
- persistent worker session management

## Maintenance Rule

Keep the worker package focused on execution inside the worker process. Batch
planning, resume logic, and higher-level orchestration should stay elsewhere.
