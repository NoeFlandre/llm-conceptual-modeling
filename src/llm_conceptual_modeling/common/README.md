# `common`

Shared utilities that are used across the repository live here.

## Contents

- graph and baseline helpers
- evaluation and factorial-analysis primitives
- parsing, schema, and literal handling
- path resolution and repository-root helpers
- failure classification, retry logic, and spec serialization
- shared client protocols for chat and embedding providers
- `hf_transformers`: packaged runtime, client, parsing, and policy helpers

## Maintenance Rule

Move reusable logic here only when it is truly shared and stable. Keep this
package narrow, explicit, and easy to import without triggering runtime work.
