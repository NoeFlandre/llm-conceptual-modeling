# HF Quality Sweep Design

**Goal**

Reduce orchestration complexity and technical debt in the HF/Vast execution path without changing
experiment semantics, outputs, or operational behavior.

**Scope**

This sweep is limited to the HF execution/orchestration layer and the related Vast helper scripts.
The selected models, algorithms, DOE structure, decoding grid, retry policy semantics, and artifact
formats must remain unchanged.

**Problems**

- [`src/llm_conceptual_modeling/hf_experiments.py`](../../../src/llm_conceptual_modeling/hf_experiments.py)
  is too large and mixes planning, resume ordering, worker dispatch, run execution, artifact
  handling, and policy resolution.
- Worker mode / retry / timeout policy resolution is spread across multiple helper blocks rather
  than grouped around the execution path they control.
- The Vast operational scripts have grown useful features quickly, but some sync/launch behavior is
  still duplicated or loosely coupled.
- Local result artifacts create working-tree noise, which makes code review and cleanup harder.

**Recommended Approach**

Do a focused architecture sweep in micro-steps:

1. extract execution dispatch and policy resolution seams from `hf_experiments.py`
2. extract resume-ordering and seeded-run accounting seams from `hf_experiments.py`
3. remove duplicated helper logic in the Vast sync/launch scripts
4. sweep dead code, stale branches, and documentation drift created by the extraction

Each step must be red/green TDD and must leave the codebase runnable before the next step starts.

**Architecture Direction**

- Keep [`hf_experiments.py`](../../../src/llm_conceptual_modeling/hf_experiments.py) as the public
  orchestration façade, but move narrow responsibilities into dedicated modules with explicit
  interfaces.
- Group execution-path helpers together instead of scattering them across unrelated sections.
- Keep the new persistent-worker machinery intact, but reduce the amount of orchestration knowledge
  required to understand or modify it.
- Keep script behavior simple: shell should remain a thin operational wrapper, while Python owns
  semantics and policy.

**Verification Strategy**

- Every extraction starts with a narrow failing test or test adjustment.
- Every micro-step runs a narrow `pytest` slice first.
- Then run the broader HF-focused suite, plus `ruff` and `ty`, before commit.
- No remote rollout is required for pure refactor steps unless the step changes operational
  behavior.

**Non-Goals**

- no experiment redesign
- no model/decode/DOE changes
- no artifact schema changes unless strictly required for cleanup
- no broad refactor outside the HF/Vast execution path

**Success Criteria**

- `hf_experiments.py` is materially smaller and easier to read
- duplicated orchestration code is reduced
- tests still cover the critical execution paths
- docs stay aligned with the actual operational entrypoints
- no regression in local verification or live batch behavior
