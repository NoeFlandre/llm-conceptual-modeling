# Algorithm 1: End-to-End Implementation Guide

This document explains how Algorithm 1 works in the codebase, from experiment construction through prompting, verification, artifact writing, and factorial analysis.

The goal is to let a reader verify, at a high level, whether the implementation matches the paper's Method 1 design.

## What Method 1 Does

Method 1 is the direct edge-linking workflow.

In plain terms:

1. The model sees two causal maps.
2. It proposes candidate cross-map edges, including edges that may introduce new nodes.
3. A second prompt verifies each proposed edge with a yes/no judgment.
4. Only edges that pass verification are kept.

There is no tree expansion, no iterative embedding loop, and no thesaurus normalization in Method 1.

## The Full Runtime Flow

Method 1 is a two-call pipeline wrapped in a probe runner.

```text
                +----------------------+
                |      Start run       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Load spec and state  |
                | model, pair, prompt  |
                +----------+-----------+
                           |
                 resume? --+-- no
                   |       |
                   yes      v
                   |   +----------------------+
                   |   | Build edge prompt    |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | LLM call #1         |
                   |   | edge generation     |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Candidate edges      |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Build CoVe prompt    |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | LLM call #2         |
                   |   | Y/N verification     |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Filter verified edges |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Write summary/artifacts|
                   |   +----------+-----------+
                   |              |
                   v              v
            +------------------------------+
            | Return cached summary         |
            +------------------------------+
```

If an exception occurs, the run is marked failed and the error is recorded in the run directory.

## The Prompt Sequence

Method 1 uses two prompts in order.

### Prompt 1: Edge Generation

This is the main prompt that asks for new links between the two maps.

It is built from these blocks:

1. A fixed opener:

```text
You are a helpful assistant who understands Knowledge Maps.
```

2. Optional explanation text, controlled by the `include_explanation` factor.

3. A representation description, controlled by the adjacency/list factors.

4. Optional example text, controlled by `include_example`.

5. Optional counterexample text, controlled by `include_counterexample`.

6. The two input maps, rendered according to the chosen representation mode.

7. The task instruction:

```text
Your task is to recommend more links between the two maps. These links can use new nodes.
Do not suggest links that are already in the maps. Do not suggest links between nodes of the same map.
Return the recommended links as a list of edges in the format [(A, Z), ..., (X, D)].
```

8. The output restriction:

```text
Your output must only be the list of proposed edges. Do not repeat any instructions I have given you
and do not add unnecessary words or phrases.
```

### Prompt 2: Chain-of-Verification

The second prompt checks whether each candidate edge is actually supported.

Its logic is simple:

```text
Return whether a causal relationship exists between the source and target concepts for each pair in a list.
For example, given [('smoking', 'cancer'), ('ice cream sales', 'shark attacks')], return ['Y', 'N']
with no other text.
```

The candidate edges from Prompt 1 are inserted into that template.

## Exact Prompt Skeletons

This section shows the prompts as they are assembled, with placeholders for the dynamic parts.

### Generation prompt skeleton

```text
You are a helpful assistant who understands Knowledge Maps.
{optional explanation text}
{representation text}
{optional example text}
{optional counterexample text}
You will get two inputs: Knowledge map 1: {rendered subgraph 1} Knowledge map 2: {rendered subgraph 2}
Your task is to recommend more links between the two maps. These links can use new nodes. Do not
suggest links that are already in the maps. Do not suggest links between nodes of the same map.
Return the recommended links as a list of edges in the format [(A, Z), ..., (X, D)].
Your output must only be the list of proposed edges. Do not repeat any instructions I have given you
and do not add unnecessary words or phrases.
```

### Verification prompt skeleton

```text
Return whether a causal relationship exists between the source and target concepts for each pair in a list.
For example, given [('smoking', 'cancer'), ('ice cream sales', 'shark attacks')], return ['Y', 'N']
with no other text.
Candidate pairs: {candidate_edges}
```

### Dynamic placeholders

The generation prompt contains these dynamic values:

- `{optional explanation text}` is included only when `include_explanation` is on.
- `{representation text}` changes with the adjacency/list factors.
- `{optional example text}` is included only when `include_example` is on.
- `{optional counterexample text}` is included only when `include_counterexample` is on.
- `{rendered subgraph 1}` is the first graph rendered in the chosen representation.
- `{rendered subgraph 2}` is the second graph rendered in the chosen representation.

The verification prompt contains this dynamic value:

- `{candidate_edges}` is the list of candidate edges returned by the generation step.

## Representation Rendering

The representation text and graph rendering are controlled by two binary factors:

- `use_adjacency_notation`
- `use_array_representation`

The code produces one of four renderings.

### Case 1: `use_adjacency_notation = 1`, `use_array_representation = 1`

Prompt text:

```text
The knowledge map is encoded using a list of nodes and an associated adjacency matrix. The adjacency matrix
is an n*n square matrix that represents whether each edge exists. In the matrix, each row and each column
corresponds to a node. Rows and columns come in the same order as the list of nodes. A relation between node A
and node B is represented as a 1 in the row corresponding to A and the column corresponding to B.
```

Rendered map format:

```text
{'nodes': [...], 'adjacency_matrix': [[...], ...]}
```

### Case 2: `use_adjacency_notation = 1`, `use_array_representation = 0`

Prompt text:

```text
The knowledge map is encoded using tags for nodes and an associated adjacency matrix.
```

Rendered map format:

```text
<knowledge-map><nodes><node>...</node></nodes><adjacency-matrix>...</adjacency-matrix></knowledge-map>
```

### Case 3: `use_adjacency_notation = 0`, `use_array_representation = 1`

Prompt text:

```text
The knowledge map is encoded as a list of edges. Each edge is a pair of nodes.
```

Rendered map format:

```text
[('source', 'target'), ...]
```

### Case 4: `use_adjacency_notation = 0`, `use_array_representation = 0`

Prompt text:

```text
The knowledge map is encoded using a hierarchical markup language representation.
```

Rendered map format:

```text
<knowledge-map><edge source='source' target='target' />...</knowledge-map>
```

## Example And Counterexample Text

The optional example and counterexample blocks are fixed strings. They do not change with the graph pair.

### Example block

```text
Here is an example of a desired output for your task. In knowledge map 1, we have the list of nodes
['capacity to hire', 'bad employees', 'good reputation'] and the associated adjacency matrix
[[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have the list of nodes
['work motivation', 'productivity', 'financial growth'] and the associated adjacency matrix
[[0,1,0],[0,0,1],[0,0,0]]. In this example, you could recommend 3 new links: 'quality of managers' with
'work motivation', 'productivity' with 'good reputation' and 'bad employees' with 'quality of managers'.
These links implicitly create 1 new node: 'quality of managers'. Therefore, this is the expected output:
[('quality of managers', 'work motivation'), ('productivity', 'good reputation'),
('bad employees', 'quality of managers')].
```

### Counterexample block

```text
Here is an example of a bad output that we do not want to see. A bad output would be:
[('moon', 'bad employees')]. The error is the recommended link between 'moon' and 'bad employees'.
Adding the node 'moon' would be incorrect since it has no relationship with the other nodes. The proposed
link does not represent a true causal relationship.
```

## What Happens At Runtime

The probe runner applies the prompts in a fixed order.

1. It writes the manifest record, including the model label, the subgraph inputs, and the factor flags.
2. It writes the edge-generation prompt to disk.
3. It calls the generation prompt once and receives candidate edges.
4. It calls the CoVe verifier prompt once on those candidate edges.
5. It writes the checkpoint and summary artifacts.
6. On resume, it reloads the cached execution and summary instead of calling the provider again.

## Exact Prompt Variants

Method 1 has four representation variants and three optional guidance knobs.

### Representation modes

These are controlled by two binary factors:

- `use_adjacency_notation`
- `use_array_representation`

The four combinations are:

| Adjacency | Array | Rendering style |
| --- | --- | --- |
| 0 | 0 | Hierarchical markup language representation |
| 0 | 1 | List of edges |
| 1 | 0 | Tags for nodes + adjacency matrix |
| 1 | 1 | List of nodes + adjacency matrix |

### Optional guidance

These are controlled independently:

- `include_explanation`
- `include_example`
- `include_counterexample`

When enabled, the prompt includes:

- a conceptual explanation of what a knowledge map is
- one positive example showing the desired behavior
- one negative example showing what should not be produced

## Factorial Design

Method 1 is a full factorial design with five binary factors:

1. adjacency notation
2. array representation
3. explanation
4. example
5. counterexample

This produces `2^5 = 32` prompt conditions.

The experiment repeats every condition 5 times, so each pair of graphs produces:

- 32 conditions
- 5 replications per condition
- 160 runs per graph pair

Across all three supported graph pairs, that becomes 480 runs total.

The supported pair names are:

- `sg1_sg2`
- `sg2_sg3`
- `sg3_sg1`

The condition-bit order is:

```text
adjacency, array, explanation, example, counterexample
```

So `00000` means all factors are off, and `11111` means all factors are on.

## Runtime Settings

The runtime is intentionally conservative and deterministic at the transport layer.

- The chat client sends requests with `temperature = 0.0`.
- The client uses JSON-schema response formatting for both calls.
- The model name is supplied by the caller, not hard-coded inside the algorithm. The code can
  therefore be run with different model labels, but the prompt logic stays the same.
- The provider label written to manifests is `mistral`, because the live adapter is the Mistral chat client wrapper.

The two structured-output calls use these schema names:

- `edge_list` for candidate edge generation
- `vote_list` for CoVe verification

The response shapes are:

- `edge_list` returns `{"edges": [{"source": "...", "target": "..."}, ...]}`
- `vote_list` returns `{"votes": ["Y", "N", ...]}`

Important nuance:

- The code enforces JSON-schema structured output for the actual API call.
- The human-readable prompt still tells the model to return a list of edges or a Y/N list.
- This is a transport-layer constraint, not a change to the conceptual workflow.

## Artifact Flow

Each probe run writes a complete audit trail.

Primary artifacts:

- `manifest.json`
- `edge_generation_prompt.txt`
- `cove_prompt.txt`
- `execution_checkpoint.json`
- `summary.json`
- `events.jsonl`
- `error.json` when the run fails

What each artifact represents:

- `manifest.json` records the run specification: model, subgraphs, prompt flags, and run metadata.
- `edge_generation_prompt.txt` captures the exact generation prompt sent to the model.
- `cove_prompt.txt` captures the exact verification prompt sent to the model.
- `execution_checkpoint.json` stores the intermediate candidate and verified edges.
- `summary.json` stores the final run result.
- `events.jsonl` stores the run lifecycle events.
- `error.json` stores failure details when an exception is raised.

## How Verification Works

Method 1 uses chain-of-verification, not a second generation pass.

The process is:

1. Generate candidate edges.
2. Ask the verifier prompt whether each candidate edge represents a real causal relationship.
3. Keep only edges with a `Y` vote.

This means the final edge set is always a subset of the candidate edge set.

## What the Code Does Not Add

Method 1 does not add hidden post-processing beyond verification.

There is no:

- synonym normalization
- antonym collapse
- embedding loop
- recursive expansion
- edge deduplication beyond what the model and verifier naturally produce

The algorithm is deliberately simple: generate, verify, keep.

## How the Experiment Runner Uses the Prompt

The experiment runner builds every prompt condition ahead of time and stores the full prompt text in the manifest.

That means you can verify the design without running a live model:

1. check the condition bits
2. inspect the generated prompt preview
3. confirm the factor setting
4. confirm the replication count

This is the practical bridge between the paper's factorial design and the codebase.

## How To Check Faithfulness Against the Paper

Use this checklist:

- The prompt starts with the knowledge-map opener.
- The prompt changes with the five binary factors.
- The input contains two rendered maps.
- The output instruction asks for direct links.
- The CoVe prompt asks for `Y` or `N` judgments over edge pairs.
- The runtime uses `temperature = 0.0`.
- Each condition is replicated 5 times.
- The experiment grid contains 32 conditions per graph pair.
- Candidate edges are generated first and verified second.

If all of those are true, the implementation is aligned with Method 1's intended structure.

## Short Summary

Method 1 is a two-stage direct-linking workflow:

1. generate candidate cross-map edges
2. verify each candidate with a yes/no CoVe prompt

The experiment varies five prompt factors across 32 conditions and repeats each condition 5 times, with deterministic temperature and auditable run artifacts.
