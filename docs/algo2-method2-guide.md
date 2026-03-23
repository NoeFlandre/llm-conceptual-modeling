# Method 2 Audit Guide

This document explains Method 2 end to end, at the level needed to verify the implementation against the paper.

Method 2 is the label-expansion pipeline:
- start from two subgraphs
- iteratively propose new labels
- stop when embedding similarity stabilizes
- suggest cross-map edges among the expanded labels
- normalize edge terms with the thesaurus
- remove edges that already existed in either input graph
- run chain-of-verification on the remaining candidate edges

The goal here is not to summarize the code abstractly. The goal is to expose the full runtime sequence, the exact prompt assembly, the convergence rule, the artifacts written to disk, and the places where the implementation can be compared against the paper.

## What Method 2 Takes As Input

Each Method 2 run is built from one pair of subgraphs:
- `sg1_sg2`
- `sg2_sg3`
- `sg3_sg1`

The pair is selected at experiment construction time. The full DOE is a `2^6 x 5` design:
- 5 binary prompt factors
- 1 binary convergence factor
- 5 replications
- 64 conditions per pair
- 320 runs per pair overall

The six binary factors are:
- adjacency vs non-adjacency notation
- array vs list representation
- explanation on/off
- example on/off
- counterexample on/off
- relaxed convergence on/off

## High-Level Runtime State Machine

```text
┌──────────────────────────┐
│ START                    │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ Build probe spec         │
│ - run name               │
│ - model                  │
│ - seed labels            │
│ - subgraph1/subgraph2    │
│ - prompt config          │
│ - convergence threshold  │
│ - output dir             │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ Resume check             │
│ summary.json + finished? │
└───────┬─────────┬────────┘
        │yes      │no
        v         v
┌──────────────┐  ┌────────────────────────────┐
│ Return cached│  │ Write manifest + prompt    │
│ summary      │  │ artifacts                  │
└──────────────┘  └─────────────┬──────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ Run label expansion loop  │
                    │ - propose 5 labels        │
                    │ - embed                   │
                    │ - measure similarity      │
                    │ - stop at threshold       │
                    └────────────┬─────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ Save execution checkpoint │
                    └────────────┬─────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ Build edge-suggestion     │
                    │ prompt                    │
                    └────────────┬─────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ LLM suggests edges       │
                    └────────────┬─────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ Normalize with thesaurus │
                    │ Remove pre-existing edges│
                    │ CoVe verify remaining    │
                    └────────────┬─────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │ Write summary.json       │
                    │ Mark probe finished      │
                    └────────────┬─────────────┘
                                 │
                                 v
                          ┌──────────────┐
                          │ COMPLETE     │
                          └──────────────┘
```

## The Exact Prompt Phases

Method 2 has two distinct LLM phases:
1. label expansion
2. edge suggestion

The prompt factorization is shared between these two phases, but the task text differs.

### Prompt Prefix Shared By Both Phases

The prefix is assembled in the same order each time:

```text
You are a helpful assistant who understands Knowledge Maps.
```

If `include_explanation` is on:

```text
A knowledge map is a network consisting of nodes and edges. Nodes must have a clear meaning, such that we can interpret having 'more' or 'less' of a node. Edges represent the existence of a direct relation between two nodes.
```

Then one notation section is added, chosen from four possibilities:

```text
The knowledge map is encoded using a list of nodes and an associated adjacency matrix. The adjacency matrix is an n*n square matrix that represents whether each edge exists. In the matrix, each row and each column corresponds to a node. Rows and columns come in the same order as the list of nodes. A relation between node A and node B is represented as a 1 in the row corresponding to A and the column corresponding to B.
```

```text
The knowledge map is encoded using a hierarchical markup language representation. The list of nodes is defined between the opening tag <NODES> and the matching closing tag </NODES>. For each node, we list all other nodes by ID and indicate whether there is a connection ('True') or not ('False').
```

```text
The knowledge map is encoded using the RDF representation. The RDF representation uses tags denoted by <>. <S> is the start of the map. <E> is the end of the map. In between <S> and <E>, we list all edges of the map. Each edge is represented with two tags: <H> precedes the node, then <T> precedes the target node.
```

```text
The knowledge map is encoded as a list of edges. Each edge is a pair of nodes.
```

If `include_example` is on, an example block is appended.
If `include_counterexample` is on, a counterexample block is appended.

## Label Expansion Prompt

This is the first LLM task.

The prompt structure is:

```text
You are a helpful assistant who understands Knowledge Maps.
[optional explanation]
[notation section]
[optional example]
[optional counterexample]
You will get two inputs: Knowledge map 1: ...
Knowledge map 2: ...
Your task is to recommend 5 more nodes in relation to those already in the two knowledge maps. Do not suggest nodes that are already in the maps. Return the recommended nodes as a list of nodes in the format ['A', 'B', 'C', 'D', 'E'].
Your output must only be the list of proposed concepts. Do not repeat any instructions I have given you and do not add unnecessary words or phrases.
```

### Dynamic parts of the label prompt

The dynamic pieces are:
- the selected pair of subgraphs
- the chosen representation mode
- the example/counterexample inclusion flags
- the list of seed labels if the prompt is built in the fallback mode

In the actual DOE run, the prompt uses the two input subgraphs, not just the seed label list.

### The label output schema

The label phase uses structured output with schema name:
- `label_list`

Expected shape:

```json
{"labels": ["...", "...", "...", "...", "..."]}
```

The runtime then appends these labels to the original seed label set.

## Convergence Loop

Method 2 does not take a fixed number of expansion iterations. It runs until the similarity stabilizes.

The loop behavior is:
1. start with the seed labels from both subgraphs
2. ask the model for 5 new labels
3. embed the candidate labels and the source labels
4. compute average best-match similarity
5. compare the current similarity against the previous iteration
6. stop when the absolute difference is at or below the threshold

The paper describes this as:
- stop at 1% consecutive difference
- or stop half as early

The code implements that as:
- `0.01` when the convergence bit is off
- `0.02` when the convergence bit is on

This is an implementation choice that captures the paper’s intent as an absolute delta threshold.

### Convergence state machine

```text
┌────────────────────────┐
│ seed labels ready      │
└───────────┬────────────┘
            │
            v
┌────────────────────────┐
│ propose 5 labels       │
└───────────┬────────────┘
            │
            v
┌────────────────────────┐
│ embed labels           │
└───────────┬────────────┘
            │
            v
┌────────────────────────┐
│ compute similarity     │
└───────────┬────────────┘
            │
            v
┌────────────────────────┐
│ |current - previous| ? │
└───────┬─────────┬──────┘
        │ yes      │ no
        v          v
┌──────────────┐  ┌────────────────────┐
│ stop loop    │  │ add labels and     │
│              │  │ repeat             │
└──────────────┘  └────────────────────┘
```

## Embeddings Layer

The expansion loop uses embeddings to measure similarity.

The sequence is:
- deduplicate labels
- embed the unique candidate labels and seed labels
- compute similarity from those embeddings

The implementation supports two embedding backends:
- `mistral` with `mistral-embed-2312` as the current default
- `openrouter` with the paper-faithful `text-embedding-3-large`

The sequence is the same in both cases:
- one embedding request per deduplicated label batch
- cosine similarity as the scoring basis

For paper fidelity, the OpenRouter path is the one that matches the manuscript. The Mistral path remains available for the current repo defaults and for backward-compatible runs.

The chat provider is also configurable:
- `mistral` current default
- `openrouter` via the OpenAI-compatible OpenRouter API
- `anthropic` remains a debug/research-only path in this repo

For Method 2, the chat provider and embedding provider are separate settings so you can mix them if needed.

This is the part of Method 2 that is most distinct from Method 1.

## Edge Suggestion Prompt

After convergence, the runtime builds a second prompt over the expanded label context.

The structure is:

```text
You are a helpful assistant who understands Knowledge Maps.
[optional explanation]
[notation section]
[optional example]
[optional counterexample]
You will get two inputs: Knowledge map 1: ...
Knowledge map 2: ...
Your task is to recommend more links between the two maps. These links can use new nodes. Do not suggest links that are already in the maps. Do not suggest links between nodes of the same map. Return the recommended links as a list of edges in the format [(A, Z), ..., (X, D)].
Available concepts: ...
```

### Dynamic parts of the edge prompt

The dynamic pieces are:
- the original subgraph pair
- the expanded label context
- the chosen representation mode
- the example/counterexample inclusion flags

The `Available concepts:` line is the additional context introduced after expansion.

### The edge output schema

The edge phase uses structured output with schema name:
- `edge_list`

Expected shape:

```json
{"edges": [{"source": "...", "target": "..."}, ...]}
```

The code now also rejects null endpoints instead of quietly converting them to `"None"`.

## Thesaurus Normalization

After the edge LLM call completes, the candidate edges are normalized through the thesaurus.

This does two things:
- map synonyms to base forms
- map antonyms to base forms

The normalizer is applied before final filtering. This means that two differently worded edges can collapse to the same canonical edge.

## Existing-Edge Removal

This step is explicit in the code and also described in the paper.

After normalization:
- edges that already existed in `subgraph1` or `subgraph2` are removed
- only novel cross-subgraph edges remain for verification

This is important because Method 2 is not supposed to recommend edges the model already had.

## Chain of Verification

The remaining candidate edges are passed through CoVe.

The CoVe prompt is built from the candidate edge list and asks for a `Y` / `N` vote list.
The output schema is:
- `vote_list`

Expected shape:

```json
{"votes": ["Y", "N", "Y", ...]}
```

Those votes are used to keep only the verified edges.

## What Gets Written To Disk

Every Method 2 condition writes a resumable set of artifacts:
- `manifest.json`
- `manifest.yaml`
- `label_expansion_prompt.txt`
- `edge_suggestion_prompt.txt`
- `execution_checkpoint.json`
- `summary.json`
- `events.jsonl`
- `run.log`
- `state.json`

The checkpointing is stage-aware:
- if label expansion completed, it does not rerun that phase
- if execution completed, it does not rerun the expansion/edge suggestion stage
- if the summary exists and the run is marked finished, `--resume` returns it immediately

## Method 2 Probe Flow In Plain English

1. Create the run context.
2. If `--resume` and the summary already exists, return it.
3. Write the manifest and the initial label prompt.
4. Build the label proposer, edge suggester, embeddings client, and thesaurus.
5. Run label expansion until similarity stabilizes.
6. Save the execution checkpoint.
7. Build the edge suggestion prompt.
8. Suggest edges across the expanded label set.
9. Normalize the edges with the thesaurus.
10. Remove edges that already existed in either input map.
11. Run CoVe on the remaining edges.
12. Save the summary and mark the probe complete.

## What To Check Against The Paper

If you want to verify fidelity, check these items:
- whether the paper’s convergence rule matches the `0.01` / `0.02` threshold split
- whether the paper expects the label-expansion prompt to include the same explanation/example/counterexample structure
- whether the paper expects the edge suggestion phase to receive the expanded label context in addition to the two subgraphs
- whether the paper explicitly removes pre-existing edges before final verification
- whether the paper’s Method 2 uses thesaurus normalization before verification
- whether the OpenRouter embedding option is selected when you want the paper-faithful `text-embedding-3-large` backend

## Bottom Line

Method 2 is implemented as a two-stage, resumable pipeline:
- iterative label expansion with embedding-based convergence
- edge suggestion over the expanded label set
- thesaurus normalization
- pre-existing edge removal
- CoVe verification

The code is structured enough that you can audit each step separately against the paper. The two places most worth double-checking are:
- the convergence threshold semantics
- the exact wording of the prompt templates versus the paper’s prose
