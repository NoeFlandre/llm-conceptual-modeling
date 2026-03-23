# Algorithm 3: End-to-End Implementation Guide

This document explains Method 3 in the codebase from end to end.

The goal is to make the implementation auditable against the paper and the archived run structure:
- what the method is trying to do
- the exact prompt text and which parts change
- how the tree expansion recursion works
- how the factorial experiment is assembled
- what gets written to disk
- how resume works
- how the offline evaluation computes recall

Method 3 is the indirect, tree-based approach.

## What Method 3 Does

Method 3 starts from a source label set and tries to reach target labels by repeatedly expanding each source concept into a tree of related concepts.

In practical terms:

1. The model receives a list of source concept names.
2. It proposes a fixed number of related child concepts for each label.
3. If a child concept already matches one of the target labels, that branch stops.
4. If it does not match, the child can itself be expanded again, up to a maximum depth.
5. The final output is the tree of generated labels and the subset that matched the target labels.

Unlike Methods 1 and 2:
- there is no cross-map edge generation prompt
- there is no chain-of-verification prompt
- there is no embedding loop
- there is no thesaurus normalization

The method is purely recursive tree expansion plus target matching.

## High-Level Runtime Flow

```text
                +----------------------+
                |      Start run       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Build probe spec     |
                | model, pair, prompt  |
                | child_count, depth   |
                +----------+-----------+
                           |
                 resume? --+-- no
                   |       |
                   yes      v
                   |   +----------------------+
                   |   | Write manifest       |
                   |   | Write prompt text    |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Expand source tree    |
                   |   | breadth-first         |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Per-node LLM call    |
                   |   | propose children     |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Match child to target |
                   |   | stop or recurse       |
                   |   +----------+-----------+
                   |              |
                   |              v
                   |   +----------------------+
                   |   | Write checkpoint     |
                   |   | Write summary        |
                   |   +----------+-----------+
                   |              |
                   v              v
            +------------------------------+
            | Return cached summary         |
            +------------------------------+
```

If a run fails, the failure is recorded and the partial artifacts remain on disk.

## The Experiment Design

The method uses a full factorial design with replication.

### Factor Structure

There are four binary factors:

1. include example vs not
2. include counterexample vs not
3. child count 3 vs 5
4. maximum depth 1 vs 2

That gives:
- `2^4 = 16` conditions per pair
- `5` replications
- `80` runs per pair

The pairings are:
- `subgraph_1_to_subgraph_3`
- `subgraph_2_to_subgraph_1`
- `subgraph_2_to_subgraph_3`

So the complete Method 3 DOE is:
- `16 conditions * 5 replications = 80 runs` per pair
- `3 pairs` total in the full design

### What the bits mean

The condition bit order is:

1. example
2. counterexample
3. child count
4. depth

The code maps those bits to:
- `child_count = 3` when the third bit is `0`
- `child_count = 5` when the third bit is `1`
- `max_depth = 1` when the fourth bit is `0`
- `max_depth = 2` when the fourth bit is `1`

## Input Pairing

Method 3 does not combine two graphs directly.

Instead, it takes:
- a source label set
- a target label set

The pair name decides which graph contributes source labels and which graph contributes target labels.

For example:
- `subgraph_1_to_subgraph_3` means labels from subgraph 1 are expanded toward labels from subgraph 3
- `subgraph_2_to_subgraph_1` means labels from subgraph 2 are expanded toward labels from subgraph 1
- `subgraph_2_to_subgraph_3` means labels from subgraph 2 are expanded toward labels from subgraph 3

## Prompt Structure

Method 3 uses one prompt template for the recursive child-proposal call.

The prompt is built from these pieces:

1. A fixed system-style opener.
2. A short explanation of what a concept name should look like.
3. The source label list.
4. The task instruction.
5. An optional example block.
6. An optional counterexample block.
7. A closing constraint that forbids extra text.

### Exact Prompt Skeleton

```text
You are a helpful assistant who understands Knowledge Maps.
Your input is a set of concept names. All concept names must have a clear meaning, such that we can
interpret having 'more' or 'less' of a concept.
Your input is the following list of concept names: [source_labels]
Your task is to recommend {child_count} related concept names for each of the names in the input.
Do not suggest names that are in the input. Your output must include the list of the {child_count}
proposed names for each of the input names. Do not include any other text. Return your proposed names
in a dictionary format { 'A' : ['B' , 'C', 'D'], 'E' : ['F' , 'G' , 'H'], ..., 'U' : ['V' , 'W' , 'X'] }.
[optional example]
[optional counterexample]
Your output must only be the list of proposed concepts. Do not repeat any instructions I have given
you and do not add unnecessary words or phrases.
```

### Dynamic placeholders

The dynamic parts are:
- `[source_labels]`: the current source label list at the current recursion step
- `{child_count}`: either `3` or `5`
- `[optional example]`: present only when `include_example` is on
- `[optional counterexample]`: present only when `include_counterexample` is on

Important nuance:
- `max_depth` is **not** part of the prompt text
- `max_depth` controls the recursion logic in the tree expander

## Example And Counterexample Blocks

The example and counterexample are fixed archive-style blocks.

### `child_count = 3`

Example:

```text
Here is an example of a desired output for your task. We have the list of concepts ['capacity to hire',
'bad employees', 'good reputation']. In this example, you could recommend these 9 new concepts:
'employment potential', 'hiring capability', 'staffing ability', 'underperformers', 'inefficient staff',
'problematic workers', 'positive image', 'favorable standing', 'high regard'. Therefore, this is the
expected output: { "capacity to hire": ['employment potential', 'hiring capability', 'staffing ability'],
"bad employees": ['underperformers', 'inefficient staff', 'problematic workers'], "good reputation":
['positive image', 'favorable standing', 'high regard'] }.
```

Counterexample:

```text
Here is an example of a bad output that we do not want to see. We have the list of nodes ['capacity to hire',
'bad employees', 'good reputation']. A bad output would be: { "capacity to hire": ['moon', 'dog',
'thermodynamics'], "bad employees": ['swimming', 'red', 'happiness'], "good reputation": ['judo',
'canada', 'light'] }. Adding the proposed concepts would be incorrect since they have no relationship with
the concepts in the input.
```

### `child_count = 5`

Example:

```text
Here is an example of a desired output for your task. We have the list of concepts ['capacity to hire',
'bad employees', 'good reputation']. In this example, you could recommend these 15 new concepts:
'employment potential', 'hiring capability', 'staffing ability', 'underperformers', 'inefficient staff',
'problematic workers', 'positive image', 'favorable standing', 'high regard'. Therefore, this is the
expected output: { "capacity to hire": ["employment potential", "hiring capability", "staffing ability",
"recruitment capacity", "talent acquisition"], "bad employees": ["underperformers", "inefficient staff",
"problematic workers", "low performers", "unproductive staff"], "good reputation": ["positive image",
"favorable standing", "high regard", "excellent reputation", "commendable status"] }.
```

Counterexample:

```text
Here is an example of a bad output that we do not want to see. We have the list of nodes ['capacity to hire',
'bad employees', 'good reputation']. A bad output would be: { "capacity to hire": ['moon', 'dog',
'thermodynamics', 'country', 'pillow'], "bad employees": ['swimming', 'red', 'happiness', 'food',
'shoe'], "good reputation": ['judo', 'canada', 'light', 'phone', 'electricity'] }. Adding the proposed
concepts would be incorrect since they have no relationship with the concepts in the input.
```

## The Structured Output Contract

The child-proposal call uses structured JSON output.

The schema name is:
- `children_by_label`

The expected shape is:

```json
{
  "children_by_label": {
    "source_label": ["child_1", "child_2", "child_3"]
  }
}
```

The runtime then reads the dictionary for the current label and treats those values as the children for that node.

This means:
- the prompt asks for a dictionary of label -> children
- the structured output wrapper makes that machine-readable
- the tree-expansion logic consumes only the parsed dictionary, not free-form text

## Tree Expansion Logic

The tree is expanded breadth-first.

### How it works

1. Each source label starts as a root node.
2. The current node is expanded with a child-proposal prompt.
3. The model returns `child_count` related labels.
4. Each child is checked against the target label set.
5. If a child matches a target label, that branch stops.
6. If it does not match, the child is queued for another expansion step.
7. Expansion stops when the maximum depth is reached.

### Tree recursion state machine

```text
┌─────────────────────────┐
│ source labels loaded    │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ queue root labels       │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ pop next parent label   │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ depth limit reached?    │
└───────┬─────────┬───────┘
        │ yes      │ no
        v          v
┌──────────────┐  ┌──────────────────────┐
│ stop branch  │  │ propose children     │
└──────────────┘  └──────────┬───────────┘
                              │
                              v
                   ┌──────────────────────┐
                   │ child matches target?│
                   └───────┬───────┬──────┘
                           │ yes    │ no
                           v       v
                    ┌──────────┐  ┌───────────────────┐
                    │ record   │  │ queue child for   │
                    │ match     │  │ deeper expansion  │
                    └──────────┘  └───────────────────┘
```

### What gets recorded

For each generated child, the runtime stores:
- `root_label`
- `parent_label`
- `label`
- `depth`
- `matched_target`

This is the evidence trail used later in the summary and checkpoint artifacts.

## Probe Runtime And Artifacts

The probe runner is what actually executes one Method 3 condition.

It writes these artifacts:
- `manifest.json`
- `tree_expansion_prompt.txt`
- `execution_checkpoint.json`
- `summary.json`
- `events.jsonl`
- `run.log`

### What the manifest records

The manifest captures:
- algorithm name
- model
- provider
- source labels
- target labels
- prompt flags
- child count
- max depth

### Resume behavior

Method 3 is resumable.

On resume:
- if `summary.json` exists and the probe is marked finished, the cached summary is returned
- if the execution checkpoint exists and execution is complete, the tree expansion is not rerun
- otherwise the missing stage is rerun and the artifacts are refreshed

That means a partially completed run can continue without losing completed work.

## Output Shape

The probe returns a summary record with:
- `run_name`
- `model`
- `provider`
- `expanded_nodes`
- `matched_labels`

`matched_labels` is the subset of generated node labels that were found in the target label set.

## Offline Evaluation

The offline evaluation path computes recall from CSV files.

The evaluation logic:
- parses source edges, target edges, mother graph edges, and model results
- builds undirected connectivity
- checks whether source and target labels lie in the same connected component
- counts a true positive when the predicted structure connects a pair that is connected in the mother graph
- reports recall as `TP / actual positives`

So for Method 3:
- the online probe gives the tree trace
- the offline evaluation checks whether the generated tree actually bridges the source and target sets

## What Is Controlled By The Paper Factors

The paper-facing factors in Method 3 are:
- example on/off
- counterexample on/off
- child count 3 vs 5
- depth 1 vs 2

What is not a prompt factor:
- provider
- model label
- temperature
- resume state

Those are runtime settings, not experimental prompt dimensions.

## Paper Fidelity Notes

What is aligned with the paper:
- tree-based indirect matching
- source/target pair execution
- two child-count levels
- two depth levels
- example and counterexample toggles
- recall-focused evaluation

What is intentionally implementation-specific:
- the structured JSON wrapper around the child dictionary
- the exact BFS queue mechanics
- the cached checkpoint and resume model
- the offline CSV evaluation pipeline

## Short Verification Checklist

If you want to verify Method 3 quickly, check these points:

1. The pair name resolves to the right source and target label sets.
2. The prompt contains the fixed opener, the concept explanation, the current source labels, and the child-count instruction.
3. The example/counterexample blocks appear only when the relevant bits are enabled.
4. The structured output schema name is `children_by_label`.
5. The tree expansion stops descending from matched target labels.
6. Depth is enforced by recursion, not by the prompt text.
7. The output artifacts include the manifest, prompt, checkpoint, summary, and event log.
8. Resume reuses completed stages instead of rerunning them.
9. The offline evaluation reads the CSV output and computes recall from connectivity.

If those are all true, the Method 3 implementation is behaving as designed.
