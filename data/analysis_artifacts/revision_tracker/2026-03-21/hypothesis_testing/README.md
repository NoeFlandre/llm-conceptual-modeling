# Hypothesis-Testing Audit Artifacts

These files support the formal statistical-testing findings recorded in `paper/revision-tracker.md`.

## Source Data

The source data comes from the imported evaluated result files under `data/results/`.

## Method

The command surface for this slice is `lcm analyze hypothesis`. It performs paired two-level tests with explicit pairing columns and applies Benjamini-Hochberg correction within each generated output file.

Benjamini-Hochberg was chosen because each output file contains a family of related tests across metrics or source files. The intent is to control the false discovery rate without adopting a more conservative familywise correction that would be poorly matched to this exploratory reviewer-facing analysis layer.

The pairing keys used for the audited outputs are:

- ALGO1 `Explanation`
  `Repetition`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO2 `Convergence`
  `Repetition`, `Explanation`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`
- ALGO2 `Explanation`
  `Repetition`, `Example`, `Counterexample`, `Array/List(1/-1)`, `Tag/Adjacency(1/-1)`, `Convergence`
- ALGO3 `Depth`
  `Repetition`, `Example`, `Counter-Example`, `Number of Words`, `Source Subgraph Name`, `Target Subgraph Name`
- ALGO3 `Number of Words`
  `Repetition`, `Example`, `Counter-Example`, `Depth`, `Source Subgraph Name`, `Target Subgraph Name`

## Files

- `algo1_explanation_hypothesis.csv`
  Paired tests for the ALGO1 `Explanation` factor.
- `algo1_explanation_hypothesis_significance_summary.csv`
  Count summary of ALGO1 `Explanation` tests by metric, direction, and adjusted-significance status.
- `algo2_convergence_hypothesis.csv`
  Paired tests for the ALGO2 `Convergence` factor.
- `algo2_convergence_hypothesis_significance_summary.csv`
  Count summary of ALGO2 `Convergence` tests by metric, direction, and adjusted-significance status.
- `algo2_explanation_hypothesis.csv`
  Paired tests for the ALGO2 `Explanation` factor.
- `algo2_explanation_hypothesis_significance_summary.csv`
  Count summary of ALGO2 `Explanation` tests by metric, direction, and adjusted-significance status.
- `algo3_depth_hypothesis.csv`
  Paired tests for the ALGO3 `Depth` factor.
- `algo3_depth_hypothesis_significance_summary.csv`
  Count summary of ALGO3 `Depth` tests by direction and adjusted-significance status.
- `algo3_number_of_words_hypothesis.csv`
  Paired tests for the ALGO3 `Number of Words` factor.
- `algo3_number_of_words_hypothesis_significance_summary.csv`
  Count summary of ALGO3 `Number of Words` tests by direction and adjusted-significance status.

## Interpretation Notes

These artifacts provide p-values and adjusted p-values for narrow paired factor comparisons. They are more specific than the earlier descriptive summaries, but they are not a full multi-factor ANOVA model.
