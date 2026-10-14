# CI artifact subset

The GitHub Actions workflow restores five pinned files from the public
Hugging Face bucket before running `make ci`. These files form a minimal
integration-test subset. They are not the full research corpus and cannot
support a complete rerun or audit of the experiments.

The subset contains four canonical input files and one authentic
`state.json` file under the canonical open-weights `runs` path. The total
download is 42,943 bytes. The state file makes the `runs` directory exist for
two result-layout tests. Those tests check directory and layout entries, not
that the full run bundle is present. The bootstrap does not weaken those tests
or any other test and verification gate.

The bucket has no immutable global revision. The manifest at
`.github/ci-artifact-pins.json` pins each file by its bucket path, size, Xet
hash, and SHA-256. The restore script checks the locked `huggingface-hub`
CLI version (1.8.0), then uses `hf buckets info`, `hf buckets list`, and
`hf buckets cp` without a token. It verifies
the bucket is public, compares the listed Xet hash and size before copying,
then checks the downloaded size and SHA-256. It stages all missing files and
moves them into place only after every downloaded file passes verification.
If a remote pin changes, an artifact is missing, or local bytes differ, the
restore step fails before `make ci` starts.

The script is idempotent. It accepts an existing file only when its pinned
size and SHA-256 match. It refuses to overwrite a file with different bytes.
It does not enumerate or sync the bucket's full `inputs/`, `results/`, or
`analysis_artifacts/` trees.

The repository still requires its full external corpus for research
reproduction and full-data review. CI's five-file subset does not satisfy
those uses.
