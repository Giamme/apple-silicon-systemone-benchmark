# Contributing

Issues and pull requests are welcome for model adapters, benchmark correctness, accessibility, and documentation.

1. Keep model definitions in TOML and adapter code small. Include an example response and explain how its answers map to the shared decision types.
2. Run `./benchmark.sh validate --dataset data/smoke.jsonl`, `./benchmark.sh models list`, `bash -n benchmark.sh scripts/setup_model.sh`, and `uv run --with pytest --no-project python -m pytest`.
3. Test new integration behavior with a local fake service; do not require downloaded weights in the automated test suite.
4. Never commit downloaded environments, model weights, private datasets, tokens, raw request logs, or generated `results/`. Review any report before sharing it.
5. In benchmark claims, name the dataset, selected models, runtime revisions, Apple Silicon hardware, methodology, and failures. A small smoke dataset is only a workflow check.

By contributing, you agree that your contribution is licensed under this repository's Apache-2.0 license. Check upstream code and weight licenses before proposing a built-in model.

## Changelog and versions

Add a polished fragment in `.changelog/unreleased/` for each user-visible change using the changelogger skill's `new` command, and check coverage with `status` and `check` before committing. Internal-only changes can be marked as skipped. Describe the behavior users will notice; unreleased entries do not claim a published release.

The repository version is in `VERSION` and uses **major.minor** only. Ordinary releases increment minor; major is available for breaking changes. Releases are manual. Patch, hotfix, prerelease, and build-number channels are outside this project's release policy. Tags use `v{version}`; changelog headings omit `v`. The isolated runtime environment package versions have their own metadata and do not follow `VERSION`.

Changelogger will create `CHANGELOG.md` and the structured release history when an authorized release is compiled. Setup and unreleased fragments alone do not bump versions, create release notes, or tag a release.
