# Contributing

Issues and pull requests are welcome for model adapters, benchmark correctness, accessibility, and documentation.

1. Keep model definitions in TOML and adapter code small. Include an example response and explain how its answers map to the shared decision types.
2. Run `./benchmark.sh validate --dataset data/smoke.jsonl`, `./benchmark.sh models list`, `bash -n benchmark.sh scripts/setup_model.sh`, and `uv run --with pytest --no-project python -m pytest`.
3. Test new integration behavior with a local fake service; do not require downloaded weights in the automated test suite.
4. Never commit downloaded environments, model weights, private datasets, tokens, raw request logs, or generated `results/`. Review any report before sharing it.
5. In benchmark claims, name the dataset, selected models, runtime revisions, Apple Silicon hardware, methodology, and failures. A small smoke dataset is only a workflow check.

By contributing, you agree that your contribution is licensed under this repository's Apache-2.0 license. Check upstream code and weight licenses before proposing a built-in model.
