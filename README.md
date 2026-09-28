# Apple Silicon System One Benchmark

An open-source benchmark for local decision models and compatible services on native Apple Silicon macOS. It compares the same labeled cases through a common decision interface, runs managed models one at a time, and produces an offline HTML dashboard with CSV, JSON, Markdown, and request-level records.

The built-in candidates are Laya, Von, Kev 0.8B, and Rizzo Flow 4B Q4_K_M. A plain run uses the **full** profile (all four). The **light** profile runs Laya and Kev. Profiles choose models manually; the tool does not infer quality or compatibility from a chip generation. On machines with less unified memory, start with `light` if `full` cannot load every model.

## Requirements

- macOS running natively on Apple Silicon (`arm64`), not Rosetta
- `uv` and `git` for built-in model setup
- Python 3.11+ (or `uv`, which the launcher can use to obtain Python 3.12)
- Several gigabytes of disk space for downloaded environments and weights

The benchmark harness itself uses only the Python standard library. Model packages and weights download from their own upstream projects and keep their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). This project is independent of those projects and of TypeSafe Jev.

## Start a benchmark

```bash
./benchmark.sh all
```

This validates the included synthetic dataset, sets up the full profile, starts each model in turn, prints prefixed server logs and request progress, and writes a new run under `results/`. Open `summary.html` in that run directory.

For a smaller comparison or a specific set:

```bash
./benchmark.sh all --profile light
./benchmark.sh all --models laya kev
```

Setup and measurement can be separated:

```bash
./benchmark.sh setup --profile full
./benchmark.sh run --profile full
```

The full profile is the default for `setup`, `run`, and `all`. `--models` overrides the profile. `./benchmark.sh models list` shows installed definitions and profile membership. Setup updates upstream packages or repositories; every run records the versions or commits it could read.

## Test a new dataset

Use one JSON object per line with `id`, `state`, `questions`, and `expected`. `state` can be a nonempty string or object. Every question needs a labeled expected answer. Supported question types are `choice`, `noul` (boolean), and `score` (zero-based index into criteria). The [smoke dataset](data/smoke.jsonl) is a full example; a minimal case looks like:

```json
{"id":"ticket-001","state":"The invoice was paid twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","support":"Product issues"}}},"expected":{"team":"billing"}}
```

Validate locally before any model setup or dataset transmission:

```bash
./benchmark.sh validate --dataset /path/to/labeled.jsonl
./benchmark.sh all --dataset /path/to/labeled.jsonl --profile light
```

Validation reports the file and line of a malformed case, duplicate ID, unsupported question type, or invalid label. An `all` run validates before downloading models. Private datasets should stay outside this repository. `results/` and `models/local/` are Git-ignored.

The included 24-case dataset checks the workflow. Its accuracy ranking is **not** evidence for a production choice. Use representative labeled traffic, including ambiguous cases, long inputs, negation, missing evidence, and your real decision labels. Reports show the dataset filename, case count, and SHA-256, never the dataset's contents or absolute path.

## Read the results

Each run includes:

```text
results/<timestamp>/
├── summary.html          # standalone, responsive comparison dashboard
├── summary.csv           # spreadsheet-friendly measurements
├── summary.json          # structured run and model metadata
├── summary.md            # text report
├── system.json           # non-identifying host details
├── <model>.server.log    # managed server output
├── <model>.requests.jsonl
└── <model>.decisions.jsonl
```

The terminal displays setup/startup, cold first request, warmup, measurement counts and rolling p50, cached repeats, and model-prefixed server output. Server logs remain in the run directory; setup logs are under `.bench/`. Existing services have request progress but no server output, startup time, or comparable process-memory measurement. A failed model still appears in the report, and the command exits nonzero.

Warm p50/p95/p99 are end-to-end HTTP latency after warmup. Each primary request carries a unique, decision-irrelevant ID to reduce repeated-prefix cache effects. Cached latency repeats one identical input separately. Accuracy, Brier score, ECE, and error-budget coverage use the labeled answers. Process-tree RSS is a host-memory proxy, not a Metal allocation measurement.

Tune sample size or slow cold loading with `--passes`, `--warmup`, `--cached-repeats`, `--startup-timeout`, and `--first-request-timeout`. For example:

```bash
./benchmark.sh run --dataset /path/to/labeled.jsonl --passes 5 --warmup 10 --first-request-timeout 2400
```

## Add a model or service

Model definitions live in `models/*.toml`. Put private local definitions in `models/local/`, or pass `--model-dir /path/to/definitions`. See [MODEL_PLUGINS.md](MODEL_PLUGINS.md) for the manifest and adapter contract.

Three use cases share the same definition:

1. An already-running service declares an `endpoint` and no `[launch]`.
2. A locally installed program adds `[launch]` with its command.
3. An automatically set up model adds `[setup]` as well.

The default adapter uses the `/v1/systemone` wire format. A small Python adapter can translate another HTTP API into the same normalized `answers` structure. See `examples/`. Non-loopback endpoints need `--allow-remote`, because they receive dataset state. Store authorization headers in environment variables, not manifests.

## Open source and contributing

This harness and its synthetic examples are Apache-2.0 licensed. Downloaded code and weights are not vendored. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). When sharing results, disclose the dataset, hardware, runtime revisions, and failures, and avoid publishing private request or server logs.
