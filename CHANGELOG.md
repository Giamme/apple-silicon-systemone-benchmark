# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow the format `{major}.{minor}` ([...] parts appear only when non-zero).

## [0.4] - 2026-10-14

A stuck model no longer costs you the whole run: skip, abort, exclude or resume without losing finished results, follow everything in a new terminal dashboard, and Von no longer hangs on its first request.

### Added

- A model with a problem no longer costs you the whole run. Press Ctrl+C once to skip the current model, or twice to abort and still get a report for everything that finished. Reports are saved after every model, `--exclude` leaves models out of a profile, and `--resume results/<run>` reruns only the failed, skipped and unfinished models of an earlier run.
- Running `./benchmark.sh` in a terminal opens an interactive dashboard. Pick the models and options, then follow the full log next to tabs for live progress, latency and quality charts, and results. Keys skip a stuck model, remove queued ones, abort, retry unfinished models, and open the report. `./benchmark.sh all` keeps the plain command-line run.

### Changed

- `all` now sets up each model just before benchmarking it, and a setup failure is reported for that model instead of stopping the run. The command exits 1 when any model failed or was skipped and 130 when the run was aborted.

### Fixed

- Von no longer hangs on its first request. Its setup now downloads the model weights, and benchmark runs load them with Hugging Face networking turned off, so a slow or offline network can no longer stall a full run.

## [0.3] - 2026-10-13

The full profile and default selection now include all 12 built-in System One models.

### Fixed

- The full profile and the default setup, run, and all commands include all 12 built-in models. Extended selects the same catalog; light remains Laya and Kev, and --models still selects an explicit subset.

## [0.2] - 2026-10-13

Benchmark eight additional System One models through the extended profile, with pinned native runtimes and provenance recorded in every report format.

### Added

- The extended profile adds D3 Edge, Lite, Mini and Flash, Clef Flash, Liquid D1 3B and D1 Omni 600M, and JEV 9B to the original four models. The full, light, and default selections remain unchanged; all candidates use the same labeled decision dataset.
- New native model runs use pinned local weight snapshots, locked Python environments, and an explicit MLX free-buffer cache budget. JSON, CSV, Markdown, and offline HTML reports distinguish configured model and runtime pins from observed backend metadata, including load and inference failures.
