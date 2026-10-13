# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow the format `{major}.{minor}` ([...] parts appear only when non-zero).

## [0.2] - 2026-10-13

Benchmark eight additional System One models through the extended profile, with pinned native runtimes and provenance recorded in every report format.

### Added

- The extended profile adds D3 Edge, Lite, Mini and Flash, Clef Flash, Liquid D1 3B and D1 Omni 600M, and JEV 9B to the original four models. The full, light, and default selections remain unchanged; all candidates use the same labeled decision dataset.
- New native model runs use pinned local weight snapshots, locked Python environments, and an explicit MLX free-buffer cache budget. JSON, CSV, Markdown, and offline HTML reports distinguish configured model and runtime pins from observed backend metadata, including load and inference failures.
