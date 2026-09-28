#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ "$#" -eq 0 ]]; then
  set -- all
fi

if command -v python3 >/dev/null 2>&1 && python3 -c 'import tomllib' >/dev/null 2>&1; then
  exec python3 "$ROOT/benchmark.py" "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python 3.12 python "$ROOT/benchmark.py" "$@"
fi
echo "Python 3.11+ or uv is required." >&2
exit 2
