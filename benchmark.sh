#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TEXTUAL_SPEC="textual==8.2.8"

# On a terminal, no arguments (or `tui`) opens the interactive dashboard; it is the only
# part that needs a dependency, which uv provides without touching the system Python.
if [[ "${1:-}" == "tui" ]] || [[ "$#" -eq 0 && -t 0 && -t 1 ]]; then
  [[ "${1:-}" == "tui" ]] && shift
  if ! command -v uv >/dev/null 2>&1; then
    echo "The interactive dashboard needs uv (https://docs.astral.sh/uv/)." >&2
    echo "Install it, or run './benchmark.sh all' for the plain command-line benchmark." >&2
    exit 2
  fi
  exec uv run --no-project --python 3.12 --with "$TEXTUAL_SPEC" python "$ROOT/tui.py" "$@"
fi

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
