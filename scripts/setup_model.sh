#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/.bench"
MODEL="$1"

command -v uv >/dev/null 2>&1 || { echo "uv is required; install it first." >&2; exit 1; }
mkdir -p "$WORK/envs" "$WORK/repos"

case "$MODEL" in
  laya)
    if [[ ! -x "$WORK/envs/laya/bin/python" ]]; then
      uv venv --python 3.12 "$WORK/envs/laya"
    fi
    uv pip install --python "$WORK/envs/laya/bin/python" --upgrade "laya[serve]"
    ;;
  von)
    if [[ ! -x "$WORK/envs/von/bin/python" ]]; then
      uv venv --python 3.12 "$WORK/envs/von"
    fi
    uv pip install --python "$WORK/envs/von/bin/python" --upgrade von-sdk
    ;;
  kev)
    command -v git >/dev/null 2>&1 || { echo "git is required." >&2; exit 1; }
    if [[ ! -d "$WORK/repos/kev/.git" ]]; then
      git clone --depth 1 https://github.com/jaredpalmer/kev.git "$WORK/repos/kev"
    else
      git -C "$WORK/repos/kev" pull --ff-only
    fi
    (cd "$WORK/repos/kev" && uv sync --extra serve)
    ;;
  rizzo)
    command -v git >/dev/null 2>&1 || { echo "git is required." >&2; exit 1; }
    if [[ ! -d "$WORK/repos/rizzo-flow/.git" ]]; then
      git clone --depth 1 https://github.com/Rizzo-AI-Academy/rizzo-flow.git "$WORK/repos/rizzo-flow"
    else
      git -C "$WORK/repos/rizzo-flow" pull --ff-only
    fi
    (cd "$WORK/repos/rizzo-flow" && uv sync --locked && uv run --no-sync rizzo download --size 4b --quant q4_k_m)
    ;;
  *)
    echo "Unknown built-in model: $MODEL" >&2
    exit 2
    ;;
esac
