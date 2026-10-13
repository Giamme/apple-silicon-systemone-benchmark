#!/usr/bin/env python3
"""Capture original/MLX D3 in separate processes, then compare without loading models.

Reference captures require the isolated Torch CPU environment and a complete,
pinned original snapshot. MLX captures use the managed d3-clef environment.
No downloads, automatic environment installs, or GPU fallback occur here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import load_jsonl
from runtime_backends.d3_clef import load_backend
from runtime_backends.d3_parity import candidate_prompts, compare_captures, reference_prompts
from runtime_backends.protocol import serialize_state
from runtime_catalog import get_model


def capture(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_jsonl(args.dataset)
    spec = get_model(args.model)
    if args.engine == "reference":
        from transformers import AutoModel

        manifest = json.loads((args.snapshot / "MODEL_MANIFEST.json").read_text())
        if manifest.get("repo_id") != spec["source_repo"] or args.snapshot.name != spec["source_revision"]:
            raise ValueError("reference snapshot identity must match the pinned source recipe")
        model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True,
                                         local_files_only=True, device="cpu", batch_size=1, verify="full")
    else:
        model = load_backend(spec, args.snapshot)
    rows = []
    for case in cases:
        state, questions = case["state"], case["questions"]
        if args.engine == "reference":
            prompts = reference_prompts(model, state, questions)
            prediction = model.system_one(state=serialize_state(state), questions=questions)
        else:
            prompts = candidate_prompts(model, state, questions)
            prediction = model.predict(state, questions)
        rows.append({"case_id": case["id"], "questions": questions,
                     "prompts": prompts, "prediction": prediction})
        print(f"{args.model} {args.engine}: {len(rows)}/{len(cases)}", file=sys.stderr, flush=True)
    return {"model_id": args.model, "engine": args.engine, "source_repo": spec["source_repo"],
            "source_revision": spec["source_revision"], "artifact_repo": spec["artifact_repo"],
            "artifact_revision": spec["artifact_revision"],
            "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("--engine", choices=("reference", "mlx"), required=True)
    collect.add_argument("--model", choices=("d3-edge", "d3-lite", "d3-mini"), required=True)
    collect.add_argument("--snapshot", type=Path, required=True)
    collect.add_argument("--dataset", type=Path, default=ROOT / "data" / "smoke.jsonl")
    collect.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            result = compare_captures(json.loads(args.reference.read_text()),
                                      json.loads(args.candidate.read_text()))
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        print(f"D3 parity failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
