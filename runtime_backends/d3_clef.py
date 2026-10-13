"""Managed D3/Clef family entrypoint; ML imports occur only during model loading."""
from __future__ import annotations

import hashlib
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

from runtime_backends.d3_artifacts import CLEF_RUNTIME_HASH, verify_snapshot
from runtime_backends.protocol import serialize_state, validate_prediction


def load_clef_module(path: Path) -> Any:
    name = "_benchmark_pinned_clef_mlx"
    source_path = path / "clef_mlx.py"
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != CLEF_RUNTIME_HASH:
        raise ValueError("Clef decision runtime source checksum mismatch")
    module = ModuleType(name)
    module.__file__ = str(source_path)
    sys.modules[name] = module
    # The artifact contains a .pyc; execute only the checked source bytes.
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


class ClefBackend:
    def __init__(self, spec: dict[str, Any], path: Path) -> None:
        import mlx.core as mx

        self.module = load_clef_module(path)
        self.model = self.module.load(path, head_dtype=mx.bfloat16, backend="vlm")
        self.model_id = spec["id"]
        self.observed_metadata = {
            "decision_mode": "native_readout_no_generation", "weight_checksums_verified": True,
            "artifact_revision": spec["artifact_revision"], "runtime_sha256": CLEF_RUNTIME_HASH,
            "head_dtype": "bfloat16", "native_probability_decimals": 4,
            "choice_order": "native_lexicographic",
            "max_input_tokens": 16384, "truncation": False, "score_mapping": "native",
        }

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        result = self.model.systemone({"model": self.model_id, "state": serialize_state(state),
                                       "questions": questions}, max_length=16384, truncate=False)
        return validate_prediction(result, questions, probability_decimals=4)


def load_backend(spec: dict[str, Any], snapshot_path: Path) -> Any:
    path = Path(snapshot_path)
    artifact = verify_snapshot(spec, path)
    if spec["family"] == "d3":
        from runtime_backends.d3_mlx import D3Backend

        return D3Backend(spec, path, artifact)
    if spec["family"] == "clef":
        return ClefBackend(spec, path)
    raise ValueError("this backend only supports the d3 and clef families")
