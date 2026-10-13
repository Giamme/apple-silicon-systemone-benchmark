"""Immutable D3/Clef artifact checks, without importing ML packages.

Hashes were read from the public Hugging Face LFS/git metadata for the approved
revisions. Git blob hashes cover small files; SHA-256 covers weight payloads.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

TOKENIZER_HASH = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
FLASH_READOUT_HASH = "8bbc9a9df4e1b7df85a80a1091977092b938d8d2d2869dfaeb8fd36063cf322f"
CLEF_RUNTIME_HASH = "852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5"

ARTIFACTS = {
    "d3-edge": {
        "repo": "ahmedandaloes/d3-edge-mlx", "revision": "33e8fdb0adb076d7af3d5d1a0049b60cdc1d7939",
        "hidden": 1024, "layers": 12,
        "files": {
            "weights.safetensors": "8063c1ab7b265d4f7e42f90f54fcd77fe7173d79ea0de32048fa07a60d3e6ad6",
            "readout.safetensors": "de33d5dc795bbd77168e40bfadc340cca9b9f163b88744c10e1a788f4e166d3d",
            "config.json": "b0d720318d668d113913b65b4e942c6deaa84065",
            "decision_config.json": "5bf6f72adb66002996fcf44a04947687eb2c2034",
            "chat_template.jinja": "0ef09f214eaa6d9bca297988afc1454b5827b2c7",
            "tokenizer_config.json": "fae3ce993e07c092ad024dde45e592379fde91bb",
            "tokenizer.json": TOKENIZER_HASH,
        },
    },
    "d3-lite": {
        "repo": "ahmedandaloes/d3-lite-mlx", "revision": "235b1fb6ed8d20ab911e10801a4f4cec9cadc24a",
        "hidden": 1024, "layers": 24,
        "files": {
            "weights.safetensors": "0a6db81616ae1c9de017402d83ea70edcc98b220103ba45ab2a95539583a5b67",
            "readout.safetensors": "fe11cfac5c79b6a83e35b707581b1260214279bb22b003c215c62719186ef629",
            "config.json": "fdc4f45ab464793cdf8121af141f782c8db44482",
            "decision_config.json": "5bf6f72adb66002996fcf44a04947687eb2c2034",
            "chat_template.jinja": "0ef09f214eaa6d9bca297988afc1454b5827b2c7",
            "tokenizer_config.json": "fae3ce993e07c092ad024dde45e592379fde91bb",
            "tokenizer.json": TOKENIZER_HASH,
        },
    },
    "d3-mini": {
        "repo": "ahmedandaloes/d3-mini-mlx", "revision": "c645173f5729051d473916283f9e37208b5113bb",
        "hidden": 2560, "layers": 32,
        "files": {
            "weights.safetensors": "02cb76a7557b95a595f9feb4121b7ceda34b57a1158f5d46e38cc09c21df48a7",
            "readout.safetensors": "d570665c1da72cb9fca7ad7b23a25720eb691362694dd69d4c80c17d7656ed55",
            "config.json": "ca8495ef6bb09775fc4a9c5ee615d930edc9fd65",
            "decision_config.json": "c3d9b6b56a91ccf48bf3178b7479f0cf4accb5d2",
            "chat_template.jinja": "a585dec894e63da457d9440ec6aa7caa16d20860",
            "tokenizer_config.json": "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
            "tokenizer.json": TOKENIZER_HASH,
        },
    },
    "d3-flash": {
        "repo": "thoddnn/d3-flash-q4", "revision": "5b9c0de2e6640380b50491ee2f9eef8eece100e3",
        "hidden": 4096, "layers": 32,
        "files": {
            "weights.safetensors": "97ecff62320b412288dd836c5d9a601dfdb0f5c2ea7cd51a2871e7a9ef84af0c",
            "config.json": "b013d7e417d78e76451ad2ac7a4123b3231867fd",
            "decision_config.json": "b6e99b9556f2953c4fd748737e1d40c402189f46",
            "chat_template.jinja": "a585dec894e63da457d9440ec6aa7caa16d20860",
            "tokenizer_config.json": "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
            "tokenizer.json": TOKENIZER_HASH,
            "fenn_d3.json": "d2c302a2e34e82fe59bccd9e4399bd220720a854",
            "verification/readout.safetensors": FLASH_READOUT_HASH,
        },
    },
    "clef-flash": {
        "repo": "mlx-community/clef-flash-4bit", "revision": "b5adbe2c3a95200d4bf04a0727c8ec457157d5e1",
        "files": {
            "clef_mlx.py": CLEF_RUNTIME_HASH,
            "config.json": "8118c27cba9d5b2eb8e9d1dbcbe581cd38cfc7af",
            "joint_head_config.json": "52467b6c7f7d1a7e628b518eb7a6cd7e9bdb9a57",
            "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
            "model-00001-of-00002.safetensors": "39894f085d844004c8e5c2394e45b5ab66bce4637ecdf23d92bf7c85cb66238f",
            "model-00002-of-00002.safetensors": "40258d3d4f7225a0acbb45f4d34693465b98c8e2c0a4b0a512e536750379fcf3",
            "model.safetensors.index.json": "b1846cf03478ee8111d3a7cd5dd2bd685237cce3",
            "processor_config.json": "33818c7f9e991ad735fd240209f4fa73e6c28c50",
            "chat_template.jinja": "a585dec894e63da457d9440ec6aa7caa16d20860",
            "tokenizer_config.json": "848c7c7ff98e31b321d5c2279befd857f269e861",
            "tokenizer.json": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
        },
    },
}


def verify_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"artifact checksum mismatch: {path.name}")


def verify_snapshot(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    artifact = ARTIFACTS.get(spec.get("id"))
    if artifact is None or (spec.get("artifact_repo"), spec.get("artifact_revision")) != (
        artifact["repo"], artifact["revision"]
    ):
        raise ValueError("unsupported or unpinned D3/Clef artifact")
    for name, digest in artifact["files"].items():
        verify_file(path / name, digest)
    return artifact


def read_header(path: Path) -> dict[str, Any]:
    """Read only bounded safetensors metadata, before ML packages see the file."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("missing safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 2 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    header.pop("__metadata__", None)
    return header


def check_readout_header(header: dict[str, Any], key: str, hidden: int) -> None:
    entry = header.get(key, {})
    if entry.get("dtype") != "F32" or entry.get("shape") != [255, hidden]:
        raise ValueError("decision readout must be FP32 with shape [255, hidden_size]")


def load_configs(path: Path, artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((path / "config.json").read_text())
    decision = json.loads((path / "decision_config.json").read_text())
    required = {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3",
                "attention_mode": "noncausal_full_attention", "pooling": "last",
                "readout_dtype": "float32", "temperature": 1.0}
    if any(decision.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported D3 decision configuration")
    text = config.get("text_config", {})
    layers = text.get("layer_types", [])
    if (config.get("model_type") != "qwen3_5" or text.get("hidden_size") != artifact["hidden"]
            or len(layers) != artifact["layers"]
            or set(layers) != {"linear_attention", "full_attention"}):
        raise ValueError("D3 backbone configuration does not match pinned artifact")
    return config, decision
