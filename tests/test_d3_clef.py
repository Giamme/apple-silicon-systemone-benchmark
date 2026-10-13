from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_backends import d3_format
from runtime_backends import d3_clef
from runtime_backends.d3_artifacts import check_readout_header, read_header, verify_file
from runtime_backends.d3_clef import ClefBackend
from runtime_backends.d3_contract import normalize_question, prepare_question, product_answer
from runtime_backends.d3_mlx import D3Backend, configure_layers, offset_source_norms, quantized_modules, readout_probabilities


class Tokenizer:
    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "\n".join(f"<{item['role']}>\n{item['content']}" for item in messages) + "\n<assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(text.encode())


def test_score_uses_description_only_prompt_and_zero_based_native_answers() -> None:
    question = normalize_question({"type": "score", "instructions": "Urgency?", "criteria": ["Low", "Soon", "Now"]})
    text = d3_format.user_prompt("Ticket", question.rendered, ["A", "B", "C"])
    assert "A: Low\nB: Soon\nC: Now" in text
    assert "0: Low" not in text
    answer = product_answer(question, [0.1, 0.2, 0.7])
    assert answer["score"] == pytest.approx(1.6)
    assert answer["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert answer["legend"] == {"0": "Low", "1": "Soon", "2": "Now"}


def test_choice_order_is_preserved_and_noul_true_is_second_option() -> None:
    choice = normalize_question({"type": "choice", "criteria": {"z": None, "a": "Description"}})
    assert d3_format.options(choice.rendered) == (["z", "a"], ["z", "a: Description"])
    assert product_answer(choice, [0.7, 0.3])["choice"] == "z"
    boolean = normalize_question({"type": "noul"})
    assert d3_format.options(boolean.rendered)[1] == ["No / false", "Yes / true"]
    assert product_answer(boolean, [0.2, 0.8]) == {"type": "noul", "noul": 0.8}


@pytest.mark.parametrize("question", [
    {"type": "score", "criteria": ["same", "same"]},
    {"type": "score", "criteria": ["only one"]},
    {"type": "score", "criteria": [str(i) for i in range(11)]},
    {"type": "choice", "criteria": {}},
    {"type": "noul", "criteria": {"maybe": "Unknown"}},
    {"type": "choice", "criteria": {"a": float("nan")}},
])
def test_malformed_questions_are_rejected(question: dict) -> None:
    with pytest.raises(ValueError):
        normalize_question(question)


@pytest.mark.parametrize("values", [[float("nan"), 0.5], [float("inf"), 0], [-0.1, 1.1], [0.4, 0.4]])
def test_nonfinite_or_unnormalized_readout_is_rejected(values: list[float]) -> None:
    question = normalize_question({"type": "noul"})
    with pytest.raises(ValueError):
        product_answer(question, values)


def test_context_limit_rejects_without_truncating() -> None:
    with pytest.raises(ValueError, match="no truncation"):
        prepare_question(Tokenizer(), "A long state", {"type": "noul"}, ["A", "B"], 10)


class Vector:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.dtype = "bfloat16"

    def astype(self, dtype: str) -> Vector:
        assert dtype == "float32"
        self.dtype = dtype
        return self

    def __getitem__(self, index: slice) -> Vector:
        return Vector(self.values[index])

    def __truediv__(self, divisor: float) -> Vector:
        return Vector([value / divisor for value in self.values])

    def __add__(self, value: float) -> Vector:
        assert self.dtype == "float32", "zero-centered offset must be added after FP32 casting"
        return Vector([item + value for item in self.values])

    def tolist(self) -> list[float]:
        return self.values


class Readout:
    def __matmul__(self, hidden: Vector) -> Vector:
        assert hidden.values == [1.0]
        # A/B are readout rows 0/1. Vocabulary IDs 32/33 would pick wrong rows.
        # Other rows overwhelm softmax if they are not masked BEFORE softmax.
        return Vector([2.0, -1.0] + [1000.0] * 253)


class MX:
    float32 = "float32"

    @staticmethod
    def softmax(values: Vector, axis: int) -> Vector:
        assert axis == -1 and len(values.values) == 2
        offset = max(values.values)
        exps = [math.exp(value - offset) for value in values.values]
        return Vector([value / sum(exps) for value in exps])

    @staticmethod
    def eval(_value: Vector) -> None:
        pass


def test_readout_uses_first_option_rows_before_softmax() -> None:
    values = readout_probabilities(MX, Readout(), Vector([1.0]), 2, 2.0)
    assert values[0] == pytest.approx(1 / (1 + math.exp(-1.5)))
    assert sum(values) == pytest.approx(1.0)


def test_bf16_source_norms_receive_fp32_offset_except_gated_and_vision_norms() -> None:
    keys = ["language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.post_attention_layernorm.weight",
            "language_model.model.layers.2.self_attn.q_norm.weight",
            "language_model.model.layers.2.self_attn.k_norm.weight",
            "language_model.model.norm.weight"]
    weights = {key: Vector([-0.25, 0.5]) for key in keys}
    gated = "language_model.model.layers.0.linear_attn.norm.weight"
    vision = "vision_tower.blocks.0.norm1.weight"
    weights[gated] = weights[vision] = Vector([1.0, 2.0])
    offset_source_norms(weights, MX)
    assert all(weights[key].values == [0.75, 1.5] for key in keys)
    assert weights[gated].values == weights[vision].values == [1.0, 2.0]


def test_explicit_layer_pattern_replaces_pruned_edge_layers_and_removes_lm_head() -> None:
    layers = [SimpleNamespace(linear_attn=object()) for _ in range(4)]
    lm = SimpleNamespace(model=SimpleNamespace(layers=layers), lm_head=object())
    net = SimpleNamespace(language_model=lm)
    config = SimpleNamespace(text_config=SimpleNamespace(layer_types=["linear_attention", "full_attention", "full_attention", "linear_attention"]))
    language = SimpleNamespace(Qwen3_5Attention=lambda _cfg: object(), Qwen3_5GatedDeltaNet=lambda _cfg: object())
    configure_layers(net, config, language, config.text_config.layer_types)
    assert [layer.is_linear for layer in layers] == [True, False, False, True]
    assert not hasattr(lm, "lm_head")
    assert hasattr(layers[1], "self_attn") and not hasattr(layers[1], "linear_attn")


def test_quantization_only_accepts_complete_group64_language_modules() -> None:
    name = "language_model.model.embed_tokens"
    header = {name + ".weight": {"dtype": "U32", "shape": [20, 512]},
              name + ".scales": {"dtype": "BF16", "shape": [20, 64]},
              name + ".biases": {"dtype": "BF16", "shape": [20, 64]}}
    assert quantized_modules(header) == {name}
    with pytest.raises(ValueError, match="companion"):
        quantized_modules({key: value for key, value in header.items() if not key.endswith(".biases")})
    with pytest.raises(ValueError, match="restricted"):
        quantized_modules({key.replace("language_model.model", "vision_tower"): value for key, value in header.items()})


def test_hashes_and_readout_headers_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    raw = json.dumps({"weight": {"dtype": "F32", "shape": [255, 1024]}}).encode()
    payload = struct.pack("<Q", len(raw)) + raw
    path.write_bytes(payload)
    verify_file(path, hashlib.sha256(payload).hexdigest())
    verify_file(path, hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest())
    check_readout_header(read_header(path), "weight", 1024)
    with pytest.raises(ValueError, match="checksum"):
        verify_file(path, "0" * 64)
    with pytest.raises(ValueError, match="readout"):
        check_readout_header(read_header(path), "weight", 2560)
    path.write_bytes(struct.pack("<Q", 3 * 1024 * 1024))
    with pytest.raises(ValueError, match="length"):
        read_header(path)


def test_d3_prediction_keeps_generic_questions_state_nonce_and_usage() -> None:
    backend = object.__new__(D3Backend)
    backend.tokenizer, backend.codes = Tokenizer(), ["A", "B", "C"]
    backend.max_length, backend.model_id = 4096, "d3-test"
    observed: list[str] = []

    def predict(ids: list[int], count: int) -> list[float]:
        observed.append(bytes(ids).decode())
        return [0.2, 0.8] if count == 2 else [0.1, 0.2, 0.7]

    backend._probabilities = predict
    questions = {"route": {"type": "choice", "criteria": {"b": "Billing", "t": "Tech"}},
                 "urgent": {"type": "noul"},
                 "level": {"type": "score", "criteria": ["Low", "Soon", "Now"]}}
    result = backend.predict({"content": "Ticket", "benchmark_request_id": "unique"}, questions)
    assert result["answers"]["route"]["choice"] == "t"
    assert result["answers"]["urgent"]["noul"] == 0.8
    assert result["answers"]["level"]["score"] == pytest.approx(1.6)
    assert all('"benchmark_request_id":"unique"' in text for text in observed)
    assert result["usage"]["input_tokens"] == sum(len(text.encode()) for text in observed)
    assert result["usage"]["output_tokens"] == 0


def test_clef_passes_no_truncation_and_preserves_native_rounding() -> None:
    backend = object.__new__(ClefBackend)
    backend.model_id = "clef-test"
    captured: dict = {}

    def systemone(request: dict, **kwargs: object) -> dict:
        captured.update(request=request, kwargs=kwargs)
        return {"answers": {"q": {"type": "choice", "choice": "a",
                                   "probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333}}}}

    backend.model = SimpleNamespace(systemone=systemone)
    questions = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}}
    result = backend.predict({"x": 1}, questions)
    assert captured["kwargs"] == {"max_length": 16384, "truncate": False}
    assert captured["request"]["state"] == '{"x":1}'
    assert result["answers"]["q"]["probabilities"]["a"] == 0.3333


def test_clef_executes_checked_source_and_rejects_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = b"loaded_from_checked_source = True\n"
    source_path = tmp_path / "clef_mlx.py"
    source_path.write_bytes(source)
    cache_path = Path(importlib.util.cache_from_source(str(source_path)))
    cache_path.parent.mkdir()
    stat = source_path.stat()
    cache_header = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, int(stat.st_mtime), stat.st_size)
    conflicting = compile("loaded_from_checked_source = False\n", str(source_path), "exec")
    cache_path.write_bytes(cache_header + marshal.dumps(conflicting))
    # Demonstrate that the old ordinary source loader accepts this valid cache.
    legacy_spec = importlib.util.spec_from_file_location("clef_cache_probe", source_path)
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    assert legacy.loaded_from_checked_source is False
    monkeypatch.setattr(d3_clef, "CLEF_RUNTIME_HASH", hashlib.sha256(source).hexdigest())
    module = d3_clef.load_clef_module(tmp_path)
    assert module.loaded_from_checked_source is True
    source_path.write_bytes(b"changed_source = True\n")
    with pytest.raises(ValueError, match="source checksum"):
        d3_clef.load_clef_module(tmp_path)
