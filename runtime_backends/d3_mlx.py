"""Benchmark-owned, text-only MLX Decision 3.0 native readout runtime.

The attention/prompt contract follows vLLM Semantic Router's Apache-2.0
6838193a0185c0ecfecab937896059d2c3377d61 release. Small BF16 artifacts retain
zero-centered source RMS weights; flash stores already-offset FP32 RMS weights.
Their distinct conventions are handled explicitly, without generic sanitization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime_backends import d3_format
from runtime_backends.d3_artifacts import check_readout_header, load_configs, read_header
from runtime_backends.d3_contract import prepare_question, product_answer
from runtime_backends.protocol import serialize_state, validate_prediction

ZERO_CENTERED_NORMS = (
    ".input_layernorm.weight", ".post_attention_layernorm.weight",
    "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
)


def offset_source_norms(weights: dict[str, Any], mx: Any) -> None:
    """Match Torch's (1 + w.float()) exactly for zero-centered source RMS norms.

    Gated DeltaNet norms are ordinary scales and must not receive this offset.
    This changes in-memory effective scales only, preserving downloaded tensors.
    """
    for name, value in weights.items():
        if name.startswith("language_model.model.") and name.endswith(ZERO_CENTERED_NORMS):
            weights[name] = value.astype(mx.float32) + 1.0


def configure_layers(net: Any, config: Any, language: Any, layer_types: list[str]) -> None:
    """Honor the explicit irregular layer pattern, including the pruned edge model."""
    lm = net.language_model.model
    for layer, kind in zip(lm.layers, layer_types, strict=True):
        full = kind == "full_attention"
        if full and not hasattr(layer, "self_attn"):
            layer.self_attn = language.Qwen3_5Attention(config.text_config)
            del layer.linear_attn
        elif not full and not hasattr(layer, "linear_attn"):
            layer.linear_attn = language.Qwen3_5GatedDeltaNet(config.text_config)
            del layer.self_attn
        layer.is_linear = not full
    # Generation is never used; the original decision checkpoints have no lm_head.
    if hasattr(net.language_model, "lm_head"):
        del net.language_model.lm_head


def quantized_modules(header: dict[str, Any]) -> set[str]:
    modules = {key.removesuffix(".scales") for key in header if key.endswith(".scales")}
    for name in modules:
        if not name.startswith("language_model.model."):
            raise ValueError("D3 4-bit quantization must be restricted to the language backbone")
        weight, scales, biases = (header.get(name + suffix, {}) for suffix in (".weight", ".scales", ".biases"))
        if weight.get("dtype") != "U32" or scales.get("dtype") != "BF16" or biases.get("dtype") != "BF16":
            raise ValueError("invalid affine 4-bit companion tensors")
        shape, group_shape = weight.get("shape", []), scales.get("shape", [])
        if len(shape) != 2 or group_shape != [shape[0], shape[1] // 8] or biases.get("shape") != group_shape:
            raise ValueError("invalid affine 4-bit group-64 tensor shapes")
    packed = {key.removesuffix(".weight") for key, entry in header.items() if entry.get("dtype") == "U32"}
    if not modules or packed != modules:
        raise ValueError("packed weights and quantization companion modules differ")
    return modules


def readout_probabilities(mx: Any, readout: Any, hidden: Any,
                          option_count: int, temperature: float) -> list[float]:
    """Readout rows are option-code positions, never vocabulary token IDs."""
    if not 1 <= option_count <= d3_format.MAX_OPTIONS:
        raise ValueError("invalid decision option count")
    logits = readout @ hidden.astype(mx.float32)
    probabilities = mx.softmax(logits[:option_count] / temperature, axis=-1)
    mx.eval(probabilities)
    return probabilities.tolist()


def load_parameters(net: Any, path: Path, artifact: dict[str, Any], mx: Any, nn: Any) -> Any:
    header = read_header(path / "weights.safetensors")
    weights = mx.load(str(path / "weights.safetensors"))
    if "decision_readout" in header:
        check_readout_header(header, "decision_readout", artifact["hidden"])
        modules = quantized_modules(header)
        nn.quantize(net, group_size=64, bits=4, mode="affine",
                    class_predicate=lambda name, _module: name in modules)
        readout = weights.pop("decision_readout")
        original = mx.load(str(path / "verification" / "readout.safetensors"))["weight"]
        if not mx.array_equal(readout, original).item():
            raise ValueError("embedded D3-flash readout differs from the original FP32 head")
    else:
        if any(entry.get("dtype") != "BF16" for entry in header.values()):
            raise ValueError("small D3 backbone must retain BF16 precision")
        check_readout_header(read_header(path / "readout.safetensors"), "weight", artifact["hidden"])
        readout = mx.load(str(path / "readout.safetensors"))["weight"]
        offset_source_norms(weights, mx)
    net.load_weights(list(weights.items()), strict=True)
    net.eval()
    mx.eval(net.parameters(), readout)
    if not mx.all(mx.isfinite(readout)).item():
        raise ValueError("decision readout contains nonfinite weights")
    return readout


class D3Backend:
    def __init__(self, spec: dict[str, Any], path: Path, artifact: dict[str, Any]) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models import qwen3_5
        from mlx_vlm.models.qwen3_5 import language
        from transformers import AutoTokenizer

        config, self.decision = load_configs(path, artifact)
        # This library's TextConfig does not retain layer_types. Pass them explicitly.
        model_config = qwen3_5.ModelConfig.from_dict(config)
        self.net = qwen3_5.Model(model_config)
        self.mx = mx
        self.lm = self.net.language_model.model
        configure_layers(self.net, model_config, language, config["text_config"]["layer_types"])
        self.readout = load_parameters(self.net, path, artifact, mx, nn)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.codes, code_ids = d3_format.answer_codes(self.tokenizer)
        if (self.codes, code_ids) != (self.decision.get("codes"), self.decision.get("token_ids")):
            raise ValueError("answer codes or token IDs disagree with decision configuration")
        self.max_length = config["text_config"]["max_position_embeddings"]
        if self.decision.get("max_length") is not None:
            self.max_length = min(self.max_length, self.decision["max_length"])
        self.model_id = spec["id"]
        self.observed_metadata = {
            "decision_mode": "native_readout_no_generation", "weight_checksums_verified": True,
            "artifact_revision": spec["artifact_revision"], "readout_dtype": "float32",
            "attention_mode": "noncausal_full_attention", "explicit_layer_types": True,
            "score_mapping": "native", "max_input_tokens": self.max_length,
            "normalization": ("stored_mlx_offset_fp32" if spec["id"] == "d3-flash"
                              else "source_zero_centered_runtime_fp32_offset"),
        }

    def _probabilities(self, ids: list[int], option_count: int) -> list[float]:
        mx = self.mx
        hidden = self.lm.embed_tokens(mx.array([ids], dtype=mx.int32))
        positions = mx.broadcast_to(mx.arange(len(ids))[None, None, :], (3, 1, len(ids)))
        for layer in self.lm.layers:
            # No causal attention mask or cache; DeltaNet's native scan remains causal.
            hidden = layer(hidden, mask=None, cache=None, position_ids=positions)
        hidden = self.lm.norm(hidden)
        return readout_probabilities(mx, self.readout, hidden[0, -1], option_count, self.decision["temperature"])

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        state_text = serialize_state(state)
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        answers: dict[str, Any] = {}
        tokens = 0
        for name, question in questions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("question IDs must be nonempty strings")
            normalized, ids = prepare_question(self.tokenizer, state_text, question, self.codes, self.max_length)
            probabilities = self._probabilities(ids, len(normalized.keys))
            answers[name] = product_answer(normalized, probabilities)
            tokens += len(ids)
        return validate_prediction({"model": self.model_id, "answers": answers,
                                    "usage": {"input_tokens": tokens, "output_tokens": 0}}, questions)
