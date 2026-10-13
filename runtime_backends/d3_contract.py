"""Weight-free Decision 3.0 question normalization and native answers.

Ported from the Apache-2.0 vLLM Semantic Router release, revision
6838193a0185c0ecfecab937896059d2c3377d61, d3_runtime.py.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from runtime_backends import d3_format
from runtime_backends.protocol import serialize_state


@dataclass(frozen=True)
class Question:
    kind: str
    rendered: dict[str, Any]
    keys: list[str]
    descriptions: list[Any]


def json_value(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"not representable as JSON: {exc}") from exc


def _choice(question: Mapping[str, Any], instructions: Any) -> Question:
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= d3_format.MAX_OPTIONS:
        raise ValueError("choice criteria must map 1 to 255 options to descriptions")
    if any(not isinstance(key, str) or not key for key in criteria):
        raise ValueError("choice option keys must be nonempty strings")
    json_value(dict(criteria))
    rendered = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
    return Question("choice", rendered, list(criteria), list(criteria.values()))


def _noul(question: Mapping[str, Any], instructions: Any) -> Question:
    criteria = question.get("criteria")
    if criteria is not None:
        if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
            raise ValueError('noul criteria may only describe "false" and "true"')
        json_value(dict(criteria))
    rendered = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        rendered["criteria"] = dict(criteria)
    keys, texts = d3_format.options(rendered)
    return Question("noul", rendered, keys, texts)


def _score(question: Mapping[str, Any], instructions: Any) -> Question:
    criteria = question.get("criteria")
    if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
        raise ValueError("score criteria must be an ordered list of 2 to 10 levels")
    json_value(list(criteria))
    texts = [d3_format.describe(level) for level in criteria]
    if any(not text for text in texts) or len(set(texts)) != len(texts):
        raise ValueError("score levels must be distinct and nonempty")
    # The native model sees each level's description alone, not its zero-based key.
    rendered = {"type": "choice", "instructions": instructions,
                "criteria": {text: None for text in texts}}
    return Question("score", rendered, [str(i) for i in range(len(texts))], list(criteria))


def normalize_question(question: Any) -> Question:
    if not isinstance(question, Mapping):
        raise ValueError("a question must be an object")
    instructions = question.get("instructions")
    json_value(instructions)
    handlers = {"choice": _choice, "noul": _noul, "score": _score}
    kind = question.get("type")
    if kind not in handlers:
        raise ValueError("unsupported question type (choice, noul or score required)")
    return handlers[kind](question, instructions)


def confidence(probabilities: Sequence[float]) -> float:
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities))))


def product_answer(question: Question, probabilities: Sequence[float]) -> dict[str, Any]:
    values = [float(p) for p in probabilities]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError("model probabilities must be finite and between zero and one")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError("model probabilities must sum to one")
    shares = d3_format.normalized(values, len(question.keys))
    if question.kind == "score":
        return {"type": "score", "score": sum(i * p for i, p in enumerate(shares)),
                "probabilities": dict(zip(question.keys, shares)),
                "confidence": confidence(shares),
                "legend": {key: serialize_state(text)
                           for key, text in zip(question.keys, question.descriptions)}}
    answer = d3_format.to_answer(question.rendered, shares)
    if question.kind == "choice":
        answer["confidence"] = confidence(shares)
    return answer


def prepare_question(tokenizer: Any, state: str, question: Any,
                     codes: Sequence[str], max_length: int) -> tuple[Question, list[int]]:
    normalized = normalize_question(question)
    text = d3_format.render(tokenizer, state, normalized.rendered, codes)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_length:
        raise ValueError(f"question prompt exceeds maximum context length {max_length}; no truncation")
    return normalized, ids
