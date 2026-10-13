"""Sequential D3 capture and fixed-tolerance comparison, isolated from ML imports."""
from __future__ import annotations

import math
from typing import Any

from runtime_backends import d3_format
from runtime_backends.d3_contract import normalize_question
from runtime_backends.protocol import serialize_state, validate_prediction
from runtime_catalog import get_model

MAX_PROBABILITY_DELTA = 0.03
STRONG_WINNER_MARGIN = 0.06


def candidate_prompts(backend: Any, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    text_state = serialize_state(state)
    result = {}
    for name, question in questions.items():
        normalized = normalize_question(question)
        text = d3_format.render(backend.tokenizer, text_state, normalized.rendered, backend.codes)
        result[name] = {"text": text, "input_ids": backend.tokenizer.encode(text, add_special_tokens=False)}
    return result


def reference_prompts(model: Any, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    runtime = model.runtime
    state_text = serialize_state(state)
    prepared = runtime.prepare(state_text, questions)
    if prepared.errors:
        raise ValueError(f"reference prompt preparation failed: {prepared.errors}")
    return {name: {"text": runtime.text(state_text, prepared.questions[name].rendered),
                   "input_ids": prepared.sequences[name]} for name in questions}


def probabilities(answer: dict[str, Any], question: dict[str, Any]) -> dict[str, float]:
    if question["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def compare_answer(reference: dict[str, Any], candidate: dict[str, Any],
                   question: dict[str, Any]) -> float:
    expected, actual = probabilities(reference, question), probabilities(candidate, question)
    if expected.keys() != actual.keys():
        raise ValueError("parity probability labels differ")
    delta = max(abs(expected[key] - actual[key]) for key in expected)
    if not math.isfinite(delta) or delta > MAX_PROBABILITY_DELTA:
        raise ValueError(f"probability parity exceeds fixed {MAX_PROBABILITY_DELTA}: {delta:.6f}")
    ranked = sorted(expected, key=expected.__getitem__, reverse=True)
    margin = expected[ranked[0]] - expected[ranked[1]] if len(ranked) > 1 else 1.0
    winner = max(actual, key=actual.__getitem__)
    if margin > STRONG_WINNER_MARGIN and winner != ranked[0]:
        raise ValueError("strong-margin reference winner differs")
    if question["type"] == "score":
        maximum = len(question["criteria"]) - 1
        if abs(reference["score"] - candidate["score"]) / maximum > MAX_PROBABILITY_DELTA:
            raise ValueError("normalized expected-score parity exceeds fixed tolerance")
    return delta


def require_capture_identity(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if reference["model_id"] != candidate["model_id"]:
        raise ValueError("parity captures belong to different models")
    if reference["engine"] != "reference" or candidate["engine"] != "mlx":
        raise ValueError("compare requires reference then MLX captures")
    spec = get_model(reference["model_id"])
    identity_fields = ("source_repo", "source_revision", "artifact_repo", "artifact_revision")
    if any(capture.get(field) != spec[field]
           for capture in (reference, candidate) for field in identity_fields):
        raise ValueError("parity capture source or artifact identity differs from pinned recipe")


def compare_captures(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    require_capture_identity(reference, candidate)
    original_rows, candidate_rows = reference["cases"], candidate["cases"]
    if not original_rows or not candidate_rows:
        raise ValueError("parity requires nonempty captures")
    if len(original_rows) != len(candidate_rows):
        raise ValueError("parity capture case counts differ")
    maximum_delta, decisions = 0.0, 0
    kinds: set[str] = set()
    for original, actual in zip(original_rows, candidate_rows, strict=True):
        if original["case_id"] != actual["case_id"] or original["questions"] != actual["questions"]:
            raise ValueError("parity capture case identities or questions differ")
        if original["prompts"] != actual["prompts"]:
            raise ValueError("exact prompt text or token IDs differ")
        questions = original["questions"]
        if not questions:
            raise ValueError("parity cases require decision coverage")
        validate_prediction(original["prediction"], questions)
        validate_prediction(actual["prediction"], questions)
        for name, question in questions.items():
            kinds.add(question["type"])
            delta = compare_answer(original["prediction"]["answers"][name],
                                   actual["prediction"]["answers"][name], question)
            maximum_delta = max(maximum_delta, delta)
            decisions += 1
    if kinds != {"choice", "noul", "score"}:
        raise ValueError("parity requires choice, noul and score coverage")
    return {"model_id": reference["model_id"], "cases": len(original_rows), "decisions": decisions,
            "max_absolute_probability_delta": maximum_delta,
            "probability_tolerance": MAX_PROBABILITY_DELTA,
            "strong_winner_margin": STRONG_WINNER_MARGIN, "exact_prompts_and_tokens": True}
