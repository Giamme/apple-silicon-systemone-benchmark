from __future__ import annotations

import copy

import pytest

from runtime_backends.d3_parity import compare_answer, compare_captures
from runtime_catalog import get_model


QUESTION = {"type": "choice", "criteria": {"a": "First", "b": "Second"}}


def captures() -> tuple[dict, dict]:
    row = {"case_id": "one", "questions": {"q": QUESTION, "bool": {"type": "noul"},
                                           "score": {"type": "score", "criteria": ["Low", "High"]}},
           "prompts": {"q": {"text": "exact", "input_ids": [1, 2]}},
           "prediction": {"answers": {"q": {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}},
                                      "bool": {"noul": 0.8},
                                      "score": {"probabilities": {"0": 0.2, "1": 0.8}, "score": 0.8}}}}
    spec = get_model("d3-edge")
    reference = {"model_id": "d3-edge", "engine": "reference", "cases": [row],
                 **{key: spec[key] for key in ("source_repo", "source_revision", "artifact_repo", "artifact_revision")}}
    candidate = copy.deepcopy(reference)
    candidate["engine"] = "mlx"
    return reference, candidate


def test_fixed_parity_gate_accepts_small_numeric_difference() -> None:
    reference, candidate = captures()
    candidate["cases"][0]["prediction"]["answers"]["q"]["probabilities"] = {"a": 0.79, "b": 0.21}
    result = compare_captures(reference, candidate)
    assert result["decisions"] == 3
    assert result["max_absolute_probability_delta"] == pytest.approx(0.01)


@pytest.mark.parametrize("field,replacement", [("text", "changed"), ("input_ids", [1, 3])])
def test_exact_prompt_and_token_mismatch_blocks_parity(field: str, replacement: object) -> None:
    reference, candidate = captures()
    candidate["cases"][0]["prompts"]["q"][field] = replacement
    with pytest.raises(ValueError, match="exact prompt"):
        compare_captures(reference, candidate)


def test_probability_regression_blocks_without_tolerance_waiver() -> None:
    reference, candidate = captures()
    candidate["cases"][0]["prediction"]["answers"]["q"]["probabilities"] = {"a": 0.7, "b": 0.3}
    with pytest.raises(ValueError, match="fixed"):
        compare_captures(reference, candidate)


def test_boolean_and_expected_score_are_compared_natively() -> None:
    assert compare_answer({"noul": 0.8}, {"noul": 0.81}, {"type": "noul"}) == pytest.approx(0.01)
    score = {"type": "score", "criteria": ["Low", "Medium", "High"]}
    answer = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "score": 1.6}
    with pytest.raises(ValueError, match="expected-score"):
        compare_answer(answer, {**answer, "score": 1.7}, score)


def test_nonfinite_capture_is_rejected_before_comparison() -> None:
    reference, candidate = captures()
    candidate["cases"][0]["prediction"]["answers"]["q"]["probabilities"]["a"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compare_captures(reference, candidate)


def test_empty_or_missing_decision_coverage_cannot_pass() -> None:
    reference, candidate = captures()
    reference["cases"] = candidate["cases"] = []
    with pytest.raises(ValueError, match="nonempty"):
        compare_captures(reference, candidate)
    reference, candidate = captures()
    for capture in (reference, candidate):
        capture["cases"][0]["questions"] = {}
    with pytest.raises(ValueError, match="coverage"):
        compare_captures(reference, candidate)


@pytest.mark.parametrize("field", ["source_repo", "source_revision", "artifact_repo", "artifact_revision"])
def test_stale_capture_revisions_are_rejected(field: str) -> None:
    reference, candidate = captures()
    reference[field] = candidate[field] = "stale"
    with pytest.raises(ValueError, match="identity"):
        compare_captures(reference, candidate)
