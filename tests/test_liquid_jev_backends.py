from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime_backends.jev import JevBackend
from runtime_backends.liquid import LiquidBackend


QUESTIONS = {
    "yes": {"type": "noul", "instructions": "Refund requested?", "criteria": {"false": "No refund", "true": "Wants money back"}},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Software"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "Today", "Now"]},
}


class Tokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict:
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [ord(character) + 1000 for character in text]}

    def convert_tokens_to_ids(self, text: str) -> int:
        return {"state": 2, "question": 3, "option": 4, "end": 5}[text]


class OmniPrompt:
    DELIM = {"q": "question", "opt_end": "end"}

    def __init__(self, cut: str | None = None) -> None:
        self.cut = cut

    @staticmethod
    def as_question(question: dict) -> tuple:
        return question["type"], question["instructions"], question.get("criteria")

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def serialize(state: str) -> str:
        return state

    @staticmethod
    def render_options(question: tuple) -> list[str]:
        kind, _, criteria = question
        if kind == "noul":
            criteria = criteria or {"false": "no, the statement does not hold", "true": "yes, the statement holds"}
            return ["false: " + criteria["false"], "true: " + criteria["true"]]
        if kind == "score":
            return [f"level {i}: {value}" for i, value in enumerate(criteria)]
        return [f"{key}: {value}" for key, value in criteria.items()]

    def encode(self, tokenizer: Tokenizer, state: str, question: tuple, limit: int) -> tuple:
        def tokens(text: str, segment: str) -> list[int]:
            values = tokenizer(text, add_special_tokens=False)["input_ids"]
            return values[:-1] if self.cut == segment else values

        ids = [1, 2] + tokens(state, "state") + [3] + tokens(question[1], "instructions")
        markers = []
        for index, option in enumerate(self.render_options(question)):
            markers.append(len(ids) + 1)
            ids += [4, 6] + tokens(" " + option, f"criterion {index}") + [5]
        return ids + [7], markers


class LiquidModel:
    def __init__(self, probabilities: list, *, omni: bool, length: int = 10) -> None:
        self.values = probabilities
        self.omni = omni
        self.length = length
        self.calls = 0
        self.state = None

    def decision_inputs(self, processor: object, state: str, questions: list) -> tuple:
        return [0] * self.length, [[0] * 3], {}

    def probabilities(self, processor: object, state: object, questions: list | None = None) -> object:
        self.calls += 1
        self.state = state[0][0] if self.omni else state
        result = self.values, 17
        return [result] if self.omni else result


@pytest.mark.parametrize("omni", [False, True])
def test_liquid_retains_raw_probabilities_and_boolean_direction(omni: bool) -> None:
    p = 0.123456789
    values = [[1 - p, p] if omni else [p, 1 - p], [0.73, 0.27], [0.2, 0.5, 0.3]]
    model = LiquidModel(values, omni=omni)
    prompt = OmniPrompt() if omni else SimpleNamespace(as_question=lambda value: value)
    backend = LiquidBackend(model, Tokenizer(), prompt, omni)
    result = backend.predict({"z": "è", "a": 1}, QUESTIONS)
    assert result["answers"]["yes"]["noul"] == p
    assert result["answers"]["team"]["choice"] == "billing"
    assert result["answers"]["urgency"]["probabilities"] == {"0": 0.2, "1": 0.5, "2": 0.3}
    assert model.state == '{"a":1,"z":"è"}'
    assert result["usage"] == {"input_tokens": 17, "output_tokens": 0}


@pytest.mark.parametrize("cut", ["state", "instructions", "criterion 0", "criterion 1"])
def test_omni_rejects_native_trimming_before_model_call(cut: str) -> None:
    model = LiquidModel([[0.8, 0.2]], omni=True)
    backend = LiquidBackend(model, Tokenizer(), OmniPrompt(cut), True)
    with pytest.raises(ValueError, match="truncate " + cut):
        backend.predict("Original state", {"team": QUESTIONS["team"]})
    assert model.calls == 0


def test_d1_checks_complete_branch_path_not_batch_budget() -> None:
    model = LiquidModel([[0.8, 0.2]], omni=False, length=32766)
    prompt = SimpleNamespace(as_question=lambda value: value)
    backend = LiquidBackend(model, Tokenizer(), prompt, False)
    with pytest.raises(ValueError, match="32768 tokens"):
        backend.predict("state", {"team": QUESTIONS["team"]})
    assert model.calls == 0


@pytest.mark.parametrize("values", [[[float("nan"), 0.2]], [[0.8]], [[0.8, 0.8]], []])
def test_liquid_rejects_incomplete_or_invalid_readouts(values: list) -> None:
    model = LiquidModel(values, omni=False)
    prompt = SimpleNamespace(as_question=lambda value: value)
    with pytest.raises(ValueError):
        LiquidBackend(model, Tokenizer(), prompt, False).predict("state", {"team": QUESTIONS["team"]})


class JevRunner:
    tokenizer = Tokenizer()
    labels = tuple("ABCDEFGHIJKLMNOP")

    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.calls = []
        self.tokens = 900

    def _pass(self, kind: str, state: str, media: list, instructions: str, options: list) -> list:
        self.calls.append((kind, state, media, instructions, options))
        self.tokens += 9
        return self.values[(len(self.calls) - 1) % len(self.values)]


def test_jev_adapts_score_choice_preserves_boolean_definitions_and_resets_usage() -> None:
    p = 0.812345678
    runner = JevRunner([[1 - p, p], [0.2, 0.8], [0.1, 0.3, 0.6]])
    backend = JevBackend({"choice_limit": 16}, runner)
    result = backend.predict({"z": 2, "a": "Italiano"}, QUESTIONS)
    assert runner.calls[0] == (
        "noul", '{"a":"Italiano","z":2}', [],
        "Refund requested?\n\nFalse: No refund\nTrue: Wants money back", ["false", "true"],
    )
    assert runner.calls[2][0] == "choice"
    assert runner.calls[2][3:] == ("How urgent?", ["0: Can wait", "1: Today", "2: Now"])
    assert result["answers"]["yes"]["noul"] == p
    assert result["answers"]["urgency"]["probabilities"] == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert result["answers"]["urgency"]["score"] == pytest.approx(1.5)
    assert result["usage"]["input_tokens"] == 27
    assert backend.predict("next", QUESTIONS)["usage"]["input_tokens"] == 27


def test_jev_rejects_untrained_option_count_and_long_prompt_before_inference() -> None:
    runner = JevRunner([[0.5, 0.5]])
    backend = JevBackend({}, runner)
    too_many = {"type": "choice", "instructions": "Choose", "criteria": {str(i): str(i) for i in range(17)}}
    with pytest.raises(ValueError, match="2-16"):
        backend.predict("state", {"q": too_many})
    with pytest.raises(ValueError, match="4096 tokens"):
        backend.predict("x" * 4096, {"team": QUESTIONS["team"]})
    assert runner.calls == []


def test_jev_accepts_boolean_without_optional_criteria() -> None:
    runner = JevRunner([[0.2, 0.8]])
    question = {"type": "noul", "instructions": "Is it true?"}
    result = JevBackend({}, runner).predict("state", {"q": question})
    assert result["answers"]["q"]["noul"] == 0.8
    assert runner.calls[0][3] == "Is it true?"


@pytest.mark.parametrize("omni", [False, True])
def test_liquid_accepts_boolean_without_optional_criteria(omni: bool) -> None:
    model = LiquidModel([[0.2, 0.8] if omni else [0.8, 0.2]], omni=omni)
    prompt = OmniPrompt() if omni else SimpleNamespace(as_question=lambda value: value)
    result = LiquidBackend(model, Tokenizer(), prompt, omni).predict(
        "state", {"q": {"type": "noul", "instructions": "Is it true?"}},
    )
    assert result["answers"]["q"]["noul"] == 0.8


@pytest.mark.parametrize("values", [[float("inf"), 0.0], [0.2], [0.8, 0.8]])
def test_jev_rejects_invalid_native_probability_vectors(values: list) -> None:
    with pytest.raises(ValueError):
        JevBackend({}, JevRunner([values])).predict("state", {"team": QUESTIONS["team"]})
