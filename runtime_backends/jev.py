"""JEV-9B NVFP4 native readout with explicit ordered-score choice adaptation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import canonical_answer, serialize_state, validate_prediction, validate_questions


class JevBackend:
    def __init__(self, spec: dict, runner: Any) -> None:
        self.runner = runner
        self.choice_limit = min(16, int(spec.get("choice_limit", 16)))
        self.context_limit = int(spec.get("max_input_tokens", 4096))
        self.observed_metadata = {
            "native_api": "JevText._pass",
            "probability_precision": "unrounded",
            "boolean_order": ["false", "true"],
            "score_mapping": "ordered_criteria_via_choice",
            "readout_precision": "unquantized_bfloat16_with_float32_probability_math",
            "text_only": True,
            "context_limit": self.context_limit,
        }

    def _question(self, question: dict) -> tuple[str, str, list[str]]:
        kind = question["type"]
        instructions = question["instructions"]
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria:
                instructions += f"\n\nFalse: {criteria['false']}\nTrue: {criteria['true']}"
            return kind, instructions, ["false", "true"]
        entries = criteria.items() if kind == "choice" else ((str(i), value) for i, value in enumerate(criteria))
        options = [f"{key}: {description}" if description else key for key, description in entries]
        if not 2 <= len(options) <= self.choice_limit:
            raise ValueError(f"JEV choice requires 2-{self.choice_limit} options")
        return "choice", instructions, options

    def _check_prompt(self, kind: str, state: str, instructions: str, options: list[str]) -> None:
        labels = self.runner.labels
        lines = [f"{labels[index]}) {option}" for index, option in enumerate(options)] if kind == "choice" else options
        text = f"[kind] {kind}\n[state] {state}\n[question] {instructions}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        count = len(self.runner.tokenizer(text, add_special_tokens=False)["input_ids"])
        if count > self.context_limit:
            raise ValueError(f"native JEV prompt exceeds {self.context_limit} tokens; shorten the input")

    def predict(self, state: Any, questions: dict[str, dict]) -> dict:
        questions = validate_questions(questions)
        text = serialize_state(state)
        prepared = {name: self._question(question) for name, question in questions.items()}
        for kind, instructions, options in prepared.values():
            self._check_prompt(kind, text, instructions, options)
        self.runner.tokens = 0
        answers = {}
        for name, (kind, instructions, options) in prepared.items():
            values = self.runner._pass(kind, text, [], instructions, options)
            answers[name] = canonical_answer(questions[name], values, boolean_yes_index=1)
        result = {"answers": answers, "usage": {"input_tokens": int(self.runner.tokens), "output_tokens": 0}}
        return validate_prediction(result, questions)


def load_backend(spec: dict, snapshot_path: Path) -> JevBackend:
    from mlx_vlm import load
    from mlx_vlm.models.jev_text.jev_text import JevText

    model, processor = load(str(snapshot_path))
    return JevBackend(spec, JevText(model, processor))
