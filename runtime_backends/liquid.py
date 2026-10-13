"""Pinned Liquid MLX decision APIs, retaining unrounded native probabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import canonical_answer, serialize_state, validate_prediction, validate_questions


class LiquidBackend:
    def __init__(self, model: Any, processor: Any, prompt: Any, omni: bool) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.prompt = prompt
        self.omni = omni
        self.context_limit = 16384 if omni else 32768
        self.observed_metadata = {
            "native_api": "probabilities",
            "probability_precision": "unrounded",
            "boolean_order": ["false", "true"] if omni else ["true", "false"],
            "text_only": True,
            "context_limit": self.context_limit,
            "calibration": "native_config" if omni else "native_readout",
        }

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _check_omni_prompt(self, state: str, question: Any, name: str) -> None:
        """Compare native encoded segments to their complete token sequences."""
        prompt = self.prompt
        ids, markers = prompt.encode(self.tokenizer, state, question, self.context_limit)
        options = prompt.render_options(question)
        if len(markers) != len(options) or not markers:
            raise ValueError(f"{name}: native option markers do not match criteria")
        q_token = self.tokenizer.convert_tokens_to_ids(prompt.DELIM["q"])
        option_end = self.tokenizer.convert_tokens_to_ids(prompt.DELIM["opt_end"])
        question_start = ids.index(q_token)
        segments = [
            ("state", ids[2:question_start], self._tokens(prompt.escape(prompt.serialize(state)))),
            ("instructions", ids[question_start + 1:markers[0] - 1], self._tokens(prompt.escape(question[1]))),
        ]
        for index, (marker, option) in enumerate(zip(markers, options)):
            end = ids.index(option_end, marker + 1)
            segments.append((f"criterion {index}", ids[marker + 1:end], self._tokens(prompt.escape(" " + option))))
        for segment, actual, complete in segments:
            if actual != complete:
                raise ValueError(f"{name}: native encoder would truncate {segment}; shorten the input")

    def _check_d1_prompt(self, state: str, questions: list[dict]) -> None:
        trunk, branches, _ = self.model.decision_inputs(self.processor, state, questions)
        longest = len(trunk) + max((len(branch) for branch in branches), default=0)
        if longest > self.context_limit:
            raise ValueError(f"native d1 prompt exceeds {self.context_limit} tokens; shorten the input")

    def predict(self, state: Any, questions: dict[str, dict]) -> dict:
        questions = validate_questions(questions)
        text = serialize_state(state)
        parsed = [self.prompt.as_question(question) for question in questions.values()]
        if self.omni:
            for name, question in zip(questions, parsed):
                self._check_omni_prompt(text, question, name)
            requests = self.model.probabilities(self.tokenizer, [(text, parsed)])
            if len(requests) != 1:
                raise ValueError("native omni returned an unexpected request count")
            probabilities, tokens = requests[0]
        else:
            self._check_d1_prompt(text, parsed)
            probabilities, tokens = self.model.probabilities(self.processor, text, parsed)
        if len(probabilities) != len(questions):
            raise ValueError("native Liquid returned an unexpected question count")
        result = {
            "answers": {
                name: canonical_answer(question, values, boolean_yes_index=1 if self.omni else 0)
                for (name, question), values in zip(questions.items(), probabilities)
            },
            "usage": {"input_tokens": int(tokens), "output_tokens": 0},
        }
        return validate_prediction(result, questions)


def load_backend(spec: dict, snapshot_path: Path) -> LiquidBackend:
    from mlx_vlm import load

    config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
    omni = config.get("model_type") == "d1_omni"
    if omni:
        from mlx_vlm.models.d1_omni import prompt
    else:
        from mlx_vlm.models.d1 import prompt
    model, processor = load(str(snapshot_path))
    return LiquidBackend(model, processor, prompt, omni)
