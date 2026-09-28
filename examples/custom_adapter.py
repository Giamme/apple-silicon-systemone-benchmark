"""Example adapter for a service returning {"decisions": {"qid": "label"}}."""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from model_registry import ModelSpec


class ChoiceAdapter:
    def build_request(
        self, case: dict[str, Any], spec: ModelSpec, nonce: str | None
    ) -> urllib.request.Request:
        body = {"state": case["state"], "questions": case["questions"]}
        if nonce is not None:
            body["request_id"] = nonce
        return urllib.request.Request(
            spec.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def normalize_response(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not isinstance(data.get("decisions"), dict):
            raise ValueError("Custom service must return a decisions object")
        return {
            "answers": {
                question_id: {"choice": label}
                for question_id, label in data["decisions"].items()
            }
        }
