"""HTTP request/response adapters for comparable decision results."""
from __future__ import annotations

import importlib
import json
import os
import urllib.request
from typing import Any, Protocol

from model_registry import ModelSpec


class DecisionAdapter(Protocol):
    def build_request(
        self, case: dict[str, Any], spec: ModelSpec, nonce: str | None
    ) -> urllib.request.Request: ...

    def normalize_response(self, data: Any) -> dict[str, Any]: ...


class SystemOneAdapter:
    """The common POST /v1/systemone wire contract."""

    def build_request(
        self, case: dict[str, Any], spec: ModelSpec, nonce: str | None
    ) -> urllib.request.Request:
        state: Any = case["state"]
        if nonce is not None:
            state = {
                "content": state,
                "benchmark_request_id": nonce,
                "benchmark_note": "The benchmark_request_id is metadata only; ignore it when answering.",
            }
        payload: dict[str, Any] = {"state": state, "questions": case["questions"]}
        if spec.api_model:
            payload["model"] = spec.api_model
        headers = {"Content-Type": "application/json"}
        for header, variable in spec.headers_env.items():
            value = os.environ.get(variable)
            if not value:
                raise ValueError(f"{spec.name}: required environment variable {variable} is missing")
            headers[header] = value
        return urllib.request.Request(
            spec.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def normalize_response(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("Response must be an object containing an answers object")
        return data


def load_adapter(name: str) -> DecisionAdapter:
    if name == "systemone":
        return SystemOneAdapter()
    module_name, separator, class_name = name.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"Invalid adapter {name!r}; expected module:Class")
    module = importlib.import_module(module_name)
    adapter = getattr(module, class_name)()
    if not callable(getattr(adapter, "build_request", None)) or not callable(
        getattr(adapter, "normalize_response", None)
    ):
        raise ValueError(f"Adapter {name!r} must provide build_request and normalize_response")
    return adapter
