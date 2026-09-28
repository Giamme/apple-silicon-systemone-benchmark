from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters import load_adapter
from benchmark import (
    check_answers,
    load_jsonl,
    run,
    run_one_model,
    selected_models,
    system_info,
    write_summary,
)
from model_registry import CommandSpec, ModelSpec, load_manifest, load_models


CASE = {
    "id": "ticket-1",
    "state": "The invoice was paid twice.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "support": "Software"},
        }
    },
    "expected": {"team": "billing"},
}


def make_spec(endpoint: str, launch: CommandSpec | None = None) -> ModelSpec:
    return ModelSpec(
        name="fake",
        display_name="Fake",
        endpoint=endpoint,
        api_model=None,
        note="A local test service",
        adapter="systemone",
        order=100,
        profiles=(),
        setup=None,
        launch=launch,
        version=None,
        headers_env={},
        source=Path("fake.toml"),
    )


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["questions"]["team"]["type"] == "choice"
        body = json.dumps({
            "answers": {
                "team": {
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "support": 0.1},
                }
            }
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def test_dataset_validation_reports_line_and_duplicate(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(CASE) + "\n" + json.dumps(CASE) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases.jsonl:2: duplicate id"):
        load_jsonl(dataset)
    invalid = {**CASE, "expected": {"team": "unknown"}}
    dataset.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="criteria containing its expected label"):
        load_jsonl(dataset)


def test_profiles_and_manifest_examples() -> None:
    specs = load_models()
    full = selected_models(SimpleNamespace(models=None, profile=None), specs)
    light = selected_models(SimpleNamespace(models=None, profile="light"), specs)
    assert [item.name for item in full] == ["laya", "von", "kev", "rizzo"]
    assert [item.name for item in light] == ["laya", "kev"]
    example = load_manifest(Path("examples/custom_service.toml"))
    assert example.launch is None
    adapter = load_adapter(example.adapter)
    assert adapter.normalize_response({"decisions": {"team": "billing"}}) == {
        "answers": {"team": {"choice": "billing"}}
    }


def test_existing_service_and_escaped_offline_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = make_spec(f"http://127.0.0.1:{server.server_port}/v1/systemone")
        result = run_one_model(spec, [CASE], tmp_path, 2, 1, 2, 5, 5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert result["accuracy"] == 1.0
    assert result["requests"] == 2
    assert result["startup_to_port_ms"] is None
    assert result["peak_process_tree_rss_mb"] is None
    assert "[fake] measuring" in capsys.readouterr().out
    metadata = {
        "dataset": {"name": "<img src=x>", "sha256": "a" * 64, "cases": 1, "questions": 1, "smoke": False},
        "profile": "custom",
        "models": ["fake"],
        "created_at": "2026-09-28T00:00:00+00:00",
        "passes": 2,
        "warmup": 1,
        "cached_repeats": 2,
        "system": {"chip_type": "Apple Silicon"},
    }
    write_summary(tmp_path, [result], metadata)
    report = (tmp_path / "summary.html").read_text(encoding="utf-8")
    assert "&lt;img src=x&gt;" in report
    assert "<img src=x>" not in report
    assert "Apple Silicon decision benchmark" in report
    assert (tmp_path / "summary.csv").exists()


def test_managed_server_streams_logs_and_stops(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    script = (
        "import json,sys; from http.server import BaseHTTPRequestHandler,HTTPServer; "
        "body=json.dumps({'answers':{'team':{'choice':'billing'}}}).encode(); "
        "class_source='class H(BaseHTTPRequestHandler):\\n"
        " def do_POST(self):\\n"
        "  self.rfile.read(int(self.headers[\"Content-Length\"])); "
        "self.send_response(200); self.end_headers(); self.wfile.write(body)\\n"
        " def log_message(self,*args): pass'; "
        "exec(class_source); print('server ready',flush=True); "
        "HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()"
    )
    launch = CommandSpec((sys.executable, "-u", "-c", script, str(port)), Path.cwd(), {})
    result = run_one_model(
        make_spec(f"http://127.0.0.1:{port}/v1/systemone", launch),
        [CASE], tmp_path, 1, 0, 0, 5, 5,
    )
    assert result["requests"] == 1
    assert result["startup_to_port_ms"] is not None
    assert "server ready" in (tmp_path / "fake.server.log").read_text()
    assert "[fake:server] server ready" in capsys.readouterr().out


def test_partial_failure_writes_report_and_returns_nonzero(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    spec = make_spec(f"http://127.0.0.1:{port}/v1/systemone")
    args = SimpleNamespace(
        action="run", dataset=dataset, allow_remote=False, passes=1, warmup=0,
        cached_repeats=0, startup_timeout=1, first_request_timeout=1,
        output=tmp_path / "output", models=["fake"], profile=None,
    )
    assert run(args, [spec]) == 1
    assert "Partial run" in (args.output / "summary.html").read_text()
    assert "errors" in (args.output / "summary.json").read_text()


def test_new_dataset_flows_into_report(tmp_path: Path) -> None:
    dataset = tmp_path / "fresh-cases.jsonl"
    dataset.write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = make_spec(f"http://127.0.0.1:{server.server_port}/v1/systemone")
        args = SimpleNamespace(
            action="run", dataset=dataset, allow_remote=False, passes=1, warmup=0,
            cached_repeats=0, startup_timeout=5, first_request_timeout=5,
            output=tmp_path / "output", models=["fake"], profile=None,
        )
        assert run(args, [spec]) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    report = json.loads((args.output / "summary.json").read_text())
    assert report["run"]["dataset"]["name"] == "fresh-cases.jsonl"
    assert report["run"]["dataset"]["cases"] == 1
    assert report["run"]["dataset"]["smoke"] is False
    assert "fresh-cases.jsonl" in (args.output / "summary.html").read_text()
    assert "smoke dataset" not in (args.output / "summary.md").read_text().lower()


def test_system_info_excludes_machine_identifiers() -> None:
    info = system_info()
    assert not any("serial" in key.lower() or "uuid" in key.lower() for key in info)


def test_missing_decision_and_remote_endpoint_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no decision"):
        check_answers(CASE, {"answers": {"team": {}}})
    spec = make_spec("https://example.com/v1/systemone")
    args = SimpleNamespace(
        dataset=tmp_path / "missing.jsonl", allow_remote=False, passes=1,
        warmup=0, cached_repeats=0, startup_timeout=1,
        first_request_timeout=1, output=None, models=["fake"], profile=None,
    )
    with pytest.raises(ValueError, match="--allow-remote"):
        run(args, [spec])
