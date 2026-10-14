from __future__ import annotations

import json
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters import load_adapter
from benchmark import (
    check_answers,
    load_jsonl,
    parse_args,
    run,
    run_one_model,
    selected_models,
    system_info,
    write_summary,
)
from fakes import CASE, Handler, serve, stuck_service, write_dataset, write_manifest
from model_registry import ROOT, CommandSpec, ModelSpec, load_manifest, load_models
from run_control import Session


def make_spec(endpoint: str, launch: CommandSpec | None = None, name: str = "fake") -> ModelSpec:
    return ModelSpec(
        name=name,
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


def run_args(dataset: Path, output: Path | None, models: list[str], **overrides: object) -> SimpleNamespace:
    return SimpleNamespace(**{
        "action": "run", "dataset": dataset, "allow_remote": False, "passes": 1, "warmup": 0,
        "cached_repeats": 0, "startup_timeout": 5, "first_request_timeout": 60,
        "output": output, "models": models, "profile": None, **overrides,
    })


def test_dataset_validation_reports_line_and_duplicate(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(CASE) + "\n" + json.dumps(CASE) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases.jsonl:2: duplicate id"):
        load_jsonl(dataset)
    invalid = {**CASE, "expected": {"team": "unknown"}}
    dataset.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="criteria containing its expected label"):
        load_jsonl(dataset)


@pytest.mark.parametrize(
    ('arguments', 'expected_ids'),
    [
        ([], None),
        (['--profile', 'full'], None),
        (['--profile', 'extended'], None),
        (['--profile', 'light'], ['laya', 'kev']),
        (['--models', 'jev-9b', 'laya'], ['jev-9b', 'laya']),
        (['--profile', 'light', '--exclude', 'kev'], ['laya']),
        (['--models', 'jev-9b', 'laya', '--exclude', 'jev-9b'], ['laya']),
    ],
    ids=['default', 'full', 'extended', 'light', 'explicit-model-order', 'profile-exclude', 'models-exclude'],
)
def test_cli_model_selection_contract(arguments: list[str], expected_ids: list[str] | None) -> None:
    specs = load_models()
    args = parse_args(['all', *arguments])
    selected = selected_models(args, specs)
    if expected_ids is None:
        builtins = [load_manifest(path) for path in Path('models').glob('*.toml')]
        builtin_ids = {spec.name for spec in builtins}
        actual_ids = {spec.name for spec in selected}
        assert builtin_ids <= actual_ids, f'Missing built-in models: {sorted(builtin_ids - actual_ids)}'
        selected_builtins = [spec.name for spec in selected if spec.name in builtin_ids]
        expected_ids = [spec.name for spec in sorted(builtins, key=lambda spec: (spec.order, spec.name))]
        assert selected_builtins == expected_ids
    else:
        assert [spec.name for spec in selected] == expected_ids


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--profile", "light", "--exclude", "nope"], "Unknown model.*--exclude"),
        (["--profile", "light", "--exclude", "laya", "kev"], "No models left"),
    ],
    ids=["unknown-id", "nothing-left"],
)
def test_exclude_rejects_unknown_ids_and_an_empty_selection(arguments: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        selected_models(parse_args(["all", *arguments]), load_models())


def test_manifest_example_adapter() -> None:
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


@pytest.mark.parametrize(
    ("action", "exit_code", "statuses", "run_status"),
    [
        ("skip", 1, {"a": "ok", "b": "skipped", "c": "ok"}, "complete"),
        ("abort", 130, {"a": "ok", "b": "skipped"}, "aborted"),
    ],
    ids=["skip-continues", "abort-keeps-finished"],
)
def test_stuck_first_request_can_be_skipped_or_aborted_with_a_report(
    tmp_path: Path, action: str, exit_code: int, statuses: dict[str, str], run_status: str
) -> None:
    dataset = write_dataset(tmp_path)
    session = Session()
    result: list[int] = []
    with serve(Handler) as fast, stuck_service() as stuck:
        specs = [make_spec(fast, name="a"), make_spec(stuck.url, name="b"), make_spec(fast, name="c")]
        args = run_args(dataset, tmp_path / "out", ["a", "b", "c"])
        engine = threading.Thread(target=lambda: result.append(run(args, specs, session)), daemon=True)
        engine.start()
        assert stuck.wait_for_request()
        # The report already covers the finished model while b is stuck.
        saved = json.loads((args.output / "summary.json").read_text())
        assert saved["run"]["status"] == "running"
        assert [(row["model"], row["status"]) for row in saved["models"]] == [("a", "ok")]
        requested = time.monotonic()
        if action == "skip":
            assert session.control.skip("b")
        else:
            session.control.abort()
        engine.join(timeout=10)
        assert time.monotonic() - requested < 5  # the first-request timeout is 60 s
    assert result == [exit_code]
    report = json.loads((args.output / "summary.json").read_text())
    assert report["run"]["status"] == run_status
    assert {row["model"]: row["status"] for row in report["models"]} == statuses
    skipped = next(row for row in report["models"] if row["model"] == "b")
    assert "first request" in skipped["skip_reason"]
    assert "Partial run" in (args.output / "summary.html").read_text()


def test_resume_reruns_only_unfinished_models_and_refuses_changed_settings(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path)
    hits: list[str] = []

    class Counting(Handler):
        def do_POST(self) -> None:
            hits.append(self.path)
            super().do_POST()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}/v1/systemone"
    with serve(Counting) as counted, serve(Handler) as fresh:
        out = tmp_path / "out"
        first = run_args(dataset, out, ["a", "b"], first_request_timeout=1)
        assert run(first, [make_spec(counted, name="a"), make_spec(dead, name="b")]) == 1
        created_at = json.loads((out / "summary.json").read_text())["run"]["created_at"]
        hits.clear()
        resumed = run_args(dataset, None, ["a", "b"], resume=out)
        assert run(resumed, [make_spec(counted, name="a"), make_spec(fresh, name="b")]) == 0
        assert hits == []
        changed = run_args(dataset, None, ["a", "b"], resume=out, passes=2)
        with pytest.raises(ValueError, match="--passes 1"):
            run(changed, [make_spec(counted, name="a"), make_spec(fresh, name="b")])
    report = json.loads((out / "summary.json").read_text())
    assert {row["model"]: row["status"] for row in report["models"]} == {"a": "ok", "b": "ok"}
    assert report["run"]["created_at"] == created_at
    assert len(report["run"]["resumed_at"]) == 1


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="the benchmark CLI only runs natively on Apple Silicon",
)
@pytest.mark.parametrize(
    ("models", "interrupts", "exit_code", "statuses"),
    [
        (["stuck", "fast"], 1, 1, {"stuck": "skipped", "fast": "ok"}),
        (["stuck", "stuck2"], 2, 130, {"stuck": "skipped", "stuck2": "skipped"}),
    ],
    ids=["once-skips", "twice-aborts"],
)
def test_ctrl_c_skips_the_current_model_and_twice_aborts(
    tmp_path: Path, models: list[str], interrupts: int, exit_code: int, statuses: dict[str, str]
) -> None:
    dataset = write_dataset(tmp_path)
    out = tmp_path / "out"
    with serve(Handler) as fast, stuck_service() as stuck:
        for name in ("stuck", "stuck2"):
            write_manifest(tmp_path / "models", name, stuck.url)
        write_manifest(tmp_path / "models", "fast", fast)
        proc = subprocess.Popen(
            [sys.executable, "benchmark.py", "run", "--model-dir", str(tmp_path / "models"),
             "--models", *models, "--dataset", str(dataset), "--passes", "1", "--warmup", "0",
             "--cached-repeats", "0", "--output", str(out)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for _ in range(interrupts):
                assert stuck.wait_for_request()
                proc.send_signal(signal.SIGINT)
            output, _ = proc.communicate(timeout=30)
        finally:
            proc.kill()
    assert proc.returncode == exit_code, output
    report = json.loads((out / "summary.json").read_text())
    assert {row["model"]: row["status"] for row in report["models"]} == statuses
