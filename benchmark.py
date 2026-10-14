#!/usr/bin/env python3
"""Apple Silicon decision-model benchmark with local and service adapters."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

from adapters import DecisionAdapter, load_adapter
from html_report import not_run, row_status, write_html_report, write_text_atomic
from model_registry import ModelSpec, ROOT, WORK, load_models
from run_control import (
    Activity,
    ModelDone,
    ModelSkipped,
    Progress,
    RunAborted,
    RunControl,
    RunDone,
    RunStarted,
    Session,
    call_cancellable,
    heartbeat,
    run_in_foreground,
)

RESULTS_ROOT = ROOT / "results"

def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)

def fmt_ms(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f} ms"

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read dataset {path}: {exc}") from exc
    with stream as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            where = f"{path}:{line_no}"
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: case must be a JSON object")
            case_id = obj.get("id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{where}: id must be a nonempty string")
            if case_id in seen_ids:
                raise ValueError(f"{where}: duplicate id {case_id!r}")
            seen_ids.add(case_id)
            state = obj.get("state")
            if not isinstance(state, (str, dict)) or not state:
                raise ValueError(f"{where}: state must be a nonempty string or object")
            questions = obj.get("questions")
            expected = obj.get("expected")
            if not isinstance(questions, dict) or not questions:
                raise ValueError(f"{where}: questions must be a nonempty object")
            if not isinstance(expected, dict) or set(expected) != set(questions):
                raise ValueError(f"{where}: expected must have exactly the same question ids as questions")
            for question_id, question in questions.items():
                if not isinstance(question_id, str) or not question_id:
                    raise ValueError(f"{where}: question ids must be nonempty strings")
                if not isinstance(question, dict):
                    raise ValueError(f"{where}: question {question_id!r} must be an object")
                qtype = question.get("type")
                instructions = question.get("instructions")
                if not isinstance(instructions, str) or not instructions.strip():
                    raise ValueError(f"{where}: question {question_id!r} needs instructions")
                label = expected[question_id]
                if qtype == "choice":
                    criteria = question.get("criteria")
                    if (
                        not isinstance(criteria, dict)
                        or not criteria
                        or any(
                            not isinstance(key, str) or not key
                            or not isinstance(value, str) or not value.strip()
                            for key, value in criteria.items()
                        )
                        or not isinstance(label, str)
                        or label not in criteria
                    ):
                        raise ValueError(f"{where}: choice {question_id!r} needs criteria containing its expected label")
                elif qtype == "noul":
                    if type(label) is not bool:
                        raise ValueError(f"{where}: noul {question_id!r} expected value must be boolean")
                elif qtype == "score":
                    criteria = question.get("criteria")
                    if (
                        not isinstance(criteria, list)
                        or not criteria
                        or any(not isinstance(value, str) or not value.strip() for value in criteria)
                        or type(label) is not int
                        or not 0 <= label < len(criteria)
                    ):
                        raise ValueError(f"{where}: score {question_id!r} needs criteria and a zero-based expected index")
                else:
                    raise ValueError(f"{where}: unsupported question type {qtype!r} for {question_id!r}")
            items.append(obj)
    if not items:
        raise ValueError(f"No benchmark cases found in {path}")
    return items

def request_model(
    req: urllib.request.Request, adapter: DecisionAdapter, timeout: float
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {req.full_url}: {detail[:1000]}") from e
    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response from {req.full_url}: {raw[:500]!r}") from e
    return adapter.normalize_response(data), elapsed_ms

def wait_for_port(
    host: str, port: int, proc: subprocess.Popen[Any], timeout_s: int, control: RunControl | None = None
) -> float:
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with status {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return (time.perf_counter() - start) * 1000
        except OSError:
            pass
        if control is None:
            time.sleep(0.25)
        else:
            control.sleep(0.25)
    raise TimeoutError(f"Server did not open port {port} within {timeout_s}s")

def descendant_rss_mb(root_pid: int) -> float | None:
    """Best-effort aggregate RSS for the root process and descendants."""
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    children: dict[int, list[int]] = {}
    rss_kb: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss = map(int, parts)
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss_kb[pid] = rss

    seen: set[int] = set()
    stack = [root_pid]
    total = 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss_kb.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total / 1024.0

def stop_process_group(proc: subprocess.Popen[Any], grace_s: float = 15) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

def normalize_probabilities(ans: dict[str, Any]) -> dict[str, float]:
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in probs.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return out

def extract_decision(qspec: dict[str, Any], ans: dict[str, Any]) -> tuple[Any, float | None, dict[str, float]]:
    qtype = qspec.get("type")
    probs = normalize_probabilities(ans)

    if qtype == "choice":
        pred = ans.get("choice")
        confidence = max(probs.values()) if probs else None
        return pred, confidence, probs

    if qtype == "noul":
        p = ans.get("noul")
        if p is None:
            return None, None, {}
        p = float(p)
        return bool(p >= 0.5), max(p, 1.0 - p), {"false": 1.0 - p, "true": p}

    if qtype == "score":
        if probs:
            # Keys are normally "0", "1", ...
            pred_key = max(probs, key=probs.get)
            try:
                pred: Any = int(pred_key)
            except ValueError:
                pred = pred_key
            return pred, max(probs.values()), probs
        score = ans.get("score")
        if score is None:
            return None, None, {}
        return int(round(float(score))), ans.get("confidence"), {}

    return None, None, {}

def one_hot_brier(probs: dict[str, float], expected_key: str) -> float | None:
    if not probs or expected_key not in probs:
        return None
    return sum((p - (1.0 if k == expected_key else 0.0)) ** 2 for k, p in probs.items())

def ece(rows: list[tuple[float, int]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    value = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(c, ok) for c, ok in rows if (lo <= c < hi) or (i == bins - 1 and c == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(ok for _, ok in bucket) / len(bucket)
        value += (len(bucket) / total) * abs(avg_acc - avg_conf)
    return value

def max_coverage_at_error(rows: list[tuple[float, int]], budget: float) -> float | None:
    """
    Empirical selective-classification coverage:
    sort decisions by confidence descending and return the largest prefix whose
    cumulative error rate is <= budget.
    """
    if not rows:
        return None
    ordered = sorted(rows, key=lambda x: x[0], reverse=True)
    errors = 0
    best_n = 0
    for i, (_, ok) in enumerate(ordered, 1):
        errors += 1 - ok
        if errors / i <= budget:
            best_n = i
    return best_n / len(ordered)

def evaluate_answers(case: dict[str, Any], response: dict[str, Any]) -> list[dict[str, Any]]:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return []

    rows: list[dict[str, Any]] = []
    expected = case["expected"]
    for qid, exp in expected.items():
        qspec = case["questions"].get(qid, {})
        ans = answers.get(qid, {})
        if not isinstance(ans, dict):
            ans = {}
        pred, conf, probs = extract_decision(qspec, ans)

        qtype = qspec.get("type")
        if qtype == "noul":
            exp_norm: Any = bool(exp)
            brier = None
            if probs:
                brier = (probs["true"] - (1.0 if exp_norm else 0.0)) ** 2
        elif qtype == "score":
            exp_norm = int(exp)
            brier = one_hot_brier(probs, str(exp_norm))
        else:
            exp_norm = exp
            brier = one_hot_brier(probs, str(exp_norm))

        correct = int(pred == exp_norm)
        rows.append({
            "case_id": case["id"],
            "question_id": qid,
            "type": qtype,
            "expected": exp_norm,
            "predicted": pred,
            "correct": correct,
            "confidence": conf,
            "brier": brier,
        })
    return rows

def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        try:
            info["macos_version"] = subprocess.check_output(
                ["sw_vers", "-productVersion"], text=True, timeout=10
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            raw = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType", "-json"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            hardware = json.loads(raw)["SPHardwareDataType"][0]
            # system_profiler also returns serial numbers and hardware UUIDs.
            info["chip_type"] = hardware.get("chip_type")
            info["physical_memory"] = hardware.get("physical_memory")
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            pass
    return info


def tee_output(
    stream: Any, log: Any, emit_line: Callable[[str], None], activity: Activity | None = None
) -> None:
    for line in stream:
        if activity is not None:
            activity.touch()
        if log is not None:
            try:
                log.write(line)
                log.flush()
            except (OSError, ValueError):  # the log closed while a stopped process was still talking
                log = None
        emit_line(line.rstrip())


def run_setup(spec: ModelSpec, session: Session | None = None) -> None:
    session = session or Session()
    control = session.control
    if spec.setup is None:
        session.stage(spec.name, "setup", "no setup command; using installed/running service")
        return
    command = spec.setup
    if not command.cwd.exists():
        raise RuntimeError(f"{spec.name}: setup working directory does not exist: {command.cwd}")
    WORK.mkdir(parents=True, exist_ok=True)
    log_path = WORK / f"{spec.name}.setup.log"
    session.stage(spec.name, "setup", f"starting; log: {log_path}")
    activity = Activity()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env={**os.environ, **command.env},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        control.track(proc)
        assert proc.stdout is not None
        reader = threading.Thread(
            target=tee_output,
            args=(proc.stdout, log, lambda line: session.log(spec.name, "setup", line), activity),
            daemon=True,
        )
        reader.start()
        try:
            with heartbeat(session, spec.name, "setup", activity, quiet_while_active=True):
                while proc.poll() is None:
                    control.sleep(0.2)
            reader.join(timeout=5)
            if proc.returncode:
                raise RuntimeError(f"{spec.name}: setup exited with status {proc.returncode}; see {log_path}")
        finally:
            if control.interrupted():
                session.stage(spec.name, "stopping", "stopping setup")
            stop_process_group(proc, grace_s=3 if control.interrupted() else 15)
            control.untrack(proc)
            reader.join(timeout=5)
    session.stage(spec.name, "setup", "complete")


def model_version(spec: ModelSpec) -> str | None:
    if spec.version is None or not spec.version.cwd.exists():
        return None
    try:
        return subprocess.check_output(
            spec.version.argv,
            cwd=spec.version.cwd,
            env={**os.environ, **spec.version.env},
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_answers(case: dict[str, Any], response: dict[str, Any]) -> None:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("Model response has no answers object")
    missing = set(case["questions"]) - set(answers)
    if missing:
        raise RuntimeError(f"Model response omitted answers: {', '.join(sorted(missing))}")
    for question_id, question in case["questions"].items():
        answer = answers[question_id]
        if not isinstance(answer, dict):
            raise RuntimeError(f"Model response has invalid answer for {question_id}")
        prediction, _, _ = extract_decision(question, answer)
        if prediction is None:
            raise RuntimeError(f"Model response has no decision for {question_id}")


def run_one_model(
    spec: ModelSpec,
    cases: list[dict[str, Any]],
    out_dir: Path,
    passes: int,
    warmup: int,
    cached_repeats: int,
    startup_timeout: int,
    first_request_timeout: int,
    *,
    session: Session | None = None,
) -> dict[str, Any]:
    session = session or Session()
    control = session.control

    def stage(name: str, detail: str = "") -> None:
        session.stage(spec.name, name, detail)

    def request(req: urllib.request.Request, timeout: float) -> tuple[dict[str, Any], float]:
        # Requests run on a worker thread so a skip never waits for a stuck HTTP call.
        return call_cancellable(control, request_model, req, adapter, timeout)

    log_path = out_dir / f"{spec.name}.server.log"
    raw_path = out_dir / f"{spec.name}.requests.jsonl"
    decisions_path = out_dir / f"{spec.name}.decisions.jsonl"
    adapter = load_adapter(spec.adapter)
    proc: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    startup_ms: float | None = None
    activity = Activity()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            if spec.launch is not None:
                command = spec.launch
                if not command.cwd.exists():
                    raise RuntimeError(f"{spec.name}: missing {command.cwd}; run setup first")
                if command.argv[0].startswith("/") and not Path(command.argv[0]).exists():
                    raise RuntimeError(f"{spec.name}: missing {command.argv[0]}; run setup first")
                if not command.argv[0].startswith("/") and shutil.which(command.argv[0]) is None:
                    raise RuntimeError(f"{spec.name}: command {command.argv[0]} not found")
                host = urlparse(spec.endpoint).hostname or "127.0.0.1"
                try:
                    with socket.create_connection((host, spec.port), timeout=0.5):
                        raise RuntimeError(f"{spec.name}: port {spec.port} is already in use")
                except OSError:
                    pass
                log.write(f"# command: {json.dumps(command.argv)}\n# cwd: {command.cwd}\n")
                log.flush()
                stage("launch", f"waiting for port {spec.port}")
                proc = subprocess.Popen(
                    command.argv,
                    cwd=command.cwd,
                    env={**os.environ, **command.env,
                         "BENCH_RUNTIME_REPORT": str((out_dir / f"{spec.name}.runtime.json").resolve())},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                    bufsize=1,
                )
                control.track(proc)
                assert proc.stdout is not None
                reader = threading.Thread(
                    target=tee_output,
                    args=(proc.stdout, log, lambda line: session.log(spec.name, "server", line), activity),
                    daemon=True,
                )
                reader.start()
                with heartbeat(session, spec.name, "launch", activity):
                    startup_ms = wait_for_port(host, spec.port, proc, startup_timeout, control)
                stage("ready", f"{startup_ms:.0f} ms to port open")
            else:
                log.write("# externally managed service; no server process started\n")
                stage("ready", "using existing service")

            # First request is recorded separately and never mixed into warm latency.
            stage("first request", "waiting for model load")
            with heartbeat(session, spec.name, "first request", activity if proc else None):
                first_response, first_ms = request(
                    adapter.build_request(cases[0], spec, "cold-first-request"), first_request_timeout
                )
            check_answers(cases[0], first_response)
            stage("first request", f"{first_ms:.0f} ms")

            # Warmup.
            stage("warmup", f"{warmup} requests")
            for i in range(warmup):
                case = cases[i % len(cases)]
                response, ms = request(adapter.build_request(case, spec, f"warmup-{i}"), 180)
                check_answers(case, response)
                session.emit(Progress(spec.name, "warmup", i + 1, warmup, ms, None))

            latencies: list[float] = []
            server_latencies: list[float] = []
            peak_rss_mb = (descendant_rss_mb(proc.pid) or 0.0) if proc else 0.0
            decision_rows: list[dict[str, Any]] = []
            total_requests = passes * len(cases)
            stage("measuring", f"{total_requests} requests")
            p50_at = 0.0
            p50: float | None = None
            with raw_path.open("w", encoding="utf-8") as raw_f, decisions_path.open("w", encoding="utf-8") as dec_f:
                request_n = 0
                for pass_i in range(passes):
                    for case_i, case in enumerate(cases):
                        request_n += 1
                        nonce = f"p{pass_i}-c{case_i}-n{request_n}"
                        response, elapsed_ms = request(adapter.build_request(case, spec, nonce), 180)
                        check_answers(case, response)
                        latencies.append(elapsed_ms)
                        if isinstance(response.get("latency_ms"), (int, float)):
                            server_latencies.append(float(response["latency_ms"]))

                        rss = descendant_rss_mb(proc.pid) if proc else None
                        if rss is not None:
                            peak_rss_mb = max(peak_rss_mb, rss)

                        raw_row = {
                            "model": spec.name,
                            "pass": pass_i,
                            "case_id": case["id"],
                            "latency_ms": elapsed_ms,
                            "rss_mb_after_request": rss,
                            "response": response,
                        }
                        raw_f.write(json.dumps(raw_row, ensure_ascii=False) + "\n")

                        for row in evaluate_answers(case, response):
                            row["model"] = spec.name
                            row["pass"] = pass_i
                            decision_rows.append(row)
                            dec_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        # Sorting for p50 on every request would be quadratic on large datasets.
                        now = time.monotonic()
                        if now - p50_at >= 0.5 or request_n == total_requests:
                            p50 = percentile(latencies, 0.5)
                            p50_at = now
                        session.emit(Progress(spec.name, "measure", request_n, total_requests, elapsed_ms, p50))

            # Explicit repeated-state/cache-friendly metric.
            cached_latencies: list[float] = []
            stage("cached", f"{cached_repeats} repeated requests")
            for i in range(cached_repeats):
                _, ms = request(adapter.build_request(cases[0], spec, None), 180)
                cached_latencies.append(ms)
                session.emit(Progress(spec.name, "cached", i + 1, cached_repeats, ms, None))

            correct = [r["correct"] for r in decision_rows]
            conf_rows = [
                (float(r["confidence"]), int(r["correct"]))
                for r in decision_rows
                if isinstance(r.get("confidence"), (int, float))
            ]
            briers = [
                float(r["brier"])
                for r in decision_rows
                if isinstance(r.get("brier"), (int, float))
            ]

            return {
                "model": spec.name,
                "status": "ok",
                "note": spec.note,
                "version": model_version(spec),
                "provenance": collect_provenance(spec, out_dir),
                "adapter": spec.adapter,
                "managed": spec.managed,
                "requests": len(latencies),
                "decisions": len(decision_rows),
                "startup_to_port_ms": startup_ms,
                "first_request_ms": first_ms,
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
                "mean_ms": statistics.mean(latencies) if latencies else None,
                "cached_p50_ms": percentile(cached_latencies, 0.50),
                "cached_p95_ms": percentile(cached_latencies, 0.95),
                "server_reported_p50_ms": percentile(server_latencies, 0.50),
                "peak_process_tree_rss_mb": peak_rss_mb if peak_rss_mb else None,
                "accuracy": (sum(correct) / len(correct)) if correct else None,
                "brier_mean": statistics.mean(briers) if briers else None,
                "ece_10bin": ece(conf_rows, bins=10),
                "coverage_at_1pct_error": max_coverage_at_error(conf_rows, 0.01),
                "coverage_at_5pct_error": max_coverage_at_error(conf_rows, 0.05),
                "errors": None,
            }
        finally:
            if proc is not None:
                if control.interrupted():
                    stage("stopping", "stopping server")
                stop_process_group(proc, grace_s=3 if control.interrupted() else 15)
                control.untrack(proc)
            if reader is not None:
                reader.join(timeout=5)

def collect_provenance(spec: ModelSpec, out_dir: Path) -> dict[str, Any]:
    """Retain configured pins even if a managed runtime fails before loading."""
    report: dict[str, Any] = {
        "configured": spec.provenance,
        "status": "unreported",
        "observed": {"state": "unknown"},
    }
    path = out_dir / f"{spec.name}.runtime.json"
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(observed, dict) or not isinstance(observed.get("observed"), dict):
                raise ValueError("invalid runtime report")
            report.update(observed)
            report["configured"] = spec.provenance
        except (OSError, ValueError) as exc:
            report["report_error"] = type(exc).__name__
    return report


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "model", "status", "version", "provenance_json", "adapter", "managed", "requests", "decisions",
        "startup_to_port_ms", "first_request_ms",
        "p50_ms", "p95_ms", "p99_ms", "mean_ms", "cached_p50_ms", "cached_p95_ms",
        "server_reported_p50_ms", "peak_process_tree_rss_mb", "accuracy",
        "brier_mean", "ece_10bin", "coverage_at_1pct_error",
        "coverage_at_5pct_error", "errors", "skip_reason", "note",
    ]
    buffer = io.StringIO(newline="")
    w = csv.DictWriter(buffer, fieldnames=fields)
    w.writeheader()
    for row in summaries:
        csv_row = {k: row.get(k) for k in fields}
        csv_row["status"] = row_status(row)
        csv_row["provenance_json"] = json.dumps(row.get("provenance", {}), sort_keys=True)
        w.writerow(csv_row)
    write_text_atomic(path, buffer.getvalue())


def provenance_markdown(summaries: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Model provenance", ""]
    for row in summaries:
        if row.get("provenance"):
            details = json.dumps(row["provenance"], sort_keys=True, ensure_ascii=False)
            lines.append(
                f"<details><summary>{escape(row['model'])} provenance</summary>"
                f"<pre>{escape(details)}</pre></details>"
            )

    return lines


def write_summary(
    out_dir: Path, summaries: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    write_summary_csv(out_dir / "summary.csv", summaries)

    write_text_atomic(
        out_dir / "summary.json",
        json.dumps({"run": metadata, "models": summaries}, indent=2, ensure_ascii=False),
    )
    valid = [r for r in summaries if row_status(r) == "ok" and r.get("p50_ms") is not None]
    latency_rank = sorted(valid, key=lambda r: r["p50_ms"])
    accuracy_rank = sorted(
        [r for r in valid if r.get("accuracy") is not None],
        key=lambda r: r["accuracy"],
        reverse=True,
    )

    lines = [
        "# Apple Silicon System One benchmark results",
        "",
        f"Dataset: **{metadata['dataset']['name']}** — "
        f"{metadata['dataset']['cases']} cases, "
        f"{metadata['dataset']['questions']} labeled decisions per pass. "
        f"SHA-256: {metadata['dataset']['sha256']}.",
        f"Profile: **{metadata['profile']}**"
        + (f" (excluding {', '.join(metadata['excluded'])})" if metadata.get("excluded") else "")
        + f". Selected models: {', '.join(metadata['models'])}.",
        *(
            [f"Run status: **{metadata['status']}**; this report covers the models that finished."]
            if metadata.get("status", "complete") != "complete" else []
        ),
        "",
        "Primary latency uses unique request metadata to reduce repeated-prefix cache effects.",
        "The `cached` columns deliberately repeat the exact same state and show cache-friendly behavior.",
        "",
        "## Summary",
        "",
        "| Model | p50 | p95 | p99 | Cached p50 | Peak RSS* | Accuracy** | ECE | ≤1% error coverage | ≤5% error coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if row_status(r) != "ok":
            label = "SKIPPED" if row_status(r) == "skipped" else "ERROR"
            lines.append(f"| {r['model']} | {label} | | | | | | | | |")
            continue
        pct = lambda x: "" if x is None else f"{100*x:.1f}%"
        rss = "" if r.get("peak_process_tree_rss_mb") is None else f"{r['peak_process_tree_rss_mb']:.0f} MB"
        ece_v = "" if r.get("ece_10bin") is None else f"{r['ece_10bin']:.3f}"
        lines.append(
            f"| {r['model']} | {fmt_ms(r['p50_ms'])} | {fmt_ms(r['p95_ms'])} | "
            f"{fmt_ms(r['p99_ms'])} | {fmt_ms(r['cached_p50_ms'])} | {rss} | "
            f"{pct(r.get('accuracy'))} | {ece_v} | {pct(r.get('coverage_at_1pct_error'))} | "
            f"{pct(r.get('coverage_at_5pct_error'))} |"
        )

    lines += [
        "",
        r"\* Peak RSS is a process-tree host-memory proxy for managed models; externally managed services show N/A.",
        "",
        r"\** Accuracy/ECE/coverage use the named dataset above. Small or unrepresentative datasets are not a basis for model selection.",
        "",
        "## Latency ranking",
        "",
    ]
    for i, r in enumerate(latency_rank, 1):
        lines.append(f"{i}. **{r['model']}** — p50 {r['p50_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms")

    lines += ["", "## Accuracy ranking on this dataset", ""]
    for i, r in enumerate(accuracy_rank, 1):
        lines.append(f"{i}. **{r['model']}** — {100*r['accuracy']:.1f}%")

    lines += provenance_markdown(summaries)

    failed = [r for r in summaries if row_status(r) == "failed"]
    if failed:
        lines += ["", "## Failures", ""]
        for row in failed:
            lines.append(f"- **{row['model']}**: {row['errors']}")
    skipped = [r for r in summaries if row_status(r) == "skipped"]
    if skipped:
        lines += ["", "## Skipped", ""]
        for row in skipped:
            lines.append(f"- **{row['model']}**: {row.get('skip_reason') or 'skipped'}")
    missing = not_run(summaries, metadata)
    if missing:
        reason = {
            "aborted": "the run was aborted first",
            "running": "the run is in progress or was interrupted",
        }.get(metadata.get("status", "complete"), "removed from the queue")
        lines += ["", "## Not run", "", f"{', '.join(missing)}: {reason}."]

    lines += [
        "",
        "## Interpretation",
        "",
        "- Use **p50/p95** for normal routing latency; do not rank by startup time.",
        "- Compare **cached p50** separately if your workload asks multiple decisions about the exact same state.",
        "- Prefer **coverage at a fixed error budget** over raw confidence values when deciding which requests can be auto-routed.",
        "- Re-run after changing quantization, runtime, model revision, macOS version, or calibration.",
        "",
    ]
    write_text_atomic(out_dir / "summary.md", "\n".join(lines))
    write_html_report(out_dir / "summary.html", summaries, metadata)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)

    def add_registry(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-dir", type=Path, action="append", default=[],
            help="additional directory of model TOML definitions",
        )

    def add_selection(parser: argparse.ArgumentParser) -> None:
        add_registry(parser)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--profile", choices=["light", "full", "extended"], default=None)
        group.add_argument("--models", nargs="+", metavar="ID")
        parser.add_argument(
            "--exclude", nargs="+", metavar="ID", default=[],
            help="leave these models out of the profile or --models selection",
        )

    setup = sub.add_parser("setup", help="install selected managed models")
    add_selection(setup)
    for action in ("run", "all"):
        runner = sub.add_parser(action, help="benchmark models" if action == "run" else "setup and benchmark")
        add_selection(runner)
        runner.add_argument("--dataset", type=Path, default=None, help="labeled JSONL (default: data/smoke.jsonl)")
        runner.add_argument("--passes", type=int, default=None, help="default: $BENCH_PASSES or 3")
        runner.add_argument("--warmup", type=int, default=None, help="default: $BENCH_WARMUP or 5")
        runner.add_argument(
            "--cached-repeats", type=int, default=None, help="default: $BENCH_CACHED_REPEATS or 20"
        )
        runner.add_argument(
            "--startup-timeout", type=int, default=int(os.getenv("BENCH_STARTUP_TIMEOUT", "1800"))
        )
        runner.add_argument(
            "--first-request-timeout",
            type=int,
            default=int(os.getenv("BENCH_FIRST_REQUEST_TIMEOUT", "1800")),
        )
        destination = runner.add_mutually_exclusive_group()
        destination.add_argument("--output", type=Path, help="new directory for this run")
        destination.add_argument(
            "--resume", type=Path, metavar="RUN_DIR",
            help="finish an earlier run: rerun its failed, skipped and unfinished models in place",
        )
        runner.add_argument(
            "--allow-remote", action="store_true",
            help="allow sending dataset content to non-loopback endpoints",
        )
    validate = sub.add_parser("validate", help="validate a labeled JSONL dataset without setup")
    validate.add_argument("--dataset", required=True, type=Path)
    models = sub.add_parser("models", help="list or validate model definitions")
    model_sub = models.add_subparsers(dest="models_action", required=True)
    listing = model_sub.add_parser("list")
    add_registry(listing)
    model_validate = model_sub.add_parser("validate")
    model_validate.add_argument("file", type=Path)
    return p.parse_args(argv)


def profile_members(specs: dict[str, ModelSpec], profile: str) -> list[str]:
    return [
        spec.name for spec in sorted(
            (spec for spec in specs.values() if profile in spec.profiles),
            key=lambda spec: (spec.order, spec.name),
        )
    ]


def selected_models(
    args: argparse.Namespace, specs: dict[str, ModelSpec], default: list[str] | None = None
) -> list[ModelSpec]:
    """--models or --profile (default: full, or a resumed run's models), minus --exclude."""
    exclude = list(getattr(args, "exclude", None) or [])
    unknown = [name for name in exclude if name not in specs]
    if unknown:
        raise ValueError(f"Unknown model(s) in --exclude: {', '.join(unknown)}")
    if args.models:
        unknown = [name for name in args.models if name not in specs]
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
        if len(set(args.models)) != len(args.models):
            raise ValueError("Duplicate model ids in --models")
        names = list(args.models)
    elif args.profile is None and default is not None:
        unknown = [name for name in default if name not in specs]
        if unknown:
            raise ValueError(
                f"The resumed run includes unknown model(s): {', '.join(unknown)}; "
                "pass --model-dir with their definitions or choose --models"
            )
        names = list(default)
    else:
        profile = args.profile or "full"
        names = profile_members(specs, profile)
        if not names:
            raise ValueError(f"No models in profile {profile}")
    names = [name for name in names if name not in exclude]
    if not names:
        raise ValueError("No models left to run after --exclude")
    return [specs[name] for name in names]


RUN_DEFAULTS = {
    "passes": ("BENCH_PASSES", 3),
    "warmup": ("BENCH_WARMUP", 5),
    "cached_repeats": ("BENCH_CACHED_REPEATS", 20),
}


def apply_run_defaults(args: argparse.Namespace, previous: dict[str, Any] | None = None) -> None:
    """Fill unset run options from the run being resumed, else from the environment."""
    for name, (variable, fallback) in RUN_DEFAULTS.items():
        if getattr(args, name, None) is None:
            value = previous.get(name) if previous else None
            setattr(args, name, value if type(value) is int else int(os.getenv(variable, str(fallback))))
    if getattr(args, "dataset", None) is None:
        args.dataset = ROOT / "data" / "smoke.jsonl"


def load_resume(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise ValueError(
            f"Cannot resume {run_dir}: it has no summary.json. Runs started before resume support "
            "cannot be resumed; start a new run and use --exclude to leave out a problem model."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume {run_dir}: unreadable summary.json ({exc})") from exc
    run_meta = data.get("run") if isinstance(data, dict) else None
    if (
        not isinstance(run_meta, dict)
        or not isinstance(run_meta.get("models"), list)
        or not isinstance(run_meta.get("dataset"), dict)
        or not isinstance(data.get("models"), list)
    ):
        raise ValueError(f"Cannot resume {run_dir}: summary.json is not a benchmark run summary")
    return data


def check_resumable(
    run_dir: Path, previous: dict[str, Any], dataset: dict[str, Any],
    args: argparse.Namespace, host: dict[str, Any],
) -> None:
    """Refuse to mix measurements taken under different conditions in one report."""
    old = previous["dataset"]
    if old.get("sha256") != dataset["sha256"]:
        raise ValueError(
            f"Cannot resume {run_dir}: it used dataset {old.get('name')} "
            f"(SHA-256 {str(old.get('sha256'))[:12]}…); pass that file with --dataset"
        )
    for name in RUN_DEFAULTS:
        if previous.get(name) != getattr(args, name):
            flag = "--" + name.replace("_", "-")
            raise ValueError(
                f"Cannot resume {run_dir}: it used {flag} {previous.get(name)}; omit {flag} or pass that value"
            )
    old_system = previous.get("system") or {}
    for key in ("chip_type", "physical_memory"):
        if old_system.get(key) and host.get(key) and old_system[key] != host[key]:
            raise ValueError(f"Cannot resume {run_dir}: it was recorded on different hardware ({old_system[key]})")


def resolve_selection(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> list[ModelSpec]:
    """Selected models plus run defaults; a resumed run supplies both unless overridden."""
    previous = load_resume(args.resume)["run"] if getattr(args, "resume", None) else None
    if getattr(args, "action", None) in ("run", "all"):
        apply_run_defaults(args, previous)
    return selected_models(args, specs, default=previous["models"] if previous else None)


def resumable_runs(root: Path = RESULTS_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / "summary.json").is_file()), key=lambda d: d.name, reverse=True)


def require_host(profile: str, session: Session | None = None) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Run setup and benchmarks natively on Apple Silicon macOS (arm64).")
    try:
        memory_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip())
        if profile == "full" and memory_bytes < 12 * 1024**3:
            note = "Note: the full profile may exceed available memory; use --profile light if a model cannot load."
            if session is None:
                print(note, flush=True)
            else:
                session.log(None, "harness", note)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunPlan:
    out_dir: Path
    cases: list[dict[str, Any]]
    metadata: dict[str, Any]
    specs: dict[str, ModelSpec]
    rows: dict[str, dict[str, Any]]
    setup: bool
    passes: int
    warmup: int
    cached_repeats: int
    startup_timeout: int
    first_request_timeout: int

    def ordered_rows(self) -> list[dict[str, Any]]:
        order = list(self.metadata["models"])
        return [self.rows[name] for name in order if name in self.rows] + [
            row for name, row in self.rows.items() if name not in order
        ]


def prepare_run(
    args: argparse.Namespace, selected: list[ModelSpec], session: Session | None = None
) -> RunPlan:
    """Validate everything that can fail before the first model starts, then create the run."""
    session = session or Session()
    for spec in selected:
        host = urlparse(spec.endpoint).hostname
        if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
            raise ValueError(
                f"{spec.name}: {spec.endpoint} is remote. "
                "Pass --allow-remote to send dataset content there."
            )
        load_adapter(spec.adapter)
    resume_dir: Path | None = getattr(args, "resume", None)
    previous = load_resume(resume_dir) if resume_dir else None
    apply_run_defaults(args, previous["run"] if previous else None)
    for name in ("passes", "startup_timeout", "first_request_timeout"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("warmup", "cached_repeats"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    cases = load_jsonl(args.dataset)
    dataset = {
        "name": args.dataset.name,
        "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "cases": len(cases),
        "questions": sum(len(case["questions"]) for case in cases),
        "smoke": args.dataset.resolve() == (ROOT / "data" / "smoke.jsonl").resolve(),
    }
    host = system_info()
    names = [spec.name for spec in selected]
    if previous is not None and resume_dir is not None:
        check_resumable(resume_dir, previous["run"], dataset, args, host)
        out_dir = resume_dir
        metadata = dict(previous["run"])
        metadata["models"] = list(metadata["models"]) + [n for n in names if n not in metadata["models"]]
        metadata["resumed_at"] = [*metadata.get("resumed_at", []), now_iso()]
        rows = {
            row["model"]: row for row in previous["models"]
            if isinstance(row, dict) and isinstance(row.get("model"), str)
        }
        queue = [name for name in names if name not in rows or row_status(rows[name]) != "ok"]
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out_dir = args.output or RESULTS_ROOT / stamp
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "system.json").write_text(json.dumps(host, indent=2), encoding="utf-8")
        metadata = {
            "created_at": now_iso(),
            "dataset": dataset,
            "profile": "custom" if args.models else (args.profile or "full"),
            "models": names,
            "excluded": list(getattr(args, "exclude", None) or []),
            "passes": args.passes,
            "warmup": args.warmup,
            "cached_repeats": args.cached_repeats,
            "system": host,
        }
        rows = {}
        queue = names
    session.control.set_queue(queue)
    return RunPlan(
        out_dir=out_dir,
        cases=cases,
        metadata=metadata,
        specs={spec.name: spec for spec in selected},
        rows=rows,
        setup=getattr(args, "action", "run") == "all",
        passes=args.passes,
        warmup=args.warmup,
        cached_repeats=args.cached_repeats,
        startup_timeout=args.startup_timeout,
        first_request_timeout=args.first_request_timeout,
    )


def incomplete_row(spec: ModelSpec, out_dir: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": spec.name,
        "status": status,
        "version": model_version(spec),
        "provenance": collect_provenance(spec, out_dir),
        "adapter": spec.adapter,
        "managed": spec.managed,
        "errors": reason if status == "failed" else None,
        "skip_reason": reason if status == "skipped" else None,
        "note": spec.note,
    }


def execute(plan: RunPlan, session: Session | None = None) -> int:
    """Benchmark the queued models one at a time, saving the report after each one.

    Returns 0 when every model finished, 1 when any failed or was skipped, 130 when aborted.
    """
    session = session or Session()
    control = session.control
    out_dir = plan.out_dir
    aborted = False

    def save(status: str) -> list[dict[str, Any]]:
        plan.metadata["status"] = status
        plan.metadata["updated_at"] = now_iso()
        rows = plan.ordered_rows()
        write_summary(out_dir, rows, plan.metadata)
        return rows

    save("running")
    session.emit(RunStarted(out_dir, dict(plan.metadata), tuple(control.queued())))
    try:
        while (name := control.next_model()) is not None:
            spec = plan.specs[name]
            session.stage(name, "begin", spec.note)
            try:
                if plan.setup:
                    run_setup(spec, session)
                row = run_one_model(
                    spec=spec,
                    cases=plan.cases,
                    out_dir=out_dir,
                    passes=plan.passes,
                    warmup=plan.warmup,
                    cached_repeats=plan.cached_repeats,
                    startup_timeout=plan.startup_timeout,
                    first_request_timeout=plan.first_request_timeout,
                    session=session,
                )
                accuracy = row["accuracy"]
                accuracy_label = "N/A" if accuracy is None else f"{accuracy * 100:.1f}%"
                session.stage(name, "complete", f"p50 {row['p50_ms']:.1f} ms; accuracy {accuracy_label}")
            except (ModelSkipped, RunAborted) as exc:
                aborted = isinstance(exc, RunAborted)
                row = incomplete_row(spec, out_dir, "skipped", str(exc))
                session.stage(name, "skipped", str(exc))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                row = incomplete_row(spec, out_dir, "failed", error)
                session.stage(name, "ERROR", error)
                log_path = out_dir / f"{spec.name}.server.log"
                if log_path.exists():
                    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
                        session.log(name, "log-tail", line)
            finally:
                control.finish_model()
            plan.rows[name] = row
            save("running")
            session.emit(ModelDone(name, row))
            if aborted:
                break
    except KeyboardInterrupt:
        aborted = True  # a synchronous caller pressed Ctrl+C; the current model gets no row
    finally:
        aborted = aborted or control.aborted
        status = "aborted" if aborted else "complete"
        rows = save(status)
        code = 130 if aborted else (1 if any(row_status(row) != "ok" for row in rows) else 0)
        session.emit(RunDone(out_dir, tuple(rows), status, code))
    return code


def run(args: argparse.Namespace, selected: list[ModelSpec], session: Session | None = None) -> int:
    session = session or Session()
    return execute(prepare_run(args, selected, session), session)


def setup_models(selected: list[ModelSpec], session: Session | None = None) -> int:
    """Set up each model in turn; a failure or skip moves on to the next model."""
    session = session or Session()
    control = session.control
    specs = {spec.name: spec for spec in selected}
    control.set_queue(specs)
    ready: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    try:
        while (name := control.next_model()) is not None:
            try:
                load_adapter(specs[name].adapter)
                run_setup(specs[name], session)
                ready.append(name)
            except (ModelSkipped, RunAborted) as exc:
                skipped.append(name)
                session.stage(name, "skipped", str(exc))
            except Exception as exc:
                failed.append(name)
                session.stage(name, "ERROR", f"{type(exc).__name__}: {exc}")
            finally:
                control.finish_model()
    except KeyboardInterrupt:
        control.abort()
    parts = [f"{len(ready)} ready"]
    for label, names in (("failed", failed), ("skipped", skipped), ("not run", control.queued())):
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    session.log(None, "harness", "Setup: " + "; ".join(parts))
    if control.aborted:
        return 130
    return 1 if failed or skipped else 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.action == "validate":
            cases = load_jsonl(args.dataset)
            count = sum(len(case["questions"]) for case in cases)
            print(f"Valid dataset: {len(cases)} cases, {count} labeled decisions")
            return 0
        if args.action == "models" and args.models_action == "validate":
            from model_registry import load_manifest

            spec = load_manifest(args.file)
            load_adapter(spec.adapter)
            print(f"Valid model: {spec.name} ({'managed' if spec.managed else 'existing service'})")
            return 0
        specs = load_models(args.model_dir)
        if args.action == "models":
            for spec in sorted(specs.values(), key=lambda item: (item.order, item.name)):
                print(
                    f"{spec.name:12} {'managed' if spec.managed else 'existing':8} "
                    f"{','.join(spec.profiles) or '-':12} {spec.display_name}"
                )
            return 0
        selected = resolve_selection(args, specs)
        session = Session()
        require_host("custom" if args.models else (args.profile or "full"), session)
        if args.action == "setup":
            for spec in selected:
                load_adapter(spec.adapter)
            return run_in_foreground(lambda: setup_models(selected, session), session)
        plan = prepare_run(args, selected, session)
        return run_in_foreground(lambda: execute(plan, session), session)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; managed server stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
