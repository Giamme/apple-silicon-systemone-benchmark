#!/usr/bin/env python3
"""Apple Silicon decision-model benchmark with local and service adapters."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
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
from typing import Any, Iterator
from urllib.parse import urlparse

from adapters import DecisionAdapter, load_adapter
from html_report import write_html_report
from model_registry import ModelSpec, ROOT, WORK, load_models

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

def wait_for_port(host: str, port: int, proc: subprocess.Popen[Any], timeout_s: int) -> float:
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
        time.sleep(0.25)
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

def stop_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
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


def print_progress(model: str, stage: str, detail: str = "") -> None:
    print(f"[{model}] {stage}{': ' + detail if detail else ''}", flush=True)


@contextmanager
def heartbeat(model: str, stage: str, interval_s: float = 15) -> Iterator[None]:
    stop = threading.Event()
    start = time.monotonic()

    def pulse() -> None:
        while not stop.wait(interval_s):
            print_progress(model, stage, f"still waiting; {time.monotonic() - start:.0f}s elapsed")

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def tee_output(stream: Any, log: Any, prefix: str) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def run_setup(spec: ModelSpec) -> None:
    if spec.setup is None:
        print_progress(spec.name, "setup", "no setup command; using installed/running service")
        return
    command = spec.setup
    if not command.cwd.exists():
        raise RuntimeError(f"{spec.name}: setup working directory does not exist: {command.cwd}")
    WORK.mkdir(parents=True, exist_ok=True)
    log_path = WORK / f"{spec.name}.setup.log"
    print_progress(spec.name, "setup", f"starting; log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env={**os.environ, **command.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            tee_output(proc.stdout, log, f"{spec.name}:setup")
            status = proc.wait()
            if status:
                raise RuntimeError(f"{spec.name}: setup exited with status {status}; see {log_path}")
        finally:
            stop_process_group(proc)
    print_progress(spec.name, "setup", "complete")


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
) -> dict[str, Any]:
    log_path = out_dir / f"{spec.name}.server.log"
    raw_path = out_dir / f"{spec.name}.requests.jsonl"
    decisions_path = out_dir / f"{spec.name}.decisions.jsonl"
    adapter = load_adapter(spec.adapter)
    proc: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    startup_ms: float | None = None
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
                print_progress(spec.name, "launch", f"waiting for port {spec.port}")
                proc = subprocess.Popen(
                    command.argv,
                    cwd=command.cwd,
                    env={**os.environ, **command.env},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                reader = threading.Thread(
                    target=tee_output, args=(proc.stdout, log, f"{spec.name}:server"), daemon=True
                )
                reader.start()
                with heartbeat(spec.name, "launch"):
                    startup_ms = wait_for_port(host, spec.port, proc, startup_timeout)
                print_progress(spec.name, "ready", f"{startup_ms:.0f} ms to port open")
            else:
                log.write("# externally managed service; no server process started\n")
                print_progress(spec.name, "ready", "using existing service")

            # First request is recorded separately and never mixed into warm latency.
            print_progress(spec.name, "first request", "waiting for model load")
            with heartbeat(spec.name, "first request"):
                first_response, first_ms = request_model(
                    adapter.build_request(cases[0], spec, "cold-first-request"),
                    adapter,
                    timeout=first_request_timeout,
                )
            check_answers(cases[0], first_response)
            print_progress(spec.name, "first request", f"{first_ms:.0f} ms")

            # Warmup.
            print_progress(spec.name, "warmup", f"{warmup} requests")
            for i in range(warmup):
                case = cases[i % len(cases)]
                response, _ = request_model(
                    adapter.build_request(case, spec, f"warmup-{i}"), adapter, timeout=180
                )
                check_answers(case, response)

            latencies: list[float] = []
            server_latencies: list[float] = []
            peak_rss_mb = (descendant_rss_mb(proc.pid) or 0.0) if proc else 0.0
            decision_rows: list[dict[str, Any]] = []
            total_requests = passes * len(cases)
            print_progress(spec.name, "measuring", f"{total_requests} requests")
            last_update = time.monotonic()
            with raw_path.open("w", encoding="utf-8") as raw_f, decisions_path.open("w", encoding="utf-8") as dec_f:
                request_n = 0
                for pass_i in range(passes):
                    for case_i, case in enumerate(cases):
                        request_n += 1
                        nonce = f"p{pass_i}-c{case_i}-n{request_n}"
                        response, elapsed_ms = request_model(
                            adapter.build_request(case, spec, nonce), adapter, timeout=180
                        )
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
                        now = time.monotonic()
                        if now - last_update >= 2 or request_n == total_requests:
                            print_progress(
                                spec.name,
                                "measuring",
                                f"{request_n}/{total_requests}; last {elapsed_ms:.1f} ms; "
                                f"p50 {percentile(latencies, 0.5):.1f} ms",
                            )
                            last_update = now

            # Explicit repeated-state/cache-friendly metric.
            cached_latencies: list[float] = []
            print_progress(spec.name, "cached", f"{cached_repeats} repeated requests")
            for _ in range(cached_repeats):
                _, ms = request_model(
                    adapter.build_request(cases[0], spec, None), adapter, timeout=180
                )
                cached_latencies.append(ms)

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
                "note": spec.note,
                "version": model_version(spec),
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
                stop_process_group(proc)
            if reader is not None:
                reader.join(timeout=5)

def write_summary(
    out_dir: Path, summaries: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    fields = [
        "model", "version", "adapter", "managed", "requests", "decisions",
        "startup_to_port_ms", "first_request_ms",
        "p50_ms", "p95_ms", "p99_ms", "mean_ms", "cached_p50_ms", "cached_p95_ms",
        "server_reported_p50_ms", "peak_process_tree_rss_mb", "accuracy",
        "brier_mean", "ece_10bin", "coverage_at_1pct_error",
        "coverage_at_5pct_error", "errors", "note",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row.get(k) for k in fields})

    (out_dir / "summary.json").write_text(
        json.dumps({"run": metadata, "models": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    valid = [r for r in summaries if not r.get("errors") and r.get("p50_ms") is not None]
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
        f"Profile: **{metadata['profile']}**. Selected models: "
        f"{', '.join(metadata['models'])}.",
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
        if r.get("errors"):
            lines.append(f"| {r['model']} | ERROR | | | | | | | | |")
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

    failed = [r for r in summaries if r.get("errors")]
    if failed:
        lines += ["", "## Failures", ""]
        for row in failed:
            lines.append(f"- **{row['model']}**: {row['errors']}")

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
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
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
        group.add_argument("--profile", choices=["light", "full"], default=None)
        group.add_argument("--models", nargs="+", metavar="ID")

    setup = sub.add_parser("setup", help="install selected managed models")
    add_selection(setup)
    for action in ("run", "all"):
        runner = sub.add_parser(action, help="benchmark models" if action == "run" else "setup and benchmark")
        add_selection(runner)
        runner.add_argument("--dataset", type=Path, default=ROOT / "data" / "smoke.jsonl")
        runner.add_argument("--passes", type=int, default=int(os.getenv("BENCH_PASSES", "3")))
        runner.add_argument("--warmup", type=int, default=int(os.getenv("BENCH_WARMUP", "5")))
        runner.add_argument(
            "--cached-repeats", type=int, default=int(os.getenv("BENCH_CACHED_REPEATS", "20"))
        )
        runner.add_argument(
            "--startup-timeout", type=int, default=int(os.getenv("BENCH_STARTUP_TIMEOUT", "1800"))
        )
        runner.add_argument(
            "--first-request-timeout",
            type=int,
            default=int(os.getenv("BENCH_FIRST_REQUEST_TIMEOUT", "1800")),
        )
        runner.add_argument("--output", type=Path, help="new directory for this run")
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


def selected_models(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> list[ModelSpec]:
    if args.models:
        unknown = [name for name in args.models if name not in specs]
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
        if len(set(args.models)) != len(args.models):
            raise ValueError("Duplicate model ids in --models")
        return [specs[name] for name in args.models]
    profile = args.profile or "full"
    selected = sorted(
        (spec for spec in specs.values() if profile in spec.profiles),
        key=lambda spec: (spec.order, spec.name),
    )
    if not selected:
        raise ValueError(f"No models in profile {profile}")
    return selected


def require_host(profile: str) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Run setup and benchmarks natively on Apple Silicon macOS (arm64).")
    try:
        memory_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip())
        if profile == "full" and memory_bytes < 12 * 1024**3:
            print(
                "Note: the full profile may exceed available memory; "
                "use --profile light if a model cannot load.",
                flush=True,
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass


def run(args: argparse.Namespace, selected: list[ModelSpec]) -> int:
    for spec in selected:
        host = urlparse(spec.endpoint).hostname
        if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
            raise ValueError(
                f"{spec.name}: {spec.endpoint} is remote. "
                "Pass --allow-remote to send dataset content there."
            )
        load_adapter(spec.adapter)
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
    if args.action == "all":
        for spec in selected:
            run_setup(spec)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = args.output or RESULTS_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    host = system_info()
    (out_dir / "system.json").write_text(json.dumps(host, indent=2), encoding="utf-8")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "profile": "custom" if args.models else (args.profile or "full"),
        "models": [spec.name for spec in selected],
        "passes": args.passes,
        "warmup": args.warmup,
        "cached_repeats": args.cached_repeats,
        "system": host,
    }
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        print_progress(spec.name, "begin", spec.note)
        try:
            result = run_one_model(
                spec=spec,
                cases=cases,
                out_dir=out_dir,
                passes=args.passes,
                warmup=args.warmup,
                cached_repeats=args.cached_repeats,
                startup_timeout=args.startup_timeout,
                first_request_timeout=args.first_request_timeout,
            )
            summaries.append(result)
            accuracy = result["accuracy"]
            accuracy_label = "N/A" if accuracy is None else f"{accuracy * 100:.1f}%"
            print_progress(
                spec.name, "complete",
                f"p50 {result['p50_ms']:.1f} ms; accuracy {accuracy_label}",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            summaries.append({
                "model": spec.name,
                "version": model_version(spec),
                "adapter": spec.adapter,
                "managed": spec.managed,
                "errors": error,
                "note": spec.note,
            })
            print_progress(spec.name, "ERROR", error)
            log_path = out_dir / f"{spec.name}.server.log"
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                for line in tail:
                    print(f"[{spec.name}:log-tail] {line}", file=sys.stderr, flush=True)
    write_summary(out_dir, summaries, metadata)
    print(f"\nResults: {out_dir}\nOpen: {out_dir / 'summary.html'}", flush=True)
    return 1 if any(row.get("errors") for row in summaries) else 0


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
        selected = selected_models(args, specs)
        require_host("custom" if args.models else (args.profile or "full"))
        if args.action == "setup":
            for spec in selected:
                load_adapter(spec.adapter)
                run_setup(spec)
            return 0
        return run(args, selected)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; managed server stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
