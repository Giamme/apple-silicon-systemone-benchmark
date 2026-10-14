"""Run events, reporters, and cooperative cancellation shared by the CLI and TUI."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Protocol, TypeVar, Union

T = TypeVar("T")


@dataclass(frozen=True)
class Log:
    model: str | None
    source: str  # harness | setup | server | log-tail
    text: str


@dataclass(frozen=True)
class Stage:
    model: str
    stage: str
    detail: str = ""


@dataclass(frozen=True)
class Heartbeat:
    model: str
    stage: str
    elapsed_s: float
    idle_s: float | None  # seconds since the last process output; None without a process
    stalled: bool


@dataclass(frozen=True)
class Progress:
    model: str
    phase: str  # warmup | measure | cached
    n: int
    total: int
    latency_ms: float
    p50_ms: float | None


@dataclass(frozen=True)
class ModelDone:
    model: str
    row: dict[str, Any]


@dataclass(frozen=True)
class RunStarted:
    out_dir: Path | None
    metadata: dict[str, Any]
    queue: tuple[str, ...]


@dataclass(frozen=True)
class RunDone:
    out_dir: Path | None
    rows: tuple[dict[str, Any], ...]
    status: str  # complete | aborted
    exit_code: int


Event = Union[Log, Stage, Heartbeat, Progress, ModelDone, RunStarted, RunDone]


class Reporter(Protocol):
    def emit(self, event: Event) -> None: ...


class ConsoleReporter:
    """Plain line-oriented output for the CLI and for logs captured by other tools."""

    skip_hint = "press Ctrl+C to skip"

    def __init__(self, progress_interval_s: float = 2.0) -> None:
        self._lock = threading.Lock()
        self._interval = progress_interval_s
        self._last_progress: dict[str, float] = {}

    def _print(self, text: str, err: bool = False) -> None:
        with self._lock:
            print(text, file=sys.stderr if err else sys.stdout, flush=True)

    def emit(self, event: Event) -> None:
        if isinstance(event, Stage):
            if event.stage == "measuring":
                self._last_progress[event.model] = time.monotonic()
            self._print(f"[{event.model}] {event.stage}{': ' + event.detail if event.detail else ''}")
        elif isinstance(event, Log):
            if event.source == "harness":
                self._print(event.text if event.model is None else f"[{event.model}] {event.text}")
            else:
                self._print(f"[{event.model}:{event.source}] {event.text.rstrip()}", err=event.source == "log-tail")
        elif isinstance(event, Heartbeat):
            detail = f"still waiting; {event.elapsed_s:.0f}s elapsed"
            if event.stalled:
                if event.idle_s is not None:
                    detail += f"; no server output for {event.idle_s:.0f}s"
                detail += f"; {self.skip_hint}"
            self._print(f"[{event.model}] {event.stage}: {detail}")
        elif isinstance(event, Progress):
            if event.phase != "measure":
                return
            now = time.monotonic()
            if now - self._last_progress.get(event.model, 0.0) >= self._interval or event.n == event.total:
                p50 = "N/A" if event.p50_ms is None else f"{event.p50_ms:.1f} ms"
                self._print(
                    f"[{event.model}] measuring: {event.n}/{event.total}; "
                    f"last {event.latency_ms:.1f} ms; p50 {p50}"
                )
                self._last_progress[event.model] = now
        elif isinstance(event, RunDone) and event.out_dir is not None:
            if event.status == "aborted":
                self._print("\nRun aborted; report written for finished models.")
            self._print(f"\nResults: {event.out_dir}\nOpen: {event.out_dir / 'summary.html'}")


class ModelSkipped(Exception):
    def __init__(self, model: str | None, stage: str | None, reason: str) -> None:
        super().__init__(reason)
        self.model = model
        self.stage = stage
        self.reason = reason


class RunAborted(Exception):
    def __init__(self, stage: str | None) -> None:
        super().__init__(f"run aborted during {stage or 'startup'}")
        self.stage = stage


class RunControl:
    """Mutable model queue plus skip/abort requests, shared by the engine and a UI."""

    def __init__(self, queue: Iterable[str] = ()) -> None:
        self._lock = threading.RLock()
        self._queue: list[str] = list(queue)
        self._current: str | None = None
        self._stage: str | None = None
        self._skip_reason: str | None = None
        self._aborted = False
        self._wake = threading.Event()
        self._procs: set[subprocess.Popen[Any]] = set()

    # Queue.
    def set_queue(self, names: Iterable[str]) -> None:
        with self._lock:
            self._queue = list(names)

    def queued(self) -> list[str]:
        with self._lock:
            return list(self._queue)

    def next_model(self) -> str | None:
        with self._lock:
            self._current = None
            self._stage = None
            self._skip_reason = None
            if self._aborted or not self._queue:
                return None
            self._wake.clear()
            self._current = self._queue.pop(0)
            return self._current

    def finish_model(self) -> None:
        with self._lock:
            self._current = None
            self._stage = None
            self._skip_reason = None
            if not self._aborted:
                self._wake.clear()

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._queue:
                return False
            self._queue.remove(name)
            return True

    def requeue(self, names: Iterable[str]) -> list[str]:
        with self._lock:
            added = [n for n in names if n not in self._queue and n != self._current]
            self._queue.extend(added)
            return added

    def reset_abort(self) -> None:
        with self._lock:
            self._aborted = False
            self._wake.clear()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    @property
    def stage(self) -> str | None:
        with self._lock:
            return self._stage

    def set_stage(self, model: str, stage: str) -> None:
        with self._lock:
            if model == self._current:
                self._stage = stage

    # Cancellation.
    def skip(self, model: str | None = None, reason: str | None = None) -> bool:
        """Skip the running model; a name guards against skipping the next one by accident."""
        with self._lock:
            if self._aborted or self._current is None or (model is not None and model != self._current):
                return False
            self._skip_reason = reason or f"skipped by user during {self._stage or 'startup'}"
            self._wake.set()
            return True

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._wake.set()

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    def interrupted(self) -> bool:
        with self._lock:
            return self._aborted or self._skip_reason is not None

    def check(self) -> None:
        with self._lock:
            if self._aborted:
                raise RunAborted(self._stage)
            if self._skip_reason is not None:
                raise ModelSkipped(self._current, self._stage, self._skip_reason)

    def sleep(self, seconds: float) -> None:
        self._wake.wait(seconds)
        self.check()

    # Processes, so a forced quit never leaves a model server behind.
    def track(self, proc: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._procs.add(proc)

    def untrack(self, proc: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._procs.discard(proc)

    def kill_all(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


@dataclass
class Session:
    reporter: Reporter = field(default_factory=ConsoleReporter)
    control: RunControl = field(default_factory=RunControl)
    stall_after_s: float = 60.0
    heartbeat_s: float = 15.0

    def emit(self, event: Event) -> None:
        self.reporter.emit(event)

    def stage(self, model: str, stage: str, detail: str = "") -> None:
        self.control.set_stage(model, stage)
        self.emit(Stage(model, stage, detail))

    def log(self, model: str | None, source: str, text: str) -> None:
        self.emit(Log(model, source, text))


class Activity:
    """Monotonic timestamp of the most recent process output."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def touch(self) -> None:
        self.last = time.monotonic()


@contextmanager
def heartbeat(
    session: Session, model: str, stage: str, activity: Activity | None = None,
    quiet_while_active: bool = False,
) -> Iterator[None]:
    """Periodic "still waiting" events; quiet_while_active stays silent while output flows."""
    stop = threading.Event()
    start = time.monotonic()

    def pulse() -> None:
        while not stop.wait(session.heartbeat_s):
            now = time.monotonic()
            idle = None if activity is None else now - activity.last
            if quiet_while_active and idle is not None and idle < session.heartbeat_s:
                continue
            stalled = (now - start if idle is None else idle) >= session.stall_after_s
            session.emit(Heartbeat(model, stage, now - start, idle, stalled))

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def call_cancellable(control: RunControl, fn: Callable[..., T], *args: Any, poll_s: float = 0.1) -> T:
    """Run a blocking call on a daemon thread and stop waiting for it on skip or abort.

    An abandoned call keeps running until it returns or times out; its result is discarded,
    so the callable must not write shared files.
    """
    control.check()
    done = threading.Event()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # re-raised on the caller's thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name="bench-call").start()
    while not done.wait(poll_s):
        control.check()
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_in_foreground(target: Callable[[], int], session: Session, abort_window_s: float = 3.0) -> int:
    """Run the engine on a worker thread so Ctrl+C can skip a model instead of killing the run.

    Ctrl+C once skips the current model; again within ``abort_window_s`` aborts the run (the
    report is still written); once more after that kills model servers and exits immediately.
    """
    control = session.control
    done = threading.Event()
    result: dict[str, Any] = {}

    def body() -> None:
        try:
            result["code"] = target()
        except BaseException as exc:
            result["error"] = exc
        finally:
            done.set()

    atexit.register(control.kill_all)
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, lambda *_: control.abort())
    if sys.stdin.isatty() and sys.stderr.isatty():
        print(
            "Ctrl+C skips the current model; twice within "
            f"{abort_window_s:.0f}s aborts the run and still writes the report.",
            file=sys.stderr, flush=True,
        )
    threading.Thread(target=body, name="bench-engine", daemon=True).start()
    last_interrupt: float | None = None
    try:
        while not done.is_set():
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                now = time.monotonic()
                model = control.current
                if control.aborted:
                    print("\nForce quitting; model servers killed.", file=sys.stderr, flush=True)
                    control.kill_all()
                    os._exit(130)
                if (last_interrupt is not None and now - last_interrupt <= abort_window_s) or model is None:
                    control.abort()
                    session.log(None, "harness", "Aborting: stopping the current model and writing the report.")
                elif control.skip(model):
                    session.log(
                        model, "harness",
                        f"skip requested; press Ctrl+C again within {abort_window_s:.0f}s to abort the run",
                    )
                else:
                    control.abort()
                    session.log(None, "harness", "Aborting: stopping the current model and writing the report.")
                last_interrupt = now
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if "error" in result:
        raise result["error"]
    return int(result.get("code", 130))
