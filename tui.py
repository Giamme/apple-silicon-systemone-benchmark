#!/usr/bin/env python3
"""Interactive terminal dashboard for the Apple Silicon System One benchmark.

The benchmark engine stays in benchmark.py; this module only drives it and shows its events.
Start it with ./benchmark.sh, which provides a pinned Textual through uv.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
from typing import Any, Callable

try:
    from rich.text import Text
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen, Screen
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Select,
        SelectionList,
        Sparkline,
        Static,
        Switch,
        TabbedContent,
        TabPane,
    )
except ImportError:
    print(
        "The interactive dashboard needs Textual. Start it with ./benchmark.sh (uv provides a "
        "pinned Textual), or run ./benchmark.sh all for the plain CLI.",
        file=sys.stderr,
    )
    raise SystemExit(2)

import benchmark
from html_report import not_run, row_status
from model_registry import ROOT, ModelSpec, load_models
from run_control import (
    Event,
    Heartbeat,
    Log,
    ModelDone,
    Progress,
    RunControl,
    RunDone,
    RunStarted,
    Session,
    Stage,
)

HOST_ERRORS = (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError)
# Model accents avoid red, orange and yellow, which mean failed, stopping and skipped.
MODEL_COLORS = (
    "cyan", "medium_purple1", "spring_green2", "deep_sky_blue1", "plum2", "aquamarine1",
    "light_sky_blue1", "orchid", "turquoise2", "pale_green1", "slate_blue1", "light_cyan1",
)
STATUS_STYLE = {
    "queued": ("queued", "dim"),
    "running": ("running", "bold yellow"),
    "ok": ("done", "bold green"),
    "failed": ("failed", "bold red"),
    "skipped": ("skipped", "yellow"),
    "removed": ("removed", "dim"),
    "not run": ("not run", "dim"),
}
STAGE_STYLE = {"complete": "bold green", "skipped": "bold yellow", "ERROR": "bold red", "stopping": "yellow"}
EIGHTHS = " ▏▎▍▌▋▊▉"
P50_STYLE = "bold #6c9eff"
P95_STYLE = "#3a5a99"
GOOD_STYLE = "#3ec7a8"
LOW_STYLE = "#d9a53f"


# Charts: pure functions of finished summary rows, redrawn at whatever width the pane has.

def _bar(value: float, scale: float, width: int) -> str:
    if width <= 0 or scale <= 0:
        return ""
    cells = max(0.0, min(1.0, value / scale)) * width
    full = int(cells)
    rest = int((cells - full) * 8)
    return "█" * full + (EIGHTHS[rest] if rest and full < width else "")


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _layout(rows: list[dict[str, Any]], width: int, value_width: int) -> tuple[int, int]:
    name_width = min(16, max(len(str(row["model"])) for row in rows))
    return name_width, max(4, width - name_width - 1 - value_width)


def _empty(message: str) -> Text:
    return Text(message, style="dim")


def latency_chart(rows: list[dict[str, Any]], width: int) -> Text:
    done = [row for row in rows if _number(row, "p50_ms") is not None]
    if not done:
        return _empty("No finished models yet. Warm p50/p95 latency appears here as each model completes.")
    value_width = 22
    name_width, bar_width = _layout(done, width, value_width)
    scale = max(_number(row, "p95_ms") or _number(row, "p50_ms") or 0.0 for row in done) or 1.0
    out = Text()
    out.append("Warm latency  ", style="bold")
    out.append("█ p50  ", style=P50_STYLE)
    out.append("░ p95", style=P95_STYLE)
    out.append("   ms, shorter is better\n\n", style="dim")
    for row in sorted(done, key=lambda item: item["p50_ms"]):
        p50 = _number(row, "p50_ms") or 0.0
        p95 = _number(row, "p95_ms") or p50
        solid = _bar(p50, scale, bar_width)
        extension = max(0, round(min(1.0, p95 / scale) * bar_width) - len(solid))
        out.append(f"{str(row['model'])[:name_width]:<{name_width}} ", style="bold")
        out.append(solid, style=P50_STYLE)
        out.append("░" * extension, style=P95_STYLE)
        out.append(" " * max(0, bar_width - len(solid) - extension))
        out.append(f"{p50:.1f} / {p95:.1f} ms".rjust(value_width) + "\n")
    cached = [row for row in done if _number(row, "cached_p50_ms") is not None]
    if cached:
        scale = max(_number(row, "cached_p50_ms") or 0.0 for row in cached) or 1.0
        out.append("\nCached p50  ", style="bold")
        out.append("the same input repeated\n\n", style="dim")
        for row in sorted(cached, key=lambda item: item["cached_p50_ms"]):
            value = _number(row, "cached_p50_ms") or 0.0
            solid = _bar(value, scale, bar_width)
            out.append(f"{str(row['model'])[:name_width]:<{name_width}} ", style="bold")
            out.append(solid, style=P95_STYLE)
            out.append(" " * max(0, bar_width - len(solid)))
            out.append(f"{value:.1f} ms".rjust(value_width) + "\n")
    return out


def quality_chart(rows: list[dict[str, Any]], width: int) -> Text:
    done = [row for row in rows if _number(row, "p50_ms") is not None]
    if not done:
        return _empty("No finished models yet. Accuracy, coverage and calibration appear here as models complete.")
    value_width = 8
    name_width, bar_width = _layout(done, width, value_width)
    out = Text()
    sections: list[tuple[str, str, str, bool, Callable[[float], str]]] = [
        ("accuracy", "Accuracy", "higher is better", True, lambda v: f"{v * 100:.1f}%"),
        ("coverage_at_5pct_error", "Coverage at ≤5% error", "share of decisions safe to automate", True,
         lambda v: f"{v * 100:.1f}%"),
        ("coverage_at_1pct_error", "Coverage at ≤1% error", "", True, lambda v: f"{v * 100:.1f}%"),
        ("ece_10bin", "Calibration error (ECE)", "lower is better", False, lambda v: f"{v:.3f}"),
        ("brier_mean", "Brier score", "lower is better", False, lambda v: f"{v:.3f}"),
    ]
    for key, title, note, higher_better, fmt in sections:
        values = [(row, _number(row, key)) for row in done]
        values = [(row, value) for row, value in values if value is not None]
        if not values:
            continue
        scale = 1.0 if higher_better else (max(value for _, value in values) or 1.0)
        out.append(title, style="bold")
        out.append(f"  {note}\n" if note else "\n", style="dim")
        for row, value in sorted(values, key=lambda item: item[1], reverse=higher_better):
            solid = _bar(value, scale, bar_width)
            out.append(f"{str(row['model'])[:name_width]:<{name_width}} ", style="bold")
            out.append(solid, style=GOOD_STYLE if higher_better else LOW_STYLE)
            out.append(" " * max(0, bar_width - len(solid)))
            out.append(f"{fmt(value):>{value_width}}\n")
        out.append("\n")
    return out


def fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def fmt_ms(value: Any) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else ""


def fmt_pct(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else ""


class Chart(Widget):
    DEFAULT_CSS = "Chart { height: auto; padding: 1 2; }"

    def __init__(self, draw: Callable[[list[dict[str, Any]], int], Text], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.draw = draw
        self.rows: list[dict[str, Any]] = []

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.refresh(layout=True)

    def render(self) -> Text:
        return self.draw(self.rows, max(20, self.content_size.width))

    def on_resize(self) -> None:
        self.refresh(layout=True)


# Engine bridge.

class TuiReporter:
    """Engine and output-reader threads only enqueue; the UI drains in batches."""

    def __init__(self) -> None:
        self.events: queue.SimpleQueue[Event] = queue.SimpleQueue()

    def emit(self, event: Event) -> None:
        self.events.put(event)


@dataclass
class ModelState:
    name: str
    status: str = "queued"
    stage: str = ""
    detail: str = ""
    hint: str = ""
    started: float | None = None
    finished: float | None = None
    stage_started: float | None = None
    phase: str = ""
    n: int = 0
    total: int = 0
    p50: float | None = None
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    row: dict[str, Any] | None = None

    def reset(self) -> None:
        self.status, self.stage, self.detail, self.hint, self.phase = "queued", "", "", "", ""
        self.started = self.finished = self.stage_started = self.p50 = None
        self.n = self.total = 0
        self.latencies.clear()


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    #dialog { width: 64; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    #dialog .detail { color: $text-muted; margin: 1 0; }
    #dialog Horizontal { height: auto; align: right middle; }
    #dialog Button { margin-left: 2; }
    """

    def __init__(self, question: str, detail: str = "") -> None:
        super().__init__()
        self.question = question
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text(self.question, style="bold"))
            yield Static(self.detail, classes="detail", markup=False)
            with Horizontal():
                yield Button("Yes (y)", variant="warning", id="yes")
                yield Button("No (n)", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


# Pre-run configuration.

PROFILES = [
    ("Full profile", "full"),
    ("Light profile", "light"),
    ("Extended profile", "extended"),
    ("Custom selection", "custom"),
]


class SetupScreen(Screen[None]):
    BINDINGS = [Binding("ctrl+s", "start", "Start", priority=True)]
    DEFAULT_CSS = """
    SetupScreen #setup { height: 1fr; }
    SetupScreen #models-pane { width: 1fr; padding: 1 2; }
    SetupScreen #options-pane { width: 1fr; padding: 1 2; }
    SetupScreen #models { height: 1fr; }
    SetupScreen Label.heading { text-style: bold; margin-top: 1; }
    SetupScreen .row { height: auto; }
    SetupScreen .row > Vertical { width: 1fr; height: auto; }
    SetupScreen .muted { color: $text-muted; }
    SetupScreen #setup-error { color: $error; margin-top: 1; }
    SetupScreen #start { margin-top: 1; width: 100%; }
    SetupScreen Switch { margin-right: 1; }
    SetupScreen .row > Static { width: 1fr; }
    """

    def __init__(self, specs: dict[str, ModelSpec], model_dirs: list[Path]) -> None:
        super().__init__()
        self.specs = specs
        self.model_dirs = model_dirs
        self.ordered = sorted(specs.values(), key=lambda spec: (spec.order, spec.name))
        self.resume_runs = benchmark.resumable_runs()

    def compose(self) -> ComposeResult:
        full = set(benchmark.profile_members(self.specs, "full"))
        yield Header()
        with Horizontal(id="setup"):
            with Vertical(id="models-pane"):
                yield Label("Models", classes="heading")
                yield Select(PROFILES, value="full", allow_blank=False, id="profile")
                yield SelectionList[str](
                    *[(self.model_label(spec), spec.name, spec.name in full) for spec in self.ordered],
                    id="models",
                )
                yield Static(id="model-count", classes="muted")
            with VerticalScroll(id="options-pane"):
                yield Label("Dataset", classes="heading")
                yield Input(value="data/smoke.jsonl", id="dataset")
                yield Static(id="dataset-info", classes="muted")
                with Horizontal(classes="row"):
                    for key, label in (("passes", "Passes"), ("warmup", "Warmup"), ("cached", "Cached repeats")):
                        with Vertical():
                            yield Label(label, classes="heading")
                            yield Input(placeholder="default", type="integer", id=key)
                yield Label("Setup", classes="heading")
                with Horizontal(classes="row"):
                    yield Switch(value=True, id="setup-switch")
                    yield Static(
                        "Install or update each model just before it runs. Turn off to reuse what is "
                        "already set up.", classes="muted",
                    )
                yield Label("Resume an earlier run", classes="heading")
                yield Select(
                    [(self.run_label(path), str(path)) for path in self.resume_runs],
                    prompt="Start a new run", id="resume", disabled=not self.resume_runs,
                )
                yield Static(id="resume-info", classes="muted")
                yield Button("Start benchmark", variant="primary", id="start")
                yield Static(id="setup-error")
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = "choose models and options"
        self.update_count()
        self.check_dataset()

    def on_screen_resume(self) -> None:
        """Back from a run: offer it (and any other new run) for resuming."""
        self.app.sub_title = "choose models and options"
        self.resume_runs = benchmark.resumable_runs()
        resume = self.query_one("#resume", Select)
        resume.set_options([(self.run_label(path), str(path)) for path in self.resume_runs])
        resume.disabled = not self.resume_runs

    @staticmethod
    def model_label(spec: ModelSpec) -> Text:
        label = Text(f"{spec.name:<14}", style="bold")
        label.append(f"{'managed' if spec.managed else 'existing':<9}", style="dim")
        label.append(" ".join(spec.profiles) or "-", style="dim italic")
        return label

    @staticmethod
    def run_label(path: Path) -> str:
        try:
            data = benchmark.load_resume(path)
        except ValueError:
            return f"{path.name} (unreadable)"
        rows = data["models"]
        ok = sum(1 for row in rows if row_status(row) == "ok")
        return f"{path.name} · {data['run'].get('status', 'complete')} · {ok}/{len(data['run']['models'])} ok"

    def selection(self) -> list[str]:
        chosen = set(self.query_one("#models", SelectionList).selected)
        return [spec.name for spec in self.ordered if spec.name in chosen]

    def update_count(self) -> None:
        names = self.selection()
        profile = self.query_one("#profile", Select).value
        text = f"{len(names)} of {len(self.ordered)} models selected"
        if profile in ("full", "light", "extended"):
            missing = [n for n in benchmark.profile_members(self.specs, str(profile)) if n not in names]
            if missing:
                text += f" · excluding {', '.join(missing)}"
        self.query_one("#model-count", Static).update(text)

    @on(Select.Changed, "#profile")
    def profile_changed(self, event: Select.Changed) -> None:
        if event.value in ("full", "light", "extended"):
            members = benchmark.profile_members(self.specs, str(event.value))
            selection = self.query_one("#models", SelectionList)
            selection.deselect_all()
            for name in members:
                selection.select(name)
        self.update_count()

    @on(SelectionList.SelectedChanged, "#models")
    def models_changed(self) -> None:
        self.update_count()

    def dataset_path(self) -> Path:
        path = Path(self.query_one("#dataset", Input).value.strip()).expanduser()
        if not path.is_absolute() and not path.exists():
            path = ROOT / path
        return path.resolve()

    @on(Input.Changed, "#dataset")
    def check_dataset(self) -> None:
        info = self.query_one("#dataset-info", Static)
        path = self.dataset_path()
        if not path.is_file():
            info.update(Text("File not found", style="red"))
            return
        try:
            cases = benchmark.load_jsonl(path)
        except ValueError as exc:
            info.update(Text(str(exc), style="red"))
            return
        questions = sum(len(case["questions"]) for case in cases)
        info.update(f"{len(cases)} cases · {questions} labeled decisions per pass")

    @on(Select.Changed, "#resume")
    def resume_changed(self, event: Select.Changed) -> None:
        info = self.query_one("#resume-info", Static)
        if event.value is Select.NULL:
            info.update("")
            return
        try:
            data = benchmark.load_resume(Path(str(event.value)))
        except ValueError as exc:
            info.update(Text(str(exc), style="red"))
            return
        run = data["run"]
        for key, field_name in (("passes", "passes"), ("warmup", "warmup"), ("cached", "cached_repeats")):
            self.query_one(f"#{key}", Input).value = str(run.get(field_name, ""))
        known = [name for name in run["models"] if name in self.specs]
        self.query_one("#profile", Select).value = "custom"
        selection = self.query_one("#models", SelectionList)
        selection.deselect_all()
        for name in known:
            selection.select(name)
        if run["dataset"].get("smoke"):
            self.query_one("#dataset", Input).value = "data/smoke.jsonl"
        rows = {row.get("model"): row for row in data["models"]}
        remaining = [n for n in run["models"] if n not in rows or row_status(rows[n]) != "ok"]
        info.update(
            f"Dataset {run['dataset'].get('name')} · finished: {len(run['models']) - len(remaining)} · "
            f"to run: {', '.join(remaining) or 'nothing'}"
        )

    def build_argv(self) -> list[str]:
        names = self.selection()
        if not names:
            raise ValueError("Select at least one model")
        argv = ["all" if self.query_one("#setup-switch", Switch).value else "run"]
        for directory in self.model_dirs:
            argv += ["--model-dir", str(directory)]
        profile = str(self.query_one("#profile", Select).value)
        members = benchmark.profile_members(self.specs, profile) if profile != "custom" else []
        if members and set(names) <= set(members):
            argv += ["--profile", profile]
            missing = [name for name in members if name not in names]
            if missing:
                argv += ["--exclude", *missing]
        else:
            argv += ["--models", *names]
        argv += ["--dataset", str(self.dataset_path())]
        for key, flag in (("passes", "--passes"), ("warmup", "--warmup"), ("cached", "--cached-repeats")):
            value = self.query_one(f"#{key}", Input).value.strip()
            if value:
                argv += [flag, value]
        resume = self.query_one("#resume", Select).value
        if resume is not Select.NULL:
            argv += ["--resume", str(resume)]
        return argv

    @on(Button.Pressed, "#start")
    def action_start(self) -> None:
        error = self.query_one("#setup-error", Static)
        try:
            argv = self.build_argv()
        except ValueError as exc:
            error.update(str(exc))
            return
        message = self.app.start_run(argv)  # type: ignore[attr-defined]
        error.update(message or "")
        if message:
            self.notify(message, severity="error", timeout=8)


# Live run.

class RunScreen(Screen[None]):
    BINDINGS = [
        Binding("s", "skip", "Skip model"),
        Binding("x", "remove", "Remove queued"),
        Binding("a", "abort", "Abort run"),
        Binding("r", "retry", "Retry unfinished"),
        Binding("o", "open_report", "Open report"),
        Binding("f", "follow", "Follow log"),
        Binding("l", "filter", "Filter log"),
        Binding("n", "new_run", "New run", show=False),
        Binding("q", "quit_app", "Quit"),
    ]
    DEFAULT_CSS = """
    RunScreen #main { height: 1fr; }
    RunScreen #log-pane { width: 1fr; border-right: solid $panel; }
    RunScreen #log-bar { height: 1; padding: 0 1; background: $boost; color: $text-muted; }
    RunScreen #log { height: 1fr; scrollbar-size-horizontal: 1; }
    RunScreen #tabs { width: 1fr; }
    RunScreen #queue { height: auto; max-height: 60%; }
    RunScreen #current { height: auto; padding: 1 1 0 1; border-top: solid $panel; }
    RunScreen #current-title { text-style: bold; }
    RunScreen #current-bar { margin: 1 0 0 0; }
    RunScreen #current-spark { height: 3; margin: 1 0 0 0; }
    RunScreen #results { height: auto; max-height: 70%; }
    RunScreen #results-notes { padding: 1 1; }
    RunScreen.narrow #main { layout: vertical; }
    RunScreen.narrow #log-pane { width: 1fr; height: 40%; border-right: none; border-bottom: solid $panel; }
    RunScreen.narrow #tabs { height: 1fr; }
    """

    def __init__(self, args: argparse.Namespace, selected: list[ModelSpec], session: Session) -> None:
        super().__init__()
        self.args = args
        self.selected = selected
        self.session = session
        self.control = session.control
        self.events: queue.SimpleQueue[Event] = session.reporter.events  # type: ignore[attr-defined]
        self.plan: benchmark.RunPlan | None = None
        self.states: dict[str, ModelState] = {}
        self.model_colors: dict[str, str] = {}
        self.log_store: deque[tuple[str | None, Text]] = deque(maxlen=20000)
        self.log_filter: str | None = None
        self.running = True
        self.quitting = False
        self.exit_code = 0
        self.results_dirty = True
        self.last_finished: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="log-pane"):
                yield Static(id="log-bar")
                yield RichLog(id="log", max_lines=10000, wrap=False, markup=False, highlight=False, min_width=20)
            with TabbedContent(id="tabs"):
                with TabPane("Progress", id="tab-progress"):
                    yield DataTable(id="queue", cursor_type="row")
                    with Vertical(id="current"):
                        yield Static(id="current-title")
                        yield Static(id="current-hint")
                        yield ProgressBar(id="current-bar", show_eta=False)
                        yield Sparkline([], id="current-spark")
                        yield Static(id="current-stats")
                with TabPane("Latency", id="tab-latency"):
                    with VerticalScroll():
                        yield Chart(latency_chart, id="latency-chart")
                with TabPane("Quality", id="tab-quality"):
                    with VerticalScroll():
                        yield Chart(quality_chart, id="quality-chart")
                with TabPane("Results", id="tab-results"):
                    with VerticalScroll():
                        yield DataTable(id="results", cursor_type="row")
                        yield Static(id="results-notes")
        yield Footer()

    def on_mount(self) -> None:
        queue_table = self.query_one("#queue", DataTable)
        for label, key in (("Model", "model"), ("Status", "status"), ("Stage", "stage"), ("Elapsed", "elapsed"),
                           ("Progress", "progress"), ("p50", "p50"), ("Accuracy", "accuracy")):
            queue_table.add_column(label, key=key)
        results = self.query_one("#results", DataTable)
        for label in ("Model", "Status", "p50", "p95", "p99", "Cached p50", "Peak RSS", "Accuracy",
                      "ECE", "≤1% err cov.", "≤5% err cov."):
            results.add_column(label)
        for i, spec in enumerate(self.selected):
            self.ensure_state(spec.name, i).status = "queued"
        self.update_log_bar()
        self.on_resize()
        self.app.sub_title = "preparing run"
        self.set_interval(0.1, self.drain)
        self.set_interval(1.0, self.refresh_views)
        self.refresh_views()
        self.run_engine(None)

    def on_resize(self, event: Any = None) -> None:
        self.set_class(self.size.width < 110, "narrow")
        log = self.query_one("#log", RichLog)
        if log.auto_scroll:
            self.call_after_refresh(log.scroll_end, animate=False)

    # Engine worker.
    @work(thread=True, exclusive=True, group="engine", exit_on_error=False)
    def run_engine(self, plan: benchmark.RunPlan | None) -> None:
        try:
            if plan is None:
                plan = benchmark.prepare_run(self.args, self.selected, self.session)
                self.plan = plan
            benchmark.execute(plan, self.session)
        except Exception as exc:  # surfaced in the UI; the CLI reports these as configuration errors
            self.app.call_from_thread(self.engine_failed, exc)

    def engine_failed(self, exc: Exception) -> None:
        self.running = False
        message = f"{type(exc).__name__}: {exc}"
        self.write_log(None, Text(f"ERROR: {message}", style="bold red"))
        if self.plan is None and len(self.app.screen_stack) > 2:
            self.app.pop_screen()
            self.app.notify(message, severity="error", timeout=10)
        else:
            self.notify(message, severity="error", timeout=10)
        if self.quitting:
            self.app.exit(2)

    # Events.
    def ensure_state(self, name: str, index: int | None = None) -> ModelState:
        if name not in self.states:
            self.states[name] = ModelState(name, status="not run")
            self.model_colors[name] = MODEL_COLORS[(len(self.model_colors) if index is None else index) % len(MODEL_COLORS)]
        return self.states[name]

    def drain(self) -> None:
        changed = False
        for _ in range(5000):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle(event)
            changed = True
        if changed:
            self.refresh_views()

    def handle(self, event: Event) -> None:
        now = time.monotonic()
        if isinstance(event, Log):
            self.write_log(event.model, self.format_log(event))
        elif isinstance(event, Stage):
            state = self.ensure_state(event.model)
            self.write_log(event.model, self.format_stage(event))
            if event.stage in ("complete", "skipped", "ERROR"):
                return
            if event.stage == "begin":
                state.reset()
                state.started = now
            if event.stage != state.stage:
                state.stage_started = now
                state.hint = ""
            state.status, state.stage, state.detail = "running", event.stage, event.detail
        elif isinstance(event, Heartbeat):
            state = self.ensure_state(event.model)
            hint = f"{event.stage}: still waiting, {fmt_elapsed(event.elapsed_s)}"
            if event.stalled:
                if event.idle_s is not None:
                    hint += f"; no server output for {fmt_elapsed(event.idle_s)}"
                hint += "; press s to skip"
            state.hint = hint
            self.write_log(event.model, self.prefix(event.model).append(hint, style="yellow" if event.stalled else "dim"))
        elif isinstance(event, Progress):
            state = self.ensure_state(event.model)
            state.phase, state.n, state.total = event.phase, event.n, event.total
            if event.phase == "measure":
                state.latencies.append(event.latency_ms)
                state.p50 = event.p50_ms
                if event.n == event.total:
                    self.write_log(event.model, self.prefix(event.model).append(
                        f"measuring: {event.n}/{event.total}; p50 {fmt_ms(event.p50_ms)}"))
        elif isinstance(event, ModelDone):
            state = self.ensure_state(event.model)
            state.row = event.row
            state.status = row_status(event.row)
            state.finished = now
            state.stage, state.hint = "", ""
            state.detail = event.row.get("errors") or event.row.get("skip_reason") or ""
            self.last_finished = event.model
            self.results_dirty = True
        elif isinstance(event, RunStarted):
            self.running = True
            plan = self.plan
            for i, name in enumerate(event.metadata["models"]):
                state = self.ensure_state(name, i)
                row = plan.rows.get(name) if plan else None
                if name in event.queue:
                    state.reset()
                    state.row = row
                elif row is not None:
                    state.row, state.status = row, row_status(row)
                else:
                    state.status = "not run"
            metadata = event.metadata
            self.app.sub_title = (
                f"{event.out_dir.name if event.out_dir else ''} · {metadata.get('profile')}"
                + (f" − {', '.join(metadata['excluded'])}" if metadata.get("excluded") else "")
                + f" · {metadata['dataset']['name']} · {metadata.get('passes')} passes"
            )
            self.results_dirty = True
        elif isinstance(event, RunDone):
            self.running = False
            self.exit_code = event.exit_code
            for name in self.control.queued():
                self.states[name].status = "not run"
            self.results_dirty = True
            counts = self.counts()
            summary = ", ".join(f"{n} {label}" for label, n in counts.items() if n)
            self.write_log(None, Text(
                f"Run {event.status}: {summary}. Report: {event.out_dir / 'summary.html' if event.out_dir else ''}",
                style="bold green" if event.exit_code == 0 else "bold yellow"))
            self.notify(f"Run {event.status}: {summary}", timeout=8)
            if self.quitting:
                self.app.exit(event.exit_code)

    @staticmethod
    def short_reason(state: ModelState) -> str:
        if state.status == "skipped":
            return state.detail[state.detail.find("during"):] if "during" in state.detail else "skipped"
        if state.status == "failed":
            return state.detail.split(":", 1)[0][:28]
        return ""

    def counts(self) -> dict[str, int]:
        counts = {"ok": 0, "failed": 0, "skipped": 0, "not run": 0}
        for state in self.states.values():
            key = "not run" if state.status in ("removed", "queued") else state.status
            if key in counts:
                counts[key] += 1
        return counts

    # Log.
    def prefix(self, model: str | None, source: str | None = None) -> Text:
        line = Text()
        if model is not None:
            line.append(f"[{model}{':' + source if source else ''}] ", style=self.model_colors.get(model, "white"))
        return line

    def format_log(self, event: Log) -> Text:
        if event.source == "harness":
            return self.prefix(event.model).append(event.text, style="bold")
        style = {"setup": "dim", "log-tail": "red"}.get(event.source, "")
        return self.prefix(event.model, event.source).append(event.text.rstrip(), style=style)

    def format_stage(self, event: Stage) -> Text:
        line = self.prefix(event.model)
        line.append(event.stage, style=STAGE_STYLE.get(event.stage, "bold"))
        if event.detail:
            line.append(f": {event.detail}")
        return line

    def write_log(self, model: str | None, text: Text) -> None:
        self.log_store.append((model, text))
        if self.log_filter is None or model == self.log_filter:
            self.query_one("#log", RichLog).write(text)

    def update_log_bar(self) -> None:
        log = self.query_one("#log", RichLog)
        scope = "all models" if self.log_filter is None else self.log_filter
        follow = "following" if log.auto_scroll else "paused"
        self.query_one("#log-bar", Static).update(f"Log · {scope} · {follow}   (l filter · f follow)")

    # Views.
    def refresh_views(self) -> None:
        now = time.monotonic()
        table = self.query_one("#queue", DataTable)
        for name, state in self.states.items():
            label, style = STATUS_STYLE.get(state.status, (state.status, ""))
            end = state.finished if state.status != "running" else now
            elapsed = fmt_elapsed(end - state.started) if state.started and end else ""
            progress = f"{state.n}/{state.total} {state.phase}" if state.status == "running" and state.total else ""
            row = state.row or {}
            if state.status == "ok" and isinstance(row.get("requests"), int):
                progress = f"{row['requests']} requests"
            cells = {
                "model": Text(name, style=self.model_colors.get(name, "")),
                "status": Text(label, style=style),
                "stage": state.stage if state.status == "running" else self.short_reason(state),
                "elapsed": elapsed,
                "progress": progress,
                "p50": fmt_ms(row.get("p50_ms") if state.status == "ok" else state.p50),
                "accuracy": fmt_pct(row.get("accuracy")) if state.status == "ok" else "",
            }
            if name not in table.rows:
                table.add_row(*cells.values(), key=name)
            else:
                for key, value in cells.items():
                    table.update_cell(name, key, value, update_width=True)
        self.refresh_current(now)
        if self.results_dirty:
            self.results_dirty = False
            self.refresh_results()

    def refresh_current(self, now: float) -> None:
        current = self.control.current
        title = self.query_one("#current-title", Static)
        hint = self.query_one("#current-hint", Static)
        stats = self.query_one("#current-stats", Static)
        bar = self.query_one("#current-bar", ProgressBar)
        spark = self.query_one("#current-spark", Sparkline)
        state = self.states.get(current) if current else None
        bar.display = spark.display = state is not None
        if state is None:
            if self.running:
                title.update("Preparing…" if self.plan is None else "Between models…")
                hint.update("")
            else:
                counts = self.counts()
                title.update("Run finished: " + ", ".join(f"{n} {label}" for label, n in counts.items() if n))
                retry = " · r retries unfinished models" if counts["failed"] or counts["skipped"] or counts["not run"] else ""
                hint.update(Text(f"o opens the report{retry} · n starts a new run · q quits", style="dim"))
            stats.update("")
            return
        stage_time = fmt_elapsed(now - state.stage_started) if state.stage_started else ""
        title.update(Text.assemble(
            (state.name, f"bold {self.model_colors.get(state.name, '')}"), " · ", (state.stage, "bold"),
            f" · {stage_time} in stage · {fmt_elapsed(now - state.started) if state.started else ''} total",
        ))
        if state.hint:
            hint.update(Text(state.hint, style="yellow" if "press s" in state.hint else "dim"))
        else:
            hint.update(Text(state.detail, style="dim"))
        if state.total:
            bar.update(total=state.total, progress=state.n)
        else:
            bar.update(total=None, progress=0)
        spark.data = list(state.latencies)
        if state.total:
            stats.update(
                f"{state.phase} {state.n}/{state.total}"
                + (f" · last {state.latencies[-1]:.1f} ms" if state.latencies and state.phase == "measure" else "")
                + (f" · p50 {state.p50:.1f} ms" if state.p50 is not None else "")
            )
        else:
            stats.update("")

    def finished_rows(self) -> list[dict[str, Any]]:
        return [state.row for state in self.states.values() if state.row is not None and state.status == "ok"]

    def refresh_results(self) -> None:
        rows = self.finished_rows()
        self.query_one("#latency-chart", Chart).set_rows(rows)
        self.query_one("#quality-chart", Chart).set_rows(rows)
        table = self.query_one("#results", DataTable)
        table.clear()
        for name, state in self.states.items():
            row = state.row or {}
            label, style = STATUS_STYLE.get(state.status, (state.status, ""))
            ok = state.status == "ok"
            ece = row.get("ece_10bin")
            rss = row.get("peak_process_tree_rss_mb")
            table.add_row(
                Text(name, style=self.model_colors.get(name, "")),
                Text(label, style=style),
                *(fmt_ms(row.get(key)) if ok else "" for key in ("p50_ms", "p95_ms", "p99_ms", "cached_p50_ms")),
                f"{rss:.0f} MB" if ok and isinstance(rss, (int, float)) else "",
                fmt_pct(row.get("accuracy")) if ok else "",
                f"{ece:.3f}" if ok and isinstance(ece, (int, float)) else "",
                fmt_pct(row.get("coverage_at_1pct_error")) if ok else "",
                fmt_pct(row.get("coverage_at_5pct_error")) if ok else "",
            )
        notes = Text()
        for state in self.states.values():
            if state.status == "failed":
                notes.append(f"{state.name} failed: ", style="bold red")
                notes.append(f"{(state.row or {}).get('errors', '')}\n")
            elif state.status == "skipped":
                notes.append(f"{state.name} skipped: ", style="bold yellow")
                notes.append(f"{(state.row or {}).get('skip_reason', '')}\n")
        if self.plan is not None:
            missing = not_run(self.finished_or_known_rows(), self.plan.metadata)
            pending = [name for name in missing if self.states.get(name) and self.states[name].status == "queued"]
            missing = [name for name in missing if name not in pending]
            if missing:
                notes.append("Not run: ", style="bold")
                notes.append(", ".join(missing) + "\n")
            notes.append("\nReport  ", style="bold")
            notes.append("o opens it\n", style="dim")
            notes.append(str(self.plan.out_dir / "summary.html") + "\n")
        self.query_one("#results-notes", Static).update(notes)

    def finished_or_known_rows(self) -> list[dict[str, Any]]:
        return [state.row for state in self.states.values() if state.row is not None]

    # Actions.
    def action_skip(self) -> None:
        model, stage = self.control.current, self.control.stage
        if model is None:
            self.notify("No model is running.", severity="warning")
            return

        def answer(confirmed: bool | None) -> None:
            if not confirmed:
                return
            if self.control.skip(model):
                self.notify(f"Skipping {model}; stopping it…")
            else:
                self.notify(f"{model} already finished.", severity="warning")

        self.app.push_screen(
            ConfirmScreen(f"Skip {model} during {stage or 'startup'}?",
                          "Its server is stopped and the run continues with the next model. "
                          "The report marks it as skipped."),
            answer,
        )

    def action_remove(self) -> None:
        table = self.query_one("#queue", DataTable)
        if table.row_count == 0:
            return
        name = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        if name and self.control.remove(name):
            self.states[name].status = "removed"
            self.notify(f"Removed {name} from the queue.")
            self.results_dirty = True
            self.refresh_views()
        else:
            self.notify("Highlight a queued model to remove it.", severity="warning")

    def action_abort(self) -> None:
        if not self.running:
            self.notify("The run already finished.", severity="warning")
            return

        def answer(confirmed: bool | None) -> None:
            if confirmed:
                self.control.abort()
                self.notify("Aborting; stopping the current model and writing the report…")

        self.app.push_screen(
            ConfirmScreen("Abort the run?", "The current model is stopped, queued models are not run, "
                          "and the report is written for the models that finished."),
            answer,
        )

    def action_retry(self) -> None:
        if self.running:
            self.notify("Retry is available once the run finishes or is aborted.", severity="warning")
            return
        if self.plan is None:
            return
        names = [name for name, state in self.states.items()
                 if state.status in ("failed", "skipped", "not run", "removed") and name in self.plan.specs]
        if not names:
            self.notify("Nothing to retry: every model finished.")
            return
        self.control.reset_abort()
        self.control.requeue(names)
        for name in names:
            self.states[name].reset()
        self.plan.metadata["resumed_at"] = [*self.plan.metadata.get("resumed_at", []), benchmark.now_iso()]
        self.running = True
        self.write_log(None, Text(f"Retrying {', '.join(names)} in {self.plan.out_dir}", style="bold"))
        self.run_engine(self.plan)

    def action_open_report(self) -> None:
        if self.plan is None:
            self.notify("No report yet.", severity="warning")
            return
        self.app.opener(self.plan.out_dir / "summary.html")  # type: ignore[attr-defined]

    def action_follow(self) -> None:
        log = self.query_one("#log", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)
        self.update_log_bar()

    def action_filter(self) -> None:
        if self.log_filter is None:
            table = self.query_one("#queue", DataTable)
            highlighted = (
                table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
                if table.row_count else None
            )
            self.log_filter = self.control.current or highlighted
            if self.log_filter is None:
                return
        else:
            self.log_filter = None
        log = self.query_one("#log", RichLog)
        log.clear()
        for model, text in self.log_store:
            if self.log_filter is None or model == self.log_filter:
                log.write(text)
        self.update_log_bar()

    def action_new_run(self) -> None:
        if self.running:
            self.notify("Finish or abort this run first.", severity="warning")
        elif len(self.app.screen_stack) > 2:
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        if not self.running:
            self.app.exit(self.exit_code)
            return
        if self.quitting:
            self.control.kill_all()
            self.app.exit(130)
            return

        def answer(confirmed: bool | None) -> None:
            if confirmed:
                self.quitting = True
                self.control.abort()
                self.notify("Stopping the current model and writing the report… press q again to force quit.")

        self.app.push_screen(
            ConfirmScreen("Abort the run and quit?",
                          "Model servers are stopped and the report is written for the models that finished."),
            answer,
        )


class BenchmarkApp(App[int]):
    TITLE = "System One benchmark"
    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
        Binding("ctrl+c", "quit_app", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        model_dirs: list[Path] | None = None,
        start_argv: list[str] | None = None,
        opener: Callable[[Path], Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_dirs = list(model_dirs or [])
        self.specs = load_models(self.model_dirs)
        self.start_argv = start_argv
        self.opener = opener or (lambda path: subprocess.Popen(["open", str(path)]))
        self.controls: list[RunControl] = []

    def on_mount(self) -> None:
        if not self.is_headless:
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGTERM, signal.SIGHUP):
                loop.add_signal_handler(signum, self.terminate)
        if self.start_argv is None:
            self.push_screen(SetupScreen(self.specs, self.model_dirs))
        else:
            message = self.start_run(self.start_argv)
            if message:
                self.exit(2, message=f"ERROR: {message}")

    def start_run(self, argv: list[str]) -> str | None:
        """Validate options exactly like the CLI, then open a live run screen."""
        session = Session(reporter=TuiReporter(), control=RunControl())
        try:
            args = benchmark.parse_args(argv)
            selected = benchmark.resolve_selection(args, self.specs)
            benchmark.require_host("custom" if args.models else (args.profile or "full"), session)
        except SystemExit:
            return "Invalid options: " + " ".join(argv)
        except HOST_ERRORS as exc:
            return str(exc)
        self.controls.append(session.control)
        self.push_screen(RunScreen(args, selected, session))
        return None

    def kill_all(self) -> None:
        for control in self.controls:
            control.abort()
            control.kill_all()

    def terminate(self) -> None:
        """SIGTERM or a closed terminal: stop model servers now; the engine writes the report."""
        self.kill_all()
        self.exit(130)

    def action_quit_app(self) -> None:
        if isinstance(self.screen, RunScreen):
            self.screen.action_quit_app()
        elif isinstance(self.screen, ModalScreen):
            self.screen.dismiss(False)
        else:
            self.exit(0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("run", "all"):
        try:
            model_dirs = benchmark.parse_args(argv).model_dir
        except SystemExit as exc:
            return int(exc.code or 2)
        start_argv: list[str] | None = argv
    else:
        parser = argparse.ArgumentParser(
            description="Interactive dashboard. Pass `run ...` or `all ...` benchmark options to start at once.",
        )
        parser.add_argument("--model-dir", type=Path, action="append", default=[],
                            help="additional directory of model TOML definitions")
        model_dirs = parser.parse_args(argv).model_dir
        start_argv = None
    try:
        app = BenchmarkApp(model_dirs, start_argv)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    atexit.register(app.kill_all)
    result = app.run()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
