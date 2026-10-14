from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

import pytest

pytest.importorskip("textual")

import benchmark  # noqa: E402
import tui  # noqa: E402
from fakes import Handler, serve, stuck_service, write_dataset, write_manifest  # noqa: E402

ROWS = [
    {"model": "d1-omni-600m", "p50_ms": 12.3, "p95_ms": 1234.5, "cached_p50_ms": 8.0, "accuracy": 0.917,
     "coverage_at_1pct_error": 0.25, "coverage_at_5pct_error": 0.5, "ece_10bin": 0.05, "brier_mean": 0.1},
    {"model": "laya", "p50_ms": 2.0, "p95_ms": 3.0, "cached_p50_ms": 1.5, "accuracy": 1.0,
     "coverage_at_1pct_error": 1.0, "coverage_at_5pct_error": 1.0, "ece_10bin": 0.0, "brier_mean": 0.0},
]


@pytest.mark.parametrize("width", [40, 80, 120])
def test_chart_bars_fit_the_pane_width(width: int) -> None:
    for chart in (tui.latency_chart, tui.quality_chart):
        bar_lines = [line for line in chart(ROWS, width).plain.splitlines() if line.startswith(("laya", "d1-omni"))]
        assert bar_lines
        assert max(len(line) for line in bar_lines) <= width


async def wait_until(pilot: object, condition: Callable[[], bool]) -> None:
    for _ in range(300):
        await pilot.pause(0.05)  # type: ignore[attr-defined]
        if condition():
            return
    raise AssertionError("condition not reached")


def test_skip_key_stops_a_stuck_model_and_the_run_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "require_host", lambda *_args, **_kwargs: None)
    dataset = write_dataset(tmp_path)
    out = tmp_path / "out"
    opened: list[Path] = []
    with serve(Handler) as fast, stuck_service() as stuck:
        write_manifest(tmp_path / "models", "stuck", stuck.url)
        write_manifest(tmp_path / "models", "fast", fast)
        app = tui.BenchmarkApp(
            [tmp_path / "models"],
            ["run", "--model-dir", str(tmp_path / "models"), "--models", "stuck", "fast",
             "--dataset", str(dataset), "--passes", "1", "--warmup", "0", "--cached-repeats", "0",
             "--output", str(out)],
            opener=opened.append,
        )

        async def drive() -> None:
            async with app.run_test(size=(140, 40)) as pilot:
                await wait_until(pilot, lambda: isinstance(app.screen, tui.RunScreen))
                screen = app.screen
                await wait_until(pilot, stuck.arrived.is_set)  # stuck's first request is in flight
                await pilot.press("s")
                await pilot.press("y")
                await wait_until(pilot, lambda: not screen.running)
                await pilot.press("o")
                await pilot.press("q")

        asyncio.run(drive())
    report = json.loads((out / "summary.json").read_text())
    assert {row["model"]: row["status"] for row in report["models"]} == {"stuck": "skipped", "fast": "ok"}
    assert opened == [out / "summary.html"]
    assert app.return_value == 1
