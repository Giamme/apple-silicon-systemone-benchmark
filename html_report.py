"""Dependency-free, offline comparison report."""
from __future__ import annotations

import json
import os

from html import escape
from pathlib import Path
from typing import Any


def row_status(row: dict[str, Any]) -> str:
    """ok, failed or skipped; rows written before statuses existed fall back to errors."""
    return row.get("status") or ("failed" if row.get("errors") else "ok")


def not_run(rows: list[dict[str, Any]], run: dict[str, Any]) -> list[str]:
    """Selected models with no result row: removed from the queue or cut off by an abort."""
    done = {row.get("model") for row in rows}
    return [name for name in run.get("models", []) if name not in done]


def write_text_atomic(path: Path, text: str) -> None:
    """Replace a report in one step so an interrupted write never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _number(value: Any, digits: int = 1, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:,.{digits}f}{suffix}"


def _percent(value: Any) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value * 100:.1f}%"


def _metric(label: str, value: str, detail: str = "") -> str:
    return (
        '<div class="metric"><span class="metric-label">'
        + _text(label)
        + '</span><strong class="metric-value">'
        + _text(value)
        + '</strong><span class="metric-detail">'
        + _text(detail)
        + "</span></div>"
    )


def _latency_chart(valid: list[dict[str, Any]]) -> str:
    if not valid:
        return '<p class="empty">No completed model runs to plot.</p>'
    maximum = max(float(row["p95_ms"]) for row in valid) or 1.0
    lines = []
    for row in sorted(valid, key=lambda item: item["p50_ms"]):
        p50 = float(row["p50_ms"])
        p95 = float(row["p95_ms"])
        width50 = max(1.5, min(100, 100 * p50 / maximum))
        width95 = max(1.5, min(100, 100 * p95 / maximum))
        lines.append(
            '<div class="bar-row" role="img" aria-label="'
            + _text(f"{row['model']}: p50 {p50:.1f} milliseconds, p95 {p95:.1f} milliseconds")
            + '"><span class="bar-name">'
            + _text(row["model"])
            + '</span><div class="bar-track"><span class="bar-p95" style="width:'
            + f"{width95:.2f}"
            + '%"></span><span class="bar-p50" style="width:'
            + f"{width50:.2f}"
            + '%"></span></div><span class="bar-value">'
            + _text(f"{p50:.1f} ms")
            + "</span></div>"
        )
    return "\n".join(lines)


def _accuracy_chart(valid: list[dict[str, Any]]) -> str:
    rows = [row for row in valid if isinstance(row.get("accuracy"), (int, float))]
    if not rows:
        return '<p class="empty">No labeled decisions were returned.</p>'
    lines = []
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True):
        value = max(0.0, min(100.0, float(row["accuracy"]) * 100))
        lines.append(
            '<div class="bar-row" role="img" aria-label="'
            + _text(f"{row['model']}: {value:.1f}% accuracy")
            + '"><span class="bar-name">'
            + _text(row["model"])
            + '</span><div class="bar-track"><span class="bar-accuracy" style="width:'
            + f"{value:.2f}"
            + '%"></span></div><span class="bar-value">'
            + _text(f"{value:.1f}%")
            + "</span></div>"
        )
    return "\n".join(lines)


def _table_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        status = {
            "ok": '<span class="status ok">Complete</span>',
            "failed": '<span class="status error">Failed</span>',
            "skipped": '<span class="status skip">Skipped</span>',
        }[row_status(row)]
        cells = [
            f'<th scope="row">{_text(row["model"])}</th>',
            f"<td>{status}</td>",
            f'<td>{_text(_number(row.get("p50_ms"), suffix=" ms"))}</td>',
            f'<td>{_text(_number(row.get("p95_ms"), suffix=" ms"))}</td>',
            f'<td>{_text(_number(row.get("p99_ms"), suffix=" ms"))}</td>',
            f'<td>{_text(_percent(row.get("accuracy")))}</td>',
            f'<td>{_text(_percent(row.get("coverage_at_5pct_error")))}</td>',
            f'<td>{_text(_number(row.get("peak_process_tree_rss_mb"), digits=0, suffix=" MB"))}</td>',
        ]
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


def _details(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for row in rows:
        items = [
            ("Runtime revision", row.get("version") or "Unavailable"),
            ("Adapter", row.get("adapter") or "Unknown"),
            ("Configured / observed provenance", json.dumps(row.get("provenance", {}), sort_keys=True, ensure_ascii=False)),
            ("Management", "Started by benchmark" if row.get("managed") else "Existing service"),
            ("Requests", str(row.get("requests", "N/A"))),
            ("Decisions", str(row.get("decisions", "N/A"))),
            ("Startup to port", _number(row.get("startup_to_port_ms"), suffix=" ms")),
            ("First request", _number(row.get("first_request_ms"), suffix=" ms")),
            ("Cached p50", _number(row.get("cached_p50_ms"), suffix=" ms")),
            ("Cached p95", _number(row.get("cached_p95_ms"), suffix=" ms")),
            ("Brier mean", _number(row.get("brier_mean"), digits=3)),
            ("ECE (10 bins)", _number(row.get("ece_10bin"), digits=3)),
            ("Coverage at ≤1% error", _percent(row.get("coverage_at_1pct_error"))),
            ("Coverage at ≤5% error", _percent(row.get("coverage_at_5pct_error"))),
        ]
        grid = "".join(
            '<div><dt>' + _text(label) + '</dt><dd>' + _text(value) + "</dd></div>"
            for label, value in items
        )
        error = (
            '<p class="failure"><strong>Run failed:</strong> ' + _text(row["errors"]) + "</p>"
            if row.get("errors") else ""
        )
        if row_status(row) == "skipped":
            error = (
                '<p class="failure skip"><strong>Skipped:</strong> '
                + _text(row.get("skip_reason") or "skipped") + "</p>"
            )
        blocks.append(
            '<details class="model-detail"><summary><span>'
            + _text(row["model"])
            + '</span><span class="detail-hint">See metrics and provenance</span></summary>'
            + '<div class="detail-body"><p>'
            + _text(row.get("note", ""))
            + '</p>'
            + error
            + '<dl class="detail-grid">'
            + grid
            + "</dl></div></details>"
        )
    return "\n".join(blocks)


CSS = """
:root{color-scheme:light;--paper:#f4f5f1;--surface:#fff;--ink:#17212c;--muted:#526170;--line:#cdd8db;--teal:#087f75;--teal-soft:#d6eee8;--blue:#335a9e;--blue-soft:#dce7fa;--red:#a52833;--red-soft:#fff0f0;--amber:#7a4b00;--amber-soft:#fdf0d5}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}figure{margin:0}
main{max-width:1160px;margin:auto;padding:36px 26px 80px}
h1{font-size:clamp(2.2rem,5vw,4.6rem);line-height:1.06;letter-spacing:-.035em;max-width:850px;margin:0 0 18px}
h2{font-size:1.55rem;letter-spacing:-.03em;margin:0 0 6px}h3{font-size:1.1rem;margin:0}
p{margin:0 0 14px}.lede{font-size:1.1rem;color:var(--muted);max-width:820px}
.run-line{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0 34px}.pill{border:1px solid var(--line);background:var(--surface);border-radius:100px;padding:6px 12px;font-size:.84rem;font-weight:650}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:38px}.metric,.panel,.table-wrap,.model-detail,.notice{background:var(--surface);border:1px solid var(--line);border-radius:16px}
.metric{min-height:140px;padding:18px;display:flex;flex-direction:column}.metric-label{font-size:.82rem;color:var(--muted);font-weight:700}.metric-value{font-size:1.9rem;line-height:1.2;letter-spacing:-.04em;margin:13px 0 5px;font-variant-numeric:tabular-nums}.metric-detail{font-size:.82rem;color:var(--muted)}
.section-heading{display:flex;justify-content:space-between;align-items:baseline;gap:20px;margin:0 0 14px}.section-heading p{font-size:.88rem;color:var(--muted);margin:0}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:38px}.panel{padding:22px;min-width:0}.panel h3{margin-bottom:6px}.panel .subtle{font-size:.85rem;color:var(--muted);margin-bottom:22px}
.bar-row{display:grid;grid-template-columns:72px minmax(0,1fr) 82px;gap:12px;align-items:center;margin:16px 0;font-variant-numeric:tabular-nums}.bar-name{overflow:hidden;text-overflow:ellipsis;font-weight:700}.bar-value{text-align:right;font-size:.83rem;font-weight:700}.bar-track{height:18px;background:#e9edf0;border-radius:4px;position:relative;overflow:hidden}.bar-p95,.bar-p50,.bar-accuracy{display:block;height:100%;position:absolute;left:0;top:0}.bar-p95{background:var(--blue-soft)}.bar-p50{background:var(--blue)}.bar-accuracy{background:var(--teal)}
.legend{display:flex;gap:18px;font-size:.8rem;color:var(--muted)}.legend span:before{content:"";display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}.legend .p50:before{background:var(--blue)}.legend .p95:before{background:var(--blue-soft);border:1px solid var(--blue)}
.table-wrap{overflow-x:auto;margin-bottom:36px;padding:8px}table{border-collapse:collapse;width:100%;min-width:800px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:14px 16px;border-bottom:1px solid var(--line);white-space:nowrap}thead th{font-size:.77rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#f8faf9}th:first-child,td:first-child{text-align:left}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}.status{font-size:.75rem;font-weight:750;border-radius:100px;padding:4px 8px}.status.ok{background:var(--teal-soft);color:#065449}.status.error{background:var(--red-soft);color:var(--red)}.status.skip{background:var(--amber-soft);color:var(--amber)}
.model-detail{margin:10px 0;overflow:hidden}.model-detail summary{cursor:pointer;padding:15px 20px;display:flex;justify-content:space-between;gap:18px;font-weight:750}.model-detail summary:hover{background:#f7faf9}.model-detail summary:focus-visible{outline:3px solid var(--blue);outline-offset:-3px}.detail-hint{font-weight:500;color:var(--muted);font-size:.82rem}.detail-body{padding:0 20px 22px}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 0}.detail-grid div{border-top:1px solid var(--line);padding-top:8px;min-width:0}.detail-grid dt{font-size:.77rem;color:var(--muted)}.detail-grid dd{margin:2px 0 0;font-weight:700;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.failure,.notice{background:var(--red-soft);border-color:#efb9be;padding:14px 18px;color:#7d1c26}.notice{margin:0 0 32px}.failure.skip{background:var(--amber-soft);border-color:#ecd08f;color:var(--amber)}.empty{color:var(--muted);padding:12px 0}
.method{margin-top:38px;border-top:1px solid var(--line);padding-top:26px;color:var(--muted);font-size:.92rem}.method p{max-width:900px}.method strong{color:var(--ink)}code{background:#e8eeed;padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}footer{border-top:1px solid var(--line);margin-top:34px;padding-top:18px;color:var(--muted);font-size:.82rem}
@media(max-width:800px){main{padding:26px 16px 60px}.metric-grid,.chart-grid{grid-template-columns:1fr 1fr}.detail-grid{grid-template-columns:1fr 1fr}}
@media(max-width:580px){.metric-grid,.chart-grid,.detail-grid{grid-template-columns:1fr}.metric{min-height:110px}.bar-row{grid-template-columns:58px minmax(0,1fr) 72px;gap:8px}.section-heading{display:block}.detail-hint{display:none}}
@media print{body{background:#fff}main{max-width:none;padding:0}.metric,.panel,.table-wrap,.model-detail{break-inside:avoid}.model-detail .detail-body{display:block}.model-detail summary{list-style:none}.table-wrap{overflow:visible}table{min-width:0;font-size:9pt}th,td{padding:7px}.metric-grid,.chart-grid{gap:7px}}
"""


def write_html_report(path: Path, rows: list[dict[str, Any]], run: dict[str, Any]) -> None:
    valid = [row for row in rows if row_status(row) == "ok" and isinstance(row.get("p50_ms"), (int, float))]
    failed = [row for row in rows if row_status(row) == "failed"]
    skipped = [row for row in rows if row_status(row) == "skipped"]
    missing = not_run(rows, run)
    run_status = run.get("status", "complete")
    fastest = min(valid, key=lambda row: row["p50_ms"]) if valid else None
    scored = [row for row in valid if isinstance(row.get("accuracy"), (int, float))]
    most_accurate = max(scored, key=lambda row: row["accuracy"]) if scored else None
    dataset = run["dataset"]
    system = run["system"]
    stats = "".join([
        _metric(
            "Fastest warm p50",
            _number(fastest["p50_ms"], suffix=" ms") if fastest else "N/A",
            fastest["model"] if fastest else "No completed runs",
        ),
        _metric(
            "Highest accuracy",
            _percent(most_accurate["accuracy"]) if most_accurate else "N/A",
            most_accurate["model"] if most_accurate else "No labeled decisions",
        ),
        _metric("Dataset cases", str(dataset["cases"]), f"{dataset['questions']} labeled decisions"),
        _metric(
            "Models completed",
            f"{len(valid)} / {len(rows) + len(missing)}",
            f"{len(failed)} failed · {len(skipped)} skipped",
        ),
    ])
    warning = ""
    if failed or skipped or missing or run_status != "complete":
        parts = [
            f"{len(items)} {label}" for items, label in
            ((failed, "failed"), (skipped, "skipped"), (missing, "not run")) if items
        ]
        reason = {
            "aborted": " (run aborted)",
            "running": " (run in progress or interrupted)",
        }.get(run_status, "")
        warning = (
            '<aside class="notice" role="status"><strong>Partial run.</strong> '
            + _text(f"{', '.join(parts) or 'No models finished'}{reason}. Details and logs are available below.")
            + "</aside>"
        )
    smoke = (
        '<aside class="notice" role="note"><strong>Smoke dataset.</strong> '
        "These few synthetic cases check the workflow; use representative labeled data "
        "before choosing a model.</aside>"
        if dataset.get("smoke") else ""
    )
    host_label = " · ".join(
        str(part) for part in (
            system.get("chip_type") or "Apple Silicon",
            system.get("physical_memory"),
            system.get("macos_version"),
        ) if part
    )
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>System One benchmark report</title><style>"
        + CSS
        + "</style></head><body><main>"
        '<header><h1>Apple Silicon decision benchmark</h1>'
        '<p class="lede">A local comparison of latency, decision quality and resource use. '
        "Models run one at a time on the same labeled dataset.</p>"
        '<div class="run-line"><span class="pill">Dataset: '
        + _text(dataset["name"])
        + '</span><span class="pill">Profile: '
        + _text(run["profile"])
        + '</span><span class="pill">'
        + _text(host_label)
        + '</span><span class="pill">'
        + _text(run["created_at"])
        + "</span></div></header>"
        + warning + smoke
        + '<section aria-label="Run highlights" class="metric-grid">' + stats + "</section>"
        '<section aria-labelledby="charts-title"><div class="section-heading"><h2 id="charts-title">At a glance</h2>'
        "<p>Shorter latency and higher accuracy are better.</p></div>"
        '<div class="chart-grid"><figure class="panel"><h3>Warm latency</h3>'
        '<p class="subtle">Median and 95th percentile · milliseconds</p>'
        + _latency_chart(valid)
        + '<figcaption class="legend"><span class="p50">p50</span><span class="p95">p95</span></figcaption></figure>'
        '<figure class="panel"><h3>Decision accuracy</h3>'
        '<p class="subtle">Correct labeled decisions · percent</p>'
        + _accuracy_chart(valid)
        + "</figure></div></section>"
        '<section aria-labelledby="table-title"><div class="section-heading"><h2 id="table-title">All models</h2>'
        "<p>Full numbers remain available in summary.csv.</p></div>"
        '<div class="table-wrap"><table><thead><tr><th scope="col">Model</th>'
        '<th scope="col">Status</th><th scope="col">p50</th><th scope="col">p95</th>'
        '<th scope="col">p99</th><th scope="col">Accuracy</th>'
        '<th scope="col">≤5% error coverage</th><th scope="col">Peak RSS*</th>'
        "</tr></thead><tbody>"
        + _table_rows(rows)
        + "</tbody></table></div></section>"
        '<section aria-labelledby="detail-title"><div class="section-heading"><h2 id="detail-title">Model details</h2>'
        "<p>Open a model to inspect its metrics and revision.</p></div>"
        + _details(rows) + "</section>"
        '<section class="method" aria-labelledby="method-title"><h2 id="method-title">Method and provenance</h2>'
        "<p>Each managed model runs alone. The first request and warmup are excluded from warm latency. "
        "Warm requests carry a unique, decision-irrelevant identifier to reduce repeated-prefix cache effects; "
        "cached latency repeats the same input separately.</p>"
        "<p>Accuracy, Brier score, calibration and error-budget coverage come from labeled answers. "
        "Peak RSS is a process-tree host-memory proxy for managed models; it is not a Metal allocation measurement. "
        "An existing service has no comparable startup or process-memory reading.</p>"
        "<p><strong>Dataset SHA-256:</strong> <code>"
        + _text(dataset["sha256"])
        + "</code>. <strong>Passes:</strong> "
        + _text(run["passes"])
        + ". <strong>Warmup requests:</strong> "
        + _text(run["warmup"])
        + ". <strong>Cached repeats:</strong> "
        + _text(run["cached_repeats"])
        + ".</p></section>"
        "<footer>Generated locally. This standalone report uses no external assets or scripts. "
        "Model response logs and detailed CSV/JSON files stay in the run directory.</footer>"
        "</main></body></html>"
    )
    write_text_atomic(path, page)
