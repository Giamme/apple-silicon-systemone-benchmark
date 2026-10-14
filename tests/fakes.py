"""Loopback /v1/systemone services for tests; no model weights are involved."""
from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator


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


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/systemone"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class Stuck:
    """A service whose requests hang, like a model stuck downloading weights."""

    def __init__(self) -> None:
        self.url = ""
        self.release = threading.Event()
        self.arrived = threading.Event()
        self.arrivals = threading.Semaphore(0)
        stuck = self

        class StuckHandler(Handler):
            def do_POST(self) -> None:
                stuck.arrived.set()
                stuck.arrivals.release()
                stuck.release.wait(30)
                try:
                    super().do_POST()
                except OSError:  # the benchmark gave up on this request long ago
                    pass

        self.handler = StuckHandler

    def wait_for_request(self, timeout: float = 15) -> bool:
        return self.arrivals.acquire(timeout=timeout)


@contextmanager
def stuck_service() -> Iterator[Stuck]:
    stuck = Stuck()
    with serve(stuck.handler) as url:
        stuck.url = url
        try:
            yield stuck
        finally:
            stuck.release.set()


def write_dataset(directory: Path) -> Path:
    path = directory / "cases.jsonl"
    path.write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    return path


def write_manifest(directory: Path, name: str, endpoint: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.toml").write_text(
        f'id = "{name}"\nendpoint = "{endpoint}"\nnote = "test service"\n', encoding="utf-8"
    )
