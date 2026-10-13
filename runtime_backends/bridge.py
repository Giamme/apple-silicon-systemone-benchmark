"""Foreground loopback bridge. Models load fully before the port is opened."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

from runtime_catalog import get_model, snapshot_path
from runtime_backends.protocol import serialize_state, validate_prediction, validate_questions


def public_error(exc: Exception) -> str:
    """Keep machine-local absolute paths out of portable runtime reports."""
    message = re.sub(r'/(?:[^\s\'"<>:]+/)*[^\s\'"<>:]*', '[local-path]', str(exc))
    return f'{type(exc).__name__}: {message}'


def package_versions() -> dict[str, Any]:
    versions = {'python': platform.python_version()}
    for package in ('mlx', 'mlx-lm', 'mlx-vlm', 'transformers', 'huggingface-hub', 'numpy'):
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = distribution.version
            direct_url = distribution.read_text('direct_url.json')
            if direct_url:
                source = json.loads(direct_url)
                if source.get('vcs_info'):
                    versions[package + '_source'] = {
                        'url': source.get('url'),
                        'commit_id': source['vcs_info'].get('commit_id'),
                    }
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return versions


def write_runtime_report(path: Path | None, spec: dict, status: str,
                         observed: dict | None = None, error: str | None = None) -> None:
    if path is None:
        return
    report = {'configured': spec, 'status': status, 'observed': observed or {'state': 'unknown'}}
    if error:
        report['error'] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configure_allocator(spec: dict, mlx_runtime: Any | None = None) -> dict:
    """Bound unused buffers without changing live allocations or model tensors."""
    if mlx_runtime is None:
        mlx_runtime = importlib.import_module('mlx.core')
    policy = dict(spec['allocator_policy'])
    previous_limit = mlx_runtime.set_cache_limit(policy['free_cache_limit_bytes'])
    return {**policy, 'previous_free_cache_limit_bytes': previous_limit, 'state': 'applied'}


def load_runtime(spec: dict, path: Path, report_path: Path | None,
                 loader: Callable | None = None, mlx_runtime: Any | None = None) -> Any:
    observed: dict = {'state': 'unknown'}
    write_runtime_report(report_path, spec, 'loading', observed)
    try:
        if not path.is_dir():
            raise RuntimeError(f'missing pinned local snapshot for {spec["id"]}; run setup first')
        receipt = json.loads((path / '.benchmark-snapshot.json').read_text(encoding='utf-8'))
        if receipt != {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}:
            raise ValueError('local snapshot receipt does not match pinned artifact')
        observed['allocator_policy'] = configure_allocator(spec, mlx_runtime)
        write_runtime_report(report_path, spec, 'loading', observed)
        if loader is None:
            loader = importlib.import_module(spec['backend_module']).load_backend
        backend = loader(spec, path)
        observed = {**dict(getattr(backend, 'observed_metadata', {})),
                    'allocator_policy': observed['allocator_policy']}
        observed['artifact'] = receipt
        observed['packages'] = package_versions()
        observed.setdefault('state', 'loaded')
        backend.observed_metadata = observed
        write_runtime_report(report_path, spec, 'loaded', observed)
        return backend
    except Exception as exc:
        write_runtime_report(report_path, spec, 'failed', observed, public_error(exc))
        raise


def handler_for(backend: Any, spec: dict, report_path: Path | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != '/v1/systemone':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError('request must have a valid Content-Length below 16 MiB')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or 'state' not in payload:
                    raise ValueError('request must contain state and questions')
                # Never forward unrelated fields, including benchmark expected labels.
                state, questions = payload['state'], validate_questions(payload.get('questions'))
                serialize_state(state)  # Reject invalid/non-finite object state before inference.
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, {'error': public_error(exc)})
                return
            try:
                started = time.perf_counter()
                result = validate_prediction(backend.predict(state, questions), questions,
                                             spec.get("native_probability_decimals"))
                result['latency_ms'] = (time.perf_counter() - started) * 1000
                self.respond(200, result)
            except Exception as exc:
                observed = dict(getattr(backend, 'observed_metadata', {}))
                observed['packages'] = package_versions()
                write_runtime_report(report_path, spec, 'inference_failed', observed, public_error(exc))
                self.respond(500, {'error': public_error(exc)})

        def respond(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args(argv)
    spec = get_model(args.model)
    report = os.environ.get('BENCH_RUNTIME_REPORT')
    report_path = Path(report) if report else None
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        backend = load_runtime(spec, snapshot_path(spec), report_path)
        server = HTTPServer(('127.0.0.1', args.port), handler_for(backend, spec, report_path))
        print(f'{args.model}: loaded pinned {spec["precision"]} runtime; listening on {args.port}', flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(public_error(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
