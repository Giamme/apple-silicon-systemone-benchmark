from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import collect_provenance
from model_registry import load_models
from runtime_catalog import MODELS, get_model
from runtime_backends.bridge import handler_for, load_runtime
from runtime_backends.protocol import canonical_answer, serialize_state, validate_prediction, validate_questions


class FakeMlx:
    def __init__(self) -> None:
        self.cache_limits = []

    def set_cache_limit(self, limit: int) -> int:
        self.cache_limits.append(limit)
        return 16 * 1024**3

    def set_memory_limit(self, _limit: int) -> int:
        raise AssertionError('active memory limit must not change')


QUESTIONS = {
    'team': {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': 'First', 'b': 'Second'}},
    'urgent': {'type': 'noul', 'instructions': 'Urgent?'},
    'severity': {'type': 'score', 'instructions': 'Severity?', 'criteria': ['Low', 'High']},
}


def prediction() -> dict:
    return {'answers': {name: canonical_answer(question, [0.3, 0.7])
                        for name, question in QUESTIONS.items()}, 'usage': {'output_tokens': 0}}


def test_model_listing_and_manifest_validation_import_no_heavy_packages() -> None:
    script = """
import sys
from benchmark import main
assert main(['models', 'list']) == 0
assert main(['models', 'validate', 'models/jev-9b.toml']) == 0
assert not any(name.split('.')[0] in {'mlx', 'mlx_vlm', 'transformers', 'numpy', 'torch'} for name in sys.modules)
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)


def test_object_serialization_retains_nonce_and_is_deterministic() -> None:
    state = {'content': {'z': 1, 'a': 2}, 'benchmark_request_id': 'p0-c0-n1'}
    assert serialize_state(state) == '{"benchmark_request_id":"p0-c0-n1","content":{"a":2,"z":1}}'
    assert serialize_state('plain text') == 'plain text'
    with pytest.raises(ValueError):
        serialize_state({'value': float('nan')})


def test_complete_strict_distributions_and_boolean_orientation() -> None:
    assert validate_prediction(prediction(), validate_questions(QUESTIONS))
    yes_first = canonical_answer(QUESTIONS['urgent'], [0.7, 0.3], boolean_yes_index=0)
    assert yes_first['noul'] == 0.7
    assert yes_first['probabilities'] == {'false': 0.3, 'true': 0.7}
    for values in ([float('nan'), 0], [0.2, 0.2], [-0.2, 1.2], [1.0]):
        with pytest.raises(ValueError):
            canonical_answer(QUESTIONS['team'], values)
    result = prediction()
    result['answers']['team']['probabilities'].pop('a')
    with pytest.raises(ValueError, match='complete'):
        validate_prediction(result, QUESTIONS)
    result = prediction()
    result['answers']['severity']['score'] = 2
    with pytest.raises(ValueError, match='criterion index'):
        validate_prediction(result, QUESTIONS)


def test_rounding_tolerance_only_when_declared() -> None:
    questions = {'x': {'type': 'choice', 'instructions': 'Which?', 'criteria': dict.fromkeys(['a', 'b', 'c'], '')}}
    result = {'answers': {'x': {'choice': 'a', 'probabilities': dict.fromkeys(['a', 'b', 'c'], 0.3333)}}}
    with pytest.raises(ValueError, match='sum to one'):
        validate_prediction(result, questions)
    assert validate_prediction(result, questions, probability_decimals=4) == result


def test_load_report_observed_unknown_on_failure_and_snapshot_pin(tmp_path: Path) -> None:
    spec = get_model('d3-edge')
    path = tmp_path / 'snapshot'
    path.mkdir()
    receipt = {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}
    (path / '.benchmark-snapshot.json').write_text(json.dumps(receipt))
    report_path = tmp_path / 'report.json'
    events = []
    def loader(_spec: dict, _path: Path) -> object:
        events.append(json.loads(report_path.read_text())['status'])
        raise RuntimeError(f'could not load {tmp_path}/secret/file')
    with pytest.raises(RuntimeError):
        load_runtime(spec, path, report_path, loader, FakeMlx())
    report = json.loads(report_path.read_text())
    assert events == ['loading']
    assert report['status'] == 'failed'
    assert report['observed']['state'] == 'unknown'
    assert report['observed']['allocator_policy']['state'] == 'applied'
    assert report['observed']['allocator_policy']['free_cache_limit_bytes'] == 256 * 1024**2
    assert str(tmp_path) not in report_path.read_text()
    assert report['configured']['artifact_revision'] == spec['artifact_revision']


def test_successful_load_reports_observed_packages_and_artifact(tmp_path: Path) -> None:
    spec = get_model('jev-9b')
    receipt = {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}
    (tmp_path / '.benchmark-snapshot.json').write_text(json.dumps(receipt))
    backend = SimpleNamespace(observed_metadata={'actual_dtype': 'bfloat16'})
    report_path = tmp_path / 'report.json'
    assert load_runtime(spec, tmp_path, report_path, lambda *_: backend, FakeMlx()) is backend
    report = json.loads(report_path.read_text())
    assert report['observed']['artifact'] == receipt
    assert report['observed']['actual_dtype'] == 'bfloat16'
    assert 'python' in report['observed']['packages']


def test_loopback_bridge_never_forwards_expected_and_reports_inference_failure(tmp_path: Path) -> None:
    seen = []
    class Backend:
        observed_metadata = {'native_api': 'test'}
        def predict(self, state: object, questions: dict) -> dict:
            seen.append((state, questions))
            if len(seen) == 2:
                raise ValueError('bad native distribution')
            return prediction()
    spec = get_model('d3-edge')
    report_path = tmp_path / 'runtime.json'
    receipt = {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}
    (tmp_path / '.benchmark-snapshot.json').write_text(json.dumps(receipt))
    backend = load_runtime(spec, tmp_path, report_path, lambda *_: Backend(), FakeMlx())
    loaded_observed = json.loads(report_path.read_text())['observed']
    server = HTTPServer(('127.0.0.1', 0), handler_for(backend, spec, report_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = {'state': {'content': 'Text', 'benchmark_request_id': 'unique'},
               'questions': {name: {**question, 'expected': 'secret'} for name, question in QUESTIONS.items()},
               'expected': {'secret': 'label'}}
    request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/systemone',
              data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response)['answers']['urgent']['noul'] == 0.7
        assert seen[0] == (payload['state'], QUESTIONS)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 500
        failed_report = json.loads(report_path.read_text())
        assert failed_report['status'] == 'inference_failed'
        assert failed_report['observed'] == loaded_observed
        assert failed_report['observed']['artifact'] == receipt
        assert failed_report['observed']['state'] == 'loaded'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_provenance_keeps_configured_and_observed_distinct(tmp_path: Path) -> None:
    spec = load_models()['jev-9b']
    report = collect_provenance(spec, tmp_path)
    assert report['observed'] == {'state': 'unknown'}
    assert report['configured']['precision'].startswith('NVFP4')
    (tmp_path / 'jev-9b.runtime.json').write_text(json.dumps({'configured': {'wrong': True},
         'status': 'loaded', 'observed': {'actual_dtype': 'bfloat16'}}))
    report = collect_provenance(spec, tmp_path)
    assert report['configured'] == spec.provenance
    assert report['observed'] == {'actual_dtype': 'bfloat16'}


def test_report_provenance_is_escaped_offline_and_csv_is_structured(tmp_path: Path) -> None:
    import csv
    from benchmark import write_summary
    value = '<script>secret</script>`'
    row = {'model': 'fake', 'errors': 'load failed',
           'provenance': {'configured': {'artifact_repo': value}, 'observed': {'state': 'unknown'}}}
    metadata = {'dataset': {'name': 'smoke.jsonl', 'cases': 24, 'questions': 72, 'sha256': 'a' * 64, 'smoke': True},
                'profile': 'custom', 'models': ['fake'], 'created_at': '2026-10-13T00:00:00Z',
                'passes': 1, 'warmup': 1, 'cached_repeats': 2, 'system': {}}
    write_summary(tmp_path, [row], metadata)
    for name in ('summary.html', 'summary.md'):
        report = (tmp_path / name).read_text()
        assert '<script>' not in report
        assert '&lt;script&gt;' in report
    with (tmp_path / 'summary.csv').open(newline='') as stream:
        csv_row = next(csv.DictReader(stream))
    assert json.loads(csv_row['provenance_json']) == row['provenance']
    assert json.loads((tmp_path / 'summary.json').read_text())['models'][0]['provenance'] == row['provenance']


def test_bridge_loads_before_binding_and_does_not_bind_on_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime_backends import bridge
    events = []
    backend = SimpleNamespace()
    def load(*_args: object) -> object:
        events.append('loaded')
        return backend
    class Server:
        def __init__(self, address: tuple, _handler: type) -> None:
            assert address == ('127.0.0.1', 18005)
            assert events == ['loaded']
            events.append('bound')
        def serve_forever(self) -> None:
            raise KeyboardInterrupt
        def server_close(self) -> None:
            events.append('closed')
    monkeypatch.setattr(bridge, 'load_runtime', load)
    monkeypatch.setattr(bridge, 'HTTPServer', Server)
    assert bridge.main(['--model', 'd3-edge', '--port', '18005']) == 130
    assert events == ['loaded', 'bound', 'closed']
    events.clear()
    def fail(*_args: object) -> object:
        raise ValueError('load failed')
    monkeypatch.setattr(bridge, 'load_runtime', fail)
    assert bridge.main(['--model', 'd3-edge', '--port', '18005']) == 1
    assert events == []


def test_package_provenance_distinguishes_mlx_vlm_git_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime_backends import bridge
    source = {'url': 'https://github.com/example/mlx-vlm',
              'vcs_info': {'commit_id': 'a' * 40}}
    distribution = SimpleNamespace(version='0.7.7', read_text=lambda _name: json.dumps(source))
    monkeypatch.setattr(bridge.importlib.metadata, 'distribution', lambda _package: distribution)
    versions = bridge.package_versions()
    assert versions['mlx-vlm_source'] == {'url': source['url'], 'commit_id': 'a' * 40}


@pytest.mark.parametrize('model_id', list(MODELS))
def test_managed_allocator_policy_applied_before_every_backend_load(model_id: str, tmp_path: Path) -> None:
    spec = get_model(model_id)
    receipt = {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}
    (tmp_path / '.benchmark-snapshot.json').write_text(json.dumps(receipt))
    runtime = FakeMlx()
    report_path = tmp_path / 'runtime.json'
    def loader(_spec: dict, _path: Path) -> object:
        assert runtime.cache_limits == [256 * 1024**2]
        loading = json.loads(report_path.read_text())
        assert loading['observed']['state'] == 'unknown'
        assert loading['observed']['allocator_policy']['state'] == 'applied'
        return SimpleNamespace(observed_metadata={'native_api': 'fake'})
    backend = load_runtime(spec, tmp_path, report_path, loader, runtime)
    report = json.loads(report_path.read_text())
    assert report['configured']['allocator_policy'] == spec['allocator_policy']
    assert report['observed']['allocator_policy'] == {
        **spec['allocator_policy'], 'previous_free_cache_limit_bytes': 16 * 1024**3, 'state': 'applied',
    }
    assert backend.observed_metadata['allocator_policy'] == report['observed']['allocator_policy']


def test_allocator_policy_failure_fails_startup_without_backend_fallback(tmp_path: Path) -> None:
    spec = get_model('d3-edge')
    receipt = {'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision']}
    (tmp_path / '.benchmark-snapshot.json').write_text(json.dumps(receipt))
    report_path = tmp_path / 'runtime.json'
    class FailedMlx(FakeMlx):
        def set_cache_limit(self, _limit: int) -> int:
            raise RuntimeError('allocator policy rejected')
    def loader(*_args: object) -> object:
        raise AssertionError('backend must not load after allocator policy failure')
    with pytest.raises(RuntimeError, match='allocator policy rejected'):
        load_runtime(spec, tmp_path, report_path, loader, FailedMlx())
    report = json.loads(report_path.read_text())
    assert report['status'] == 'failed'
    assert report['observed'] == {'state': 'unknown'}
    assert report['configured']['allocator_policy']['free_cache_limit_bytes'] == 256 * 1024**2
