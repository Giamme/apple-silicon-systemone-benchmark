#!/usr/bin/env python3
"""Install one locked runtime and cache its exact local artifact snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_catalog import WORK, get_model, snapshot_path


def download(spec: dict) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    destination = snapshot_path(spec)
    snapshot_download(repo_id=spec['artifact_repo'], revision=spec['artifact_revision'],
                      local_dir=destination, cache_dir=WORK / 'hf-cache')
    for item in spec.get('verification_files', []):
        path = Path(hf_hub_download(repo_id=spec['source_repo'], revision=spec['source_revision'],
                    filename=item['filename'], local_dir=destination / 'verification', cache_dir=WORK / 'hf-cache'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('source verification file hash mismatch')
    (destination / '.benchmark-snapshot.json').write_text(json.dumps({
        'repo_id': spec['artifact_repo'], 'revision': spec['artifact_revision'],
    }, indent=2), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--download-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec = get_model(args.model)
    if args.download_only:
        download(spec)
        return 0
    env_path = WORK / 'envs' / spec['environment']
    environment = {**os.environ, 'UV_PROJECT_ENVIRONMENT': str(env_path),
                   'UV_CACHE_DIR': str(WORK / 'uv-cache')}
    subprocess.run(['uv', 'sync', '--project', str(ROOT / 'runtimes' / spec['environment']),
                    '--python', '3.12', '--frozen'], check=True, env=environment)
    subprocess.run([str(env_path / 'bin' / 'python'), str(Path(__file__).resolve()),
                    spec['id'], '--download-only'], check=True, env=environment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
