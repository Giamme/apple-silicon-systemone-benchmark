"""Load the small, declarative model and service registry."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".bench"
MODEL_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    endpoint: str
    api_model: str | None
    note: str
    adapter: str
    order: int
    profiles: tuple[str, ...]
    setup: CommandSpec | None
    launch: CommandSpec | None
    version: CommandSpec | None
    headers_env: dict[str, str]
    source: Path

    @property
    def managed(self) -> bool:
        return self.launch is not None

    @property
    def port(self) -> int:
        parsed = urlparse(self.endpoint)
        return parsed.port or (443 if parsed.scheme == "https" else 80)


def _expand(value: str) -> str:
    return value.replace("{root}", str(ROOT)).replace("{work}", str(WORK))


def _command(data: object, label: str) -> CommandSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a table")
    argv = data.get("command")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError(f"{label}.command must be a nonempty string array")
    cwd = data.get("cwd", "{root}")
    env = data.get("env", {})
    if not isinstance(cwd, str) or not isinstance(env, dict):
        raise ValueError(f"{label}.cwd and .env must be strings/a table")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError(f"{label}.env must contain string values")
    return CommandSpec(
        argv=tuple(_expand(x) for x in argv),
        cwd=Path(_expand(cwd)),
        env={k: _expand(v) for k, v in env.items()},
    )


def load_manifest(path: Path) -> ModelSpec:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    name = data.get("id")
    display_name = data.get("display_name", name)
    endpoint = data.get("endpoint")
    adapter = data.get("adapter", "systemone")
    order = data.get("order", 100)
    note = data.get("note", "")
    api_model = data.get("api_model")
    profiles = data.get("profiles", [])
    headers_env = data.get("headers_env", {})
    if not isinstance(name, str) or not MODEL_ID.fullmatch(name):
        raise ValueError(f"{path}: id must match {MODEL_ID.pattern}")
    if not all(isinstance(x, str) for x in (display_name, endpoint, adapter, note)):
        raise ValueError(f"{path}: display_name, endpoint, adapter and note must be strings")
    if api_model is not None and not isinstance(api_model, str):
        raise ValueError(f"{path}: api_model must be a string")
    if type(order) is not int:
        raise ValueError(f"{path}: order must be an integer")
    if not isinstance(profiles, list) or any(not isinstance(x, str) or not x for x in profiles):
        raise ValueError(f"{path}: profiles must be a string array")
    if not isinstance(headers_env, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v
        for k, v in headers_env.items()
    ):
        raise ValueError(f"{path}: headers_env must map header names to environment variable names")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{path}: endpoint must be an HTTP(S) URL without embedded credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{path}: invalid endpoint port") from exc
    setup = _command(data.get("setup"), f"{path}: setup")
    launch = _command(data.get("launch"), f"{path}: launch")
    version = _command(data.get("version"), f"{path}: version")
    if launch and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(f"{path}: managed models must use a loopback endpoint")
    return ModelSpec(
        name=name,
        display_name=display_name,
        endpoint=endpoint,
        api_model=api_model,
        note=note,
        adapter=adapter,
        order=order,
        profiles=tuple(profiles),
        setup=setup,
        launch=launch,
        version=version,
        headers_env=headers_env,
        source=path,
    )


def load_models(extra_dirs: list[Path] | None = None) -> dict[str, ModelSpec]:
    dirs = [ROOT / "models", ROOT / "models" / "local", *(extra_dirs or [])]
    specs: dict[str, ModelSpec] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.toml")):
            try:
                spec = load_manifest(path)
            except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
                raise ValueError(f"Invalid model manifest {path}: {exc}") from exc
            if spec.name in specs:
                raise ValueError(f"Duplicate model id {spec.name}: {specs[spec.name].source} and {path}")
            specs[spec.name] = spec
    if not specs:
        raise ValueError("No model manifests found")
    return specs
