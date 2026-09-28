# Model and service definitions

The benchmark loads `models/*.toml`, `models/local/*.toml`, and any directory passed with `--model-dir`. A local definition may also select `light` or `full`. Use `./benchmark.sh models validate path/to/model.toml` before running it.

## Manifest

```toml
id = "my-service"
display_name = "My Service"
endpoint = "http://127.0.0.1:19001/v1/systemone"
api_model = "my-model"              # optional
adapter = "systemone"                # default
profiles = []                        # or ["light"], ["full"]
order = 100                          # optional profile display order
note = "Short description for the report"

[headers_env]
Authorization = "MY_SERVICE_AUTH_HEADER" # env value includes Bearer prefix

[setup]                               # optional: omit for installed/existing services
command = ["bash", "{root}/scripts/setup_my_service.sh"]
cwd = "{root}"

[launch]                              # optional: omit for already-running services
command = ["{work}/bin/my-service", "--port", "19001"]
cwd = "{root}"

[launch.env]
MY_SERVICE_DEVICE = "mps"

[version]                             # optional: captured in each report
command = ["{work}/bin/my-service", "--version"]
```

Only `id` and `endpoint` are required. The endpoint must be HTTP(S), without embedded credentials. Managed launch endpoints must be loopback URLs. The tool executes command arrays directly, without a shell; use an explicit script for conditional setup. `{root}` and `{work}` expand to this repository and its ignored `.bench/` directory. Setup commands should be idempotent. The launcher captures and prints their output.

With `[launch]`, the benchmark starts and stops the process group. Without it, the endpoint must already be serving. For an existing service, startup and process RSS are reported as N/A. A non-loopback endpoint requires `--allow-remote` at run time. Treat third-party manifests and Python adapters as executable code and review them before use. Put local secrets only in environment variables; the example `[headers_env]` map reads the complete header value from an environment variable.

## Other HTTP APIs

Use `adapter = "package.module:ClassName"` when the service does not return the System One format. The class must implement:

```python
def build_request(self, case: dict, spec: ModelSpec, nonce: str | None) -> urllib.request.Request: ...
def normalize_response(self, data: object) -> dict: ...
```

`build_request` receives the validated labeled case and a unique nonce for primary requests (`None` for the repeated-state metric). Preserve decision semantics and use the nonce only as irrelevant metadata. `normalize_response` must return an object with `answers` keyed by question ID. Each answer uses the common shape: `{"choice": "label", "probabilities": {...}}`, `{"noul": 0.7}`, or `{"score": 1, "probabilities": {...}}`. Optional top-level `latency_ms` is captured separately from measured HTTP latency. The shared evaluator then computes every model's metrics.

The copyable [example manifest](examples/custom_service.toml) and [adapter](examples/custom_adapter.py) show a choice-only HTTP service with a different response shape. A plugin should reject responses it cannot normalize, rather than silently return an empty `answers` object.

## Fair comparisons

For each model, check that the adapter sends the same state and question intent, preserves labels, and does not insert the expected answer into the request. Record model/runtime revisions with `[version]`; the built-in setup currently updates upstream packages and repositories, so compare revisions before comparing results across machines or dates. Model weights and dependencies retain their own licenses.
