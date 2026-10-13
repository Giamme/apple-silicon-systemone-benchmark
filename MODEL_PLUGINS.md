# Model and service definitions

The benchmark loads `models/*.toml`, `models/local/*.toml`, and any directory passed with `--model-dir`. A local definition may also select `light`, `full`, or `extended`. Use `./benchmark.sh models validate path/to/model.toml` before running it.

## Manifest

```toml
id = "my-service"
display_name = "My Service"
endpoint = "http://127.0.0.1:19001/v1/systemone"
api_model = "my-model"              # optional
adapter = "systemone"                # default
profiles = []                        # or ["light"], ["full", "extended"]
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

## Pinned built-in runtime bridge

Built-in native readout manifests use `runtime_model = "<id>"` to attach the standard-library-only [catalog](runtime_catalog.py) recipe as configured provenance. This field must match the manifest ID. It does not import the backend during model listing or manifest validation. The default, `full`, and `extended` selections include all 12 built-in candidates: the original four and eight pinned native readout models. `light` selects Laya and Kev. Explicit `--models` selection is preserved; use `--models laya von kev rizzo` to run only the original four.

Family modules expose `load_backend(spec: dict, snapshot_path: Path) -> backend`, with lazy ML imports. The backend provides `predict(state, questions)` returning `{"answers": {...}, "usage": {...}}` and an `observed_metadata` dictionary. The snapshot is local, complete, and pinned before launch. The shared bridge binds only `127.0.0.1` and opens the port only after loading succeeds. Before loading any of the eight new managed backends, it applies `mlx.core.set_cache_limit(256 * 1024**2)` to bound unused allocator buffers. This fixed 256 MiB soft limit leaves active memory limits and model tensors unchanged; cached buffers may temporarily exceed it until allocator reclamation. Configured provenance declares the budget, and observed provenance records application and the previous limit, including when subsequent backend loading fails. Failure to apply the policy fails startup without fallback. It forwards state and typed questions without expected labels and preserves the benchmark nonce in deterministic object-state serialization.

Choice answers contain every original label's probability. Boolean `noul` is always `P(true)` regardless of the native head's class order. Score distributions use every zero-based criterion index as a string key; the evaluator is unchanged. Probabilities must be finite, in range, complete, and normalized. Only Clef's declared native four-decimal output permits its bounded rounding error; other native vectors use strict validation. Clef's pinned native encoder sorts choice labels lexicographically internally; the bridge preserves the request criteria and returns probabilities under the original label names. Other families retain their own native ordering. Backends must reject unsupported inputs and native truncation, and may not silently select another precision or runtime.

The runner provides `BENCH_RUNTIME_REPORT` to managed launches. The bridge atomically records configured/loading metadata, then actual package/backend metadata or a failed-load record with observed state unknown. Inference failure preserves loaded observations. Portable provenance must not expose local absolute paths. JSON reports retain the nested structure, CSV adds `provenance_json`, and HTML/Markdown provide model details; the `version` field remains separate. Existing service adapters remain supported without a sidecar.

## BF16 D3 reference parity

[`scripts/verify_d3_parity.py`](scripts/verify_d3_parity.py) captures and compares `d3-edge`, `d3-lite`, and `d3-mini`. Run the original pinned source model in an isolated Torch CPU environment, then the pinned MLX artifact in the managed `d3-clef` environment, as separate sequential processes. The verifier does not download snapshots or install environments. The original snapshot must be complete, have a `MODEL_MANIFEST.json` with the catalog's source repository ID, and use the pinned source revision as its directory name.

For each model, set `REFERENCE_PYTHON` to the isolated CPU environment's Python, `MODEL` to its ID, `SOURCE_SNAPSHOT` to that model's pinned original snapshot, and `MLX_SNAPSHOT` to its pinned managed artifact snapshot. For example, after those paths are prepared:

```bash
"$REFERENCE_PYTHON" scripts/verify_d3_parity.py capture \
  --engine reference --model "$MODEL" --snapshot "$SOURCE_SNAPSHOT" \
  --output ".bench/parity/$MODEL-reference.json"
.bench/envs/d3-clef/bin/python scripts/verify_d3_parity.py capture \
  --engine mlx --model "$MODEL" --snapshot "$MLX_SNAPSHOT" \
  --output ".bench/parity/$MODEL-mlx.json"
python3 scripts/verify_d3_parity.py compare \
  --reference ".bench/parity/$MODEL-reference.json" \
  --candidate ".bench/parity/$MODEL-mlx.json"
```

Capture defaults to the included smoke dataset; both captures must use the same cases and questions. Comparison loads no models. It checks pinned source/artifact identities, exact prompt text and token IDs, and coverage of choice, boolean, and score decisions. Every absolute probability delta and normalized expected-score delta must be at most **0.03**. A reference winner with a probability margin above **0.06** must remain the candidate winner. These fixed thresholds have no tolerance fallback; a mismatch fails the comparison.

Only the three BF16 D3 artifacts store source zero-centered RMS weights that require an in-memory conversion: cast the affected weights to FP32 **before** adding 1, matching the original Torch computation. Downloaded tensors remain unchanged, and ordinary Gated DeltaNet norm scales do not receive this offset. D3 Flash's already-offset norm weights remain unchanged; Flash is outside this BF16 capture workflow.
