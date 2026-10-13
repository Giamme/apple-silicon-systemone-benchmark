# Third-party projects

This repository provides a benchmark harness. It downloads no model code or weights into Git and does not relicense them. Built-in setup currently fetches:

| Candidate | Code or package | Weights |
| --- | --- | --- |
| Laya | [laya on PyPI](https://pypi.org/project/laya/) | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) |
| Von | [wfzyx/von](https://github.com/wfzyx/von) | [wfzyx/von](https://huggingface.co/wfzyx/von) |
| Kev | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) and its linked base model |
| Rizzo Flow | [Rizzo-AI-Academy/rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) | Downloaded from the upstream model repository by the Rizzo Flow CLI |

Each dependency and weight file retains its own terms. Check its current upstream license before distributing a bundle, container, or modified copy. The Apache-2.0 license in this repository applies to this harness and its synthetic examples.

## Extended profile artifacts

The source and converted artifact are separate identities. Exact revisions and selected precision are in [runtime_catalog.py](runtime_catalog.py); dependency closure is pinned in the three `runtimes/*/uv.lock` files. Setup downloads weights into the ignored `.bench/` directory.

| Candidate | Original source | Selected artifact |
| --- | --- | --- |
| D3 Edge | [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) | [ahmedandaloes/d3-edge-mlx](https://huggingface.co/ahmedandaloes/d3-edge-mlx) |
| D3 Lite | [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) | [ahmedandaloes/d3-lite-mlx](https://huggingface.co/ahmedandaloes/d3-lite-mlx) |
| D3 Mini | [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) | [ahmedandaloes/d3-mini-mlx](https://huggingface.co/ahmedandaloes/d3-mini-mlx) |
| D3 Flash | [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash) | [thoddnn/d3-flash-q4](https://huggingface.co/thoddnn/d3-flash-q4) |
| Clef Flash | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit) |
| Liquid D1 3B | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | [LiquidAI/d1-3B-MLX-fp16](https://huggingface.co/LiquidAI/d1-3B-MLX-fp16) |
| Liquid D1 Omni 600M | [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | [LiquidAI/d1-omni-600M-MLX-fp16](https://huggingface.co/LiquidAI/d1-omni-600M-MLX-fp16) |
| JEV 9B | [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) | [nativ-community/JEV-9B-MLX-NVFP4](https://huggingface.co/nativ-community/JEV-9B-MLX-NVFP4) |

D3/Clef use MLX 0.32.3, mlx-lm 0.32.0, mlx-vlm 0.7.4, and Transformers 5.17.0. Liquid uses MLX 0.32.3, Transformers 5.14.0, and [Blaizzy/mlx-vlm at the pinned commit](https://github.com/Blaizzy/mlx-vlm/tree/952d4f6bc65bd5095e74abe71c78038764ee08aa). JEV uses MLX 0.32.3, Transformers 5.14.0, and [Lazarus-931/mlx-vlm at the pinned commit](https://github.com/Lazarus-931/mlx-vlm/tree/597941d0d4eb900f1d066eba8d89abf474c0dd42). Native reference prompt and readout implementations retain their upstream notices and terms. D3 Flash setup also obtains the original pinned readout for verification. The JEV score compatibility mapping is a benchmark adapter choice; the source model's ordered criteria become numeric string choice keys.

The five D3/Clef original and selected artifact repositories include Apache-2.0 license files. D3 native prompt formatting and decision normalization derive from the pinned D3 Flash `d3_format.py` and `d3_runtime.py`; attribution is retained in the backend source. Clef loads its pinned and hash-verified artifact `clef_mlx.py`, derived from the Cloudflare joint schema model. Their Qwen3.5 base family uses Apache-2.0. The Liquid source/artifacts include their `LICENSE` files. The selected JEV artifact declares Apache-2.0 in its model card and does not include a standalone `LICENSE` file; verify the original model and backbone terms before redistribution. No model weights or copied distribution bundles are committed here.
