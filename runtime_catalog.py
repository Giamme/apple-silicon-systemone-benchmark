"""Pinned managed model recipes; safe to import without ML dependencies."""
from __future__ import annotations

import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / '.bench'


def _model(model_id: str, family: str, source: str, source_rev: str,
           artifact: str, artifact_rev: str, precision: str, quantization: str = 'none') -> dict:
    return {'id': model_id, 'family': family, 'source_repo': source,
            'source_revision': source_rev, 'artifact_repo': artifact,
            'artifact_revision': artifact_rev, 'precision': precision,
            'quantization': quantization, 'environment': 'd3-clef' if family in ('d3', 'clef') else family,
            'backend_module': 'runtime_backends.d3_clef' if family in ('d3', 'clef') else f'runtime_backends.{family}',
            'score_mapping': 'native', 'decision_mode': 'native_readout_no_generation'}


MODELS = {
    'd3-edge': _model('d3-edge', 'd3', 'vllm-sr/d3-edge', 'fe71f8e1dad1d0198386de78345b13f25c319b36', 'ahmedandaloes/d3-edge-mlx', '33e8fdb0adb076d7af3d5d1a0049b60cdc1d7939', 'BF16'),
    'd3-lite': _model('d3-lite', 'd3', 'vllm-sr/d3-lite', '99e5b1ce5a718454cd112cfe00caa8d2ae8df2ff', 'ahmedandaloes/d3-lite-mlx', '235b1fb6ed8d20ab911e10801a4f4cec9cadc24a', 'BF16'),
    'd3-mini': _model('d3-mini', 'd3', 'vllm-sr/d3-mini', '6d8971f464d6f9d4c5af7d9a8d75f3aa03be4f59', 'ahmedandaloes/d3-mini-mlx', 'c645173f5729051d473916283f9e37208b5113bb', 'BF16'),
    'd3-flash': _model('d3-flash', 'd3', 'vllm-sr/d3-flash', '6838193a0185c0ecfecab937896059d2c3377d61', 'thoddnn/d3-flash-q4', '5b9c0de2e6640380b50491ee2f9eef8eece100e3', 'backbone INT4, vision BF16, readout FP32', 'affine 4-bit, group 64'),
    'clef-flash': _model('clef-flash', 'clef', 'Cloudflare/clef-flash', '8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3', 'mlx-community/clef-flash-4bit', 'b5adbe2c3a95200d4bf04a0727c8ec457157d5e1', 'INT4 backbone, BF16 head', 'affine 4-bit, group 64'),
    'd1-3b': _model('d1-3b', 'liquid', 'LiquidAI/d1-3B', '051bcc464b01b9f92942b364d9586b0ef5912432', 'LiquidAI/d1-3B-MLX-fp16', 'c7b140a64fc7466d5f9d08be19f261a704febef1', 'FP16'),
    'd1-omni-600m': _model('d1-omni-600m', 'liquid', 'LiquidAI/d1-omni-600M', '02b55d7076f15129e59ab3f94783f32c4b088674', 'LiquidAI/d1-omni-600M-MLX-fp16', '05e99e49eb1799f1f9f98ec619b6fef2b0dd0960', 'FP16'),
    'jev-9b': _model('jev-9b', 'jev', 'autotrust/JEV-9B', '7ac669b8fb476ba926491e3d1c39f9330d68dbed', 'nativ-community/JEV-9B-MLX-NVFP4', '1680ddb8dded11933558fa57f79a93bbdfcccd46', 'NVFP4 backbone, BF16 readout, FP32 probability math', 'NVFP4 4-bit, group 16'),
}
for _spec in MODELS.values():
    _spec['allocator_policy'] = {'free_cache_limit_bytes': 256 * 1024**2,
                                 'scope': 'unused_mlx_buffers', 'enforcement': 'soft_limit'}
    _spec['runtime_pins'] = {'python': '3.12', 'mlx': '0.32.3'}
    if _spec['environment'] == 'd3-clef':
        _spec['runtime_pins'].update({'mlx-lm': '0.32.0', 'mlx-vlm': '0.7.4', 'transformers': '5.17.0'})
    else:
        _spec['runtime_pins']['transformers'] = '5.14.0'
        _spec['runtime_pins']['mlx-vlm_source'] = (
            'https://github.com/Blaizzy/mlx-vlm.git@952d4f6bc65bd5095e74abe71c78038764ee08aa'
            if _spec['environment'] == 'liquid' else
            'https://github.com/Lazarus-931/mlx-vlm.git@597941d0d4eb900f1d066eba8d89abf474c0dd42'
        )

MODELS['clef-flash']['native_probability_decimals'] = 4
MODELS['jev-9b'].update(score_mapping='ordered_criteria_via_choice', choice_limit=16,
                        conversion_source_revision='b63f651ce8ed64481d3f5e73ecdb05f740042f01')
MODELS['d3-flash']['verification_files'] = [
    {'filename': 'readout.safetensors', 'sha256': '8bbc9a9df4e1b7df85a80a1091977092b938d8d2d2869dfaeb8fd36063cf322f'}
]


def get_model(model_id: str) -> dict:
    if not isinstance(model_id, str):
        raise ValueError('runtime model id must be a string')
    try:
        return copy.deepcopy(MODELS[model_id])
    except KeyError as exc:
        raise ValueError(f'Unknown runtime model: {model_id}') from exc


def snapshot_path(spec: dict) -> Path:
    return WORK / 'models' / spec['id'] / spec['artifact_revision']
