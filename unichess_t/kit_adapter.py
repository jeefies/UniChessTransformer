"""UniChessKit 接入：把 TransformerEngine 的批量前向包装成 kit 的 BatchEvaluator / PlayerFactory。

kit 负责搜索（PUCT，与 Python MCTS 同一算法）、跨局攒批、裁决与统计；本模块只负责加载权重。
本仓库自带的 C++ MCTS 不在这条路径上（它自带搜索循环，无法与其他对局拼批），加载时关掉以免 JIT 编译；
搜索默认用 kit 的 C++ PUCT（``search_impl="auto"``，与 kit 的 Python PUCT 逐位一致，叶子编码由 C++ 写出、
经 ``evaluate_planes`` 前向）。``search_impl="python"`` 可切回 Python PUCT 对照，结果相同，宜放在 runtime 里。

批量对弈配置示例（``python -m unichess_kit.match``）::

    {"factory": "unichess_t.kit_adapter:make_player_factory",
     "root": "/home/jeefy/UniChess/Transformer",
     "kwargs": {"preset": "max_mcts", "simulations": 800}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from unichess_kit.contrib.planes19 import BatchFnEvaluator, make_search_player_factory

KIT_SPI_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "runs" / "transformer_20m" / "best_model.pt"

# config.json 预设里与搜索相关的键 → make_player_factory 参数
_PRESET_KEYS = {"ckpt": "checkpoint", "mcts_sims": "simulations", "mcts_batch": "batch_size",
                "device": "device", "precision": "precision", "syzygy_path": "syzygy_path",
                "book_path": "book_path", "temperature": "temperature",
                "root_top_k": "root_top_k"}


def load_preset(name: str) -> dict:
    presets = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if name not in presets:
        raise KeyError(f"config.json 没有预设 {name!r}（可选：{', '.join(presets)}）")
    return {_PRESET_KEYS[k]: v for k, v in presets[name].items() if k in _PRESET_KEYS}


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_evaluators(checkpoint=DEFAULT_CKPT, *, device: str = "cuda", precision: str = "fp16",
                    max_batch: Optional[int] = 256) -> tuple:
    """(棋盘评估器, 编码评估器)，共用同一个引擎。编码评估器的负载是 (19, 8, 8) float32。"""
    import numpy as np

    from unichess_t.engine.engine import TransformerEngine
    ckpt = _resolve(checkpoint).resolve()
    engine = TransformerEngine(ckpt, device=device, precision=precision,
                               mcts_sims=0, use_cpp_mcts=False)
    # 同一权重 + 同一精度才能拼进同一批前向
    key = f"T:{ckpt}:{precision}"
    return (BatchFnEvaluator(key, engine.evaluate_batch, max_batch),
            BatchFnEvaluator(key + ":planes", lambda xs: engine.evaluate_planes(np.stack(xs)),
                             max_batch))


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "cuda", precision: str = "fp16",
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    return make_evaluators(checkpoint, device=device, precision=precision,
                           max_batch=max_batch)[0]


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "T",
                        device: Optional[str] = None, precision: Optional[str] = None,
                        max_batch: Optional[int] = 256, **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余参数透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / search_impl / PUCTConfig 字段）。"""
    explicit = dict(checkpoint=checkpoint, device=device, precision=precision)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None}, **search_kwargs}
    evaluator, planes_evaluator = make_evaluators(
        opts.pop("checkpoint", DEFAULT_CKPT), device=opts.pop("device", "cuda"),
        precision=opts.pop("precision", "fp16"), max_batch=max_batch)
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, planes_evaluator=planes_evaluator, **opts)
