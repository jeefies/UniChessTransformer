"""UniChessKit 接入：把 TransformerEngine 的批量前向包装成 kit 的 BatchEvaluator / PlayerFactory。

kit 负责搜索（PUCT，与 Python MCTS 同一算法）、跨局攒批、裁决与统计；本模块只负责加载权重。
C++ MCTS 不在这条路径上（它自带搜索循环，无法与其他对局拼批），因此加载时关掉以免 JIT 编译。

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
                "book_path": "book_path", "temperature": "temperature"}


def load_preset(name: str) -> dict:
    presets = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if name not in presets:
        raise KeyError(f"config.json 没有预设 {name!r}（可选：{', '.join(presets)}）")
    return {_PRESET_KEYS[k]: v for k, v in presets[name].items() if k in _PRESET_KEYS}


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "cuda", precision: str = "fp16",
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    from unichess_t.engine.engine import TransformerEngine
    ckpt = _resolve(checkpoint).resolve()
    engine = TransformerEngine(ckpt, device=device, precision=precision,
                               mcts_sims=0, use_cpp_mcts=False)
    # 同一权重 + 同一精度才能拼进同一批前向
    return BatchFnEvaluator(f"T:{ckpt}:{precision}", engine.evaluate_batch, max_batch)


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "T",
                        device: Optional[str] = None, precision: Optional[str] = None,
                        max_batch: Optional[int] = 256, **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余参数透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / PUCTConfig 字段）。"""
    explicit = dict(checkpoint=checkpoint, device=device, precision=precision)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None}, **search_kwargs}
    evaluator = make_evaluator(opts.pop("checkpoint", DEFAULT_CKPT),
                               device=opts.pop("device", "cuda"),
                               precision=opts.pop("precision", "fp16"), max_batch=max_batch)
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, **opts)
