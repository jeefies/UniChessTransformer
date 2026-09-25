"""UniChessKit 接入：把 T 的批量前向包装成 kit 的 BatchEvaluator / PlayerFactory / TrainTask。

kit 的 PUCT / C++ PUCT 负责搜索、跨局攒批、裁决与统计；``Kit.planes19.task`` 负责训练循环、
数据流与损失口径。本模块只做三件事：加载权重、暴露模型结构、转发参数。

三个入口：

- ``make_player_factory``：对局用（``python -m Kit match`` / Server 的 models/T）。
- ``make_task``：训练用（``python -m Kit train``），实现 ``Kit.train.TrainTask``。
- ``make_evaluators``：只要评估器（基准复现、离线评估）。

对局配置示例::

    {"factory": "Transformer.kit:make_player_factory",
     "root": "/home/jeefy/UniChess",
     "kwargs": {"preset": "max_mcts", "simulations": 800}}

训练配置示例（``configs/t20m.json``）：``{"factory": "Transformer.kit:make_task", ...}``。

**旧的网页对弈走的是本仓库自带的 C++ MCTS**，那套搜索无法与其他对局拼批，已随旧管线删除；
现在所有对局都是 kit 的 C++ PUCT（``search_impl="auto"``，与 kit 的 Python PUCT 逐位一致）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Optional

from Kit.planes19 import BatchFnEvaluator, make_search_player_factory

from .evaluator import TransformerEngine
from .model import (ChessTransformer, STRATIFIED_CFG, StratifiedChessTransformer, TransformerConfig,
                    cfg_conflicts, preset_config, stratified_20m)

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "runs" / "transformer_20m" / "best_model.pt"

# config.json（Server 预设）里与搜索相关的键 → make_player_factory 参数。
# description 只是说明文字，不进参数表；新增无法映射的键要显式忽略而不是静默丢弃。
_PRESET_IGNORED = {"description"}
_PRESET_KEYS = {"ckpt": "checkpoint", "mcts_sims": "simulations", "mcts_batch": "batch_size",
                "device": "device", "precision": "precision", "syzygy_path": "syzygy_path",
                "book_path": "book_path", "temperature": "temperature",
                "root_top_k": "root_top_k"}


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_preset(name: str) -> dict:
    """config.json 的预设 → make_player_factory 的关键字参数。"""
    presets = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if name not in presets:
        raise KeyError(f"config.json 没有预设 {name!r}（可选：{', '.join(presets)}）")
    unknown = set(presets[name]) - set(_PRESET_KEYS) - _PRESET_IGNORED
    if unknown:
        raise KeyError(f"预设 {name!r} 里有无法映射的键 {sorted(unknown)}"
                       f"（已知可忽略：{sorted(_PRESET_IGNORED)}）")
    return {_PRESET_KEYS[k]: v for k, v in presets[name].items() if k in _PRESET_KEYS}


# ---------------------------------------------------------------- 对局


def make_evaluators(checkpoint=DEFAULT_CKPT, *, device: str = "cuda", precision: str = "fp16",
                    max_batch: Optional[int] = 256) -> tuple:
    """(棋盘评估器, 编码评估器)，共用同一个模型。编码评估器的负载是 (19,8,8) float32。"""
    import numpy as np

    engine = TransformerEngine(_resolve(checkpoint), device=device, precision=precision)
    # 同一权重 + 同一精度才能拼进同一批前向
    key = f"T:{engine.ckpt_path}:{engine.precision}"
    return (BatchFnEvaluator(key, engine.evaluate_batch, max_batch),
            BatchFnEvaluator(key + ":planes", lambda xs: engine.evaluate_planes(np.stack(xs)),
                             max_batch))


def make_evaluator(checkpoint=DEFAULT_CKPT, *, device: str = "cuda", precision: str = "fp16",
                   max_batch: Optional[int] = 256) -> BatchFnEvaluator:
    return make_evaluators(checkpoint, device=device, precision=precision,
                           max_batch=max_batch)[0]


def make_player_factory(checkpoint=None, *, preset: Optional[str] = None, name: str = "T",
                        device: Optional[str] = None, precision: Optional[str] = None,
                        max_batch: Optional[int] = 256, search_impl: str = "auto", **search_kwargs):
    """``preset`` 取 config.json 的值作默认，显式参数优先；其余透传给
    ``make_search_player_factory``（simulations / batch_size / syzygy_path / book_path /
    book_plies / temperature / search_impl / PUCTConfig 字段）。"""
    # Server 的模型插件把 config.json 的预设 dict **原样**当 kwargs 传进来（不经 preset=，
    # 见 Server/models/__init__.py::resolve_kwargs）。这里与 preset= 路径共用同一套键映射，
    # 否则 ckpt / mcts_sims 会漏到 PUCTConfig 上（typcls: unexpected keyword）。
    search_kwargs = {_PRESET_KEYS.get(k, k): v for k, v in search_kwargs.items()
                     if k not in _PRESET_IGNORED}
    explicit = dict(checkpoint=checkpoint, device=device, precision=precision)
    opts = {**(load_preset(preset) if preset else {}),
            **{k: v for k, v in explicit.items() if v is not None},
            "search_impl": search_impl, **search_kwargs}
    evaluator, planes_evaluator = make_evaluators(
        opts.pop("checkpoint", DEFAULT_CKPT),
        device=opts.pop("device", "cuda"),
        precision=opts.pop("precision", "fp16"),
        max_batch=opts.pop("max_batch", max_batch))
    for key in ("syzygy_path", "book_path"):
        if opts.get(key):
            opts[key] = str(_resolve(opts[key]))
    return make_search_player_factory(name, evaluator, planes_evaluator=planes_evaluator, **opts)


# ---------------------------------------------------------------- 训练


#: 分层模型的三个专家与 curriculum 流里的阶段过滤一一对应
EXPERTS = {"opening": "opening", "middlegame": "middlegame", "endgame": "endgame"}


def _make_cfg(cfg) -> TransformerConfig:
    """``cfg`` 既可以是预设名，也可以是 TransformerConfig 字段的 dict。"""
    if isinstance(cfg, str):
        return preset_config(cfg)
    if isinstance(cfg, dict):
        known = {f.name for f in fields(TransformerConfig)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise KeyError(f"cfg 有未知字段 {unknown}（可选：{sorted(known)}）")
        base = replace(TransformerConfig(), **cfg)
        # 与预设工厂一致：d_ff / hidden_dim 缺一个时互相补齐
        if base.d_ff is None and base.hidden_dim is not None:
            base = replace(base, d_ff=base.hidden_dim)
        return base
    raise TypeError(f"cfg 只能是预设名或字段 dict，得到 {type(cfg).__name__}")


class TTrainAdapter:
    """``Kit.planes19.task`` 需要的模型适配器（见该模块 docstring）。

    ``stratified``：True 建三专家集合，False 建单个 ``ChessTransformer``。
    ``trainable_expert``：只训练某个专家（T 的 curriculum 配方），其余专家的参数冻结——
    导出的 checkpoint 仍是**完整**的三专家 state_dict，Server 与 kit 都能直接加载。
    """

    def __init__(self, *, cfg: str = "transformer_20m", stratified: bool = False,
                 base_ckpt=None, trainable_expert: Optional[str] = None,
                 strict_base: bool = False):
        if stratified:
            self.model_type = "stratified_20m"
            self.cfg = STRATIFIED_CFG
        else:
            self.model_type = None
            self.cfg = _make_cfg(cfg)
        if trainable_expert is not None:
            if not stratified:
                raise ValueError("trainable_expert 只对分层模型有意义")
            if trainable_expert not in EXPERTS:
                raise KeyError(f"没有专家 {trainable_expert!r}（可选：{sorted(EXPERTS)}）")
        self.stratified = bool(stratified)
        self.trainable_expert = trainable_expert
        self.base_ckpt = str(base_ckpt) if base_ckpt else None
        self.strict_base = strict_base
        self.step = 0

    # ------------------------------------------------------------ 模型
    def build(self):
        step = 0
        if self.stratified:
            model = stratified_20m()
            if self.base_ckpt:
                model.load_base_checkpoint(_resolve(self.base_ckpt))
                step = _ckpt_step(_resolve(self.base_ckpt))
        else:
            model = ChessTransformer(self.cfg)
            if self.base_ckpt:
                step = _ckpt_step(_resolve(self.base_ckpt))
                model.load_state_dict(_clean(self.base_ckpt), strict=self.strict_base)
        self.step = int(step)
        return model

    def trainable(self, model):
        """只训练指定专家；其余参数由 ``Planes19Task.param_groups`` 置 requires_grad_(False)。

        返回的就是模型里的那个子模块本身，训练更新直接落在完整模型上，导出时不需要回写。
        """
        if self.trainable_expert is None:
            return model
        return getattr(model, self.trainable_expert)

    def forward(self, model, x, bucket=None, *, mlh: bool = False):
        if bucket is not None:
            raise ValueError("Transformer 没有分桶头（那是 R 的 num_buckets）")
        if mlh:
            return model(x, return_mlh=True)
        return model(x)

    def export(self, model, step: int) -> dict:
        """与旧脚本一致：完整 state_dict + cfg + step；分层模型带 ``preset`` 供加载端识别。

        curriculum 配方只训一个专家，但训练用的模型本来就是完整的三专家集合（其余专家参数
        被冻结），所以 ``model.state_dict()`` 已经是 Server / kit 能直接加载的完整权重，
        与旧 ``save_curriculum_checkpoint`` 的产物结构相同。
        """
        return {"model": model.state_dict(), "cfg": asdict(self.cfg), "step": int(step),
                "preset": "stratified_20m" if self.stratified else None}


def _ckpt_step(path) -> int:
    import torch
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    return int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0


def _clean(ckpt_path) -> dict:
    """checkpoint → 干净的 model state_dict（去 ``_orig_mod.`` 前缀）。"""
    import torch
    ckpt = torch.load(_resolve(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

def make_adapter(**kwargs) -> TTrainAdapter:
    """``Kit.planes19.task`` 的 ``model.factory``：返回训练用的模型适配器。

    参数：``cfg``（预设名或 TransformerConfig 字段 dict）、``stratified``（三专家）、
    ``base_ckpt``（底座权重）、``trainable_expert``（只训某个专家）、``strict_base``。
    """
    allowed = {"cfg", "stratified", "base_ckpt", "trainable_expert", "strict_base"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"make_adapter 收到未知参数 {unknown}")
    kw = {"cfg": kwargs.pop("cfg", "transformer_20m"),
          "stratified": kwargs.pop("stratified", False),
          "base_ckpt": kwargs.pop("base_ckpt", kwargs.pop("checkpoint", None)),
          "trainable_expert": kwargs.pop("trainable_expert", None),
          "strict_base": kwargs.pop("strict_base", False)}
    return TTrainAdapter(**kw)


def make_task(runtime=None, **kwargs):
    """``Transformer.kit:make_task`` → 一个 ``Kit.train.TrainTask``。

    模型相关的键（``cfg`` / ``stratified`` / ``base_ckpt`` / ``trainable_expert`` /
    ``strict_base``）走顶层参数并转发给 ``make_adapter``，其余（``data`` / ``loss`` /
    ``validation``）原样交给 ``Planes19Task``——这样配置文件里模型与数据各写一边，
    不会和 ``model.kwargs`` 的嵌套混在一起。
    """
    from Kit.planes19.task import Planes19Task

    model_keys = ("cfg", "stratified", "base_ckpt", "trainable_expert", "strict_base")
    model_kw = dict(kwargs.pop("model_kwargs", None) or {})
    merged = {**model_kw, **{k: kwargs.pop(k) for k in model_keys if k in kwargs}}
    merged.setdefault("cfg", "transformer_20m")
    merged.setdefault("stratified", False)
    return Planes19Task(model={"factory": "Transformer.kit:make_adapter", "kwargs": merged},
                        runtime=runtime, **kwargs)
