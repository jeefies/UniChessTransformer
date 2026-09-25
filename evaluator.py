"""T 的批量前向：给 kit 的 Player / Trainer 用（``evaluate_planes`` / ``evaluate_batch``）。

与旧 ``unichess_t/engine/engine.py`` 的区别：那是一个完整引擎（开局书、Syzygy、
自带 C++/Python MCTS、四级优先级链、采样），对弈路径已经整体交给 kit 的 PUCT；这里只保留
**批量前向**，不再有第二个搜索实现。

口径与旧引擎逐位一致（推理黄金对拍就是拿它当基准）：

- ``precision="fp16"``（默认）走 ``torch.autocast(cuda, float16)``，``fp32`` 完全不开 autocast；
- 三个头都在 ``.float()`` 之后 softmax，CPU 上取回；
- 分层路由模型从平面 0-11 数子力决定专家，不需要棋盘，kit 的 C++ PUCT 只给平面也能走。
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .model import TransformerConfig, load_model

ROOT = Path(__file__).resolve().parent


def _resolve_ckpt(path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"权重不存在：{p}")
    return p


def load_checkpoint(path) -> tuple[torch.nn.Module, TransformerConfig, int, dict]:
    """→ (eval 模式的模型, TransformerConfig, step, 原始 checkpoint 字典)。"""
    return load_model(_resolve_ckpt(path), device="cpu")


def describe(path) -> dict:
    """只读权重元信息（不建模型也能报结构信息；用于 Server 的模型列表与配置自检）。"""
    _, cfg, step, ckpt = load_checkpoint(path)
    return {"cfg": asdict(cfg), "step": step, "name": f"{cfg.num_layers}x{cfg.d_model}",
            "stratified": any(k.startswith(("experts.", "opening."))
                              for k in ckpt.get("model", {})),
            "has_mlh": any("mlh_head" in k for k in ckpt.get("model", {}))}


class TransformerEngine:
    """只做批量前向的评估器。

    ``cfg`` 是加载的 ``TransformerConfig``（分层模型取第一个专家的）；``step`` 是 checkpoint
    里记的训练步。构造时把模型放到目标设备并置 eval 模式，之后只读。
    """

    def __init__(self, ckpt_path, *, device: str = "auto", precision: str = "fp16"):
        self.device = torch.device(device if device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"未知精度 {precision!r}（可选 fp16 / bf16 / fp32）")
        self.precision = precision
        self.model, self.cfg, self.step, self.ckpt = load_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.ckpt_path = str(_resolve_ckpt(ckpt_path))
        # 只有 CUDA 上的半精度 autocast 有意义；CPU 上开 fp16 只会更慢更差
        self.dtype = ({"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
                      if self.device.type == "cuda" else None)
        self.stratified = any(k.startswith(("experts.", "opening."))
                              for k in self.ckpt.get("model", {}))

    # ---------------------------------------------------------------- 前向
    @torch.no_grad()
    def evaluate_planes(self, xs: np.ndarray):
        """xs: (N, 19, 8, 8) float32 → (policy[N,4096], promo[N,4], wdl[N,3]) 行棋方视角概率。"""
        x = torch.from_numpy(np.ascontiguousarray(xs, dtype=np.float32)).to(self.device)
        if self.dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                p_l, pr_l, w_l = self.model(x)
        else:
            p_l, pr_l, w_l = self.model(x)
        return (torch.softmax(p_l.float(), dim=-1).cpu().numpy(),
                torch.softmax(pr_l.float(), dim=-1).cpu().numpy(),
                torch.softmax(w_l.float(), dim=-1).cpu().numpy())

    @torch.no_grad()
    def evaluate_batch(self, boards):
        from Kit.planes19 import encode
        return self.evaluate_planes(np.stack([encode(b) for b in boards]))

    def evaluate(self, board):
        """单个局面 → (policy[4096], promo[4], wdl[3])。"""
        p, pr, w = self.evaluate_batch([board])
        return p[0], pr[0], w[0]

    # ---- 与 TrainTask 的接口 ----
    def forward(self, model, x, bucket=None, *, mlh: bool = False):
        """TrainTask 约定的前向：x 是 (N,19,8,8) float32。

        ``mlh=True`` 时多返回一路 moves-left logit（P3 配方）。
        """
        if bucket is not None:
            raise ValueError("Transformer 没有分桶头（那是 R 的 num_buckets）")
        if mlh:
            return model(x, return_mlh=True)
        return model(x)

    def export(self, model, step: int) -> dict:
        """写出的 checkpoint 与旧脚本一致：``{"model": state_dict, "cfg": cfg, "step": ...}``，
        外加 ``preset`` 便于加载端识别分层模型。"""
        out = {"model": model.state_dict(), "cfg": asdict(self.cfg), "step": int(step),
               "preset": "stratified_20m" if self.stratified else None}
        return out


__all__ = ["TransformerEngine", "TransformerConfig", "describe", "load_checkpoint"]
