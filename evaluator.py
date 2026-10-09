"""T 的批量前向：给 kit 的 Player / Trainer 用（``evaluate_planes`` / ``evaluate_batch``）。

与旧 ``unichess_t/engine/engine.py`` 的区别：那是一个完整引擎（开局书、Syzygy、
自带 C++/Python MCTS、四级优先级链、采样），对弈路径已经整体交给 kit 的 PUCT；这里只保留
**批量前向**，不再有第二个搜索实现。

口径与旧引擎逐位一致（推理黄金对拍就是拿它当基准）：

- ``precision="fp16"``（默认）走 ``torch.autocast(cuda, float16)``，``fp32`` 完全不开 autocast；
- 三个头都在 ``.float()`` 之后 softmax，CPU 上取回；
- 分层路由模型从平面 0-11 数子力决定专家，不需要棋盘，kit 的 C++ PUCT 只给平面也能走。

CUDA graph（2026-10-09，``_CudaGraphRunner``，**默认关**）：自对弈/arena 的前向是
~300 次 kernel 发射的 launch-bound 批，§18.27 微基准（干净 GPU、m=64）显示图回放
5.83×/7.26×。工程化按 ``(N, k0,k1,k2)``（批大小 × 三专家各行数）分桶捕获：
拆批/散射用静态 GPU 索引张量在捕获区内做，H2D 全部在图外；与 eager 逐位一致
（同形状回放同 kernel，单测 17 种形状 + 两种回退路径全绿）。**原位实测前向
7.5 ms → 0.75-1.8 ms（6-8×）**、capture-on-repeat 后捕获税仅 15-25 s/worker。
但自对弈的关键路径是每次搜索 ~375 个顺序减半波次往返（§18.33），前向与编排
重叠 ⇒ 端到端 0.93-0.98×，故默认关；待 C++ 多波合并后再评估。完整数据见
experiments.md §18.31/33。softmax/取回保持在图外与旧路径一致。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .model import TransformerConfig, load_model

ROOT = Path(__file__).resolve().parent

#: 图前向的启用开关（环境变量）。**默认关**：2026-10-09 完整实测（§18.33）——
#: 图机制本身有效（原位前向 7.5 ms → 0.75-1.8 ms，6-8×，逐位一致），但自对弈的
#: 关键路径是**每次搜索 ~375 个顺序减半波次往返**（Python↔C++ 协程周期 ~0.32ms，
#: 与批大小无关），模型前向与波次编排**重叠**，不是关键路径——故端到端 0.93-0.98×。
#: 留待 C++ 多波合并（kg_collect 一次多收几个波）落地、前向重新成为瓶颈时再开；
#: 届时 capture-on-repeat（_GRAPH_CAPTURE_AFTER）已把捕获税压到 15-25 s/worker。
#: ``UNICHESS_T_NO_CUDAGRAPH=1`` 可随时硬关。
_GRAPH_ON_ENV = "UNICHESS_T_CUDAGRAPH"
_NO_GRAPH_ENV = "UNICHESS_T_NO_CUDAGRAPH"
#: 建图的批大小上限（与 BatchFnEvaluator 的 max_batch=256 对齐；再大收益趋零）
_GRAPH_MAX_N = 256
#: 桶数上限（每个桶持一份图私池显存；实测 ~0.3-0.9 GB / 512 桶 / worker）。
#: 512 是实测生产 key 空间（单 worker 最多 ~650 个 (N,计数) 组合）的上沿
_GRAPH_MAX_BUCKETS = 512
#: 同一形状第几次出现才建图。生产 key 空间是长尾（~29k 次调用摊到 ~500 个形状），
#: 一次性形状走 eager 不付捕获税；捕获一次要预热+autotune，实测 ~0.3-1 s/桶
_GRAPH_CAPTURE_AFTER = 2
#: 进程退出时打印图前向统计（桶数/回退/命中/批大小分布），用于生产 key 空间普查
_GRAPH_STATS_ENV = "UNICHESS_T_GRAPH_STATS"


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


def _routes_from_planes(x32: np.ndarray) -> list:
    """平面 0-11 数子力 → 专家号（与模型内默认路由同一判据，CPU 侧零同步）。"""
    pc = x32[:, :12].sum(axis=(1, 2, 3))
    return np.where(pc >= 24, 0, np.where(pc > 12, 1, 2)).tolist()


class _CudaGraphRunner:
    """按 (N, k0,k1,k2) 分桶的 CUDA graph 专家前向。

    捕获区只含 GPU 操作：``x[idx_e]``（index_select）→ 专家前向（autocast 与原路径
    一致）→ ``p[idx_e] = sub_p``（index_put）。索引是**静态 GPU 张量**，每次调用在
    图外 H2D 拷进去；输出张量在捕获区内分配（图私池），回放后直接读。
    与 eager 逐位一致：index_select/put 是精确拷贝，专家看到的行与顺序完全相同。
    """

    def __init__(self, engine: "TransformerEngine"):
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.dtype = engine.dtype                    # None = 不开 autocast（fp32 模式）
        self.experts = list(engine.model.experts) if engine.stratified else [engine.model]
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets: dict = {}                      # key -> dict | None（None = 已回退）
        self.lock = threading.Lock()
        self.hits = 0
        self.fallbacks = 0
        self.capture_fails = 0
        self.capture_t = 0.0                       # 捕获税累计（秒）
        self.seen: dict = {}                       # key -> 出现过几次（复现才建图）
        # 生产形态普查：批大小分布 + 单/混合专家占比
        self.n_le16 = self.n_le64 = self.n_gt64 = 0
        self.single = self.mixed = 0

    # ---------------------------------------------------------------- 捕获区
    def _region(self, x: torch.Tensor, idx: list, k: tuple):
        """静态输入 → (p, pr, v)；与 StratifiedChessTransformer 拆批路径同序同核。"""
        p = pr = v = None
        for e, expert in enumerate(self.experts):
            if idx[e] is None:
                continue
            sub_x = x[idx[e]]
            if self.dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    sub_p, sub_pr, sub_v = expert(sub_x)
            else:
                sub_p, sub_pr, sub_v = expert(sub_x)
            if p is None:
                n = x.shape[0]
                p = torch.empty(n, 4096, device=x.device, dtype=sub_p.dtype)
                pr = torch.empty(n, 4, device=x.device, dtype=sub_pr.dtype)
                v = torch.empty(n, 3, device=x.device, dtype=sub_v.dtype)
            p[idx[e]] = sub_p
            pr[idx[e]] = sub_pr
            v[idx[e]] = sub_v
        return p, pr, v

    def _capture(self, key: tuple):
        """建一个桶；任何失败/超限返回 None（该 key 此后永久 eager）。"""
        n = key[0]
        live = sum(1 for b in self.buckets.values() if b)
        if n > _GRAPH_MAX_N or live >= _GRAPH_MAX_BUCKETS:
            self.fallbacks += 1
            return None
        k = key[1:]
        try:
            x = torch.zeros(n, 19, 8, 8, device=self.device, dtype=torch.float32)
            idx = [torch.zeros(int(kk), dtype=torch.long, device=self.device)
                   if kk else None for kk in k]
            # 预热：侧流上跑几遍，让 cudnn/cublas 定算法（捕获期不接受 autotune）
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._region(x, idx, k)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                p, pr, v = self._region(x, idx, k)
            return {"x": x, "idx": idx, "graph": g, "p": p, "pr": pr, "v": v}
        except Exception:                              # noqa: BLE001 —— 任何捕获失败都回退
            self.capture_fails += 1
            return None

    # ---------------------------------------------------------------- 前向
    def forward(self, x32: np.ndarray, routes: list):
        """→ (p_l, pr_l, v_l)，与 eager 逐位一致（同 key 回放同 kernel）。

        整段加锁：Server 的 T engine 是单实例多线程共享，eager 路径天然线程安全，
        但同一 bucket 的并发回放会互相覆盖输出张量。
        """
        n = x32.shape[0]
        k = tuple(routes.count(e) for e in range(len(self.experts)))
        if sum(k) != n:
            raise ValueError(f"路由计数 {k} 与批大小 {n} 不符（routes={routes[:8]}…）")
        if n <= 16:
            self.n_le16 += 1
        elif n <= 64:
            self.n_le64 += 1
        else:
            self.n_gt64 += 1
        if sum(1 for kk in k if kk) <= 1:
            self.single += 1
        else:
            self.mixed += 1
        key = (n, *k)
        with self.lock:
            cnt = self.seen.get(key, 0) + 1
            self.seen[key] = cnt
            bucket = self.buckets.get(key, False)     # False = 尚未决定
            # 复现形状才建图：首次见到的走 eager（生产 key 空间是长尾，
            # 一次性形状不付捕获税；capture 每次要预热+autotune，~0.3-1s）
            if bucket is False and cnt >= _GRAPH_CAPTURE_AFTER:
                _tc = time.perf_counter()
                bucket = self._capture(key)
                self.capture_t += time.perf_counter() - _tc
                self.buckets[key] = bucket            # dict=建成 / None=失败或超限
            if bucket is None or bucket is False:     # False=首次出现（缓建）/ None=已判缓
                self.fallbacks += 1
                return self._eager(x32, routes)
            self.hits += 1
            bucket["x"].copy_(torch.from_numpy(x32))
            for e, ie in enumerate(bucket["idx"]):
                if ie is not None:
                    ie.copy_(torch.tensor([i for i, r in enumerate(routes)
                                           if r == e], dtype=torch.long))
            bucket["graph"].replay()
            return bucket["p"], bucket["pr"], bucket["v"]

    def _eager(self, x32: np.ndarray, routes: list):
        x = torch.from_numpy(x32).to(self.device)
        if self.dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self._model_eager(x, routes)
        return self._model_eager(x, routes)

    def _model_eager(self, x: torch.Tensor, routes: list):
        if self.engine.stratified:
            return self.model(x, route_indices=routes)
        return self.model(x)

    def stats(self) -> dict:
        return {"buckets": sum(1 for b in self.buckets.values() if b),
                "eager_keys": sum(1 for b in self.buckets.values() if b is None),
                "hits": self.hits, "fallbacks": self.fallbacks,
                "capture_fails": self.capture_fails,
                "capture_t": round(self.capture_t, 1),
                "n_le16": self.n_le16, "n_le64": self.n_le64, "n_gt64": self.n_gt64,
                "single": self.single, "mixed": self.mixed}


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
        # 原位前向计时（诊断用：evaluate_planes 的墙钟累计，随 graph stats 转储）
        self._fwd_t = 0.0
        self._fwd_n = 0
        # CUDA graph 前向：**默认关**（生产 A/B 无增益，见 _GRAPH_ON_ENV 注释与
        # experiments.md §18.31）；置 UNICHESS_T_CUDAGRAPH=1 启用，CPU 自动走 eager
        self.graphs: Optional[_CudaGraphRunner] = None
        if (self.device.type == "cuda" and os.environ.get(_GRAPH_ON_ENV)
                and not os.environ.get(_NO_GRAPH_ENV)):
            try:
                self.graphs = _CudaGraphRunner(self)
            except Exception:                          # noqa: BLE001 —— 建不起就纯 eager
                self.graphs = None
        # 统计转储与图开关解耦：eager 臂也要量原位前向耗时（诊断 A/B 用）
        if os.environ.get(_GRAPH_STATS_ENV):
            import atexit
            atexit.register(self._dump_graph_stats)

    # ---------------------------------------------------------------- 前向
    @torch.no_grad()
    def evaluate_planes(self, xs: np.ndarray):
        """xs: (N, 19, 8, 8) float32 → (policy[N,4096], promo[N,4], wdl[N,3]) 行棋方视角概率。

        分层模型的路由在 CPU 侧按同一子力数判据直接算好再传 ``route_indices``（模型内
        GPU 归约 + 每行一次 ``.item()`` 同步的默认路径在大批量下是纯开销）。子力数是
        0/1 平面的小整数和，fp32 求和任意顺序都精确，判据与顺序完全一致，**数值逐位不变**。
        CUDA graph 开启时走 ``_CudaGraphRunner``（同形状回放同 kernel，与 eager 逐位一致）。
        """
        x32 = np.ascontiguousarray(xs, dtype=np.float32)
        routes = _routes_from_planes(x32) if self.stratified else [0] * len(x32)
        _t0 = time.perf_counter()
        if self.graphs is not None:
            p_l, pr_l, w_l = self.graphs.forward(x32, routes)
        elif self.stratified:
            x = torch.from_numpy(x32).to(self.device)
            if self.dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    p_l, pr_l, w_l = self.model(x, route_indices=routes)
            else:
                p_l, pr_l, w_l = self.model(x, route_indices=routes)
        else:
            x = torch.from_numpy(x32).to(self.device)
            if self.dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    p_l, pr_l, w_l = self.model(x)
            else:
                p_l, pr_l, w_l = self.model(x)
        self._fwd_t += time.perf_counter() - _t0
        self._fwd_n += 1
        return (torch.softmax(p_l.float(), dim=-1).cpu().numpy(),
                torch.softmax(pr_l.float(), dim=-1).cpu().numpy(),
                torch.softmax(w_l.float(), dim=-1).cpu().numpy())

    def graph_stats(self) -> Optional[dict]:
        return self.graphs.stats() if self.graphs is not None else None

    def _dump_graph_stats(self) -> None:
        try:
            fwd = (f"fwd: n={self._fwd_n} total={self._fwd_t:.1f}s "
                   f"avg={self._fwd_t / max(1, self._fwd_n) * 1000:.2f}ms/call"
                   if self._fwd_n else "fwd: n=0")
            gst = self.graphs.stats() if self.graphs is not None else {"graphs": "off"}
            print(f"[T graph stats] pid={os.getpid()} {gst} {fwd}",
                  file=sys.stderr, flush=True)
        except Exception:                              # noqa: BLE001 —— 退出路径不抛
            pass

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
