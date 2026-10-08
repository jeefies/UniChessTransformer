"""CUDA graph 前向的回归：与 eager 逐位一致（同形状回放同 kernel）。

覆盖：多种 (N, 路由计数) 组合、fp16/fp32 两种精度、非分层模型、桶上限回退、
以及 engine 层 evaluate_planes 的端到端一致。无 CUDA 自动跳过。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Transformer import evaluator as ev  # noqa: E402
from Transformer.evaluator import TransformerEngine  # noqa: E402

_CUDA = torch.cuda.is_available()


def _fake_stratified_ckpt(tmp: Path, step: int = 7):
    """随机权重的分层 checkpoint（结构与生产同族；load_model 按 preset 选三专家结构）。"""
    import dataclasses

    from Transformer.model import STRATIFIED_CFG, stratified_20m
    model = stratified_20m()
    p = tmp / "fake_stratified.pt"
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(STRATIFIED_CFG),
                "step": step, "preset": "stratified_20m"}, p)
    return p


def _rand_planes(n: int, seed: int) -> np.ndarray:
    """随机但确定性的 (n,19,8,8) 平面：随机摆子 + 随机 side/castling/ep 元信息。"""
    rng = np.random.default_rng(seed)
    xs = np.zeros((n, 19, 8, 8), dtype=np.float32)
    for i in range(n):
        g = np.random.default_rng(seed * 1000 + i)
        n_pieces = int(g.integers(4, 33))
        squares = g.choice(64, size=min(n_pieces, 64), replace=False)
        for sq in squares:
            xs[i, int(g.integers(0, 12)), sq // 8, sq % 8] = 1.0
        xs[i, 12:, :, :] = g.random((7, 8, 8), dtype=np.float32) > 0.7
        xs[i, 12:16] = 0.0
    return np.ascontiguousarray(xs)


@unittest.skipUnless(_CUDA, "需要 CUDA")
class TestCudaGraphParity(unittest.TestCase):
    """graph 前向 vs eager：逐位相等。"""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = Path(tempfile.mkdtemp(prefix="t_cudagraph_"))
        cls.ckpt = _fake_stratified_ckpt(cls.tmp)

    def _pair(self, precision: str):
        """同一权重的两个 engine：一个全 eager（环境变量关图），一个开图。"""
        old = os.environ.get(ev._NO_GRAPH_ENV)
        os.environ[ev._NO_GRAPH_ENV] = "1"
        try:
            eager = TransformerEngine(str(self.ckpt), device="cuda", precision=precision)
        finally:
            if old is None:
                os.environ.pop(ev._NO_GRAPH_ENV, None)
            else:
                os.environ[ev._NO_GRAPH_ENV] = old
        graphed = TransformerEngine(str(self.ckpt), device="cuda", precision=precision)
        self.assertIsNotNone(graphed.graphs, "CUDA 可用时图前向应启用")
        return eager, graphed

    def _check(self, precision: str, n: int, seed: int):
        eager, graphed = self._pair(precision)
        xs = _rand_planes(n, seed)
        a = eager.evaluate_planes(xs)
        b = graphed.evaluate_planes(xs)
        c = graphed.evaluate_planes(xs)          # 再回放一次：缓存命中也要逐位一致
        for name, x, y, z in zip(("policy", "promo", "wdl"), a, b, c):
            self.assertTrue(torch.equal(torch.from_numpy(x), torch.from_numpy(y)),
                            f"{precision} N={n} {name} 不一致 "
                            f"max|Δ|={np.abs(np.asarray(x) - np.asarray(y)).max()}")
            self.assertTrue(torch.equal(torch.from_numpy(y), torch.from_numpy(z)),
                            f"{precision} N={n} {name} 回放两次不一致")
        st = graphed.graph_stats()
        self.assertGreaterEqual(st["buckets"], 1)
        self.assertEqual(st["capture_fails"], 0)
        del eager, graphed
        torch.cuda.empty_cache()

    def test_fp16_shapes(self):
        for n in (1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 48, 64):
            self._check("fp16", n, seed=n)

    def test_fp32_shapes(self):
        for n in (1, 4, 9, 16, 33):
            self._check("fp32", n, seed=100 + n)

    def test_bucket_limit_falls_back(self):
        """桶数压到 1：第二个 key 必须走 eager 且结果仍正确。"""
        old = ev._GRAPH_MAX_BUCKETS
        ev._GRAPH_MAX_BUCKETS = 1
        try:
            eager, graphed = self._pair("fp16")
            for n in (4, 7, 9):
                xs = _rand_planes(n, seed=7 + n)
                a, b = eager.evaluate_planes(xs), graphed.evaluate_planes(xs)
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(torch.from_numpy(x), torch.from_numpy(y)),
                                    f"N={n} 回退路径不一致")
            self.assertGreaterEqual(graphed.graph_stats()["eager_keys"], 2)
            del eager, graphed
            torch.cuda.empty_cache()
        finally:
            ev._GRAPH_MAX_BUCKETS = old

    def test_n_over_limit_eager(self):
        old = ev._GRAPH_MAX_N
        ev._GRAPH_MAX_N = 8
        try:
            eager, graphed = self._pair("fp16")
            xs = _rand_planes(16, seed=3)
            a, b = eager.evaluate_planes(xs), graphed.evaluate_planes(xs)
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(torch.from_numpy(x), torch.from_numpy(y)))
            self.assertEqual(graphed.graph_stats()["buckets"], 0)
            del eager, graphed
            torch.cuda.empty_cache()
        finally:
            ev._GRAPH_MAX_N = old


if __name__ == "__main__":
    unittest.main()
