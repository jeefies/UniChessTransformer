"""Transformer 重建后的测试：模型结构、分层路由、检查点兼容、kit 接入、Server 插件、配方配置。

CPU 可跑的部分（不需要权重）：结构 / 参数量 / 路由 / cfg_conflicts / export 协议 /
preset 映射 / 训练适配器 / 配置钉规模。需要权重的部分标 ``_HAS_CKPT``，远端存在时自动跑。
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import chess  # noqa: E402

from Transformer import kit as tkit  # noqa: E402
from Transformer.evaluator import TransformerEngine, describe, load_checkpoint  # noqa: E402
from Transformer.model import (PRESET_CONFIGS, ChessTransformer, StratifiedChessTransformer,
                               TransformerConfig, cfg_conflicts, count_params, stratified_20m)

_HAS_CKPT = (ROOT / "runs" / "transformer_20m" / "best_model.pt").exists()


def _fake_ckpt(tmp: Path, cfg: TransformerConfig, step: int = 5, *, stratified: bool = False):
    """写一个随机权重的 checkpoint（export / 加载往返，不碰真实权重）。"""
    import torch
    model = stratified_20m() if stratified else ChessTransformer(cfg)
    p = tmp / "fake.pt"
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), "step": step,
                "preset": "stratified_20m" if stratified else None}, p)
    return p


class TestTransformerStructure(unittest.TestCase):
    def test_preset_configs(self):
        self.assertEqual(sorted(PRESET_CONFIGS),
                         ["transformer_20m", "transformer_50m", "transformer_large",
                          "transformer_medium", "transformer_small", "transformer_tiny"])
        cfg = PRESET_CONFIGS["transformer_20m"]
        self.assertEqual((cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff),
                         (384, 11, 12, 1024))
        # 20M 的真实参数量（去重后；state_dict 里 experts.N.* 与 opening.* 是同一批模块）
        self.assertEqual(sum(p.numel() for p in stratified_20m().parameters()), 61031064)
        self.assertEqual(sum(p.numel() for p in ChessTransformer(cfg).parameters()), 20343688)

    def test_state_dict_key_names(self):
        net = ChessTransformer(PRESET_CONFIGS["transformer_small"])
        keys = set(net.state_dict())
        for k in ("stem.weight", "blocks.0.attn.rel_pos_bias", "blocks.0.attn.qkv.weight",
                  "blocks.0.mlp.w1.weight", "final_norm.weight", "policy_head.wq.weight",
                  "policy_head.bias_move", "promo_head.0.weight", "value_head.1.weight",
                  "mlh_head.0.weight"):
            self.assertIn(k, keys)
        p, pr, w = net(torch_zeros(2))
        self.assertEqual((tuple(p.shape), tuple(pr.shape), tuple(w.shape)),
                         ((2, 4096), (2, 4), (2, 3)))

    def test_mlh_head_optional_output(self):
        net = ChessTransformer(PRESET_CONFIGS["transformer_small"])
        p, pr, w, m = net(torch_zeros(3), return_mlh=True)
        self.assertEqual(tuple(m.shape), (3,))
        # 没有 mlh 目标时旧模型也能加载：mlh_head 是 P3 才加的辅助头
        self.assertTrue(any("mlh_head" in k for k in net.state_dict()))

    def test_cfg_conflicts(self):
        from dataclasses import replace
        base = dict(dataclasses.asdict(PRESET_CONFIGS["transformer_20m"]))
        self.assertEqual(cfg_conflicts(base, PRESET_CONFIGS["transformer_20m"]), [])
        self.assertEqual(len(cfg_conflicts(dict(base, num_layers=8),
                                           PRESET_CONFIGS["transformer_20m"])), 1)
        # checkpoint 里没有的字段：当前值必须等于默认值才算兼容
        missing = {k: v for k, v in base.items() if k != "dropout"}
        self.assertEqual(cfg_conflicts(missing, PRESET_CONFIGS["transformer_20m"]), [])
        cur = replace(PRESET_CONFIGS["transformer_20m"], dropout=0.1)
        self.assertEqual(cfg_conflicts(missing, cur),
                         ["dropout: checkpoint 无此字段（等同默认值 0.0），当前为 0.1"])


class TestRouting(unittest.TestCase):
    def test_route_index_from_board(self):
        route = StratifiedChessTransformer.route_index
        self.assertEqual(route(chess.Board()), 0)                  # 32 子，开局
        # ply<=20 也走开局专家（与 docstring 的口径一致）
        early = chess.Board("8/8/8/8/8/8/k1K5/8 w - - 0 5")
        self.assertEqual(route(early), 0)
        # ply>20 后只看子力数
        endgame = chess.Board("8/8/8/8/8/8/k1K5/8 w - - 0 30")
        self.assertEqual(route(endgame), 2)
        middlegame = chess.Board("8/8/8/3pp3/3PP3/8/k1K5/8 w - - 0 30")   # 6 子 → 残局口径
        self.assertEqual(route(middlegame), 2)
        many = chess.Board("r3k3/pp3ppp/8/8/8/8/PPP3PP/R3K3 w - - 0 30")  # 14 子 → 中局
        self.assertEqual(route(many), 1)

    def test_routes_from_planes_without_board(self):
        model = stratified_20m()
        xs = torch_zeros(4)
        xs[0, :12] = 0.0                                     # 0 子 → 残局专家
        xs[0, 0] = 1.0
        xs[1, :12] = 1.0                                     # 12 子 → 中局专家（>12 才开局）
        p, pr, w = model(xs)
        self.assertEqual((tuple(p.shape), tuple(w.shape)), ((4, 4096), (4, 3)))


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_export_and_reload(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = _fake_ckpt(Path(d), PRESET_CONFIGS["transformer_small"], step=42)
            model, cfg, step, _ = load_checkpoint(p)
            self.assertEqual(step, 42)
            self.assertEqual(cfg, PRESET_CONFIGS["transformer_small"])
            self.assertEqual(describe(p)["step"], 42)
            q = Path(d) / "again.pt"
            import torch
            out = tkit.TTrainAdapter(cfg="transformer_small").export(model, 7)
            torch.save(out, q)
            _, _, step2, _ = load_checkpoint(q)
            self.assertEqual(step2, 7)

    def test_stratified_checkpoint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = _fake_ckpt(Path(d), PRESET_CONFIGS["transformer_20m"], step=9,
                           stratified=True)
            model, cfg, step, _ = load_checkpoint(p)
            self.assertIsInstance(model, StratifiedChessTransformer)
            self.assertEqual(step, 9)
            self.assertTrue(describe(p)["stratified"])

    def test_missing_mlh_head_is_allowed(self):
        """P3 之前的检查点没有 mlh_head：只许缺这一组键，缺别的直接报错。"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = _fake_ckpt(Path(d), PRESET_CONFIGS["transformer_small"])
            import torch
            ckpt = torch.load(p, weights_only=False)
            ckpt["model"] = {k: v for k, v in ckpt["model"].items()
                             if not k.startswith("mlh_head.")}
            torch.save(ckpt, p)
            load_checkpoint(p)                                  # 不该抛
            bad = ckpt["model"].pop("value_head.1.weight", None)
            torch.save({"model": ckpt["model"], "cfg": ckpt["cfg"], "step": 0}, p)
            from Transformer.model import load_model
            with self.assertRaises(ValueError):
                load_model(p)

    def test_reject_wrong_structure(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.pt"
            import torch
            torch.save({"model": {}, "cfg": {"d_model": 8}}, p)
            with self.assertRaises(ValueError):
                load_checkpoint(p)


class TestKitAdapters(unittest.TestCase):
    def test_preset_mapping(self):
        for name in ("max_mcts", "max_t"):
            kw = tkit.load_preset(name)
            self.assertEqual(kw["simulations"], 2400)
            self.assertEqual(kw["precision"], "fp16")
            self.assertIn("syzygy_path", kw)
        self.assertEqual(tkit.load_preset("max_t")["temperature"], 1.0)
        self.assertEqual(tkit.load_preset("max_t")["root_top_k"], 3)
        self.assertNotIn("description", tkit.load_preset("max_mcts"))
        with self.assertRaises(KeyError):
            tkit.load_preset("nope")

    def test_task_construction(self):
        # 单个 20M：全部参数可训
        task = tkit.make_task(cfg="transformer_small",
                              data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                              loss={"kind": "t_chess"})
        self.assertEqual(task.adapter.cfg, PRESET_CONFIGS["transformer_small"])
        self.assertFalse(task.adapter.stratified)
        self.assertIsNone(task.adapter.trainable_expert)
        model = task.build_model()
        self.assertEqual(len(task.param_groups(model)[0]["params"]),
                         len(list(model.parameters())))
        model = task.build_model()
        self.assertEqual(len(list(model.parameters())),
                         len(list(ChessTransformer(PRESET_CONFIGS["transformer_small"]).parameters())))

        # 分层 + 只训一个专家：其余参数冻结，可训参数=单个专家的数量
        cur = tkit.make_task(stratified=True, trainable_expert="opening",
                             data={"kind": "curriculum", "shards": {"dir": "x"},
                                   "batch_size": 8, "filter": "opening"},
                             loss={"kind": "t_chess"})
        self.assertTrue(cur.adapter.stratified)
        self.assertEqual(cur.adapter.cfg, PRESET_CONFIGS["transformer_20m"])
        full = cur.build_model()
        groups = cur.param_groups(full)
        self.assertEqual(len(groups[0]["params"]),
                         len(list(cur.adapter.trainable(full).parameters())))
        frozen = [p for p in full.parameters() if not p.requires_grad]
        self.assertEqual(len(frozen), len(list(full.parameters()))
                         - len(list(cur.adapter.trainable(full).parameters())))
        # 导出仍是完整三专家权重（Server / kit 能直接加载）
        out = cur.export(full, 3)
        self.assertEqual(sorted(out), ["cfg", "model", "preset", "step"])
        self.assertEqual(out["preset"], "stratified_20m")

    def test_task_rejects_bad_model_args(self):
        base = dict(data={"kind": "loader", "shards": {"dir": "x"}, "batch_size": 8},
                    loss={"kind": "t_chess"})
        with self.assertRaises(KeyError):
            tkit.make_task(cfg="giant", **base)
        with self.assertRaises(KeyError):
            tkit.make_task(stratified=True, trainable_expert="opening42", **base)
        with self.assertRaises(ValueError):
            tkit.make_task(trainable_expert="opening", **base)      # 非分层模型不给单训
        with self.assertRaises(TypeError):
            tkit.make_task(cfg=15, **base)
        with self.assertRaises(KeyError):
            tkit.make_task(cfg={"d_model": 384, "nope": 1}, **base)

    def test_engine_module_contract(self):
        """Server 按路径加载本仓库的 engine.py：KIT_FACTORY 与 GameEngine 必须存在。

        服务端用 ``spec_from_file_location`` 起了一个合成的模块名，与本测试里
        直接 exec_module 得到的是**两个**不同 module 对象，所以类不能按身份比较——
        这里断言契约本身。
        """
        from Kit.serving import load_engine_class

        spec = importlib.util.spec_from_file_location("_t_engine_test", ROOT / "engine.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.KIT_FACTORY, "Transformer.kit:make_player_factory")
        self.assertTrue(mod.GameEngine.IMPLEMENTED)
        self.assertEqual(mod.GameEngine.__name__, "TransformerEngine")
        cls = load_engine_class(ROOT / "engine.py")
        self.assertEqual(cls.KIT_FACTORY, "Transformer.kit:make_player_factory")
        self.assertTrue(cls.IMPLEMENTED)


class TestConfigs(unittest.TestCase):
    """``configs/*.json`` 必须逐字段复刻旧脚本（train.py / curriculum_*.py / p3）。

    历史坑（都对拍时才暴露）：

    - 数据必须关掉 castling 修复：R 的旧解码器会修，T 的旧解码器不修，共用一个
      默认 True 的解码器会让策略目标差几个 ulp，损失从第一步就偏 5.6e-6。
    - optimizer 必须显式 ``fused=false``：T 旧脚本写的是
      ``fused = cuda and hasattr(AdamW, "_step_supports_fused")``，而新版 torch 没有
      这个属性，于是永远走 foreach；不写死就改成自动选核，更新结果逐位不同。
    """

    def _load(self, name: str) -> dict:
        cfg = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        self.assertEqual(cfg["task"]["factory"], "Transformer.kit:make_task")
        return cfg

    def test_t20m(self):
        cfg = self._load("t20m.json")
        kw = cfg["task"]["kwargs"]
        self.assertEqual(kw["cfg"], "transformer_20m")
        self.assertFalse(kw["stratified"])
        self.assertIs(kw["data"]["repair_castling"], False)
        self.assertEqual(kw["data"]["shards"]["slice"], [None, -1])
        self.assertEqual(kw["data"]["batch_size"], 1024)
        self.assertEqual(kw["validation"]["shards"]["slice"], [-1, None])
        self.assertEqual((cfg["steps"], cfg["accum"], cfg["seed"], cfg["clip"]),
                         (15000, 1, 42, 1.0))
        self.assertEqual(cfg["optimizer"]["lr"], 0.001)
        self.assertIs(cfg["optimizer"]["fused"], False)
        self.assertEqual(cfg["schedule"], {"kind": "cosine", "t_max": 15000, "eta_min": 1e-05})
        task = tkit.make_task(**kw)
        self.assertEqual(sum(p.numel() for p in task.build_model().parameters()), 20343688)

    def test_curriculum(self):
        for name, expert, filt in (("stratified_opening", "opening", "opening"),
                                   ("stratified_middlegame", "middlegame", "middlegame"),
                                   ("stratified_endgame", "endgame", "endgame")):
            cfg = self._load(f"{name}.json")
            kw = cfg["task"]["kwargs"]
            self.assertTrue(kw["stratified"])
            self.assertEqual(kw["trainable_expert"], expert)
            self.assertIs(kw["data"]["repair_castling"], False)
            self.assertEqual(kw["data"]["filter"], filt)
            self.assertEqual(kw["data"]["chunk_size"], 65536)
            self.assertEqual(kw["data"]["score_lambda"], 1.0)
            self.assertEqual((kw["data"]["batch_size"], cfg["steps"], cfg["seed"]),
                             (512, 10000, 42))
            self.assertEqual(cfg["optimizer"]["lr"], 0.0002)
            self.assertIs(cfg["optimizer"]["fused"], False)
            self.assertEqual(cfg["schedule"], {"kind": "cosine", "t_max": 10000,
                                               "eta_min": 1e-05})
            task = tkit.make_task(**kw)
            model = task.build_model()
            self.assertEqual(len(task.param_groups(model)[0]["params"]),
                             len(list(getattr(model, expert).parameters())))

    def test_p3_mlh(self):
        cfg = self._load("p3_mlh.json")
        kw = cfg["task"]["kwargs"]
        self.assertTrue(kw["stratified"])
        self.assertTrue(kw["loss"]["mlh"])
        self.assertEqual(kw["loss"]["mlh_weight"], 0.05)
        self.assertIs(kw["data"]["repair_castling"], False)
        self.assertEqual(kw["data"]["batch_size"], 512)
        self.assertEqual((cfg["optimizer"]["lr"], cfg["seed"], cfg["steps"]),
                         (0.0001, 42, 10000))
        task = tkit.make_task(**kw)
        model = task.build_model()
        # 三个专家全都可训
        self.assertEqual(len(task.param_groups(model)[0]["params"]),
                         len(list(model.parameters())))
        # mlh 目标由 kit 算：(2·子力数 + 20·(1−|W−L|)) / 100；子力数从平面 0-11 数
        import torch
        from Kit.planes19.losses import mlh_target
        x = torch.zeros(2, 19, 8, 8)
        x[:, :12] = 0.0
        x[0, 0, 0, :4] = 1.0                                   # 4 个自己的兵
        x[1, 0, 0, :6] = 1.0                                   # 6 个
        w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # |W−L| = 1 / 0
        got = mlh_target(x, w)
        want = torch.tensor([(2.0 * 4 + 20.0 * (1.0 - 1.0)) / 100.0,
                             (2.0 * 6 + 20.0 * (1.0 - 0.0)) / 100.0])
        self.assertTrue(torch.allclose(got, want), (got, want))

    def test_loop_p4_v2(self):
        """激进版循环：1024 局自对弈 + lr×步数枚举搜索 + 256 对 arena。

        与 loop_p4.json 的差别只在规模与枚举搜索；训练口径（P4）不变。
        """
        cfg = json.loads((ROOT / "configs" / "loop_p4_v2.json").read_text(encoding="utf-8"))
        # 4096 局：600 步变体抽 36.9 万条 / 54.5 万记录 = 0.7x，1200 步 1.4x，
        # 两边都不陷入反复过拟合（1024 局时分别是 2.7x / 5.4x，正是初版 gen3/4 掉负 Elo 的原因）
        self.assertEqual(cfg["games"], 4096)
        self.assertEqual(cfg["window"], 2)
        self.assertEqual(cfg["selfplay"]["concurrency"], 32)
        # initial 是已跑出来的冠军（gen 0 的候选），不是最初的生产权重
        self.assertTrue(cfg["initial"].endswith(
            "runs/loop_p4/gen_0000/train/final.pt"), cfg["initial"])
        self.assertEqual(cfg["engine"]["kwargs"]["simulations"], 800)

        train = cfg["train"]
        variants = train["variants"]
        labels = [v["label"] for v in variants]
        self.assertEqual(len(labels), len(set(labels)))
        # lr × steps 网格：5 个 lr × 2 个步数
        self.assertEqual({v["optimizer"]["lr"] for v in variants},
                         {1e-06, 2e-06, 5e-06, 1e-05, 2e-05})
        self.assertEqual({v["steps"] for v in variants}, {600, 1200})
        for v in variants:                      # 每个变体只覆盖 lr 与 steps
            self.assertEqual(set(v) - {"label"}, {"optimizer", "steps"})
            self.assertEqual(set(v["optimizer"]), {"lr"})
        # 筛选赛：候选对冠军，800 sims（与自对弈同档）+ 192 对。
        # 实测 GPU 吞吐相同（约 1.2 万 positions/s），2400 sims 每局只多干 3.75 倍活；
        # 降到 800 后同样墙钟能跑 3 倍局数，得分 SE 从 ±6.2% 降到 ±3.6%。
        self.assertEqual(train["screen"]["pairs"], 192)
        self.assertEqual(train["screen"]["simulations"], 800)
        self.assertEqual(train["screen"]["openings"], "bundled")
        # 最终 arena：256 对（512 局）且仍是 2400 sims —— 换代的判定绝不为省时而降规格
        self.assertEqual(cfg["arena"]["match"]["pairs"], 256)
        self.assertEqual(cfg["arena"]["match"]["simulations"], 2400)
        self.assertEqual(cfg["arena"]["match"]["sprt"]["elo1"], 60.0)
        self.assertEqual(cfg["arena"]["gate"], {"kind": "sprt"})
        # 训练口径与 loop_p4.json 一致（KL / accum 4 / bf16 / 前 4 片 / 自对弈 files）
        kw = train["task"]["kwargs"]
        self.assertEqual(kw["loss"]["policy_loss_type"], "kl_divergence")
        self.assertEqual(train["accum"], 4)
        self.assertEqual(train["precision"], "bf16")
        self.assertEqual(kw["data"]["sources"][0]["shards"]["slice"], [0, 4])
        self.assertEqual(kw["data"]["sources"][1]["shards"], {"files": "{selfplay_files}"})

    def test_loop_p4_and_v2_differ_only_in_scale(self):
        a = json.loads((ROOT / "configs" / "loop_p4.json").read_text(encoding="utf-8"))
        b = json.loads((ROOT / "configs" / "loop_p4_v2.json").read_text(encoding="utf-8"))
        for key in ("engine", "selfplay", "sink", "export"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["train"]["task"], b["train"]["task"])
        self.assertEqual(a["train"]["accum"], b["train"]["accum"])
        self.assertEqual(a["train"]["precision"], b["train"]["precision"])
        self.assertEqual(a["train"]["schedule"], b["train"]["schedule"])
        # v2 只多了枚举搜索，训练模板其余一致
        a_train = {k: v for k, v in a["train"].items() if k not in ("steps", "optimizer")}
        b_train = {k: v for k, v in b["train"].items()
                   if k not in ("steps", "optimizer", "variants", "screen")}
        self.assertEqual(a_train, b_train)

    def test_loop_p4(self):
        """换代循环配方：钉死 P4 自对弈口径（30% 自对弈 + 70% 监督、lr 5e-6、KL）。

        与旧 ``tools/gumbel_selfplay_corrected.py`` 的 argparse 默认值逐项对应；
        自对弈那 30% 走 ``*.sp.bin`` 显式分片列表（loop 的 ``{selfplay_files}`` 占位符），
        监督那 70% 取前 4 片（旧 ``--max-original-shards 4``）。
        """
        cfg = json.loads((ROOT / "configs" / "loop_p4.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["initial"].endswith(
            "runs/stratified_p4_selfplay_corrected/best_model.pt"), cfg["initial"])
        self.assertEqual(cfg["export"], "final.pt")
        self.assertEqual(cfg["train"]["export"]["final"], "final.pt")
        self.assertEqual(cfg["arena"]["gate"], {"kind": "sprt"})
        self.assertEqual(cfg["arena"]["match"]["simulations"], 2400)
        self.assertEqual(cfg["arena"]["match"]["sprt"],
                         {"elo0": 0.0, "elo1": 40.0, "alpha": 0.05, "beta": 0.1})
        self.assertEqual(cfg["engine"]["kwargs"]["simulations"], 800)
        self.assertEqual(cfg["engine"]["kwargs"]["checkpoint"], "{weights}")
        # 开局库路径必须绝对：loop 的三个子进程都以 <out> 为 cwd 运行
        self.assertEqual(cfg["selfplay"]["openings"],
                         "/home/jeefy/UniChess/Kit/data/openings.txt")
        self.assertEqual(cfg["sink"]["kwargs"]["path"], "{selfplay_dir}/selfplay.sp.bin")

        kw = cfg["train"]["task"]["kwargs"]
        self.assertIs(kw["stratified"], True)
        self.assertNotIn("trainable_expert", kw)              # 三专家一起训
        self.assertEqual(kw["base_ckpt"], "{weights}")
        data = kw["data"]
        self.assertEqual(data["kind"], "mix")
        self.assertEqual(data["batch_size"], 512)
        sup, sp = data["sources"]
        self.assertEqual(sup["weight"], 0.7)
        self.assertEqual(sup["shards"]["slice"], [0, 4])
        self.assertIs(sup["repair_castling"], False)
        self.assertEqual(sp["weight"], 0.3)
        self.assertEqual(sp["shards"], {"files": "{selfplay_files}"})
        self.assertEqual(kw["loss"], {"kind": "t_chess", "policy_weight": 1.0,
                                      "promo_weight": 0.1, "wdl_weight": 1.0,
                                      "policy_loss_type": "kl_divergence"})
        self.assertEqual((cfg["train"]["optimizer"]["lr"], cfg["train"]["accum"]),
                         (5e-06, 4))
        self.assertEqual(cfg["train"]["clip"], 1.0)
        self.assertEqual(cfg["train"]["schedule"], {"kind": "constant"})

        # 训练的模型侧参数要能真的构造出三专家模型（数据源只做形状校验，不读分片）
        task = tkit.make_task(stratified=True, base_ckpt=None,
                              data={"kind": "mix", "batch_size": 8,
                                    "sources": [{"weight": 0.7, "shards": {"dir": str(ROOT)}},
                                                {"weight": 0.3, "shards": {"dir": str(ROOT)}}]},
                              loss={"kind": "t_chess", "policy_loss_type": "kl_divergence"})
        model = task.build_model()
        self.assertEqual(len(task.param_groups(model)[0]["params"]),
                         len(list(model.parameters())))
        self.assertEqual(task._chess_loss.policy_loss_type, "kl_divergence")


def torch_zeros(n: int):
    """(n, 19, 8, 8) fp32 全零平面。"""
    import torch
    return torch.zeros(n, 19, 8, 8)


if __name__ == "__main__":
    unittest.main()
