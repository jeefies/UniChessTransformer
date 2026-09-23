"""kit_adapter 预设映射（纯 JSON，不需要 torch / GPU）。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KIT_ROOT = PROJECT_ROOT.parent / "Kit"
for p in (PROJECT_ROOT, KIT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unichess_kit.search import PUCTConfig  # noqa: E402
from unichess_t.kit_adapter import _PRESET_KEYS, load_preset  # noqa: E402


class TestPresets(unittest.TestCase):
    def test_every_search_key_of_every_preset_is_mapped(self):
        presets = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
        ignored = {"description", "use_cpp", "type", "engine"}
        for name, preset in presets.items():
            unknown = set(preset) - set(_PRESET_KEYS) - ignored
            self.assertFalse(unknown, f"预设 {name} 有未映射的键 {unknown}（kit 路径会静默丢弃）")

    def test_root_top_k_reaches_puct_config(self):
        opts = load_preset("max_t")
        self.assertEqual(opts["root_top_k"], 3)
        self.assertEqual(opts["temperature"], 1.0)
        self.assertEqual(PUCTConfig(root_top_k=opts["root_top_k"]).root_top_k, 3)


if __name__ == "__main__":
    unittest.main()
