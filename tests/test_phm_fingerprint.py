#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资产指纹测试：预测来源必须可核对

为什么单开一个文件：指纹是"审计链"的最后一环 —— 报告里写"运行 FD001_xxx_final"，
但如果那份 npz 被人替换过，结论就在讲另一份数据。指纹让这件事可查。

三条不变量：
1. 同一文件 → 同一指纹（含 sha256 前 8 位、字节数、mtime）
2. 不同内容 → 不同指纹（内容寻址，与路径无关）
3. 每个工具的事实里都带指纹，且答案里的指纹不破坏数字回查
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.core.llm import OfflineStub
from insight.domains import get_domain
from insight.domains.phm import config, data, tools


def _fixture(root: Path, pred=(12.0, 22.0, 32.0), sigma=(1.0, 2.0, 3.0)):
    """在 root/artifacts/<run>/predictions.npz 写一份最小合法资产。"""
    values = {
        "u_test": np.array([1, 2, 3]),
        "y_test": np.array([10.0, 20.0, 30.0]),
        "pred_ensemble": np.array(pred, dtype=float),
        "sigma": np.array(sigma, dtype=float),
        "end_cycle": np.array([100, 101, 102]),
        "pred_members": np.array([[11.0, 21.0, 31.0], [13.0, 23.0, 33.0]]),
    }
    path = root / "artifacts" / config.RUN / "predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return path


class TestFingerprint(unittest.TestCase):
    def test_shape_and_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            _fixture(Path(root))
            df = data.load(root=Path(root))
        info = data.asset_summary(df)
        self.assertTrue(info["asset_id"].startswith("npz:"))
        self.assertEqual(len(info["asset_id"].split(":")[1]), 8)
        self.assertEqual(len(info["asset_sha256"]), 64)
        self.assertGreater(info["asset_bytes"], 0)
        self.assertGreater(info["asset_mtime_ns"], 0)
        self.assertTrue(info["asset_mtime_iso"])

    def test_sha_matches_file_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            path = _fixture(Path(root))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()   # 必须在临时目录销毁前算
            df = data.load(root=Path(root))
        self.assertEqual(df.attrs["asset_sha256"], expected)
        self.assertEqual(df.attrs["asset_id"], "npz:" + expected[:8])

    def test_same_content_same_id_regardless_of_path(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _fixture(Path(a))
            _fixture(Path(b))
            ida = data.load(root=Path(a)).attrs["asset_id"]
            idb = data.load(root=Path(b)).attrs["asset_id"]
        self.assertEqual(ida, idb, "指纹应内容寻址，不随路径变化")

    def test_content_change_changes_id(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _fixture(Path(a), pred=(12.0, 22.0, 32.0))
            _fixture(Path(b), pred=(12.0, 22.0, 99.0))
            ida = data.load(root=Path(a)).attrs["asset_id"]
            idb = data.load(root=Path(b)).attrs["asset_id"]
        self.assertNotEqual(ida, idb, "内容变了指纹必须变")

    def test_repeated_load_is_stable(self):
        with tempfile.TemporaryDirectory() as root:
            _fixture(Path(root))
            first = data.load(root=Path(root)).attrs["asset_id"]
            second = data.load(root=Path(root)).attrs["asset_id"]
        self.assertEqual(first, second)

    def test_sample_has_distinct_prefix(self):
        df = data.load_sample()
        self.assertTrue(df.attrs["asset_id"].startswith("sample:"))
        self.assertEqual(df.attrs["asset_kind"], "sample")

    def test_every_tool_carries_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            _fixture(Path(root))
            df = data.load(root=Path(root))
            for name in ("fleet_overview", "uncertainty_audit"):
                res = tools.REGISTRY.call(name, df)
                self.assertEqual(res["facts"]["asset_id"], df.attrs["asset_id"],
                                 f"{name} 未携带资产指纹")
                self.assertIn("asset_mtime_iso", res["facts"])

    def test_scope_desc_shows_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            _fixture(Path(root))
            df = data.load(root=Path(root))
            text = data.scope_desc(df)
        self.assertIn(df.attrs["asset_id"], text)
        self.assertIn("预测资产", text)

    def test_grounding_accepts_fingerprint_in_answer(self):
        """指纹里含数字（日期、字节数），不能把 groundedness 打挂。"""
        with tempfile.TemporaryDirectory() as root:
            _fixture(Path(root))
            df = data.load(root=Path(root))
            with patch.object(data, "load", return_value=df):
                agent = Agent(get_domain("phm"), backend=OfflineStub())
                res = agent.ask("机队总览", use_llm=False)
        self.assertTrue(res["grounding"]["passed"],
                        f"未支撑：{res['grounding']['unsupported']}")
        self.assertIn(df.attrs["asset_id"], res["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
