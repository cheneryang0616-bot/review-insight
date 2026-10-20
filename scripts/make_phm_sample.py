#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 FD001 全部机器的预测摘要，不复制权重、传感器序列或 artifacts。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight.domains.phm import config, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None, help="C-MAPSS 仓库，默认 RI_PHM_ROOT")
    ap.add_argument("--output", type=Path, default=config.SAMPLE_PATH)
    args = ap.parse_args()
    df = data.load(args.root)
    if df.attrs["source"] != "full":
        ap.error("需要完整预测资产，不能从样例重新生成样例")
    if len(df) != 100:
        ap.error("FD001 样例应包含 100 台机器")
    out = df.loc[:, list(data.BASE_COLUMNS)].copy()
    out["dataset"], out["run"] = df.attrs["dataset"], df.attrs["run"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"已从 {df.attrs['path']} 导出 {len(out)} 台机器到 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
