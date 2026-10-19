#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成可公开的脱敏样例（两个场景各一份）。

为什么需要：完整数据里，评论表含 27,686 个用户的昵称与 ID（个人信息），
信贷表受 Kaggle 竞赛规则约束，都不适合公开转分发。仓库只放样例，
完整数据按 data/README.md 自行获取。

用法：
    python scripts/make_samples.py                 # 两个场景都生成
    python scripts/make_samples.py --only credit
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def reviews(n: int = 300) -> None:
    src = os.path.join(DATA, "meituan_reviews.xlsx")
    if not os.path.exists(src):
        print(f"[skip] 缺 {src}")
        return
    df = pd.read_excel(src)
    per = max(1, n // df["评论用户星级"].nunique())
    parts = [g.sample(min(per, len(g)), random_state=42)
             for _, g in df.groupby("评论用户星级")]
    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    # 个人信息必须删除：昵称 + 平台用户 ID
    out = out.drop(columns=[c for c in ["评论用户姓名", "评论用户id"] if c in out.columns])
    dst = os.path.join(DATA, "sample_reviews.csv")
    out.to_csv(dst, index=False, encoding="utf-8")
    print(f"[review] {dst}  行数 {len(out)}  已删除：昵称/用户ID")


def credit(n: int = 3000) -> None:
    src = os.path.join(DATA, "cs-training.csv")
    if not os.path.exists(src):
        print(f"[skip] 缺 {src}")
        return
    df = pd.read_csv(src)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    pos = df[df["SeriousDlqin2yrs"] == 1]
    neg = df[df["SeriousDlqin2yrs"] == 0]
    n_pos = min(len(pos), max(1, int(n * 0.07)))     # 保持约 7% 的坏样本比例
    out = pd.concat([
        pos.sample(n_pos, random_state=42),
        neg.sample(n - n_pos, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    out["_dataset"] = "give_me_some_credit"
    dst = os.path.join(DATA, "sample_credit.csv")
    out.to_csv(dst, index=False, encoding="utf-8")
    print(f"[credit] {dst}  行数 {len(out)}  坏样本 {n_pos}"
          f"（{n_pos / len(out) * 100:.2f}%）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["review", "credit"], default=None)
    args = ap.parse_args()
    if args.only != "credit":
        reviews()
    if args.only != "review":
        credit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
