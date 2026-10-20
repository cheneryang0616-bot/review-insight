#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 PHM 的 4 张交付图，并把它们嵌进报告。

用法：
    python scripts/make_phm_figures.py                          # 默认资产，写 figures/phm 并更新报告
    python scripts/make_phm_figures.py --root /path/to/cmapss   # 指定外部资产
    python scripts/make_phm_figures.py --out-dir /tmp/figs      # 只渲染（CI 冒烟用，不改报告）

为什么单独一个脚本：图必须和问答用同一批计算，所以它调的是 domains/phm/report.py，
而 report.py 又只调 tools/decision —— 不另算一套，避免"图文不一致"。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "<!-- FIGURES:BEGIN -->"
END = "<!-- FIGURES:END -->"


def section(paths: dict) -> str:
    rel = lambda p: os.path.relpath(p, ROOT / "results").replace(os.sep, "/")
    return "\n".join([
        BEGIN,
        "## 图表（4 张，均由 domains/phm/report.py 从同一批计算生成）",
        "",
        f"![机队紧急度]({rel(paths['fleet'])})",
        "",
        f"> 哪几台进了两周窗口。矩形按预测 RUL 升序，虚线是 {14} 天窗口与 30 循环警戒线。",
        "",
        f"![单机预测区间]({rel(paths['interval'])})",
        "",
        "> 最紧急 10 台的点估计与 ±1.96σ 区间，叉号是离线复盘的真值 —— 可以看出区间偏窄。",
        "",
        f"![校准曲线]({rel(paths['calibration'])})",
        "",
        "> **最重要的一张**：横轴名义覆盖率、纵轴实测覆盖率。贴近对角线才叫校准过；"
        "本例 k=1.96 时名义 95% 实测只有 56%，所以 coverage_gate 阻断停机决策。",
        "",
        f"![代价曲线]({rel(paths['cost'])})",
        "",
        "> 告警线扫描：最低点即期望代价最优阈值；与默认 30 循环相比的差额标在图上。"
        "成本与 σ 都是未校准假设，只作情景分析。",
        END,
    ])


def update_report(paths: dict) -> str:
    report = ROOT / "results" / "REPORT_phm.md"
    if not report.is_file():
        return "（未找到 REPORT_phm.md，跳过嵌图）"
    text = report.read_text(encoding="utf-8")
    block = section(paths)
    if BEGIN in text and END in text:
        head = text.split(BEGIN)[0]
        tail = text.split(END)[1]
        text = head + block + tail
        action = "已替换报告中的图表段"
    else:
        text = text.rstrip() + "\n\n---\n\n" + block + "\n"
        action = "已在报告末尾追加图表段"
    report.write_text(text, encoding="utf-8")
    return action


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, help="C-MAPSS 资产根目录，默认 RI_PHM_ROOT")
    ap.add_argument("--out-dir", type=Path, help="只渲染到该目录，不改报告（CI 冒烟）")
    args = ap.parse_args()
    if args.root is not None:
        os.environ["RI_PHM_ROOT"] = str(args.root.resolve())
    os.chdir(ROOT)

    from insight.domains.phm import config, data, report
    if args.out_dir is not None:
        config.FIG_DIR = args.out_dir.resolve()

    df = data.load()
    paths = report.build_all(df)
    print(f"数据来源：{df.attrs['source']}（{len(df)} 台，{df.attrs['dataset']}）")
    for key in ("fleet", "interval", "calibration", "cost"):
        size = Path(paths[key]).stat().st_size
        print(f"  {key:12s} {paths[key]}  {size // 1024} KB")
    if args.out_dir is None:
        print(update_report(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
