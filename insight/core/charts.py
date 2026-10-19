# -*- coding: utf-8 -*-
"""图表公共设施：中文字体 + MPLCONFIGDIR。

必须在 import pyplot 之前设置 MPLCONFIGDIR，否则 matplotlib 会尝试写 ~/.matplotlib，
在沙箱/CI 里通常不可写。字体也必须显式指定，否则中文全变方框。
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_CANDIDATES = ["Heiti TC", "Arial Unicode MS", "PingFang HK", "Songti SC",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]


def setup_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]


def save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
