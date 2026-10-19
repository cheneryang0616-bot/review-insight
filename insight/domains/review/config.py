# -*- coding: utf-8 -*-
"""reviewinsight.config —— 全局配置

把"口径"集中在一处，是这类分析项目最容易被质疑的地方：
口径散落在代码里，数字就永远对不上。
"""
import os

# ---------------------------------------------------------------- 路径
# insight/domains/review/config.py → 上溯 4 级才是项目根
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE = os.path.join(ROOT, "data", "meituan_reviews.xlsx")      # 完整数据（不进仓库）
SAMPLE_FILE = os.path.join(ROOT, "data", "sample_reviews.csv")       # 脱敏样例（进仓库）
FIG_DIR = os.path.join(ROOT, "figures", "review")
OUT_DIR = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- 口径
# 星级是 5 分制 ×10 存储（10/20/.../50），存在 15/25/35/45 半星
STAR_DIVISOR = 10.0
# 差评口径：stars <= 2。美团点评体系里 1-2 星是明确负面，3 星是中性
BAD_STAR_MAX = 2.0
# 文本分析最小长度：低于 4 字的评论（"好吃"/"不错"）只产出噪声
MIN_TEXT_LEN = 4
# 排名类分析的最小样本量，低于此不参与排名（避免 1 条评论 0% 或 100%）
MIN_N_SHOP = 200
MIN_N_DEAL = 300
MIN_N_SLOT = 100

# ---------------------------------------------------------------- 统计
LOGO_ODDS_MIN_COUNT = 20      # log-odds 关键词的最小出现次数
LOGO_ODDS_ALPHA0 = None       # None = 用 Monroe 的信息先验（语料词数）
TREND_FREQ = "Q"             # 趋势粒度：季度
ANOMALY_Z = 1.5              # 最近一期差评率相对历史基线的 z 阈值

# ---------------------------------------------------------------- 可解释
ASPECT_MIN_RATE = 0.02       # 方面词在任一侧提及率低于此、且 lift 不显著则不报
