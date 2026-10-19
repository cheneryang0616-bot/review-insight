# -*- coding: utf-8 -*-
"""credit 域 —— 口径配置

风控的口径比文本分析更"硬"：阈值是行业惯例，写死在配置里，
这样闸门（gates.py 里那些断言）才有意义，也方便在面试里解释来源。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = os.path.join(ROOT, "data")
GMC_TRAIN = os.path.join(DATA_DIR, "cs-training.csv")      # Kaggle Give Me Some Credit
GERMAN = os.path.join(DATA_DIR, "german_credit.xlsx")      # UCI German Credit
SAMPLE = os.path.join(DATA_DIR, "sample_credit.csv")       # 仓库自带脱敏样例

FIG_DIR = os.path.join(ROOT, "figures", "credit")

# ---------------------------------------------------------------- 标签
TARGET_GMC = "SeriousDlqin2yrs"          # 1 = 90 天以上严重逾期
TARGET_GERMAN = "Creditability"
# 注意：UCI 原始文档写 1=Good / 2=Bad，但这张 xlsx 已被重编码成 1=好客户 / 0=坏客户。
# 按文档写会导致坏样本数为 0、IV 全部变成 inf —— 这个坑实测踩过，别改回去。
GERMAN_BAD_VALUE = 0

# ---------------------------------------------------------------- 分箱与评估
DEFAULT_BINS = 10
MIN_BIN_COUNT = 50            # 单个分箱最小样本，太少的箱要合并
MIN_N_SEGMENT = 500           # 分群结论的最小样本
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# ---------------------------------------------------------------- 闸门阈值
IV_SUSPICIOUS = 0.5           # IV > 0.5：教科书说"极强"，实际上该先怀疑泄漏/极端值
KS_FLOOR = 0.3                # KS < 0.3 的模型不可用（信贷行业常用下限）
PSI_ALERT = 0.25              # PSI > 0.25 视为分布显著漂移
IMBALANCE_FLOOR = 0.02        # 坏样本占比 < 2% 时需提示重采样/权重

# ---------------------------------------------------------------- 合规字段
# 硬阻断：性别/婚姻、外籍劳工 —— 直接进模型属歧视性变量（fair lending / 就业歧视）
FORBIDDEN_HARD = ["Sex & Marital Status", "Foreign Worker"]      # 原始列名（文档用）
FORBIDDEN_HARD_COLS = ["sex_marital", "foreign_worker"]          # 重命名后的列名（代码用）
# 敏感但非一律禁止：年龄在部分地区/产品中可用，但必须显式说明
SENSITIVE_SOFT = ["Age (years)"]
SENSITIVE_SOFT_COLS = ["age"]

# 展示用中文短标签（真实字段名太长，且含数字会干扰"数字回查"）
SHORT_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "额度使用率",
    "NumberOfTime30-59DaysPastDueNotWorse": "中短期逾期次数",
    "NumberOfTimes90DaysLate": "严重逾期次数",
    "NumberOfTime60-89DaysPastDueNotWorse": "长期逾期次数",
    "DebtRatio": "负债率",
    "MonthlyIncome": "月收入",
    "NumberOfOpenCreditLinesAndLoans": "信贷账户数",
    "NumberRealEstateLoansOrLines": "房产贷款数",
    "NumberOfDependents": "抚养人数",
    "age": "年龄",
    "account_balance": "账户余额档",
    "duration_month": "贷款期限",
    "payment_status_prev": "历史还款",
    "credit_amount": "贷款金额",
    "savings": "储蓄档",
    "employment_length": "在职时长",
    "instalment_pct": "分期比例",
    "sex_marital": "性别婚姻",
    "foreign_worker": "外籍劳工",
    "purpose": "贷款用途",
}


def label(col: str) -> str:
    return SHORT_LABELS.get(col, col[:16])


GERMAN_RENAME = {
    "Account Balance": "account_balance",
    "Duration of Credit (month)": "duration_month",
    "Payment Status of Previous Credit": "payment_status_prev",
    "Purpose": "purpose",
    "Credit Amount": "credit_amount",
    "Value Savings/Stocks": "savings",
    "Length of current employment": "employment_length",
    "Instalment per cent": "instalment_pct",
    "Sex & Marital Status": "sex_marital",
    "Guarantors": "guarantors",
    "Duration in Current address": "address_years",
    "Most valuable available asset": "asset",
    "Age (years)": "age",
    "Concurrent Credits": "concurrent_credits",
    "Type of apartment": "apartment",
    "No of Credits at this Bank": "bank_credits",
    "Occupation": "occupation",
    "No of dependents": "dependents",
    "Telephone": "telephone",
    "Foreign Worker": "foreign_worker",
}
