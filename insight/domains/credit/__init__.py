# -*- coding: utf-8 -*-
"""credit 域 —— 信用风险建模（场景二）

同一个框架，换一整套分析原语：
WOE/IV 分箱、WOE+LR 评分卡、KS/AUC/Lift、通过率权衡、合规字段拦截。
"""
from ...core import gates as G
from ...core.agent import Domain
from . import bullets as bullets_mod
from . import config
from . import data as data_mod
from . import planner_spec, tools

SYSTEM_PROMPT = (
    "你是信贷风险建模分析师。你只能使用给定工具结果表中的数字，禁止引入任何外部知识或估算。"
    "每条结论必须标注来源编号，形如 [T2]。输出 Markdown：先 3~5 条要点"
    "（每条一个事实 + 一句风险解读），再一段不超过 120 字的总判断。"
    "如果数据没有时间列，不要声称做过时间外验证。不要输出表格。")


GATES = [
    # 硬约束：KS 低于行业下限的模型不可交付
    G.fact_gate("ks_floor", f"KS 不得低于行业下限 {config.KS_FLOOR}",
                "ks", lambda v: v >= config.KS_FLOOR, G.ERROR,
                "KS 低于行业下限，模型不可用于决策"),
    # 硬约束：模型不得使用性别/婚姻、外籍劳工等字段
    G.fact_in_gate("fair_lending", "模型不得使用硬阻断字段（性别/婚姻、外籍劳工）",
                   "forbidden_used", set(config.FORBIDDEN_HARD_COLS), G.ERROR),
    # 硬约束：通过率-坏账率曲线必须单调，否则排序逻辑有问题
    G.fact_gate("cutoff_monotone", "通过率越低，坏账率必须越低",
                "cutoff_monotone", lambda v: int(v) == 1, G.ERROR,
                "通过率与坏账率曲线非单调，排序逻辑有问题"),
    # 告警：IV 异常偏高时先查数据，别急着夸变量强
    G.fact_gate("iv_ceiling", f"IV > {config.IV_SUSPICIOUS} 应先怀疑泄漏或极端值",
                "iv_max", lambda v: v <= config.IV_SUSPICIOUS, G.WARN,
                "存在 IV 异常偏高的变量，需先排查泄漏/极端值"),
    # 告警：WOE 非单调需要手工合并分箱
    G.monotone_list_gate("woe_monotonicity", "WOE 分箱应保持单调",
                         "nonmono", G.WARN),
    # 告警：敏感字段入模需在报告里说明理由
    G.fact_in_gate("sensitive_fields", "敏感字段（年龄等）入模需说明理由",
                   "sensitive_used", set(config.SENSITIVE_SOFT_COLS), G.WARN),
    # 告警：数据集本身含硬阻断字段 —— 不是错，但要提醒建模前必须处理
    G.monotone_list_gate("forbidden_in_data", "数据集含硬阻断字段，建模前必须剔除",
                         "forbidden_present_hard", G.WARN),
    # 告警：坏样本太少，需要重采样或调整权重
    G.fact_gate("imbalance", f"坏样本占比低于 {config.IMBALANCE_FLOOR} 需重采样",
                "bad_rate", lambda v: v >= config.IMBALANCE_FLOOR, G.WARN,
                "坏样本占比过低，需重采样或显著调整样本权重"),
]


class CreditDomain(Domain):
    name = "credit"
    registry = tools.REGISTRY
    plan_spec = planner_spec.PLAN_SPEC
    system_prompt = SYSTEM_PROMPT
    gates = GATES

    def load(self):
        return data_mod.load()

    def bullets(self, res, idx):
        return bullets_mod.bullets(res, idx)

    def conclusion(self, results):
        return bullets_mod.conclusion(results)

    def scope_desc(self, dataset=None, feature=None, segment=None, bins=None,
                   keep=None, kind=None):
        return data_mod.scope_desc(dataset, feature, segment, bins, keep, kind)

    def objects_hint(self, df):
        feats = df.attrs.get("features", [])
        return (f"数据集：{df.attrs.get('dataset')}；变量（共 {len(feats)} 个）："
                + ", ".join(feats[:12]))
