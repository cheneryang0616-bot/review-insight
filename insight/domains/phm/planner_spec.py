# -*- coding: utf-8 -*-
"""Phase 0 仅声明已经实现的两个工具，不提前承诺单机与排期能力。"""
from ...core.planner import PlanSpec

PLAN_SPEC = PlanSpec(
    intents={
        "uncertain": ["区间", "可信", "敢不敢", "置信", "覆盖率", "停机", "检修"],
        "fleet": ["机队", "总览", "多少台", "哪些机器", "整体"],
    },
    sequence={
        "uncertain": ["uncertainty_audit", "fleet_overview"],
        "fleet": ["fleet_overview", "uncertainty_audit"],
    },
    default_sequence=["fleet_overview", "uncertainty_audit"],
    entity_fn=lambda q, df: {},
    anaphora=["这个", "这些", "它", "那", "同样"],
    scope_keys=(),
    llm_hints=("当前仅实现 Phase 0。每轮都调用 uncertainty_audit 与 fleet_overview。"
               "询问区间/可信度/停机时先审计；不能生成排期或工单。"),
)
