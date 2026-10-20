# -*- coding: utf-8 -*-
"""PHM 第三领域：Phase 0 区间决策准入原型。"""
from ...core import gates as G
from ...core.agent import Domain
from . import bullets as bullets_mod
from . import config, data, planner_spec, tools

GATES = [
    G.fact_gate("coverage_gate", "区间实测覆盖率须达到名义覆盖率减去容忍差",
                "empirical_coverage", lambda v: v >= config.MIN_COVERAGE, G.ERROR,
                "区间覆盖率未达到准入线，禁止用该区间安排停机检修"),
]


class PHMDomain(Domain):
    name = "phm"
    registry = tools.REGISTRY
    plan_spec = planner_spec.PLAN_SPEC
    gates = GATES
    system_prompt = (
        "你是设备健康分析助手，当前为 Phase 0 原型。只能使用工具 facts/table 中的数字，"
        "每个结论标注 [T编号]。必须声明 C-MAPSS 是仿真数据，明确成本与提前期是假设。"
        "usable_for_decision 为 false 时，必须明确回答不能用该区间安排停机检修，"
        "说明 coverage_gate 阻断并建议独立校准集上的共形预测校准。"
        "无审计结果也不得批准停机。覆盖率通过仅代表此项检查通过。"
        "本阶段不输出具体停机日期、排期或工单；RUL 单位为循环，不能当成天。")
    clarify_hint = "- 机队 RUL 总览\n- 预测区间可信度与覆盖率审计（Phase 0）"

    def load(self):
        self._df = data.load()
        return self._df

    def bullets(self, res, idx):
        return bullets_mod.bullets(res, idx)

    def conclusion(self, results):
        return bullets_mod.conclusion(results)

    def scope_desc(self, **scope):
        return data.scope_desc(getattr(self, "_df", None), **scope)

    def objects_hint(self, df):
        return f"数据集 {df.attrs['dataset']}；机器编号 1..{len(df)}（本阶段只支持机队与区间审计）"
