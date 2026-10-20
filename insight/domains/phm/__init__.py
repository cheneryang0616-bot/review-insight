# -*- coding: utf-8 -*-
"""PHM：八工具、六工业闸门，复用现有框架闭环。"""
from ...core import gates as G
from ...core.agent import Domain
from . import bullets as bullets_mod
from . import config, data, planner_spec, tools

GATES = [
    G.fact_gate("coverage_gate", "区间实测覆盖率须达到准入线",
                "empirical_coverage", lambda v: v >= config.MIN_COVERAGE, G.ERROR,
                "区间覆盖率未达到准入线，禁止用该区间安排停机检修"),
    G.fact_gate("lead_time_gate", "维护候选必须等得到备件",
                "n_missed", lambda v: v == 0, G.ERROR,
                "存在今天下单也无法及时到货的机器，当前候选方案不可执行"),
    G.monotone_list_gate("rul_monotone", "预测 RUL 必须非负", "bad_units", G.ERROR),
    G.monotone_list_gate("threshold_monotone", "漏报成本比上升时告警阈值不得下降",
                         "threshold_violations", G.WARN),
    G.fact_gate("model_selection", "所选比较模型的 NASA-Score 不得劣于最优值",
                "nasa_gap", lambda v: v <= 1e-9, G.WARN,
                "所选模型的 NASA-Score 更差，晚报警惩罚更大"),
    G.fact_gate("pipeline_selfcheck", "预测、原始 RUL 和原始序列必须按机器对齐",
                "pipeline_ok", lambda v: v is True, G.ERROR,
                "数据对齐检查未通过或缺少原始资产无法验证，不能准入"),
]


class PHMDomain(Domain):
    name = "phm"
    registry = tools.REGISTRY
    plan_spec = planner_spec.PLAN_SPEC
    gates = GATES
    system_prompt = (
        "你是设备健康分析助手。只能使用工具 facts/table 的数字，每条结论标注 [T编号]。"
        "必须声明 C-MAPSS 是仿真数据，披露工具结果中的成本、提前期与循环/天换算假设。"
        "usable_for_decision=false 时必须回答不能用这个预测区间安排停机检修。"
        "n_missed>0、bad_units非空或pipeline_ok=false时必须说明相应闸门阻断。"
        "没有覆盖率审计不能批准区间决策。成本阈值基于未校准σ时只能作为期望代价近似情景。"
        "传感器贡献只是 z 偏离代理，不是梯度/SHAP或故障因果。"
        "维护单只是点预测规则候选，未核验产能/真实维护窗口，不批准生产停机。")
    clarify_hint = (
        "- 机队 RUL 总览\n- 单机健康（需要机器编号）\n- 退化斜率与传感器代理贡献（需要编号）\n"
        "- 区间可信度与覆盖率审计\n- 告警阈值与代价情景\n- 维护候选排期与备件约束\n- 模型对比")

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
        return f"数据集 {df.attrs['dataset']}；机器编号 1..{len(df)}；可选择 FD001..FD004"
