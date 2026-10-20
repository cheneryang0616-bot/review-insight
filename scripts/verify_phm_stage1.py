#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条命令复现 PHM Stage 1：强制离线验收、原领域回归、单测与证据报告。"""
import argparse
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.core.agent import Agent
from insight.core.llm import OfflineStub
from insight.domains import get_domain
from insight.domains.phm import GATES, config, data, tools
from insight.domains.phm import report as phm_report
from scripts.run_eval import check_one
from scripts.run_loop_eval import CASES


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluation(domain):
    agent = Agent(get_domain(domain), backend=OfflineStub())
    cases = [json.loads(line) for line in
             (ROOT / "insight/domains" / domain / "eval/questions.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]
    active = agent.df.attrs.get("dataset", "")
    rows, skipped = [], []
    for case in cases:
        need = case.get("dataset")
        if domain == "credit" and need and need not in active:
            skipped.append(case["id"])
            continue
        rows.append(check_one(agent, case, use_llm=False))
    return {"domain": domain, "mode": "离线", "dataset": active, "total": len(rows),
            "passed": sum(row["passed"] for row in rows), "skipped": skipped, "cases": rows}


def loop_evaluation():
    rows = []
    for domain, cases in CASES.items():
        agent = Agent(get_domain(domain), backend=OfflineStub())
        for case in cases:
            result = agent.ask(case["question"], use_llm=False)
            called = [row["tool"] for row in result["plan"]]
            ok = (result["status"] == case["expect_status"]
                  and all(name in called for name in case.get("expect_tools", [])))
            rows.append({"domain": domain, "id": case["id"], "question": case["question"],
                         "status": result["status"], "tools": called, "passed": ok, "why": case["why"]})
    return {"total": len(rows), "passed": sum(row["passed"] for row in rows), "cases": rows}


def source_manifest(df):
    root = Path(df.attrs["root"])
    files = [Path(df.attrs["path"]), root / "src/cmapss/metrics.py",
             root / "data/train_FD001.txt", root / "data/test_FD001.txt",
             root / "data/RUL_FD001.txt", root / "artifacts/summary_FD001.json"]
    files += [Path(row["source"]) for row in data.runs(**data.asset_context(df))]
    return [{"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, help="完整 C-MAPSS 资产根目录，默认 RI_PHM_ROOT")
    args = ap.parse_args()
    if args.root is not None:
        os.environ["RI_PHM_ROOT"] = str(args.root.resolve())
    os.chdir(ROOT)
    try:
        df = data.load()
        if df.attrs["source"] != "full":
            ap.error("完整 Stage 1 验收需要原始序列和模型报告；摘要样例只能运行受限演示")
        manifest = source_manifest(df)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))

    # 图表与问答共用同一批计算（report.py 只调 tools/decision），避免图文不一致；
    # 必须在 checks 之前生成，否则 figures_rendered 引用不到
    figures = phm_report.build_all(df)

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test = unittest.TextTestRunner(verbosity=1).run(suite)
    tests = {"total": test.testsRun, "passed": test.testsRun - len(test.skipped)
             - len(test.failures) - len(test.errors), "skipped": len(test.skipped),
             "failed": len(test.failures), "errors": len(test.errors)}
    evaluations = {name: evaluation(name) for name in ("phm", "review", "credit")}
    loops = loop_evaluation()
    agent = Agent(get_domain("phm"), backend=OfflineStub())
    interval = agent.ask("能不能用这个预测区间安排停机检修？", use_llm=False)
    blocked = agent.ask("哪几台两周内必须修", use_llm=False)
    followup = agent.ask("那提前期改成7天呢", use_llm=False)
    cost = agent.ask("漏报是误报的10倍，告警阈值怎么选", use_llm=False)
    phase0_ok = (
        "56%" in interval["answer"] and interval["grounding"]["passed"]
        and not interval["gates"]["passed"]
        and "coverage_gate" in {g["name"] for g in interval["gates"]["blocking"]}
        and "不能用这个预测区间安排停机检修" in interval["answer"])
    p = next(row for row in blocked["results"] if row["tool"] == "maintenance_plan")
    after = next(row for row in followup["results"] if row["tool"] == "maintenance_plan")
    c = next(row for row in cost["results"] if row["tool"] == "cost_threshold")
    checks = {
        "eight_tools": len(tools.REGISTRY) == 8, "six_industrial_gates": len(GATES) == 6,
        "phase0_preserved": phase0_ok,
        "two_week_candidate_units": p["facts"]["order_units"] == [34, 81],
        "missed_before_clipping": p["facts"]["n_missed"] == 2 and
            "lead_time_gate" in {g["name"] for g in blocked["gates"]["blocking"]},
        "acceptance_grounded": blocked["status"] == "ok" and blocked["grounding"]["passed"],
        "followup_passes_lead_time": after["facts"]["n_missed"] == 0
            and after["facts"]["window_days"] == 14 and after["facts"]["lead_time_days"] == 7
            and followup["gates"]["passed"] and followup["grounding"]["passed"],
        "unit_tests": test.wasSuccessful(),
        "offline_evaluations": all(v["total"] == v["passed"] for v in evaluations.values()),
        "loop_evaluations": loops["total"] == loops["passed"],
        "no_torch": "torch" not in sys.modules,
        "figures_rendered": all(Path(figures[key]).is_file()
                                and Path(figures[key]).stat().st_size > 10_000
                                for key in ("fleet", "interval", "calibration", "cost")),
    }
    output = {"stage": 1, "mode": "offline_rule_and_template", "passed": all(checks.values()),
              "checks": checks, "tests": tests, "evaluations": evaluations, "loop": loops,
              "sources": manifest, "array_locator": "predictions.npz::field[u_test == unit]",
              "snapshots": {"phase0_interval": interval, "two_week_plan": blocked,
                            "seven_day_followup": followup, "cost_scenario": cost}}
    save_json(ROOT / "results/phm_stage1.json", output)
    save_json(ROOT / "results/eval_phm_offline.json", evaluations["phm"])
    save_json(ROOT / "results/eval_loop.json", loops)

    p_f, c_f = p["facts"], c["facts"]
    phm_loop = [row for row in loops["cases"] if row["domain"] == "phm"]
    report = [
        "# PHM Stage 1 验收报告", "",
        "执行日期：2026-10-21。依据新版 /Users/zhuan/agent/PHM_DATA_AGENT_PLAN.md 的 Phase 1；"
        "当前定位为可复现、可审计、可准入的工程交付。", "",
        "## 验收结果", "",
        "| 检查 | 结果 |", "|---|---|",
    ]
    report.extend(f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks.items())
    report += [
        "", "## 真实问答与约束", "",
        f"- 默认每天 1 循环、维护窗口 14 天、备件提前期 14 天，"
        f"候选机器为 {p_f['order_units']}，共 {p_f['n_orders']} 台，{p_f['n_missed']} 台来不及到货。",
        "- 原始下单日先参与提前期判断，然后截为零；具体原始值、候选日和建议下单日见下方问答及 JSON。",
        "- 同会话追问提前期改 7 天：继承两周窗口，提前期闸门通过；候选仍需人工复核。",
        "- 区间名义覆盖率 95%、实测 56%，低于 85% 准入线；"
        "coverage_gate 持续阻断区间决策，明确拒绝据此安排停机。",
        f"- 固定 30 循环风险事件、成本比 10:1：告警线 {c_f['threshold_rul']:.4f} 循环，"
        f"机队期望代价 {c_f['expected_cost']:.4f}，默认告警线代价 {c_f['default_cost']:.4f}，"
        f"降低 {c_f['compare_to_default']:.4f} 个假设成本单位。未校准 σ 下仅作情景分析。",
        "", "## 验证证据", "",
        f"- 单元测试：{tests['total']} 项，{tests['passed']} 项通过，"
        f"{tests['skipped']} 项跳过，{tests['failed']} 项失败，{tests['errors']} 项错误。",
    ]
    for domain, result in evaluations.items():
        report.append(f"- {domain} 离线评测：{result['passed']}/{result['total']}；"
                      f"数据 {result['dataset'] or '本地评论数据'}。")
    figure_lines = [
        "", "## 图表", "",
        "四张图由 `insight/domains/phm/report.py` 生成，数据来自与问答相同的工具计算，"
        "不另算一套。", "",
        f"![机队紧急度]({os.path.relpath(figures['fleet'], ROOT / 'results')})", "",
        f"> 哪几台进了两周窗口；虚线为 {config.WINDOW_DAYS} 天窗口与 "
        f"{config.SERVICE_NEAR} 循环警戒线。", "",
        f"![单机预测区间]({os.path.relpath(figures['interval'], ROOT / 'results')})", "",
        "> 最紧急 10 台的点估计与 ±1.96σ 区间，叉号为离线复盘真值 —— 区间偏窄一目了然。", "",
        f"![校准曲线]({os.path.relpath(figures['calibration'], ROOT / 'results')})", "",
        "> **最重要的一张**：横轴名义覆盖率、纵轴实测覆盖率。贴近对角线才算校准过。", "",
        f"![代价曲线]({os.path.relpath(figures['cost'], ROOT / 'results')})", "",
        "> 告警线扫描；最低点即期望代价最优阈值，差额相对默认 30 循环标注。", "",
    ]
    report += [
        f"- 闭环回归 {loops['passed']}/{loops['total']}，其中 PHM "
        f"{sum(row['passed'] for row in phm_loop)}/{len(phm_loop)}"
        "（缺机器编号、领域外意图、单机序列不足）。",
        "- tests/test_phm_stage1.py 对六条工业闸门分别验证触发与不触发，"
        "并以完整小型原始资产验证八工具契约、标签错位与负 RUL、参数跨工具覆盖。",
        "- 4 项 German Credit 数据专属测试在缺少该完整数据时跳过；"
        "PHM 的便携 fixture 测试不依赖本机 C-MAPSS 资产。",
        "", "## 一条命令复现", "",
        "    python3 scripts/verify_phm_stage1.py --root /Users/zhuan/c-mapss_train", "",
        "命令强制离线，不调用模型、不重训、不复制 artifacts；验证失败返回非零退出码。",
        "问答 CLI 在预期业务阻断时返回 2；本验收脚本在确认该阻断正确时返回 0。",
        "", "## 数字回查与来源", "",
        "[完整机器证据、回归明细及来源 SHA-256](phm_stage1.json)；"
        "[标准 PHM 评测](eval_phm_offline.json)；[设计说明](../docs/PHM_DESIGN.md)。",
        "单机数字回查至 predictions.npz 的 field[u_test == unit]；原始标签为"
        " RUL_FD00X.txt 第 unit 行；通道为原始 test 文件的 unit/cycle 选择器、"
        "最后窗口与训练统计计算。模型指标可回查 summary 与各 report.json 的 ensemble 字段。",
        "", "## 假设与边界", "",
        "- C-MAPSS 是仿真数据；成本、提前期、每天循环数和维护窗口均为显式假设，可在提问中覆盖。",
        "- 排期是规则候选清单，没有维修记录，未校验真实停机窗口、维修容量、库存或工作日历。",
        "- 成本目标固定风险时间窗 H，单独搜索告警线 t；原方案同时改变风险事件与告警线会退化。"
        "C_fp 在这个近似中是每次告警成本，包含真告警，并非完整的生产成本模型。",
        "- 传感器贡献是相对全部训练集统计的 z 偏离代理，不是梯度/SHAP或故障因果；"
        "健康指数以训练首末窗口趋势定向，多工况变化可能影响趋势。",
        "- 摘要样例可以问答，但没有原始序列或模型报告。相关工具返回 insufficient，"
        "pipeline_selfcheck 明确阻断不可验证的准入。",
        "- 本次只验收离线规则与模板；在线 LLM 规划可能漏强制审计，"
        "通用框架也不保证改写合成中的错误语义结论，不能把本次通过视作在线 LLM 安全验证。",
        "- 共享核心仅修复一处 0 值继承的兼容缺陷，保留所有既有编排与闸门机制；"
        "否则零提前期会在追问中被静默恢复默认值。",
        "- Phase 2 的图表已并入本报告；评测扩展到 12 题；可选 RAG 未执行。",
        "",
    ]
    report += figure_lines
    report += ["## Stage 1 主验收原文", "", Agent.render(blocked), "",
        "## 同会话追问原文", "", Agent.render(followup), "",
    ]
    (ROOT / "results/REPORT_phm.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"checks": checks, "tests": tests,
                      "evaluations": {k: f"{v['passed']}/{v['total']}" for k, v in evaluations.items()},
                      "report": str(ROOT / "results/REPORT_phm.md")}, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
