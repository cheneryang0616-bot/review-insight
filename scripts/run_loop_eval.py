#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自修正用例（Loop eval）—— 先证明现状会翻车，再证明改动真的解决

这组用例不看"结论对不对"，只看**失败之后的行为对不对**：

| 场景 | 单轮版本会怎样 | 期望行为 |
|---|---|---|
| 意图没听懂 | 静默给默认答案（用户以为答了） | 明确说"我没理解"，列出能答什么 |
| 必填参数缺失 | 工具报错，但答案照出 | 反问：你要看哪个变量？并给候选 |
| 切片太小 | 只说"样本不足"就结束 | 自动放宽切片重算一轮 |
| 工具异常 | 答案里出现"工具执行失败" | 剔除失败工具，用剩下的继续 |

用法：python scripts/run_loop_eval.py [--domain review|credit] [--verbose]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.domains import available, get_domain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = {
    "review": [
        {"id": "L1", "question": "把 Income 分箱看一下",
         "expect_status": "clarification", "why": "review 域没有分箱能力，不该静默给默认答案"},
        {"id": "L2", "question": "这个指标的口径是怎么定义的",
         "expect_status": "clarification", "why": "完全没听懂的问题应澄清而不是猜"},
        {"id": "L3", "question": "2004 年的差评归因",
         "expect_status": "ok", "expect_tools": ["aspect_gap"],
         "why": "2004 年样本极少，应放宽切片后仍给出结论"},
        {"id": "L4", "question": "店铺 4134768 的套餐差评率排名",
         "expect_status": "ok", "expect_tools": ["deal_ranking"],
         "why": "shop 实体不该把套餐排名工具整个换掉"},
        {"id": "L5", "question": "为什么差评这么多？短板在哪？",
         "expect_status": "ok", "expect_tools": ["aspect_gap"],
         "why": "正常问题不该被误判成需要澄清"},
    ],
    "credit": [
        {"id": "C-L1", "question": "看下这个变量的分箱",
         "expect_status": "clarification", "why": "woe_table 缺必填 feature，应反问并给候选"},
        {"id": "C-L2", "question": "把 Income 分箱看一下",
         "expect_status": "clarification", "why": "字段名不存在，应给候选而不是报错"},
        {"id": "C-L3", "question": "模型效果怎么样？",
         "expect_status": "ok", "expect_tools": ["perf_metrics"],
         "why": "正常问题不该被误判"},
        {"id": "C-L4", "question": "评分刻度怎么映射？",
         "expect_status": "ok", "expect_tools": ["score_scale"],
         "why": "正常问题不该被误判"},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None, choices=available())
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    domains = [args.domain] if args.domain else list(CASES)
    total = passed = 0
    rows = []
    for dom in domains:
        agent = Agent(get_domain(dom))
        for c in CASES[dom]:
            total += 1
            r = agent.ask(c["question"], use_llm=False)
            status = r.get("status", "ok")
            tools = [x["tool"] for x in r["plan"]]
            ok_status = status == c["expect_status"]
            ok_tools = all(t in tools for t in c.get("expect_tools", []))
            ok = ok_status and ok_tools
            passed += ok
            rows.append({"domain": dom, "id": c["id"], "question": c["question"],
                         "status": status, "tools": tools, "passed": ok,
                         "why": c["why"]})
            if args.verbose or not ok:
                mark = "PASS" if ok else "FAIL"
                print(f"  [{mark}] {dom}/{c['id']} {c['question']}")
                print(f"         status={status}（期望 {c['expect_status']}）"
                      f" tools={tools}")
                if not ok:
                    print(f"         原因：{c['why']}")
                    ans = r["answer"].replace("\n", " ")[:110]
                    print(f"         实际回答：{ans}")

    print(f"\n{'=' * 66}")
    print(f"自修正用例：{passed}/{total} 通过")
    out = os.path.join(ROOT, "results", "eval_loop.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"total": total, "passed": passed, "cases": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"明细已写入 {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
