#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归评测：跑某个领域的 eval/questions.jsonl

每条问题检查四件事：
1. 工具选择：期望的工具是否都被调用
2. 事实正确：期望的 facts 键值是否对得上（浮点按 1% 相对误差）
3. 切片正确：实体（店铺/年份/变量）是否抽对
4. 质检与闸门：数字回查是否通过、断言闸门是否按预期通过/阻断

问题可带 "dataset" 字段：只在当前数据集匹配时执行，否则跳过。
用法：python scripts/run_eval.py [--domain review|credit] [--llm] [--verbose]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core.agent import Agent
from insight.domains import available, get_domain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def close(a, b, rel=0.01):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(a - b) <= max(1e-6, rel * abs(b))


def check_one(agent, case, use_llm=False, verbose=False):
    res = agent.ask(case["question"], use_llm=use_llm)
    called = [c["tool"] for c in res["plan"]]
    facts = {}
    for r in res["results"]:
        facts.update(r.get("facts") or {})

    checks = []
    missing = [t for t in case.get("expect_tools", []) if t not in called]
    checks.append(("tools", not missing, f"缺 {missing}" if missing else "OK"))

    bad_facts = [(k, v, facts.get(k)) for k, v in (case.get("expect_facts") or {}).items()
                 if not close(facts.get(k), v)]
    checks.append(("facts", not bad_facts, f"{bad_facts[:2]}" if bad_facts else "OK"))

    bad_scope = [(k, v, res["scope"].get(k)) for k, v in (case.get("expect_scope") or {}).items()
                 if str(res["scope"].get(k)) != str(v)]
    checks.append(("scope", not bad_scope, f"{bad_scope}" if bad_scope else "OK"))

    g = res["grounding"]
    checks.append(("grounding", g["passed"],
                   "OK" if g["passed"] else f"未支撑 {[u['raw'] for u in g['unsupported']]}"))

    want_gate = case.get("expect_gates")          # "pass" / "block" / None
    if want_gate == "pass":
        ok = res["gates"]["passed"]
        detail = "OK" if ok else f"被阻断：{[b['name'] for b in res['gates']['blocking']]}"
        checks.append(("gates", ok, detail))
    elif want_gate == "block":
        names = {b["name"] for b in res["gates"]["blocking"]}
        want = set(case.get("expect_block_gates") or ["fair_lending"])
        ok = want <= names
        checks.append(("gates", ok, f"阻断项 {sorted(names)}，期望含 {sorted(want)}"))

    ok = all(c[1] for c in checks)
    if verbose:
        print(f"\n--- {case['id']} {case['question']}")
        print(f"    工具：{' → '.join(called)}")
        for name, passed, detail in checks:
            print(f"    {'PASS' if passed else 'FAIL'} {name:10s} {detail}")
    return {"id": case["id"], "question": case["question"], "passed": ok,
            "tools": called, "composer": res["composer"],
            "grounding_rate": g["rate"],
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="review", choices=available())
    ap.add_argument("--llm", action="store_true", help="用 LLM 规划+合成（默认离线）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qs = os.path.join(ROOT, "insight", "domains", args.domain, "eval", "questions.jsonl")
    with open(qs, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]

    agent = Agent(get_domain(args.domain))
    active = agent.df.attrs.get("dataset", "")

    ran, skipped = [], []
    for c in cases:
        need = c.get("dataset")
        if need and need not in active and args.domain == "credit":
            skipped.append(c["id"])
            continue
        ran.append(check_one(agent, c, args.llm, args.verbose))

    n_pass = sum(r["passed"] for r in ran)
    mode = "LLM" if args.llm else "离线"
    print(f"\n{'=' * 66}")
    print(f"评测（{args.domain} / {mode} / 数据 {active}）：{n_pass}/{len(ran)} 通过"
          f"　数字回查平均 {sum(r['grounding_rate'] for r in ran) / max(len(ran), 1):.3f}")
    if skipped:
        print(f"跳过（数据集不匹配）：{skipped}")
    for r in ran:
        if not r["passed"]:
            print(f"  ✗ {r['id']} {r['question']}")
            for c in r["checks"]:
                if not c["passed"]:
                    print(f"      {c['name']}: {c['detail']}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"eval_{args.domain}_{'llm' if args.llm else 'offline'}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"domain": args.domain, "mode": mode, "dataset": active,
                   "total": len(ran), "passed": n_pass, "skipped": skipped,
                   "cases": ran}, f, ensure_ascii=False, indent=1)
    print(f"明细已写入 {out}")
    return 0 if n_pass == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
