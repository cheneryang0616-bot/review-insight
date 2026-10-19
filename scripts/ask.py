#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一入口：问一句话，拿到带出处、经过质检和断言闸门的分析。

用法：
    python scripts/ask.py "为什么差评这么多？"                      # 默认 review 领域
    python scripts/ask.py "模型效果怎么样？" --domain credit
    python scripts/ask.py "合规吗？" --domain credit --llm
    python scripts/ask.py "哪家店差评最多" --json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight.core import llm as llm_mod
from insight.core.agent import Agent
from insight.domains import available, get_domain


def main() -> int:
    ap = argparse.ArgumentParser(description="可复现的分析 Agent（review / credit）")
    ap.add_argument("question", help="自然语言问题")
    ap.add_argument("--domain", default="review", choices=available())
    ap.add_argument("--llm", action="store_true", help="强制使用 LLM 规划与合成")
    ap.add_argument("--no-llm", action="store_true", help="强制离线模板合成")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON（含工具原始结果）")
    args = ap.parse_args()

    use_llm = True if args.llm else (False if args.no_llm else None)
    backend = llm_mod.OpenAICompat() if args.llm else None   # --llm 必须真用上模型
    agent = Agent(get_domain(args.domain), backend=backend)
    res = agent.ask(args.question, use_llm=use_llm)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    else:
        print(Agent.render(res))
    return 0 if (res["grounding"]["passed"] and res["gates"]["passed"]) else 2


if __name__ == "__main__":
    sys.exit(main())
