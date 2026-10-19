# -*- coding: utf-8 -*-
"""insight.core.gates —— 断言闸门层（本项目相对"随便问问 LLM"的第二层防线）

和 groundedness 的分工：

| 层 | 管什么 | 例子 | 是否跨领域 |
|---|---|---|---|
| grounding | 数字有没有出处（语法层） | 答案里"86.41%"在工具结果里找不到 | 是 |
| gates     | 结论有没有违反业务约束（语义层） | 某变量 IV=1.11 → 疑似泄漏；模型用了性别字段 | 否，每域自定义 |

为什么需要 gates：**风控、评测、合规这类场景，错误的形态不是"算错数字"，而是
"数字都对，但结论违反了一条硬约束"**。比如 AUC 0.80 却把 `Sex & Marital Status`
喂进了模型 —— 每个数字都能回查到，但整套东西不可用。

三个等级：
- error：阻断。答案必须改，不能就这么给出去
- warn ：告警。附在答案末尾，交给人判断
- info ：记录，不改输出
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

ERROR, WARN, INFO = "error", "warn", "info"


@dataclass
class GateResult:
    name: str
    desc: str
    level: str
    passed: bool
    detail: str = ""
    value: object = None

    def to_dict(self) -> dict:
        return {"name": self.name, "desc": self.desc, "level": self.level,
                "passed": self.passed, "detail": self.detail, "value": self.value}


class Gate:
    def __init__(self, name: str, desc: str, level: str, fn: Callable):
        self.name, self.desc, self.level, self.fn = name, desc, level, fn

    def run(self, ctx: dict) -> GateResult:
        try:
            res = self.fn(ctx)
        except Exception as e:  # 闸门自身出错不能拖垮整轮，但要显式记录
            return GateResult(self.name, self.desc, INFO, True,
                              f"闸门执行异常，已跳过：{type(e).__name__}: {e}")
        if isinstance(res, GateResult):
            return res
        passed, detail = res
        return GateResult(self.name, self.desc, self.level, bool(passed), str(detail or ""))


# ---------------------------------------------------------------- 证据辅助
def merged_facts(results: List[dict]) -> Dict[str, object]:
    facts: Dict[str, object] = {}
    for r in results:
        facts.update(r.get("facts") or {})
    return facts


def _skip_if_missing(name, desc, level, why):
    return GateResult(name, desc, level, True, f"本轮未产出该指标，跳过（{why}）")


# ---------------------------------------------------------------- 通用闸门
def trace_gate(check_fn) -> Gate:
    """把 groundedness.check 包成闸门。check_fn(answer, results) -> dict"""
    def fn(ctx):
        g = check_fn(ctx["answer"], ctx["results"])
        return GateResult("numeric_trace", "答案中所有数字必须可追溯到工具输出",
                          ERROR, g["passed"],
                          f"{g['supported']}/{g['checked']} 可追溯"
                          + ("" if g["passed"] else
                             f"；无出处：{[u['raw'] for u in g['unsupported']]}"),
                          g["rate"])
    return Gate("numeric_trace", "答案中所有数字必须可追溯到工具输出", ERROR, fn)


def tool_health_gate() -> Gate:
    def fn(ctx):
        bad = [r["tool"] for r in ctx["results"] if r.get("error")]
        return GateResult("tool_health", "所有工具调用必须成功", ERROR, not bad,
                          "全部成功" if not bad else f"失败工具：{bad}")
    return Gate("tool_health", "所有工具调用必须成功", ERROR, fn)


def fact_gate(name: str, desc: str, key: str, predicate: Callable,
              level: str = WARN, bad_text: str = "") -> Gate:
    """声明式断言：对某个 fact 施加谓词。

    谓词返回 True 视为通过。fact 不存在时跳过（不算失败），
    这样同一套闸门可以挂在"只跑了部分工具"的轮次上。
    """
    def fn(ctx):
        facts = merged_facts(ctx["results"])
        if key not in facts:
            return _skip_if_missing(name, desc, level, f"缺 fact: {key}")
        val = facts[key]
        ok = bool(predicate(val))
        detail = "通过" if ok else (bad_text or f"{key}={val} 不满足约束")
        return GateResult(name, desc, level, ok, detail, val)
    return Gate(name, desc, level, fn)


def fact_in_gate(name: str, desc: str, key: str, forbidden: set,
                 level: str = ERROR) -> Gate:
    """断言某个列表型 fact 与禁用集合没有交集（用于禁忌字段）。"""
    def fn(ctx):
        facts = merged_facts(ctx["results"])
        if key not in facts:
            return _skip_if_missing(name, desc, level, f"缺 fact: {key}")
        used = facts[key]
        if isinstance(used, str):
            used = [used] if used else []
        hit = sorted({u for u in used if u in forbidden})
        return GateResult(name, desc, level, not hit,
                          "未使用禁忌字段" if not hit else f"使用了禁忌字段：{hit}",
                          hit)
    return Gate(name, desc, level, fn)


def monotone_list_gate(name: str, desc: str, key: str, level: str = WARN) -> Gate:
    """断言列表型 fact 为空（例如"非单调的变量"清单）。"""
    def fn(ctx):
        facts = merged_facts(ctx["results"])
        if key not in facts:
            return _skip_if_missing(name, desc, level, f"缺 fact: {key}")
        items = facts[key]
        ok = not items
        return GateResult(name, desc, level, ok,
                          "无" if ok else f"{len(items)} 项需处理：{items[:5]}", items)
    return Gate(name, desc, level, fn)


# ---------------------------------------------------------------- 执行
def run_all(gates: Optional[List[Gate]], ctx: dict) -> dict:
    gates = gates or []
    results = [g.run(ctx) for g in gates]
    blocking = [r for r in results if not r.passed and r.level == ERROR]
    warnings = [r for r in results if not r.passed and r.level == WARN]
    return {
        "results": [r.to_dict() for r in results],
        "checked": len(results),
        "passed": not blocking,
        "blocking": [r.to_dict() for r in blocking],
        "warnings": [r.to_dict() for r in warnings],
    }


def format_report(res: dict) -> str:
    if not res["checked"]:
        return "（本领域未配置断言闸门）"
    if res["passed"] and not res["warnings"]:
        return f"✅ 断言闸门 {res['checked']}/{res['checked']} 通过"
    parts = []
    if res["passed"]:
        parts.append(f"✅ 硬约束 {res['checked']} 项全部通过")
    else:
        parts.append(f"⛔ 阻断：{len(res['blocking'])} 项硬约束未通过")
    for w in res["warnings"]:
        parts.append(f"⚠️ {w['name']}：{w['detail']}")
    for b in res["blocking"]:
        parts.append(f"⛔ {b['name']}：{b['detail']}")
    return "；".join(parts)
