# -*- coding: utf-8 -*-
"""insight.core.planner —— 规划器（领域无关骨架）

领域只需要提供一张"意图表"和"意图→工具序列"的映射，其余逻辑共用：

- 意图识别：关键词命中计数（多意图时按组合规则取）
- 实体抽取：交给领域的 entity_fn
- 指代消解：只有追问才继承上一轮范围（否则会上下文污染，见 review 域的踩坑记录）
- 实体注入：只注入工具声明过的参数，防止"参数幻觉"
- LLM 规划：可选，失败静默回退规则规划
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import llm as llm_mod
from .registry import Registry


@dataclass
class PlanSpec:
    intents: Dict[str, List[str]]                       # 意图 → 触发词
    sequence: Dict[str, List[str]]                      # 意图 → 工具序列
    default_sequence: List[str]
    entity_fn: Callable[[str, object], dict]
    anaphora: List[str] = field(default_factory=list)
    composite: Dict[Tuple[str, ...], List[str]] = field(default_factory=dict)
    adjust_seq: Optional[Callable] = None               # (q, ent, seq, intents) -> seq
    scope_keys: Tuple[str, ...] = ("shop", "deal", "deal_type", "year")
    llm_hints: str = ""


def is_followup(q: str, anaphora: List[str]) -> bool:
    """判断是不是"追问"。只有追问才允许继承上一轮的分析对象。

    否则会出现：上一轮问"2020 年怎么样"，下一轮问"为什么差评多"，
    模型忠实地继续用 2020 年 —— 用户看到的口径是"全量"，数字却是子集。
    """
    if any(a in q for a in anaphora):
        return True
    return len(q.strip()) <= 8


def detect_intents(q: str, intents: Dict[str, List[str]]) -> List[str]:
    scored = []
    for name, words in intents.items():
        s = sum(1 for w in words if w in q)
        if s:
            scored.append((name, s))
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored]


def _pick_sequence(intents: List[str], spec: PlanSpec) -> List[str]:
    if intents and spec.composite:
        for k in (tuple(intents[:3]), tuple(intents[:2]), tuple(intents[:1])):
            if k in spec.composite:
                return list(spec.composite[k])
    if intents:
        return list(spec.sequence.get(intents[0], spec.default_sequence))
    return list(spec.default_sequence)


def _inherit(ent: dict, q: str, prev: dict, spec: PlanSpec) -> dict:
    prev = dict(prev or {})
    for k in spec.scope_keys:
        if k not in ent and prev.get(k) and any(a in q for a in spec.anaphora):
            ent[k] = prev[k]
    return ent


def _build_calls(seq: List, ent: dict, registry: Registry) -> List[dict]:
    """序列元素可以是工具名，也可以是 (工具名, {固定参数})。

    后者用于领域把某些参数钉死，例如"合规修复"场景必须用
    perf_metrics(exclude_forbidden=1) 重新评估，而不是让模型自己决定。
    """
    calls, seen = [], set()
    for entry in seq:
        extra = {}
        if isinstance(entry, (tuple, list)):
            name, extra = entry[0], dict(entry[1] or {})
        else:
            name = entry
        if name in seen or name not in registry:
            continue
        seen.add(name)
        allowed = registry.param_names(name)
        params = {k: ent[k] for k in spec_key_order(ent) if k in allowed}
        params.update({k: v for k, v in extra.items() if k in allowed})
        calls.append({"tool": name, "params": params})
    return calls


def spec_key_order(ent: dict) -> List[str]:
    """稳定顺序，便于测试与日志对比。"""
    order = ["shop", "deal", "deal_type", "year", "feature", "dim", "freq", "top_k"]
    return [k for k in order if k in ent] + [k for k in ent if k not in order]


def rule_plan(q: str, df, spec: PlanSpec, registry: Registry,
              prev_scope: Optional[dict] = None) -> dict:
    sticky = is_followup(q, spec.anaphora)
    prev = dict(prev_scope or {}) if sticky else {}

    ent = spec.entity_fn(q, df)
    ent = _inherit(ent, q, prev, spec)

    intents = detect_intents(q, spec.intents)
    seq = _pick_sequence(intents, spec)
    if spec.adjust_seq:
        seq = spec.adjust_seq(q, ent, seq, intents)

    return {"plan": _build_calls(seq, ent, registry),
            "intent": intents[0] if intents else "default",
            "entities": ent, "planner": "rule",
            "reason": f"命中意图 {intents[:3] or ['default']}，实体 {ent or '无'}"}


# ---------------------------------------------------------------- LLM 规划
PLAN_PROMPT = """你是数据洞察 agent 的规划器。根据用户问题，从工具清单中选择 1~4 个工具，
按执行顺序返回 JSON 数组，每项形如 {{"tool": "工具名", "params": {{...}}}}。
只能使用清单里的工具名和参数；不确定就不传参数。只输出 JSON，不要解释。

领域约定：
{hints}

可用工具：
{sigs}

当前上下文：{context}
可用对象：{objects}
用户问题：{q}
"""


def llm_plan(q: str, df, spec: PlanSpec, registry: Registry, backend,
             prev_scope: Optional[dict] = None,
             objects: str = "") -> dict:
    prompt = PLAN_PROMPT.format(
        hints=spec.llm_hints or "无",
        sigs=registry.signatures(),
        context=prev_scope or "无",
        objects=objects or "无",
        q=q)
    raw = backend.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
    arr = llm_mod.parse_json_array(raw)

    calls = []
    for item in arr[:4]:
        if not isinstance(item, dict) or item.get("tool") not in registry:
            continue
        allowed = registry.param_names(item["tool"])
        params = {k: v for k, v in (item.get("params") or {}).items() if k in allowed}
        calls.append({"tool": item["tool"], "params": params})
    if not calls:
        raise llm_mod.LLMError("LLM 未给出合法工具调用")

    # 指代继承不指望模型自觉：规则层兜底，否则"那这家店呢"会丢对象
    sticky = is_followup(q, spec.anaphora)
    ent = spec.entity_fn(q, df)
    if sticky:
        ent = _inherit(ent, q, prev_scope or {}, spec)
    for call in calls:
        allowed = registry.param_names(call["tool"])
        for k in spec.scope_keys:
            if k in allowed and k in ent and k not in call["params"]:
                call["params"][k] = ent[k]

    return {"plan": calls, "intent": "llm", "entities": ent, "planner": "llm",
            "reason": "由模型选择工具（实体由规则层兜底）"}
