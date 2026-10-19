# -*- coding: utf-8 -*-
"""insight.core.agent —— 会话编排（领域无关）

一次 ask() 的完整链路：

    question
      → planner（规则 / LLM）           选工具、抽实体、指代消解
      → registry.call × N               唯一数字来源（facts + table）
      → composer（模板 / LLM）           只许用工具结果
      → grounding + gates               数字回查（通用）+ 业务断言（领域）
      → 失败：带反馈重写一次 → 仍失败删句降级

领域（Domain）需要提供的东西都在 `Domain` 的注释里，框架只依赖这些接口。
"""
from typing import List, Optional

from . import gates as gates_mod
from . import grounding, llm as llm_mod, planner
from .registry import Registry


class Domain:
    """领域插件契约（不做强制继承，鸭子类型即可）。

    必须提供：
      name: str                领域名（review / credit）
      registry: Registry       工具表
      plan_spec: PlanSpec      意图表 + 实体抽取 + 序列
      load() -> df             数据（带缓存）
      bullets(res, idx)        把一个工具结果翻译成 1~3 条人话（模板合成用）
      scope_desc(**scope)      把切片条件翻译成人话
    可选：
      gates: List[Gate]        业务断言（推荐配置）
      system_prompt: str       LLM 合成的角色设定
      conclusion(results)      模板合成的"结论"段
      objects_hint(df)         给 LLM 规划器的可选对象清单
    """
    name = "base"
    registry: Registry = None
    plan_spec = None
    gates: List = []
    system_prompt = "你是数据分析师。只能使用给定工具结果中的数字，禁止引入外部知识或估算。"


# ---------------------------------------------------------------- 渲染辅助
def pct(x) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(x)


def pt(x) -> str:
    try:
        return f"{float(x) * 100:.1f} 个百分点"
    except (TypeError, ValueError):
        return str(x)


def md_table(rows: List[dict]) -> str:
    if not rows:
        return "（无数据）"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


class Agent:
    def __init__(self, domain, backend=None):
        self.domain = domain
        self.registry: Registry = domain.registry
        self.df = domain.load()
        self.backend = backend or llm_mod.get_backend()
        self.history: List[dict] = []
        self.scope = {}
        self.gates = [gates_mod.trace_gate(self._check_grounding),
                      gates_mod.tool_health_gate()] + list(getattr(domain, "gates", []))

    # ---------------------------------------------------------------- 内部
    def _check_grounding(self, answer: str, results: List[dict]) -> dict:
        return grounding.check(answer, results, self._scope_texts())

    def _scope_texts(self) -> List[str]:
        try:
            return [self.domain.scope_desc(**self.scope)]
        except Exception:
            return []

    def _execute(self, calls: List[dict]) -> List[dict]:
        results = []
        for call in calls:
            name, params = call["tool"], call.get("params") or {}
            try:
                res = self.registry.call(name, self.df, **params)
            except Exception as e:
                res = {"tool": name, "params": params, "table": [], "facts": {},
                       "note": f"工具执行失败：{type(e).__name__}: {e}", "error": True}
            results.append(res)
        return results

    def _plan(self, question: str, prev_scope: dict, use_llm: Optional[bool]):
        if use_llm is None:
            use_llm = not isinstance(self.backend, llm_mod.OfflineStub)
        base = planner.rule_plan(question, self.df, self.domain.plan_spec,
                                 self.registry, prev_scope)
        if not use_llm:
            return base, "rule", use_llm
        try:
            plan = planner.llm_plan(question, self.df, self.domain.plan_spec,
                                    self.registry, self.backend, prev_scope,
                                    objects=getattr(self.domain, "objects_hint", lambda df: "")(
                                        self.df))
            return plan, "llm", use_llm
        except llm_mod.LLMError:
            return base, "rule(LLM失败回退)", use_llm

    # ---------------------------------------------------------------- 合成
    def _compose_offline(self, results: List[dict]) -> str:
        parts = [f"### 分析范围\n{self.domain.scope_desc(**self.scope)}", "\n### 事实"]
        for i, res in enumerate(results, 1):
            parts.extend(self.domain.bullets(res, i))
        concl = self.domain.conclusion(results) if hasattr(self.domain, "conclusion") else []
        parts.append("\n### 结论")
        if concl:
            parts.extend(f"{i}. {c}" for i, c in enumerate(concl, 1))
        else:
            parts.append("1. 本轮为结构化取数，未触及归因分析；可继续追问具体问题。")
        return "\n".join(parts)

    def _compose_llm(self, question: str, results: List[dict], feedback: str = "") -> str:
        blocks = []
        for i, res in enumerate(results, 1):
            kv = "，".join(f"{k}={v}" for k, v in (res.get("facts") or {}).items())
            blocks.append(f"[T{i}] 工具 {res['tool']}（{res.get('note','')}）\n"
                          f"{md_table(res.get('table') or [])}\n关键值：{kv}")
        user = (f"分析范围：{self.domain.scope_desc(**self.scope)}\n\n"
                + "\n\n".join(blocks) + f"\n\n用户问题：{question}")
        if feedback:
            user += (f"\n\n【上一版被质检拒绝】以下数字在工具结果里找不到，"
                     f"必须删除或改正：{feedback}")
        return self.backend.chat([{"role": "system", "content": self.domain.system_prompt},
                                  {"role": "user", "content": user}],
                                 temperature=0.2, max_tokens=800)

    # ---------------------------------------------------------------- 主入口
    def ask(self, question: str, use_llm: Optional[bool] = None) -> dict:
        prev_scope = dict(self.scope) if planner.is_followup(
            question, self.domain.plan_spec.anaphora) else {}

        plan, planner_used, use_llm = self._plan(question, prev_scope, use_llm)
        results = self._execute(plan["plan"])
        scope = {k: plan["entities"][k] for k in self.domain.plan_spec.scope_keys
                 if k in plan["entities"]}
        self.scope = scope

        composer, answer, llm_error = "template", None, None
        if use_llm:
            try:
                answer = self._compose_llm(question, results)
                composer = "llm"
            except llm_mod.LLMError as e:
                llm_error = str(e)
        if answer is None:
            answer = self._compose_offline(results)

        # ---- 质检（数字回查）
        gres = self._check_grounding(answer, results)
        attempts = 1
        if composer == "llm" and not gres["passed"]:
            bad = "、".join(f"{u['raw']}{'%' if u['percent'] else ''}"
                            for u in gres["unsupported"])
            try:
                retry = self._compose_llm(question, results, feedback=bad)
                g2 = self._check_grounding(retry, results)
                attempts = 2
                if g2["passed"]:
                    answer, gres = retry, g2
                else:
                    gres = g2
            except llm_mod.LLMError:
                pass

        degraded = False
        if not gres["passed"] and composer == "llm":
            answer = grounding.strip_bad_sentences(answer, gres["unsupported"])
            gres = self._check_grounding(answer, results)
            degraded = True

        # ---- 业务断言（领域）
        gate_ctx = {"answer": answer, "results": results, "df": self.df,
                    "scope": scope, "domain": self.domain.name}
        gate_res = gates_mod.run_all(self.gates, gate_ctx)

        out = {
            "domain": self.domain.name,
            "question": question,
            "plan": plan["plan"],
            "planner": planner_used,
            "reason": plan.get("reason", ""),
            "scope": scope,
            "results": results,
            "answer": answer,
            "composer": composer,
            "grounding": gres,
            "gates": gate_res,
            "attempts": attempts,
            "degraded": degraded,
            "llm_error": llm_error,
        }
        self.history.append(out)
        return out

    # ---------------------------------------------------------------- 渲染
    @staticmethod
    def render(res: dict) -> str:
        lines = [f"## Q: {res['question']}", ""]
        chain = " → ".join(c["tool"] for c in res["plan"])
        lines.append(f"> 领域：{res['domain']}｜规划器：{res['planner']}"
                     f"｜工具链：{chain}｜合成器：{res['composer']}")
        lines.append("")
        lines.extend([res["answer"], ""])
        lines.append("---")
        lines.append(grounding.format_report(res["grounding"]))
        lines.append(gates_mod.format_report(res["gates"]))
        if res.get("llm_error"):
            lines.append(f"\n> LLM 调用失败，已回退模板：{res['llm_error']}")
        if res.get("degraded"):
            lines.append("\n> ⚠️ 二次重写仍未通过数字回查，已删除含未支撑数字的句子（降级输出）。")
        return "\n".join(lines)
