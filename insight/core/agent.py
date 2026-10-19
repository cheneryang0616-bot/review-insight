# -*- coding: utf-8 -*-
"""insight.core.agent —— 会话编排（领域无关）

一次 ask() 现在是一个**多轮闭环**，而不是"一次规划、一次执行"：

    规划 ─┐
          ▼
    ┌── 执行 ──▶ 观察（工具报错？样本不足？参数缺失？）
    │                 │
    │      ┌──────────┼─────────────┐
    │      ▼          ▼             ▼
    │   缺必填参数   样本不足      工具异常
    │      │          │             │
    │   反问用户   放宽切片重算   剔除该工具重跑
    │      │          │             │
    └──────┴──────────┴─────────────┘（最多 max_turns 轮）
          ▼
     合成 → 数字回查 → 业务断言 → 修复/降级

为什么要有这个循环：数据分析里"算不出来"是常态（切片太小、字段没对上、口径缺失）。
单轮版本遇到这些会**照样输出一段话**，用户看不出问题；只有把它变成
"观察到失败 → 改变动作"，才谈得上 agent。
"""
from typing import List, Optional

from . import gates as gates_mod
from . import grounding, llm as llm_mod, planner
from .registry import Registry


class Domain:
    """领域插件契约（鸭子类型即可）。

    必须提供：
      name / registry / plan_spec / load() / bullets(res, idx) / scope_desc(**scope)
    可选：
      gates             业务断言（推荐）
      system_prompt     LLM 合成的角色设定
      conclusion(results) 模板合成的"结论"段
      objects_hint(df)  可选对象清单（也用于澄清时给候选）
      clarify_hint      听不懂问题时的能力清单
    """
    name = "base"
    registry: Registry = None
    plan_spec = None
    gates: List = []
    system_prompt = "你是数据分析师。只能使用给定工具结果中的数字，禁止引入外部知识或估算。"
    clarify_hint = ""


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
        self.max_turns = 3
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

    def _scope_desc(self) -> str:
        try:
            return self.domain.scope_desc(**self.scope)
        except Exception:
            return "当前数据全量"

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

    # ---------------------------------------------------------------- 观察 / 修复
    def _check_required(self, calls: List[dict]):
        runnable, missing = [], []
        for c in calls:
            miss = self.registry.missing_required(c["tool"], c["params"])
            (missing.append({"tool": c["tool"], "missing": miss}) if miss
             else runnable.append(c))
        return runnable, missing

    @staticmethod
    def _observe(results: List[dict]) -> List[dict]:
        issues = []
        for r in results:
            if r.get("error"):
                issues.append({"type": "tool_error", "tool": r["tool"],
                               "detail": r.get("note", "")})
            elif r.get("insufficient"):
                issues.append({"type": "insufficient", "tool": r["tool"],
                               "detail": r.get("note", "")})
        return issues

    @staticmethod
    def _relax(scope: dict):
        """放宽切片：从最窄的条件开始去掉（年份 → 套餐 → 店铺 → 类型）。"""
        for k in ("year", "deal", "shop", "deal_type", "keep", "kind"):
            if k in scope:
                new = dict(scope)
                new.pop(k)
                return new, k
        return scope, None

    def _clarify_missing(self, missing: List[dict]) -> str:
        params = sorted({p for m in missing for p in m["missing"]})
        tools = ", ".join(sorted({m["tool"] for m in missing}))
        options = ""
        try:
            options = self.domain.objects_hint(self.df)
        except Exception:
            pass
        return (f"这个问题缺一个必要信息：**{', '.join(params)}** 没有指定"
                f"（工具 `{tools}` 需要它）。\n\n"
                f"可选：{options}\n\n"
                f"请补充后我再查，例如：「把额度使用率分箱看一下」。")

    def _clarify_intent(self) -> str:
        hint = getattr(self.domain, "clarify_hint", "") or ""
        return ("这个问题我没把握理解对，不想直接猜一个答案给你。\n\n"
                + (f"我目前能做的是：\n{hint}\n\n" if hint else "")
                + "你可以换个说法，或者从上面挑一个。")

    # ---------------------------------------------------------------- 合成
    def _compose_offline(self, results: List[dict]) -> str:
        parts = [f"### 分析范围\n{self._scope_desc()}", "\n### 事实"]
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
        user = (f"分析范围：{self._scope_desc()}\n\n"
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
        calls = list(plan["plan"])
        scope = {k: plan["entities"][k] for k in self.domain.plan_spec.scope_keys
                 if k in plan["entities"]}
        self.scope = scope

        # 意图完全没命中：不要用默认序列硬答，先反问。
        # 但如果已经抽到了实体（店铺/年份/变量），说明问题是有指向的，交给默认序列去答。
        if (plan.get("intent") == "default" and len(question.strip()) > 8
                and not plan.get("entities")):
            return self._finish(
                question, plan, planner_used, scope, [], "clarification",
                self._clarify_intent(), [{"turn": 0, "action": "clarify", "reason": "意图未命中"}],
                use_llm=use_llm)

        results, trajectory, relaxations = [], [], []
        status, clarification = "ok", None
        for turn in range(1, self.max_turns + 1):
            runnable, missing = self._check_required(calls)
            if missing:
                status, clarification = "clarification", self._clarify_missing(missing)
                trajectory.append({"turn": turn, "action": "clarify",
                                   "reason": "缺失必填参数", "detail": missing})
                break
            if not runnable:
                status, clarification = "clarification", self._clarify_intent()
                trajectory.append({"turn": turn, "action": "clarify", "reason": "无可执行工具"})
                break

            results = self._execute(runnable)
            issues = self._observe(results)
            trajectory.append({"turn": turn,
                               "calls": [c["tool"] for c in runnable],
                               "issues": [f"{i['type']}:{i['tool']}" for i in issues]})
            if not issues:
                break

            issue = issues[0]
            if issue["type"] == "insufficient":
                new_scope, dropped = self._relax(scope)
                if dropped is None:
                    status = "degraded"
                    trajectory[-1]["action"] = "已放宽到全量仍不足，降级说明"
                    break
                scope = new_scope
                self.scope = scope
                calls = [{**c, "params": {k: v for k, v in c["params"].items()
                                          if k != dropped}} for c in runnable]
                relaxations.append(dropped)
                trajectory[-1]["action"] = f"放宽切片：去掉 {dropped} 后重算"
                continue

            if issue["type"] == "tool_error":
                calls = [c for c in runnable if c["tool"] != issue["tool"]]
                if not calls:
                    status = "degraded"
                    trajectory[-1]["action"] = "工具全部失败，降级说明"
                    break
                trajectory[-1]["action"] = f"剔除报错工具 {issue['tool']} 后重跑"
                continue
            break

        return self._finish(question, plan, planner_used, scope, results, status,
                            clarification, trajectory, use_llm,
                            relaxations=relaxations)

    def _finish(self, question, plan, planner_used, scope, results, status,
                clarification, trajectory, use_llm=None, relaxations=None) -> dict:
        composer, answer, llm_error = "template", None, None

        if status == "clarification":
            answer = clarification
        else:
            if use_llm:
                try:
                    answer = self._compose_llm(question, results)
                    composer = "llm"
                except llm_mod.LLMError as e:
                    llm_error = str(e)
            if answer is None:
                answer = self._compose_offline(results)
            if relaxations:
                # 放宽口径必须说出来：否则用户看到的是"全站排名"却以为还是那家店
                answer = (f"⚠️ 原切片样本不足，已去掉 {', '.join(relaxations)} 重新计算；"
                          f"下面的结论适用于调整后的范围。\n\n" + answer)
            if status == "degraded":
                answer = ("⚠️ 已把筛选条件放宽到全量仍然无法支撑这个问题，"
                          "下面是放宽后的结果，请谨慎使用。\n\n" + answer)

        # 澄清轮不参与数字回查（没有工具结果，也没有结论）
        if status == "clarification":
            gres = {"checked": 0, "supported": 0, "unsupported": [], "skipped_structural": 0,
                    "rate": 1.0, "passed": True, "citations": [], "cite_ok": False,
                    "pool_size": 0}
        else:
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

        gate_ctx = {"answer": answer, "results": results, "df": self.df,
                    "scope": scope, "domain": self.domain.name, "status": status}
        gate_res = gates_mod.run_all(self.gates, gate_ctx)

        out = {
            "domain": self.domain.name,
            "question": question,
            "status": status,
            "clarification": clarification,
            "plan": plan["plan"],
            "planner": planner_used,
            "reason": plan.get("reason", ""),
            "scope": scope,
            "results": results,
            "answer": answer,
            "composer": composer,
            "grounding": gres,
            "gates": gate_res,
            "trajectory": trajectory,
            "turns": len([t for t in trajectory if "calls" in t]) or 1,
            "llm_error": llm_error,
        }
        self.history.append(out)
        return out

    # ---------------------------------------------------------------- 渲染
    @staticmethod
    def render(res: dict) -> str:
        lines = [f"## Q: {res['question']}", ""]
        chain = " → ".join(c["tool"] for c in res["plan"]) or "（无）"
        lines.append(f"> 领域：{res['domain']}｜规划器：{res['planner']}"
                     f"｜工具链：{chain}｜状态：**{res['status']}**")
        if res.get("trajectory") and len(res["trajectory"]) > 1:
            lines.append("> 修复轨迹：" + "；".join(
                f"第{t['turn']}轮 {t.get('action', '执行')}" for t in res["trajectory"]))
        lines.append("")
        lines.append(res["answer"])
        if res["status"] == "clarification":
            lines.append("\n---\n⏸ 需要你补充信息后才能继续（本轮未产出结论，因此不做质检）。")
            return "\n".join(lines)
        lines.append("")
        lines.append("---")
        lines.append(grounding.format_report(res["grounding"]))
        lines.append(gates_mod.format_report(res["gates"]))
        if res.get("llm_error"):
            lines.append(f"\n> LLM 调用失败，已回退模板：{res['llm_error']}")
        return "\n".join(lines)
