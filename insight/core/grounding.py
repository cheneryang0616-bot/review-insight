# -*- coding: utf-8 -*-
"""reviewinsight.grounding —— 答案级 groundedness 质检

核心命题：**数据分析 agent 最危险的错误不是算错，而是把没算过的数字说得很像算过。**

做法（answer-level，不是 prompt 层的"请不要编造"）：
1. 工具返回的 facts / table 是唯一合法证据池
2. 从答案文本里抽出所有数字（含百分比）
3. 每个数字都要能在证据池里找到（1% 相对误差或 0.005 绝对误差内）
4. 找不到 → 该句作废；整体给出 pass / fail 与逐项失败原因

已知局限（写在 README 里，不藏）：
- 个位数整数（<= 12）与年份不参与校验：它们通常是结构性数字
  （"7 个方面""3 条建议""2014-2021"），不是结论数字。这是有意放宽，
  代价是"有 3 家店"这类小数字幻觉抓不到。
"""
import re
from typing import Iterable, List, Optional

# 前面紧跟 ASCII 字母的数字是标签的一部分（P99 / P90 / x2），不是结论数字：
# 不排除的话，"超过 P99 的 5 倍" 里的 99 会被当成无出处的编造数字。
NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z])")
PCT_NEAR = re.compile(r"^\s*%")
CITE_RE = re.compile(r"\[T(\d+)\]")

STRUCTURAL_MAX = 12          # <= 12 的整数视为结构性数字，跳过
YEAR_RANGE = (1990, 2100)
REL_TOL = 0.01
ABS_TOL = 0.005


# 引号里的内容是"引用原文"（证据本身），不参与数字回查。
# 否则评论原话里的"等了 20 分钟"会被判成编造的数字。
QUOTE_RE = re.compile(r"[「“\"][^」”\"]{0,300}[」”\"]")


def mask_quotes(text: str) -> str:
    return QUOTE_RE.sub(lambda m: " " * len(m.group(0)), text)


def _iter_numbers(text: str) -> Iterable[tuple]:
    """返回 (原始串, 数值, 是否百分比)。"""
    text = mask_quotes(text)
    for m in NUM_RE.finditer(text):
        raw = m.group(0)
        is_pct = bool(PCT_NEAR.match(text[m.end():m.end() + 2]))
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        yield raw, val, is_pct


def evidence_pool(results: List[dict], extra_texts=None) -> List[float]:
    """把所有工具返回的 facts 值与 table 里的数字收成一个可比对的证据池。

    extra_texts 用来放"分析范围"这类上下文（如店铺 ID、年份），
    它们不是结论数字，但会出现在答案里，不放进池子会造成误报。
    """
    pool: List[float] = []
    for t in (extra_texts or []):
        for _, x, is_pct in _iter_numbers(str(t)):
            pool.append(x)
            if is_pct:
                pool.append(x / 100.0)
    for r in results:
        for v in (r.get("facts") or {}).values():
            if isinstance(v, bool):
                pool.append(float(v))
            elif isinstance(v, (int, float)):
                pool.append(float(v))
            elif isinstance(v, str):
                for _, x, _ in _iter_numbers(v):
                    pool.append(x)
        for row in (r.get("table") or []):
            for cell in row.values():
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    pool.append(float(cell))
                elif isinstance(cell, str):
                    for _, x, is_pct in _iter_numbers(cell):
                        pool.append(x)
                        if is_pct:      # 表格里的 "6.42%" 要同时以 6.42 和 0.0642 两种刻度入池
                            pool.append(x / 100.0)
    return pool


def _rounding_tol(raw: str) -> float:
    """按答案里写的小数位数放宽容差。

    "2.3" 相对 2.26 只差了 0.04，这是正常的四舍五入，不该判为编造；
    但 "2.5" 相对 2.26 差 0.24，就必须判失败。
    不做这一步，质检会把"取整"和"瞎编"混为一谈——那比不做质检更糟。
    """
    if "." not in raw:
        return 0.0
    return 0.5 * 10 ** (-len(raw.split(".")[1]))


def _match(val: float, is_pct: bool, pool: List[float], raw: str = "") -> bool:
    cands = [val / 100.0] if is_pct else [val, val / 100.0]
    round_tol = _rounding_tol(raw)
    for c in cands:
        for e in pool:
            tol = max(ABS_TOL, REL_TOL * abs(e), round_tol)
            if abs(c - e) <= tol:
                return True
    return False


def _skip(raw: str, val: float, is_pct: bool) -> bool:
    if is_pct:
        return False
    if YEAR_RANGE[0] <= val <= YEAR_RANGE[1] and "." not in raw and len(raw) == 4:
        return True
    if "." not in raw and 0 <= val <= STRUCTURAL_MAX:
        return True
    return False


def check(answer: str, results: List[dict], extra_texts=None) -> dict:
    """对答案做数字回查。返回逐项结果，供 agent 决定重写还是标注。"""
    pool = evidence_pool(results, extra_texts)
    supported, unsupported, skipped = [], [], 0
    for raw, val, is_pct in _iter_numbers(answer):
        if _skip(raw, val, is_pct):
            skipped += 1
            continue
        item = {"raw": raw, "value": val, "percent": is_pct}
        (supported if _match(val, is_pct, pool, raw) else unsupported).append(item)

    # 引用覆盖：有多少工具被引用到
    cited = sorted({int(m) for m in CITE_RE.findall(answer)})
    n_tools = len(results)
    cite_ok = bool(cited) and all(1 <= c <= n_tools for c in cited)

    checked = len(supported) + len(unsupported)
    return {
        "checked": checked,
        "supported": len(supported),
        "unsupported": unsupported,
        "skipped_structural": skipped,
        "rate": round(len(supported) / checked, 4) if checked else 1.0,
        "passed": len(unsupported) == 0,
        "citations": cited,
        "cite_ok": cite_ok,
        "pool_size": len(pool),
    }


def strip_bad_sentences(answer: str, unsupported: List[dict]) -> str:
    """兜底修复：把含未支撑数字的句子整句删掉（比让它留在答案里更安全）。"""
    if not unsupported:
        return answer
    bad_raws = {u["raw"] for u in unsupported}
    keep = []
    for line in answer.split("\n"):
        nums = {raw for raw, _, _ in _iter_numbers(line)}
        if nums & bad_raws:
            continue
        keep.append(line)
    out = "\n".join(keep).strip()
    return out if out else answer  # 全被删掉就保留原文，由上层报警


def format_report(res: dict) -> str:
    """把质检结果格式化成一行结论，放进答案末尾。"""
    if res["passed"]:
        return (f"✅ 数字回查通过：{res['supported']}/{res['checked']} 个数字可追溯到工具输出"
                f"（跳过 {res['skipped_structural']} 个结构性数字）")
    bad = "、".join(u["raw"] for u in res["unsupported"][:5])
    return (f"⚠️ 数字回查失败：{len(res['unsupported'])}/{res['checked']} 个数字无法追溯（{bad}）")
