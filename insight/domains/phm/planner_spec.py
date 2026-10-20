# -*- coding: utf-8 -*-
"""PHM 意图、显式实体与多轮参数继承。

只抽取用户实际给出的假设；默认成本、提前期及循环/天留给工具处理。
机器编号不在这里裁剪，以便工具把越界编号作为 insufficient 交回闭环。
"""
import re

from ...core.planner import PlanSpec


INTENTS = {
    # 平分时优先排期，避免“哪几台两周内必须修”被“哪几台”抢成机队概览。
    "plan": ["排期", "工单", "检修", "维修", "备件", "提前期", "必须修", "要修", "该修"],
    "uncertain": ["区间", "可信", "敢不敢", "置信", "覆盖率", "停机"],
    "why": ["为什么", "原因", "哪个传感器", "传感器", "归因"],
    "threshold": ["阈值", "告警", "漏报", "误报", "代价", "成本", "风险窗口", "风险范围"],
    "model": ["哪个模型", "模型", "GBDT", "gbdt", "对比", "NASA", "nasa"],
    "unit": ["这台", "单机", "几号", "还能飞", "健康", "剩余寿命", "RUL", "rul"],
    "fleet": ["机队", "总览", "多少台", "哪些机器", "哪几台", "整体"],
}

SEQUENCE = {
    "fleet": ["fleet_overview", "uncertainty_audit"],
    "unit": ["unit_health", "degradation_rate"],
    "why": ["sensor_attribution", "unit_health"],
    "uncertain": ["uncertainty_audit", "unit_health"],
    "threshold": ["cost_threshold", "uncertainty_audit"],
    "plan": ["maintenance_plan", "fleet_overview"],
    "model": ["model_compare", "fleet_overview"],
}

SCOPE_KEYS = ("unit", "dataset", "c_fn", "c_fp", "lead_time_days", "window_days",
              "cycles_per_day", "risk_horizon_cycles", "chosen")
ANAPHORA = ["这个", "这些", "这台", "这批", "该机", "它", "那", "同样", "继续",
            "改成", "改为", "改到", "换成"]
_NUMBER = r"(?:-?\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
_DURATION = rf"({_NUMBER})\s*(天|日|周|星期)"


def _number(text: str):
    """口语中的“两周”与数字写法共用同一换算，保留小数假设。"""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        value = float(text)
        return int(value) if value.is_integer() else value
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total, section, value = 0, 0, 0
    for char in text:
        if char in digits:
            value = digits[char]
        elif units[char] == 10000:
            total += (section + value) * units[char]
            section, value = 0, 0
        else:
            section += (value or 1) * units[char]
            value = 0
    return total + section + value


def _duration(match):
    return _number(match.group(1)) * (7 if match.group(2) in ("周", "星期") else 1)


def extract_entities(q: str, df) -> dict:
    ent = {}
    # 数字必须有机器/发动机/unit/号语境；14 天、10 倍和 FD001 都不是 unit。
    unit_patterns = (
        r"(?:机器|发动机|引擎|(?<![A-Za-z])unit)\s*(?:编号|号|id|第)?\s*(?:为|是)?\s*[:：=#]?\s*"
        r"(-?\d+)(?![\d.]|\s*(?:天|日|周|星期|倍|个循环|循环))",
        r"(?:第\s*)?(-?\d+)\s*号(?:\s*(?:机器|发动机|引擎))?",
        r"^\s*(-?\d+)\s*$",
    )
    for pattern in unit_patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            ent["unit"] = int(match.group(1))
            break
    match = re.search(r"(?<![A-Za-z0-9])fd\s*(\d{3})(?![A-Za-z0-9])", q, re.IGNORECASE)
    if match:
        # 不支持的显式编号也交给 resolver 报错，不能悄悄查默认 FD001。
        ent["dataset"] = "FD" + match.group(1)

    ratio = re.search(rf"漏报(?:的)?(?:代价|成本)?\s*(?:是|为|=|：)?\s*"
                      rf"误报(?:的)?(?:代价|成本)?\s*的?\s*({_NUMBER})\s*倍", q)
    for label, key in (("漏报", "c_fn"), ("误报", "c_fp")):
        match = re.search(rf"{label}\s*(?:的)?\s*(?:代价|成本)?\s*"
                          rf"(?:是|为|设为|改成|改为|=|：|:)?\s*({_NUMBER})"
                          rf"(?![\d.零〇一二两三四五六七八九十百千万]|\s*倍)", q)
        if match:
            ent[key] = _number(match.group(1))

    lead = re.search(rf"提前期\s*(?:是|为|改成|改为|设为|=|：|:)?\s*{_DURATION}", q)
    if lead is None:
        lead = re.search(rf"备件\s*(?:需要|需|要|在|约)?\s*{_DURATION}\s*(?:后|内)?\s*(?:到|交付)", q)
    if lead:
        ent["lead_time_days"] = _duration(lead)

    window = re.search(rf"(?:窗口|未来|接下来|今后)\s*(?:是|为|改成|改为|设为|=|：|:)?\s*{_DURATION}", q)
    if window is None:
        window = next((match for match in re.finditer(rf"{_DURATION}\s*(?:之)?内", q)
                       if lead is None or not (lead.start() <= match.start() < lead.end())), None)
    if window:
        ent["window_days"] = _duration(window)

    rate = re.search(rf"(?:每天|每日|一天)\s*(?:运行|飞行|飞|按|是|为|改成|改为|设为)?\s*({_NUMBER})\s*(?:个)?\s*循环", q)
    if rate is None:
        rate = re.search(rf"({_NUMBER})\s*(?:个)?\s*循环\s*[/／]\s*(?:天|日)", q)
    if rate:
        ent["cycles_per_day"] = _number(rate.group(1))
    horizon = re.search(rf"(?:风险窗口|风险范围|风险周期)\s*(?:是|为|改成|改为|设为|=|：|:)?\s*({_NUMBER})\s*(?:个)?\s*循环", q)
    if horizon is None:
        horizon = re.search(rf"({_NUMBER})\s*(?:个)?\s*循环\s*(?:之)?内", q)
    if horizon is None:
        horizon = re.search(rf"(?:未来|接下来|今后)\s*({_NUMBER})\s*(?:个)?\s*循环", q)
    if horizon:
        ent["risk_horizon_cycles"] = _number(horizon.group(1))

    # CLI 风格参数也是显式假设，不需要用户换写法。
    for key in SCOPE_KEYS:
        if key in ("unit", "dataset", "chosen"):
            continue
        match = re.search(rf"(?<![A-Za-z0-9_]){key}\s*[=:：]\s*({_NUMBER})", q, re.IGNORECASE)
        if match:
            ent[key] = _number(match.group(1))

    if ratio:
        multiplier = _number(ratio.group(1))
        if "c_fn" in ent and "c_fp" not in ent and multiplier != 0:
            ent["c_fp"] = ent["c_fn"] / multiplier
        elif "c_fn" not in ent:
            ent["c_fn"] = multiplier * ent.get("c_fp", 1)
            ent.setdefault("c_fp", 1)

    chosen = re.search(r"(?:chosen\s*[=:：]|选择|选用|选|采用|使用|用)\s*"
                       r"([A-Za-z][A-Za-z0-9_-]*)", q, re.IGNORECASE)
    if chosen and not re.fullmatch(r"FD\d{3}", chosen.group(1), re.IGNORECASE):
        ent["chosen"] = chosen.group(1).lower()
    else:
        models = re.findall(r"(?<![A-Za-z0-9_])(?:gbdt|cnn_bilstm_attn|cnn_transformer|"
                            r"patch_transformer|legacy_cnn_bilstm|blend_nn_gbdt|ensemble)"
                            r"(?![A-Za-z0-9_])", q, re.IGNORECASE)
        if len(set(model.lower() for model in models)) == 1:
            ent["chosen"] = models[0].lower()
    return ent


def adjust_seq(q: str, ent: dict, seq: list, intents: list) -> list:
    uncertain = any(word in q for word in INTENTS["uncertain"])
    audit_question = uncertain and any(word in q for word in (
        "可信", "置信", "覆盖率", "敢不敢", "能不能用", "可不可以用", "是否能用", "可以用"))
    if audit_question or ("uncertain" in intents and "plan" not in intents):
        # 全局区间审计无需凭空要求机器编号，保留 Phase 0 验收路径。
        single_machine = "unit" in ent or any(word in q for word in ("这台", "该机", "单机", "这部发动机"))
        seq = ["uncertainty_audit", "unit_health" if single_machine else "fleet_overview"]
    elif "plan" in intents:
        seq = list(SEQUENCE["plan"])
    elif "why" in intents:
        seq = list(SEQUENCE["why"])
    elif not intents and any(word in q for word in (
            "窗口", "每天", "每日", "cycles_per_day", "window_days", "lead_time_days")):
        seq = list(SEQUENCE["plan"])
    elif not intents and any(word in q for word in ("c_fn", "c_fp", "risk_horizon_cycles")):
        seq = list(SEQUENCE["threshold"])
    elif "unit" in ent and not intents:
        # 澄清后只回复“34”或“34 号”仍应查询单机，而非退回机队。
        seq = list(SEQUENCE["unit"])
    elif "chosen" in ent and not intents:
        seq = list(SEQUENCE["model"])

    # 使用区间或停机的排期必须带审计；普通点预测排期不强加区间假设。
    if uncertain and "uncertainty_audit" not in seq:
        seq = ["uncertainty_audit"] + seq
    return seq


PLAN_SPEC = PlanSpec(
    intents=INTENTS,
    sequence=SEQUENCE,
    default_sequence=["fleet_overview", "uncertainty_audit"],
    entity_fn=extract_entities,
    anaphora=ANAPHORA,
    adjust_seq=adjust_seq,
    scope_keys=SCOPE_KEYS,
    llm_hints=(
        "机队→fleet_overview,uncertainty_audit；单机→unit_health,degradation_rate；"
        "原因/传感器→sensor_attribution,unit_health；阈值/代价→cost_threshold,uncertainty_audit；"
        "维修排期→maintenance_plan,fleet_overview；模型比较→model_compare,fleet_overview。"
        "单机/传感器工具必须提供unit，缺失时保留必填工具以便框架澄清。"
        "全局区间可信度→uncertainty_audit,fleet_overview；指定unit或明确这台/单机才使用unit_health，"
        "这台/单机缺编号时必须澄清，不能把它改成全局审计。"
        "涉及停机或使用区间做决策必须uncertainty_audit；点预测排期不代表授权停机。"
        "只传用户显式给出的参数，默认值由工具提供。dataset可传所有工具；"
        "天/周用lead_time_days、window_days，循环/天用cycles_per_day，"
        "风险循环窗口用risk_horizon_cycles；不要把循环当成天。"
        "漏报是误报的10倍对应c_fn=10,c_fp=1；chosen是所选模型名。"
    ),
)
