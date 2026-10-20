# -*- coding: utf-8 -*-
"""领域注册表。

一个框架三个场景：
- review：美团团购评论归因（文本 + 业务分组）
- credit：信用风险建模（分箱/IV/评分卡/合规）
- phm：设备健康与维护决策（Phase 0：机队总览/区间准入审计）
"""
import importlib

DOMAINS = {
    "review": "insight.domains.review.ReviewDomain",
    "credit": "insight.domains.credit.CreditDomain",
    "phm": "insight.domains.phm.PHMDomain",
}


def get_domain(name: str):
    key = (name or "review").lower()
    if key not in DOMAINS:
        raise KeyError(f"未知领域：{name}（可选：{list(DOMAINS)}）")
    path = DOMAINS[key]
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)()


def available():
    return list(DOMAINS)
