# -*- coding: utf-8 -*-
"""insight.core.registry —— 工具注册表（领域无关）

每个领域（评论归因 / 信用风险）各自建一个 Registry，把工具注册进去。
工具函数的契约固定为：

    fn(df, **params) -> {"tool", "params", "table", "facts", "note"}

- `facts`  扁平数字字典，是 groundedness 唯一认可的"事实来源"
- `table`  给人看的表格（也是证据池的一部分）
- `note`   口径说明，会随答案一起展示
"""
from typing import Callable, Dict


class Registry:
    def __init__(self, name: str = ""):
        self.name = name
        self.tools: Dict[str, dict] = {}

    def tool(self, name: str, desc: str, params: Dict[str, str], required=()):
        """required：必填参数。缺了不该让工具崩，而应由编排层去反问用户。"""
        def deco(fn: Callable):
            self.tools[name] = {"name": name, "desc": desc, "params": params,
                                "required": tuple(required), "fn": fn}
            return fn
        return deco

    def required(self, name: str):
        return self.tools[name]["required"] if name in self.tools else ()

    def missing_required(self, name: str, params: dict):
        return [p for p in self.required(name) if p not in (params or {})]

    # ------------------------------------------------------------ 调用
    def call(self, name: str, df, **params) -> dict:
        if name not in self.tools:
            raise KeyError(f"未知工具：{name}")
        allowed = self.tools[name]["params"]
        unknown = [k for k in params if k not in allowed]
        if unknown:
            raise ValueError(f"工具 {name} 不接受的参数：{unknown}")
        clean = {k: v for k, v in params.items() if v is not None}
        return self.tools[name]["fn"](df, **clean)

    # ------------------------------------------------------------ 元信息
    def names(self):
        return list(self.tools)

    def param_names(self, name: str):
        return set(self.tools[name]["params"]) if name in self.tools else set()

    def signatures(self) -> str:
        """给 LLM 看的工具清单（紧凑格式，省 token）。"""
        lines = []
        for t in self.tools.values():
            ps = ", ".join(f"{k}:{v}" for k, v in t["params"].items()) or "无参数"
            lines.append(f"- {t['name']}({ps})：{t['desc']}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.tools)

    def __contains__(self, name):
        return name in self.tools
