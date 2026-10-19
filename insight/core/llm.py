# -*- coding: utf-8 -*-
"""reviewinsight.llm —— LLM 后端

两个后端，接口一致（.chat(messages) -> str）：
- OfflineStub：不发请求，抛 LLMError，让 agent 回退到模板合成器（CI 全程离线）
- OpenAICompat：纯标准库 urllib 实现，通吃 DeepSeek / Qwen / Moonshot / Ollama / vLLM

Key 读取顺序：环境变量 → 私有 env 文件（默认 ~/.agentdoc.env，不进仓库）
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

from . import config


class LLMError(RuntimeError):
    pass


def load_key(env_name: Optional[str] = None) -> Optional[str]:
    env_name = env_name or config.LLM_KEY_ENV
    key = os.environ.get(env_name)
    if key:
        return key.strip()
    path = config.FALLBACK_ENV_FILE
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip().lstrip("export ").strip() == env_name:
                        return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return None


class OfflineStub:
    name = "offline"

    def chat(self, messages: List[dict], **kw) -> str:
        raise LLMError("offline 模式不调用 LLM（未配置 key 或显式指定 offline）")


class OpenAICompat:
    name = "openai-compat"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or load_key()
        if not self.api_key:
            raise LLMError(f"未找到 API Key（环境变量 {config.LLM_KEY_ENV}）")

    def chat(self, messages: List[dict], temperature: float = 0.2,
             max_tokens: int = 900) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise LLMError(f"HTTP {e.code}: {detail}")
        except Exception as e:  # 网络类
            raise LLMError(f"调用失败：{type(e).__name__}: {e}")
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise LLMError(f"返回格式异常：{str(data)[:200]}")


def get_backend(provider: Optional[str] = None):
    """auto：有 key 就用真模型，没有就静默降级到离线模板（保证 CI 与离线环境可跑）。"""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider in ("offline", "none", "stub"):
        return OfflineStub()
    try:
        return OpenAICompat()
    except LLMError:
        return OfflineStub()


JSON_BLOCK_RE = re.compile(r"\[.*\]", re.S)


def parse_json_array(text: str) -> list:
    """从 LLM 输出里抠出 JSON 数组。模型爱加 ```json 围栏和解释，所以只取第一个 [...]。"""
    m = JSON_BLOCK_RE.search(text)
    if not m:
        raise LLMError("模型没有返回 JSON 数组")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 解析失败：{e}")
    if not isinstance(data, list):
        raise LLMError("返回值不是数组")
    return data
