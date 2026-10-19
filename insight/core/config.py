# -*- coding: utf-8 -*-
"""框架级配置（LLM 通道）。领域自己的口径配置放在各 domain 的 config.py。"""
import os

# 后端：auto（有 key 就用，没有就离线）/ offline / openai
LLM_PROVIDER = os.environ.get("RI_LLM_PROVIDER", "auto")
LLM_MODEL = os.environ.get("RI_LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.environ.get("RI_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_KEY_ENV = "DEEPSEEK_API_KEY"
FALLBACK_ENV_FILE = os.path.expanduser("~/.agentdoc.env")   # 私有，不进仓库
LLM_TIMEOUT = 60
