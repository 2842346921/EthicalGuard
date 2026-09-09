"""OpenAI 兼容 API 后端（GPT-4o/Claude/Gemini/DeepSeek 等均走 /chat/completions）。"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from .base import LLMBackend


class APIBackend(LLMBackend):
    mode = "api"

    def __init__(self, base_url: str, api_key: Optional[str], model: str = "gpt-4o",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: int = 60,
                 chat_template_kwargs: Optional[dict] = None):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.chat_template_kwargs = chat_template_kwargs

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs:  # vLLM 扩展：本地 Qwen3 关闭 thinking，避免截断
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # 本地服务（vLLM/Ollama）无需鉴权，省略 Authorization
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
