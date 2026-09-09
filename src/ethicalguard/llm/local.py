"""本地模型后端（vLLM / Ollama 等 OpenAI 兼容服务）。

用法：
- 起一个 OpenAI 兼容服务，例如 vLLM：
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
- 然后在配置里指定（configs/default.json 或环境变量）：
    ETHICALGUARD_LOCAL_BASE_URL=http://localhost:8000/v1
    ETHICALGUARD_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct
- 运行：--mode local

本后端与 APIBackend 走同一 /chat/completions 协议（本地服务无需鉴权）。
"""
from __future__ import annotations

from .api import APIBackend
from .base import LLMBackend


class LocalBackend(LLMBackend):
    """本地模型模式：复用 OpenAI 兼容协议（vLLM 默认暴露 /v1/chat/completions）。"""

    mode = "local"

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", base_url: str = "http://localhost:8000/v1",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # 本地服务无需鉴权：不传 api_key；Qwen3 系默认开 thinking，思考 token 会吃满
        # max_tokens(1024) 导致回答截断，这里在请求体里显式关闭
        self._api = APIBackend(base_url=self.base_url, api_key=None, model=self.model,
                               temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                               chat_template_kwargs={"enable_thinking": False})

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._api.complete(system_prompt, user_prompt)


class RuleBackend(LLMBackend):
    """规则引擎模式（确定性）。规则 Agent 不经过 complete()，此后端仅作占位/兜底。"""

    mode = "rule"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "{}"
