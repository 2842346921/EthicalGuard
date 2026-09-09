"""LLM 基座抽象（三模式：api / local / rule）。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """基座模型抽象。协商 Agent 通过它生成提案/理由文本。"""

    mode: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """返回模型的文本输出。"""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LLMBackend {self.mode}>"
