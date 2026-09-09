"""后端工厂。"""
from __future__ import annotations

from ..config import LLMConfig
from .api import APIBackend
from .base import LLMBackend
from .local import LocalBackend, RuleBackend


def make_backend(cfg: LLMConfig) -> LLMBackend:
    if cfg.mode == "api":
        return APIBackend(cfg.api.base_url, cfg.api.api_key, cfg.api.model,
                          cfg.temperature, cfg.max_tokens, cfg.timeout)
    if cfg.mode == "local":
        return LocalBackend(model=cfg.local.model or cfg.api.model,
                            base_url=cfg.local.base_url,
                            temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.timeout)
    return RuleBackend()
