"""基座模型三模式（api / local / rule）。"""
from __future__ import annotations

from .base import LLMBackend
from .api import APIBackend
from .local import LocalBackend, RuleBackend
from .registry import make_backend

__all__ = ["LLMBackend", "APIBackend", "LocalBackend", "RuleBackend", "make_backend"]
