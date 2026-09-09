"""适配器注册表：导入所有适配器以触发 @register，并暴露查询接口。"""
from __future__ import annotations

from .base import get_adapter, list_adapters, register

# 导入即注册
from . import principlismqa  # noqa: F401
from . import medethiceval  # noqa: F401
from . import vital  # noqa: F401
from . import medethicsqa  # noqa: F401
from . import llmevalmed  # noqa: F401

__all__ = ["get_adapter", "list_adapters", "register"]
