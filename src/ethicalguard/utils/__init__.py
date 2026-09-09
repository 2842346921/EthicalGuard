"""工具函数：日志、随机种子。"""
from __future__ import annotations

import logging
import random

import numpy as np


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ethicalguard")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
