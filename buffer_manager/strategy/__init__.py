"""
Strategy模块 - 策略层

提供替换策略、预取策略和自适应调整。
"""

from .replacement import LRU, ARC
from .prefetch import Prefetcher
from .adaptive import AdaptiveStrategy

__all__ = [
    "LRU",
    "ARC",
    "Prefetcher",
    "AdaptiveStrategy",
]
