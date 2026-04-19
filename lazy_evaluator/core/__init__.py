"""
核心层 (Core Layer)

提供惰性计算的核心抽象和基础组件。
"""

from .lazy import Lazy, ThunkState
from .evaluation import EvaluationContext
from .exceptions import (
    LazyEvaluationError,
    CircularDependencyError,
    ThunkEvaluationError,
    CacheError
)

__all__ = [
    "Lazy",
    "ThunkState",
    "EvaluationContext",
    "LazyEvaluationError",
    "CircularDependencyError",
    "ThunkEvaluationError",
    "CacheError",
]
