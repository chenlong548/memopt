"""
惰性计算工具模块 (Lazy Evaluator Module)

该模块实现了完整的惰性计算框架，包括：
- Lazy[T] 泛型类：支持Call-by-need语义的惰性值
- Memothunk：带记忆化的thunk实现
- ThunkPool：对象池管理，减少GC压力
- LRUCache：线程安全的LRU缓存
- DependencyGraph：依赖图，支持增量计算
- StreamFusion：流操作融合优化
- LazyPipeline：惰性管道，支持链式操作

学术支撑：
1. Memothunk - 带记忆化的thunk实现，支持Call-by-need语义
2. Thunk Recycling - 对象池复用，减少GC压力
3. Self-adjusting computation - 基于依赖图的增量计算
4. Stream Fusion - 流操作融合优化
"""

from .core.lazy import Lazy, ThunkState
from .core.evaluation import EvaluationContext
from .core.exceptions import (
    LazyEvaluationError,
    CircularDependencyError,
    ThunkEvaluationError,
    CacheError
)
from .memoization import LRUCache, MultiLevelCache, memoize

__version__ = "1.0.0"
__all__ = [
    "Lazy",
    "ThunkState",
    "EvaluationContext",
    "LazyEvaluationError",
    "CircularDependencyError",
    "ThunkEvaluationError",
    "CacheError",
    "LRUCache",
    "MultiLevelCache",
    "memoize",
]
