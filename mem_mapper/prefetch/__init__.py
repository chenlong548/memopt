"""
mem_mapper 预取模块

提供内存预取策略和执行功能。
"""

from .strategy import (
    PrefetchStrategyType,
    PrefetchRange,
    PrefetchStrategy,
    SequentialPrefetchStrategy,
    RandomPrefetchStrategy,
    AdaptivePrefetchStrategy,
    NoPrefetchStrategy,
    create_prefetch_strategy,
)

from .executor import (
    PrefetchStats,
    PrefetchExecutor,
    prefault_region,
)

__all__ = [
    # 策略
    'PrefetchStrategyType',
    'PrefetchRange',
    'PrefetchStrategy',
    'SequentialPrefetchStrategy',
    'RandomPrefetchStrategy',
    'AdaptivePrefetchStrategy',
    'NoPrefetchStrategy',
    'create_prefetch_strategy',
    
    # 执行器
    'PrefetchStats',
    'PrefetchExecutor',
    'prefault_region',
]
