"""
mem_mapper 大页模块

提供大页检测和池管理功能。
"""

from .detector import (
    HugePageInfo,
    HugePageConfig,
    HugePageDetector,
    get_hugepage_detector,
    get_hugepage_config,
    is_hugepage_available,
)

from .pool import (
    HugePageAllocation,
    HugePagePoolStats,
    HugePagePool,
    get_hugepage_pool,
)

__all__ = [
    # 检测
    'HugePageInfo',
    'HugePageConfig',
    'HugePageDetector',
    'get_hugepage_detector',
    'get_hugepage_config',
    'is_hugepage_available',
    
    # 池管理
    'HugePageAllocation',
    'HugePagePoolStats',
    'HugePagePool',
    'get_hugepage_pool',
]
