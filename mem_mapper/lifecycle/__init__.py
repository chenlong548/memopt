"""
mem_mapper 生命周期管理模块

提供内存映射的生命周期管理功能。
"""

from .manager import (
    CleanupTask,
    LifecycleManager,
    RegionGuard,
    create_region_guard,
)

__all__ = [
    'CleanupTask',
    'LifecycleManager',
    'RegionGuard',
    'create_region_guard',
]
