"""
mem_monitor 集成层

提供与外部模块的集成适配器。
"""

from .psutil_adapter import (
    PsutilAdapter,
    ProcessMemoryInfo,
    SystemMemoryInfo,
)
from .tracemalloc_adapter import (
    TracemallocAdapter,
    AllocationSnapshot,
    AllocationFilter,
)

__all__ = [
    # psutil适配器
    'PsutilAdapter',
    'ProcessMemoryInfo',
    'SystemMemoryInfo',
    # tracemalloc适配器
    'TracemallocAdapter',
    'AllocationSnapshot',
    'AllocationFilter',
]


class MemMapperAdapter:
    """
    mem_mapper集成适配器

    提供与mem_mapper模块的集成。
    """

    def __init__(self):
        self._mapper = None
        self._available = False

        try:
            from mem_mapper import MemoryMapper
            self._mapper = MemoryMapper
            self._available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._available

    def get_mapped_regions(self) -> list:
        """获取映射区域"""
        if not self._available:
            return []
        # 实现细节...
        return []

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'available': self._available,
            'mapped_regions': 0,
        }


class MemOptimizerAdapter:
    """
    mem_optimizer集成适配器

    提供与mem_optimizer模块的集成。
    """

    def __init__(self):
        self._optimizer = None
        self._available = False

        try:
            from mem_optimizer.core.memory_pool import MemoryPool
            self._optimizer = MemoryPool
            self._available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._available

    def get_pool_stats(self) -> dict:
        """获取内存池统计"""
        if not self._available:
            return {}
        return {}

    def get_allocator_stats(self) -> dict:
        """获取分配器统计"""
        if not self._available:
            return {}
        return {}

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'available': self._available,
            'pool_stats': self.get_pool_stats(),
            'allocator_stats': self.get_allocator_stats(),
        }
