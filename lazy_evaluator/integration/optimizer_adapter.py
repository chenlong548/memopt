"""
Memory Optimizer适配器模块

提供与mem_optimizer模块的集成适配器。
"""

from typing import Any, Callable, Optional, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ..core.lazy import Lazy
from ..thunk.thunk_pool import ThunkPool


class OptimizerAdapter:
    """
    Memory Optimizer适配器

    该类提供了lazy_evaluator与mem_optimizer模块的集成适配器，
    允许在惰性计算中使用内存优化功能。

    Example:
        >>> adapter = OptimizerAdapter()
        >>> optimized_thunk = adapter.create_optimized_thunk(lambda: large_data)
        >>> result = optimized_thunk.get()
    """

    def __init__(self):
        """初始化OptimizerAdapter"""
        self._optimizer = None
        self._thunk_pool = ThunkPool(max_size=50)
        try:
            # 尝试导入mem_optimizer模块
            from mem_optimizer import MemoryOptimizer
            self._optimizer = MemoryOptimizer
        except ImportError:
            pass

    def is_available(self) -> bool:
        """
        检查mem_optimizer模块是否可用

        Returns:
            bool: 如果可用返回True，否则返回False
        """
        return self._optimizer is not None

    def create_optimized_thunk(self, computation: Callable, pool_size: int = 50) -> Any:
        """
        创建优化的thunk

        Args:
            computation: 计算函数
            pool_size: 对象池大小

        Returns:
            Memothunk: 带记忆化的thunk
        """
        # 使用对象池获取thunk
        thunk = self._thunk_pool.acquire(computation)
        return thunk

    def release_thunk(self, thunk: Any) -> None:
        """
        释放thunk到对象池

        Args:
            thunk: Memothunk实例
        """
        self._thunk_pool.release(thunk)

    def lazy_allocate(self, size: int, initializer: Optional[Callable] = None) -> Lazy:
        """
        惰性内存分配

        Args:
            size: 分配大小
            initializer: 初始化函数

        Returns:
            Lazy: 惰性值
        """
        def allocate():
            if not self.is_available():
                # 如果optimizer不可用，返回None
                return None

            optimizer = self._optimizer()
            if hasattr(optimizer, 'allocate'):
                memory = optimizer.allocate(size)
                if initializer and memory is not None:
                    initializer(memory)
                return memory
            else:
                return None

        return Lazy(allocate)

    def create_memory_pool(self, pool_size: int = 10, chunk_size: int = 1024) -> Dict:
        """
        创建内存池

        Args:
            pool_size: 池大小
            chunk_size: 块大小

        Returns:
            Dict: 内存池配置
        """
        from ..thunk.thunk_pool import ThunkPool

        class MemoryPool:
            def __init__(self, size, chunk_sz, adapter):
                self._size = size
                self._chunk_size = chunk_sz
                self._adapter = adapter
                self._pool = ThunkPool(max_size=size)

            def acquire(self) -> Any:
                """获取内存块"""
                return self._pool.acquire()

            def release(self, chunk: Any) -> None:
                """释放内存块"""
                self._pool.release(chunk)

            def clear(self) -> None:
                """清空池"""
                self._pool.clear()

        return MemoryPool(pool_size, chunk_size, self)

    def optimize_computation(self, computation: Callable, memory_limit: Optional[int] = None) -> Callable:
        """
        优化计算函数

        Args:
            computation: 计算函数
            memory_limit: 内存限制

        Returns:
            Callable: 优化后的函数
        """
        def optimized_computation(*args, **kwargs):
            # 使用对象池
            thunk = self._thunk_pool.acquire(lambda: computation(*args, **kwargs))
            try:
                result = thunk.get()
                return result
            finally:
                self._thunk_pool.release(thunk)

        return optimized_computation

    def get_memory_stats(self) -> Dict:
        """
        获取内存统计信息

        Returns:
            Dict: 统计信息
        """
        stats = {
            'thunk_pool_size': self._thunk_pool.size(),
            'thunk_pool_max_size': self._thunk_pool.max_size(),
            'thunk_pool_created': self._thunk_pool.created_count(),
            'optimizer_available': self.is_available(),
        }

        if self.is_available():
            optimizer = self._optimizer()
            if hasattr(optimizer, 'get_stats'):
                stats['optimizer_stats'] = optimizer.get_stats()

        return stats

    def clear_pools(self) -> None:
        """清空所有对象池"""
        self._thunk_pool.clear()

    def __repr__(self) -> str:
        """字符串表示"""
        available = "available" if self.is_available() else "not available"
        return f"OptimizerAdapter(status={available}, pool_size={self._thunk_pool.size()})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
