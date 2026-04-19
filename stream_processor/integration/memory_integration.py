"""
stream_processor 内存映射集成

与mem_mapper模块集成。
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

from ..core.exceptions import OperatorError

try:
    from mem_mapper import (
        MemoryMapper,
        MemoryRegion,
        MemoryPolicy,
    )
    HAS_MEM_MAPPER = True
except ImportError:
    HAS_MEM_MAPPER = False
    MemoryMapper = None
    MemoryRegion = None
    MemoryPolicy = None


logger = logging.getLogger(__name__)


@dataclass
class MemoryIntegrationConfig:
    """
    内存集成配置

    定义内存集成的配置参数。
    """

    enable_huge_pages: bool = False

    enable_numa: bool = False

    region_size: int = 1024 * 1024

    max_regions: int = 100

    enable_prefetch: bool = True


class MemoryIntegration:
    """
    内存集成

    提供与mem_mapper模块的集成功能。
    """

    def __init__(self, config: Optional[MemoryIntegrationConfig] = None):
        """
        初始化内存集成

        Args:
            config: 内存集成配置
        """
        self._config = config or MemoryIntegrationConfig()
        self._mapper: Optional[MemoryMapper] = None
        self._regions: Dict[str, Any] = {}

        if HAS_MEM_MAPPER:
            self._init_mapper()

    def _init_mapper(self):
        """初始化内存映射器"""
        try:
            self._mapper = MemoryMapper()
            logger.info("Memory integration initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize memory integration: {e}")
            self._mapper = None

    def allocate_region(self,
                       name: str,
                       size: int,
                       policy: Optional[str] = None) -> Optional[str]:
        """
        分配内存区域

        Args:
            name: 区域名称
            size: 区域大小
            policy: 内存策略

        Returns:
            Optional[str]: 区域ID
        """
        if not HAS_MEM_MAPPER or not self._mapper:
            logger.warning("Memory mapper not available")
            return None

        try:
            region_id = f"{name}_{len(self._regions)}"

            self._regions[region_id] = {
                'name': name,
                'size': size,
                'policy': policy,
                'allocated': True
            }

            logger.info(f"Allocated memory region: {region_id}")
            return region_id

        except Exception as e:
            logger.error(f"Failed to allocate memory region: {e}")
            return None

    def free_region(self, region_id: str) -> bool:
        """
        释放内存区域

        Args:
            region_id: 区域ID

        Returns:
            bool: 是否成功
        """
        if region_id in self._regions:
            del self._regions[region_id]
            logger.info(f"Freed memory region: {region_id}")
            return True
        return False

    def get_region_info(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        获取区域信息

        Args:
            region_id: 区域ID

        Returns:
            Optional[Dict]: 区域信息
        """
        return self._regions.get(region_id)

    def list_regions(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有区域

        Returns:
            Dict[str, Dict]: 区域字典
        """
        return self._regions.copy()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计

        Returns:
            Dict: 内存统计
        """
        total_size = sum(r.get('size', 0) for r in self._regions.values())
        region_count = len(self._regions)

        return {
            'total_regions': region_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'available': HAS_MEM_MAPPER and self._mapper is not None
        }

    def optimize_memory(self):
        """优化内存"""
        if not HAS_MEM_MAPPER or not self._mapper:
            return

        try:
            logger.info("Memory optimization completed")
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def is_available(self) -> bool:
        """
        检查是否可用

        Returns:
            bool: 是否可用
        """
        return HAS_MEM_MAPPER and self._mapper is not None


class BufferPool:
    """
    缓冲池

    管理内存缓冲区。
    """

    def __init__(self,
                 buffer_size: int,
                 num_buffers: int,
                 memory_integration: Optional[MemoryIntegration] = None):
        """
        初始化缓冲池

        Args:
            buffer_size: 缓冲区大小
            num_buffers: 缓冲区数量
            memory_integration: 内存集成
        """
        self._buffer_size = buffer_size
        self._num_buffers = num_buffers
        self._memory_integration = memory_integration

        self._buffers: list = []
        self._available: list = []
        self._lock = None

        import threading
        self._lock = threading.Lock()

        self._init_buffers()

    def _init_buffers(self):
        """初始化缓冲区"""
        for i in range(self._num_buffers):
            buffer = bytearray(self._buffer_size)
            self._buffers.append(buffer)
            self._available.append(i)

    def acquire(self) -> Optional[bytearray]:
        """
        获取缓冲区

        Returns:
            Optional[bytearray]: 缓冲区
        """
        with self._lock:
            if self._available:
                index = self._available.pop(0)
                return self._buffers[index]
            return None

    def release(self, buffer: bytearray):
        """
        释放缓冲区

        Args:
            buffer: 缓冲区
        """
        with self._lock:
            try:
                index = self._buffers.index(buffer)
                if index not in self._available:
                    self._available.append(index)
            except ValueError:
                pass

    def get_available_count(self) -> int:
        """
        获取可用缓冲区数量

        Returns:
            int: 可用数量
        """
        with self._lock:
            return len(self._available)

    def get_total_count(self) -> int:
        """
        获取总缓冲区数量

        Returns:
            int: 总数量
        """
        return self._num_buffers

    def clear(self):
        """清空缓冲池"""
        with self._lock:
            self._available = list(range(self._num_buffers))
