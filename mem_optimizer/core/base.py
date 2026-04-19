"""
mem_optimizer 核心基础模块

定义内存分配优化器的基础类和接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import time


class AllocatorType(Enum):
    """分配器类型"""
    BUDDY = "buddy"
    SLAB = "slab"
    TLSF = "tlsf"
    AUTO = "auto"


class AllocationStrategy(Enum):
    """分配策略"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    NEXT_FIT = "next_fit"


class MemoryRegionState(Enum):
    """内存区域状态"""
    FREE = "free"
    ALLOCATED = "allocated"
    FRAGMENTED = "fragmented"
    RESERVED = "reserved"


@dataclass
class MemoryBlock:
    """
    内存块

    表示一个内存分配单元。
    """

    address: int
    size: int
    state: MemoryRegionState = MemoryRegionState.FREE
    allocator_type: AllocatorType = AllocatorType.AUTO
    numa_node: int = -1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_free(self) -> bool:
        """检查是否空闲"""
        return self.state == MemoryRegionState.FREE

    def is_allocated(self) -> bool:
        """检查是否已分配"""
        return self.state == MemoryRegionState.ALLOCATED

    def contains(self, addr: int) -> bool:
        """检查地址是否在块内"""
        return self.address <= addr < self.address + self.size

    def overlaps(self, other: 'MemoryBlock') -> bool:
        """检查是否与另一块重叠"""
        return (self.address < other.address + other.size and
                self.address + self.size > other.address)

    def can_merge(self, other: 'MemoryBlock') -> bool:
        """检查是否可以合并"""
        if not self.is_free() or not other.is_free():
            return False
        if self.numa_node != other.numa_node:
            return False
        return (self.address + self.size == other.address or
                other.address + other.size == self.address)


@dataclass
class AllocationRequest:
    """
    分配请求

    表示一个内存分配请求。
    """

    size: int
    alignment: int = 8
    numa_node: int = -1
    flags: int = 0
    hint_address: Optional[int] = None
    deadline: Optional[float] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """
    分配结果

    表示内存分配的结果。
    """

    success: bool
    address: int = 0
    size: int = 0
    actual_size: int = 0
    allocator_type: AllocatorType = AllocatorType.AUTO
    numa_node: int = -1
    fragmentation: float = 0.0
    allocation_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class MemoryStatistics:
    """
    内存统计信息

    记录内存池的性能指标。
    """

    total_size: int = 0
    used_size: int = 0
    free_size: int = 0
    fragmentation_ratio: float = 0.0
    allocation_count: int = 0
    deallocation_count: int = 0
    peak_usage: int = 0
    allocation_failures: int = 0
    defragmentation_count: int = 0
    average_allocation_time: float = 0.0
    average_fragmentation: float = 0.0

    def update_usage(self, used: int, total: int):
        """更新使用率"""
        self.used_size = used
        self.total_size = total
        self.free_size = total - used
        if used > self.peak_usage:
            self.peak_usage = used

    def calculate_fragmentation(self, free_blocks: List[MemoryBlock]):
        """计算碎片率"""
        if not free_blocks:
            self.fragmentation_ratio = 0.0
            return

        total_free = sum(b.size for b in free_blocks)
        if total_free == 0:
            self.fragmentation_ratio = 0.0
            return

        max_free = max(b.size for b in free_blocks)
        self.fragmentation_ratio = 1.0 - (max_free / total_free)

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total_size_mb': self.total_size / (1024 * 1024),
            'used_size_mb': self.used_size / (1024 * 1024),
            'free_size_mb': self.free_size / (1024 * 1024),
            'usage_percent': (self.used_size / self.total_size * 100) if self.total_size > 0 else 0,
            'fragmentation_ratio': f"{self.fragmentation_ratio:.4f}",
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'peak_usage_mb': self.peak_usage / (1024 * 1024),
            'allocation_failures': self.allocation_failures,
            'defragmentation_count': self.defragmentation_count,
            'avg_allocation_time_us': self.average_allocation_time * 1000000
        }


class AllocatorBase(ABC):
    """
    分配器基类

    定义内存分配器的标准接口。
    """

    def __init__(self, total_size: int, base_address: int = 0):
        """
        初始化分配器

        Args:
            total_size: 总内存大小
            base_address: 基地址
        """
        self.total_size = total_size
        self.base_address = base_address
        self.stats = MemoryStatistics(total_size=total_size)

    @abstractmethod
    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """
        分配内存

        Args:
            request: 分配请求

        Returns:
            AllocationResult: 分配结果
        """
        pass

    @abstractmethod
    def deallocate(self, address: int) -> bool:
        """
        释放内存

        Args:
            address: 内存地址

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def reallocate(self, address: int, new_size: int) -> AllocationResult:
        """
        重新分配内存

        Args:
            address: 原内存地址
            new_size: 新大小

        Returns:
            AllocationResult: 分配结果
        """
        pass

    @abstractmethod
    def get_free_blocks(self) -> List[MemoryBlock]:
        """
        获取空闲块列表

        Returns:
            List[MemoryBlock]: 空闲块列表
        """
        pass

    @abstractmethod
    def get_allocated_blocks(self) -> List[MemoryBlock]:
        """
        获取已分配块列表

        Returns:
            List[MemoryBlock]: 已分配块列表
        """
        pass

    @abstractmethod
    def get_fragmentation_score(self) -> float:
        """
        获取碎片评分

        Returns:
            float: 碎片评分 (0-1)
        """
        pass

    @abstractmethod
    def defragment(self) -> int:
        """
        执行碎片整理

        Returns:
            int: 整理的块数量
        """
        pass

    def get_stats(self) -> MemoryStatistics:
        """
        获取统计信息

        Returns:
            MemoryStatistics: 统计信息
        """
        return self.stats

    def get_utilization(self) -> float:
        """
        获取利用率

        Returns:
            float: 利用率 (0-1)
        """
        if self.total_size == 0:
            return 0.0
        return self.stats.used_size / self.total_size


class StrategySelectorBase(ABC):
    """
    策略选择器基类

    定义分配策略选择的标准接口。
    """

    @abstractmethod
    def select_allocator(self,
                        request: AllocationRequest,
                        context: Dict[str, Any]) -> AllocatorType:
        """
        选择最优分配器

        Args:
            request: 分配请求
            context: 上下文信息

        Returns:
            AllocatorType: 选择的分配器类型
        """
        pass

    @abstractmethod
    def update_performance(self,
                          allocator: AllocatorType,
                          performance: Dict[str, Any]):
        """
        更新性能数据

        Args:
            allocator: 分配器类型
            performance: 性能数据
        """
        pass

    @abstractmethod
    def get_recommendations(self) -> Dict[AllocatorType, float]:
        """
        获取推荐权重

        Returns:
            Dict: 分配器到权重的映射
        """
        pass


class DefragmenterBase(ABC):
    """
    碎片整理器基类

    定义碎片整理的标准接口。
    """

    @abstractmethod
    def analyze(self, blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """
        分析碎片情况

        Args:
            blocks: 内存块列表

        Returns:
            Dict: 分析结果
        """
        pass

    @abstractmethod
    def plan(self, blocks: List[MemoryBlock]) -> List[Tuple[int, int, int]]:
        """
        制定整理计划

        Args:
            blocks: 内存块列表

        Returns:
            List[Tuple]: 移动计划 (源地址, 目标地址, 大小)
        """
        pass

    @abstractmethod
    def execute(self, plan: List[Tuple[int, int, int]]) -> int:
        """
        执行整理计划

        Args:
            plan: 整理计划

        Returns:
            int: 成功移动的块数量
        """
        pass


class NUMACoordinatorBase(ABC):
    """
    NUMA协调器基类

    定义NUMA感知内存分配的标准接口。
    """

    @abstractmethod
    def get_numa_nodes(self) -> List[int]:
        """
        获取NUMA节点列表

        Returns:
            List[int]: NUMA节点ID列表
        """
        pass

    @abstractmethod
    def get_node_memory_info(self, node: int) -> Dict[str, Any]:
        """
        获取节点内存信息

        Args:
            node: NUMA节点ID

        Returns:
            Dict: 内存信息
        """
        pass

    @abstractmethod
    def select_node(self, request: AllocationRequest) -> int:
        """
        选择最优NUMA节点

        Args:
            request: 分配请求

        Returns:
            int: 选择的节点ID
        """
        pass

    @abstractmethod
    def migrate(self, address: int, size: int, target_node: int) -> bool:
        """
        迁移内存到目标节点

        Args:
            address: 内存地址
            size: 内存大小
            target_node: 目标节点

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def get_interleave_policy(self) -> Dict[str, Any]:
        """
        获取交错策略

        Returns:
            Dict: 交错策略配置
        """
        pass
