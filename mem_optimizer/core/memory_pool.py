"""
mem_optimizer 内存池主类模块

提供MemoryPool主类，整合所有功能模块。
"""

import logging
import threading
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from .base import (
    AllocatorBase, AllocatorType, AllocationRequest, AllocationResult,
    MemoryBlock, MemoryStatistics, StrategySelectorBase, NUMACoordinatorBase
)
from .config import OptimizerConfig
from .exceptions import (
    MemOptimizerError, AllocationError, OutOfMemoryError,
    ConfigurationError, AllocatorError, IntegrationError
)

logger = logging.getLogger(__name__)


@dataclass
class PoolSnapshot:
    """
    内存池快照

    记录某一时刻的内存池状态。
    """
    timestamp: float
    total_size: int
    used_size: int
    free_size: int
    fragmentation_ratio: float
    allocation_count: int
    active_allocators: List[AllocatorType]


class MemoryPool:
    """
    内存池主类

    提供高性能的内存分配优化功能，整合多种分配器、策略选择、碎片整理等。
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化内存池

        Args:
            config: 优化器配置，None则使用默认配置
        """
        self.config = config or OptimizerConfig()

        if not self.config.validate():
            raise ConfigurationError("Invalid configuration")

        self.total_size = self.config.total_memory
        self.base_address = self.config.base_address

        self._allocators: Dict[AllocatorType, AllocatorBase] = {}
        self._strategy_selector: Optional[StrategySelectorBase] = None
        self._numa_coordinator: Optional[NUMACoordinatorBase] = None
        self._defragmenter = None
        self._monitor = None

        self._stats = MemoryStatistics(total_size=self.total_size)
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        self._free_blocks: List[MemoryBlock] = []
        self._block_lock = threading.RLock() if self.config.thread_safe else None

        self._initialized = False
        self._shutdown = False

        self._init_components()
        self._initialized = True

        logger.info(f"MemoryPool initialized: total_size={self.total_size}")

    def _init_components(self):
        """初始化组件"""
        self._init_allocators()

        if self.config.enable_rl_selector:
            self._init_strategy_selector()

        if self.config.enable_numa:
            self._init_numa_coordinator()

        if self.config.enable_defrag:
            self._init_defragmenter()

        if self.config.enable_monitoring:
            self._init_monitor()

        self._init_free_blocks()

    def _init_allocators(self):
        """初始化分配器"""
        try:
            from ..allocators.buddy import BuddyAllocator
            from ..allocators.slab import SlabAllocator
            from ..allocators.tlsf import TLSFAllocator

            buddy_config = self.config.get_allocator_config(AllocatorType.BUDDY)
            self._allocators[AllocatorType.BUDDY] = BuddyAllocator(
                total_size=self.total_size,
                base_address=self.base_address,
                config=buddy_config
            )

            slab_config = self.config.get_allocator_config(AllocatorType.SLAB)
            self._allocators[AllocatorType.SLAB] = SlabAllocator(
                total_size=self.total_size,
                base_address=self.base_address,
                config=slab_config
            )

            tlsf_config = self.config.get_allocator_config(AllocatorType.TLSF)
            self._allocators[AllocatorType.TLSF] = TLSFAllocator(
                total_size=self.total_size,
                base_address=self.base_address,
                config=tlsf_config
            )

            logger.debug("All allocators initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize allocators: {e}")
            raise AllocatorError(f"Allocator initialization failed: {e}")

    def _init_strategy_selector(self):
        """初始化策略选择器"""
        try:
            from ..strategies.rl_selector import RLStrategySelector
            self._strategy_selector = RLStrategySelector(self.config.rl_config)
            logger.debug("Strategy selector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize strategy selector: {e}")
            self._strategy_selector = None

    def _init_numa_coordinator(self):
        """初始化NUMA协调器"""
        try:
            from ..numa.coordinator import NUMACoordinator
            self._numa_coordinator = NUMACoordinator(self.config.numa_config)
            logger.debug("NUMA coordinator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NUMA coordinator: {e}")
            self._numa_coordinator = None

    def _init_defragmenter(self):
        """初始化碎片整理器"""
        try:
            from ..defrag.defragmenter import Defragmenter
            self._defragmenter = Defragmenter(self.config.defrag_config)
            logger.debug("Defragmenter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize defragmenter: {e}")
            self._defragmenter = None

    def _init_monitor(self):
        """初始化监控器"""
        try:
            from ..monitor.monitor import MemoryMonitor
            self._monitor = MemoryMonitor(self.config.monitor_config, self)
            logger.debug("Monitor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize monitor: {e}")
            self._monitor = None

    def _init_free_blocks(self):
        """初始化空闲块列表"""
        initial_block = MemoryBlock(
            address=self.base_address,
            size=self.total_size
        )
        self._free_blocks.append(initial_block)

    def _get_lock(self):
        """
        获取锁上下文管理器
        
        Returns:
            上下文管理器，用于with语句
        """
        if self._block_lock:
            return self._block_lock
        else:
            # 返回一个空上下文管理器
            from contextlib import nullcontext
            return nullcontext()

    def allocate(self,
                size: int,
                alignment: Optional[int] = None,
                numa_node: int = -1,
                allocator_type: Optional[AllocatorType] = None,
                metadata: Optional[Dict[str, Any]] = None) -> AllocationResult:
        """
        分配内存

        Args:
            size: 分配大小
            alignment: 对齐要求
            numa_node: NUMA节点
            allocator_type: 指定分配器类型
            metadata: 元数据

        Returns:
            AllocationResult: 分配结果
        """
        if self._shutdown:
            return AllocationResult(
                success=False,
                error_message="Memory pool is shut down"
            )

        start_time = time.time()

        try:
            with self._get_lock():

                if size <= 0:
                    return AllocationResult(
                        success=False,
                        error_message="Invalid allocation size"
                    )

                if size > self.config.max_allocation_size:
                    return AllocationResult(
                        success=False,
                        error_message=f"Size exceeds maximum: {size} > {self.config.max_allocation_size}"
                    )

                actual_alignment = alignment or self.config.default_alignment

                request = AllocationRequest(
                    size=size,
                    alignment=actual_alignment,
                    numa_node=numa_node,
                    metadata=metadata or {}
                )

                if numa_node < 0 and self._numa_coordinator:
                    request.numa_node = self._numa_coordinator.select_node(request)

                if allocator_type is None:
                    allocator_type = self._select_allocator(request)

                allocator = self._allocators.get(allocator_type)
                if allocator is None:
                    return AllocationResult(
                        success=False,
                        error_message=f"Allocator not found: {allocator_type}"
                    )

                result = allocator.allocate(request)

                if result.success:
                    self._stats.allocation_count += 1
                    self._stats.update_usage(
                        self._stats.used_size + result.actual_size,
                        self.total_size
                    )

                    block = MemoryBlock(
                        address=result.address,
                        size=result.actual_size,
                        state=result.state if hasattr(result, 'state') else None,
                        allocator_type=allocator_type,
                        numa_node=result.numa_node,
                        metadata=metadata or {}
                    )
                    self._allocated_blocks[result.address] = block

                    self._update_strategy_performance(allocator_type, result, time.time() - start_time)

                    if self._monitor:
                        self._monitor.record_allocation(result)

                else:
                    self._stats.allocation_failures += 1

                    if self.config.enable_defrag and self._should_defrag():
                        self._trigger_defrag()
                        result = allocator.allocate(request)

                return result

        except Exception as e:
            logger.error(f"Allocation failed: {e}")
            return AllocationResult(
                success=False,
                error_message=str(e)
            )

    def deallocate(self, address: int) -> bool:
        """
        释放内存

        Args:
            address: 内存地址

        Returns:
            bool: 是否成功
        """
        if self._shutdown:
            return False

        try:
            with self._get_lock():

                block = self._allocated_blocks.get(address)
                if block is None:
                    logger.warning(f"Block not found at address: 0x{address:x}")
                    return False

                allocator = self._allocators.get(block.allocator_type)
                if allocator is None:
                    return False

                success = allocator.deallocate(address)

                if success:
                    del self._allocated_blocks[address]

                    self._stats.deallocation_count += 1
                    self._stats.update_usage(
                        self._stats.used_size - block.size,
                        self.total_size
                    )

                    block.state = None
                    self._free_blocks.append(block)

                    if self._monitor:
                        self._monitor.record_deallocation(block)

                return success

        except Exception as e:
            logger.error(f"Deallocation failed: {e}")
            return False

    def reallocate(self, address: int, new_size: int) -> AllocationResult:
        """
        重新分配内存

        Args:
            address: 原内存地址
            new_size: 新大小

        Returns:
            AllocationResult: 分配结果
        """
        if self._shutdown:
            return AllocationResult(
                success=False,
                error_message="Memory pool is shut down"
            )

        try:
            with self._get_lock():

                block = self._allocated_blocks.get(address)
                if block is None:
                    return AllocationResult(
                        success=False,
                        error_message=f"Block not found at address: 0x{address:x}"
                    )

                allocator = self._allocators.get(block.allocator_type)
                if allocator is None:
                    return AllocationResult(
                        success=False,
                        error_message=f"Allocator not found: {block.allocator_type}"
                    )

                result = allocator.reallocate(address, new_size)

                if result.success:
                    del self._allocated_blocks[address]

                    new_block = MemoryBlock(
                        address=result.address,
                        size=result.actual_size,
                        state=result.state if hasattr(result, 'state') else None,
                        allocator_type=block.allocator_type,
                        numa_node=block.numa_node,
                        metadata=block.metadata
                    )
                    self._allocated_blocks[result.address] = new_block

                    size_diff = result.actual_size - block.size
                    self._stats.update_usage(
                        self._stats.used_size + size_diff,
                        self.total_size
                    )

                return result

        except Exception as e:
            logger.error(f"Reallocation failed: {e}")
            return AllocationResult(
                success=False,
                error_message=str(e)
            )

    def _select_allocator(self, request: AllocationRequest) -> AllocatorType:
        """
        选择分配器

        Args:
            request: 分配请求

        Returns:
            AllocatorType: 分配器类型
        """
        if self._strategy_selector:
            context = {
                'total_size': self.total_size,
                'used_size': self._stats.used_size,
                'fragmentation': self._stats.fragmentation_ratio
            }
            return self._strategy_selector.select_allocator(request, context)

        return self._default_allocator_selection(request)

    def _default_allocator_selection(self, request: AllocationRequest) -> AllocatorType:
        """
        默认分配器选择逻辑

        Args:
            request: 分配请求

        Returns:
            AllocatorType: 分配器类型
        """
        size = request.size

        if size <= 4096:
            return AllocatorType.SLAB
        elif size <= 1024 * 1024:
            return AllocatorType.TLSF
        else:
            return AllocatorType.BUDDY

    def _update_strategy_performance(self,
                                    allocator: AllocatorType,
                                    result: AllocationResult,
                                    duration: float):
        """更新策略性能数据"""
        if self._strategy_selector:
            performance = {
                'success': result.success,
                'allocation_time': duration,
                'fragmentation': result.fragmentation,
                'size': result.size
            }
            self._strategy_selector.update_performance(allocator, performance)

    def _should_defrag(self) -> bool:
        """判断是否需要碎片整理"""
        if not self._defragmenter:
            return False

        fragmentation = self._stats.fragmentation_ratio
        threshold = self.config.defrag_config.threshold

        return fragmentation > threshold

    def _trigger_defrag(self):
        """触发碎片整理"""
        if self._defragmenter:
            try:
                blocks = list(self._allocated_blocks.values()) + self._free_blocks
                moved = self._defragmenter.defragment(blocks)
                self._stats.defragmentation_count += 1
                logger.info(f"Defragmentation completed: {moved} blocks moved")
            except Exception as e:
                logger.error(f"Defragmentation failed: {e}")

    def get_stats(self) -> MemoryStatistics:
        """
        获取统计信息

        Returns:
            MemoryStatistics: 统计信息
        """
        self._stats.calculate_fragmentation(self._free_blocks)
        return self._stats

    def get_snapshot(self) -> PoolSnapshot:
        """
        获取内存池快照

        Returns:
            PoolSnapshot: 快照
        """
        return PoolSnapshot(
            timestamp=time.time(),
            total_size=self.total_size,
            used_size=self._stats.used_size,
            free_size=self._stats.free_size,
            fragmentation_ratio=self._stats.fragmentation_ratio,
            allocation_count=self._stats.allocation_count,
            active_allocators=list(self._allocators.keys())
        )

    def get_block_info(self, address: int) -> Optional[MemoryBlock]:
        """
        获取块信息

        Args:
            address: 内存地址

        Returns:
            MemoryBlock: 块信息
        """
        return self._allocated_blocks.get(address)

    def get_all_blocks(self) -> List[MemoryBlock]:
        """
        获取所有块

        Returns:
            List[MemoryBlock]: 块列表
        """
        return list(self._allocated_blocks.values())

    def get_free_blocks(self) -> List[MemoryBlock]:
        """
        获取空闲块

        Returns:
            List[MemoryBlock]: 空闲块列表
        """
        return self._free_blocks.copy()

    def get_allocator_stats(self) -> Dict[AllocatorType, MemoryStatistics]:
        """
        获取各分配器统计

        Returns:
            Dict: 分配器统计映射
        """
        result = {}
        for atype, allocator in self._allocators.items():
            result[atype] = allocator.get_stats()
        return result

    def compact(self) -> int:
        """
        压缩内存池

        Returns:
            int: 移动的块数量
        """
        if self._defragmenter:
            return self._defragmenter.defragment(
                list(self._allocated_blocks.values()) + self._free_blocks
            )
        return 0

    def reset(self):
        """重置内存池"""
        with self._get_lock():
            try:
                for allocator in self._allocators.values():
                    allocator.defragment()

                self._allocated_blocks.clear()
                self._free_blocks.clear()

                initial_block = MemoryBlock(
                    address=self.base_address,
                    size=self.total_size
                )
                self._free_blocks.append(initial_block)

                self._stats = MemoryStatistics(total_size=self.total_size)

                logger.info("Memory pool reset")

            except Exception as e:
                logger.error(f"Reset failed: {e}")
                raise

    def shutdown(self):
        """关闭内存池"""
        self._shutdown = True

        if self._monitor:
            self._monitor.stop()

        self.reset()
        logger.info("Memory pool shut down")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False

    def __del__(self):
        """析构函数"""
        try:
            if not self._shutdown:
                self.shutdown()
        except Exception as e:
            # 在析构函数中只记录日志，不抛出异常
            # 使用 logging 模块的 exception 方法记录完整的异常信息
            logger.exception(f"Error during MemoryPool cleanup: {e}")
