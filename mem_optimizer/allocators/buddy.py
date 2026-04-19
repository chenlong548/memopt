"""
mem_optimizer Buddy分配器

实现Buddy System内存分配算法。
"""

import math
import time
from typing import Optional, List, Dict
from dataclasses import dataclass

from ..core.base import (
    AllocatorBase, AllocatorType, AllocationRequest, AllocationResult,
    MemoryBlock, MemoryRegionState, AllocationStrategy
)
from ..core.config import BuddyAllocatorConfig
from ..core.exceptions import AllocationError, OutOfMemoryError, InvalidBlockError


@dataclass
class BuddyBlock:
    """Buddy系统块"""
    address: int
    order: int
    is_free: bool = True
    buddy_address: Optional[int] = None

    @property
    def size(self) -> int:
        return 1 << self.order


class BuddyAllocator(AllocatorBase):
    """
    Buddy分配器

    使用Buddy System算法管理内存分配，适合大块内存分配。
    """

    def __init__(self,
                total_size: int,
                base_address: int = 0,
                config: Optional[BuddyAllocatorConfig] = None):
        """
        初始化Buddy分配器

        Args:
            total_size: 总内存大小
            base_address: 基地址
            config: 分配器配置
        """
        super().__init__(total_size, base_address)

        self.config = config or BuddyAllocatorConfig()
        self.allocator_type = AllocatorType.BUDDY

        self.min_order = self.config.min_order
        self.max_order = max(
            int(math.log2(total_size)),
            self.min_order
        )

        self._free_lists: Dict[int, List[BuddyBlock]] = {
            order: [] for order in range(self.min_order, self.max_order + 1)
        }

        self._allocated: Dict[int, BuddyBlock] = {}
        self._all_blocks: Dict[int, BuddyBlock] = {}

        self._initialize_pool()

    def _initialize_pool(self):
        """初始化内存池"""
        order = self.max_order
        block = BuddyBlock(
            address=self.base_address,
            order=order,
            is_free=True
        )
        block.buddy_address = self._get_buddy_address(block.address, order)
        self._free_lists[order].append(block)
        self._all_blocks[block.address] = block

    def _get_buddy_address(self, address: int, order: int) -> int:
        """
        计算伙伴地址

        Args:
            address: 块地址
            order: 块阶数

        Returns:
            int: 伙伴地址
        """
        size = 1 << order
        return self.base_address + ((address - self.base_address) ^ size)

    def _get_order(self, size: int) -> int:
        """
        根据大小获取阶数

        Args:
            size: 请求大小

        Returns:
            int: 阶数

        Raises:
            AllocationError: 当size小于等于0时
        """
        if size <= 0:
            raise AllocationError(f"Invalid allocation size: {size}, size must be positive")
        order = max(int(math.ceil(math.log2(size))), self.min_order)
        return min(order, self.max_order)

    def _split_block(self, block: BuddyBlock, target_order: int) -> Optional[BuddyBlock]:
        """
        分裂块

        Args:
            block: 待分裂的块
            target_order: 目标阶数

        Returns:
            BuddyBlock: 分裂后的块
        """
        while block.order > target_order:
            block.order -= 1

            buddy_addr = self._get_buddy_address(block.address, block.order)
            buddy = BuddyBlock(
                address=buddy_addr,
                order=block.order,
                is_free=True
            )
            buddy.buddy_address = block.address

            block.buddy_address = buddy_addr

            self._free_lists[block.order].append(buddy)
            self._all_blocks[buddy_addr] = buddy

        return block

    def _merge_buddies(self, block: BuddyBlock) -> Optional[BuddyBlock]:
        """
        合并伙伴块

        Args:
            block: 待合并的块

        Returns:
            BuddyBlock: 合并后的块
        """
        while block.order < self.max_order:
            buddy_addr = block.buddy_address
            buddy = self._all_blocks.get(buddy_addr)

            if buddy is None or not buddy.is_free or buddy.order != block.order:
                break

            self._free_lists[block.order].remove(buddy)

            if buddy_addr < block.address:
                block.address = buddy_addr

            block.order += 1
            block.buddy_address = self._get_buddy_address(block.address, block.order)
            block.is_free = True

            del self._all_blocks[buddy_addr]

        return block

    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """
        分配内存

        Args:
            request: 分配请求

        Returns:
            AllocationResult: 分配结果
        """
        start_time = time.time()

        try:
            size = request.size
            alignment = request.alignment

            aligned_size = max(size, alignment)
            order = self._get_order(aligned_size)

            if order > self.max_order:
                return AllocationResult(
                    success=False,
                    error_message=f"Requested size too large: {size}"
                )

            for current_order in range(order, self.max_order + 1):
                if self._free_lists[current_order]:
                    block = self._free_lists[current_order].pop(0)
                    break
            else:
                self.stats.allocation_failures += 1
                return AllocationResult(
                    success=False,
                    error_message="Out of memory"
                )

            if block.order > order:
                block = self._split_block(block, order)

            block.is_free = False
            self._allocated[block.address] = block

            self.stats.allocation_count += 1
            self.stats.used_size += block.size
            self.stats.free_size -= block.size

            if self.stats.used_size > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.used_size

            allocation_time = time.time() - start_time

            return AllocationResult(
                success=True,
                address=block.address,
                size=request.size,
                actual_size=block.size,
                allocator_type=self.allocator_type,
                numa_node=request.numa_node,
                fragmentation=(block.size - request.size) / block.size if block.size > 0 else 0,
                allocation_time=allocation_time
            )

        except Exception as e:
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
        try:
            block = self._allocated.get(address)
            if block is None:
                return False

            del self._allocated[address]

            block.is_free = True

            merged = self._merge_buddies(block)

            self._free_lists[merged.order].append(merged)

            self.stats.deallocation_count += 1
            self.stats.used_size -= block.size
            self.stats.free_size += block.size

            return True

        except Exception:
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
        try:
            old_block = self._allocated.get(address)
            if old_block is None:
                return AllocationResult(
                    success=False,
                    error_message=f"Block not found at address: {address}"
                )

            new_order = self._get_order(new_size)

            if new_order <= old_block.order:
                return AllocationResult(
                    success=True,
                    address=address,
                    size=new_size,
                    actual_size=old_block.size,
                    allocator_type=self.allocator_type
                )

            self.deallocate(address)

            request = AllocationRequest(size=new_size)
            result = self.allocate(request)

            return result

        except Exception as e:
            return AllocationResult(
                success=False,
                error_message=str(e)
            )

    def get_free_blocks(self) -> List[MemoryBlock]:
        """
        获取空闲块列表

        Returns:
            List[MemoryBlock]: 空闲块列表
        """
        blocks = []
        for order, free_list in self._free_lists.items():
            for buddy in free_list:
                block = MemoryBlock(
                    address=buddy.address,
                    size=buddy.size,
                    state=MemoryRegionState.FREE,
                    allocator_type=self.allocator_type
                )
                blocks.append(block)
        return blocks

    def get_allocated_blocks(self) -> List[MemoryBlock]:
        """
        获取已分配块列表

        Returns:
            List[MemoryBlock]: 已分配块列表
        """
        blocks = []
        for address, buddy in self._allocated.items():
            block = MemoryBlock(
                address=buddy.address,
                size=buddy.size,
                state=MemoryRegionState.ALLOCATED,
                allocator_type=self.allocator_type
            )
            blocks.append(block)
        return blocks

    def get_fragmentation_score(self) -> float:
        """
        获取碎片评分

        Returns:
            float: 碎片评分 (0-1)
        """
        free_blocks = self.get_free_blocks()
        if not free_blocks:
            return 0.0

        total_free = sum(b.size for b in free_blocks)
        if total_free == 0:
            return 0.0

        max_free = max(b.size for b in free_blocks)
        return 1.0 - (max_free / total_free)

    def defragment(self) -> int:
        """
        执行碎片整理

        Returns:
            int: 整理的块数量
        """
        merged_count = 0

        for order in range(self.min_order, self.max_order):
            free_list = self._free_lists[order].copy()

            for block in free_list:
                if block.is_free:
                    buddy_addr = block.buddy_address
                    buddy = self._all_blocks.get(buddy_addr)

                    if buddy and buddy.is_free and buddy.order == block.order:
                        self._free_lists[order].remove(block)
                        self._free_lists[order].remove(buddy)

                        merged = self._merge_buddies(block)
                        self._free_lists[merged.order].append(merged)
                        merged_count += 1

        self.stats.defragmentation_count += 1
        return merged_count

    def get_order_stats(self) -> Dict[int, Dict[str, int]]:
        """
        获取各阶统计

        Returns:
            Dict: 阶数到统计的映射
        """
        stats = {}
        for order in range(self.min_order, self.max_order + 1):
            free_count = len(self._free_lists[order])
            allocated_count = sum(
                1 for b in self._allocated.values() if b.order == order
            )
            stats[order] = {
                'free': free_count,
                'allocated': allocated_count,
                'total': free_count + allocated_count,
                'block_size': 1 << order
            }
        return stats
