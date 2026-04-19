"""
mem_optimizer TLSF分配器

实现TLSF (Two-Level Segregated Fit) 内存分配算法。
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ..core.base import (
    AllocatorBase, AllocatorType, AllocationRequest, AllocationResult,
    MemoryBlock, MemoryRegionState
)
from ..core.config import TLSFAllocatorConfig
from ..core.exceptions import AllocationError, OutOfMemoryError


@dataclass
class TLSFBlock:
    """TLSF块"""
    address: int
    size: int
    is_free: bool = True
    prev_physical: Optional[int] = None
    next_physical: Optional[int] = None
    prev_free: Optional[int] = None
    next_free: Optional[int] = None

    @property
    def block_size(self) -> int:
        return self.size


class TLSFAllocator(AllocatorBase):
    """
    TLSF分配器

    使用TLSF算法管理内存分配，提供O(1)时间复杂度的分配和释放。
    """

    def __init__(self,
                total_size: int,
                base_address: int = 0,
                config: Optional[TLSFAllocatorConfig] = None):
        """
        初始化TLSF分配器

        Args:
            total_size: 总内存大小
            base_address: 基地址
            config: 分配器配置
        """
        super().__init__(total_size, base_address)

        self.config = config or TLSFAllocatorConfig()
        self.allocator_type = AllocatorType.TLSF

        self.first_level_bits = self.config.first_level_bits
        self.second_level_bits = self.config.second_level_bits
        self.min_block_size = self.config.min_block_size

        self.first_level_count = 1 << self.first_level_bits
        self.second_level_count = 1 << self.second_level_bits

        self._fl_bitmap: int = 0
        self._sl_bitmap: List[int] = [0] * self.first_level_count
        self._blocks: List[List[Optional[int]]] = [
            [None] * self.second_level_count
            for _ in range(self.first_level_count)
        ]

        self._all_blocks: dict = {}
        self._allocated: dict = {}

        self._initialize_pool()

    def _initialize_pool(self):
        """初始化内存池"""
        block = TLSFBlock(
            address=self.base_address,
            size=self.total_size,
            is_free=True
        )
        self._all_blocks[self.base_address] = block

        fl, sl = self._mapping_insert(self.total_size)
        self._block_link(block, fl, sl)

    def _log2_floor(self, size: int) -> int:
        """计算floor(log2(size))"""
        if size < 2:
            return 0
        return size.bit_length() - 1

    def _log2_ceil(self, size: int) -> int:
        """计算ceil(log2(size))"""
        if size <= 1:
            return 0
        return (size - 1).bit_length()

    def _mapping_insert(self, size: int) -> Tuple[int, int]:
        """
        映射大小到FL/SL索引

        Args:
            size: 块大小

        Returns:
            Tuple[int, int]: (FL索引, SL索引)
        """
        if size < self.min_block_size:
            size = self.min_block_size

        fl = self._log2_floor(size)

        if fl < self.first_level_bits:
            return 0, size // self.min_block_size

        fl = min(fl - self.first_level_bits + 1, self.first_level_count - 1)

        sl = (size >> (fl + self.first_level_bits - 1)) - self.second_level_count
        sl = max(0, min(sl, self.second_level_count - 1))

        return fl, sl

    def _mapping_search(self, size: int) -> Tuple[int, int]:
        """
        搜索合适大小的FL/SL索引

        Args:
            size: 请求大小

        Returns:
            Tuple[int, int]: (FL索引, SL索引)
        """
        fl, sl = self._mapping_insert(size)

        sl_mask = self._sl_bitmap[fl] & (~0 << sl)
        if sl_mask:
            sl = (sl_mask & -sl_mask).bit_length() - 1
            return fl, sl

        fl_mask = self._fl_bitmap & (~0 << (fl + 1))
        if fl_mask:
            fl = (fl_mask & -fl_mask).bit_length() - 1
            sl = (self._sl_bitmap[fl] & -self._sl_bitmap[fl]).bit_length() - 1
            return fl, sl

        return -1, -1

    def _block_link(self, block: TLSFBlock, fl: int, sl: int):
        """
        将块链接到空闲列表

        Args:
            block: 块
            fl: FL索引
            sl: SL索引
        """
        block.prev_free = None

        head_addr = self._blocks[fl][sl]
        if head_addr is not None:
            head = self._all_blocks.get(head_addr)
            if head:
                head.prev_free = block.address

        block.next_free = head_addr
        self._blocks[fl][sl] = block.address

        self._fl_bitmap |= (1 << fl)
        self._sl_bitmap[fl] |= (1 << sl)

    def _block_unlink(self, block: TLSFBlock, fl: int, sl: int):
        """
        从空闲列表移除块

        Args:
            block: 块
            fl: FL索引
            sl: SL索引
        """
        prev_addr = block.prev_free
        next_addr = block.next_free

        if prev_addr is not None:
            prev = self._all_blocks.get(prev_addr)
            if prev:
                prev.next_free = next_addr
        else:
            self._blocks[fl][sl] = next_addr

        if next_addr is not None:
            next_block = self._all_blocks.get(next_addr)
            if next_block:
                next_block.prev_free = prev_addr

        if self._blocks[fl][sl] is None:
            self._sl_bitmap[fl] &= ~(1 << sl)
            if self._sl_bitmap[fl] == 0:
                self._fl_bitmap &= ~(1 << fl)

    def _find_suitable_block(self, size: int) -> Optional[TLSFBlock]:
        """
        查找合适的块

        Args:
            size: 请求大小

        Returns:
            TLSFBlock: 合适的块
        """
        fl, sl = self._mapping_search(size)

        if fl < 0:
            return None

        block_addr = self._blocks[fl][sl]
        if block_addr is None:
            return None

        return self._all_blocks.get(block_addr)

    def _split_block(self, block: TLSFBlock, size: int) -> Optional[TLSFBlock]:
        """
        分裂块

        Args:
            block: 待分裂的块
            size: 请求大小

        Returns:
            TLSFBlock: 分裂后的块
        """
        remaining = block.size - size

        min_split = self.min_block_size + 16

        if remaining >= min_split:
            fl, sl = self._mapping_insert(block.size)
            self._block_unlink(block, fl, sl)

            block.size = size

            remaining_block = TLSFBlock(
                address=block.address + size,
                size=remaining,
                is_free=True,
                prev_physical=block.address,
                next_physical=block.next_physical
            )

            if block.next_physical is not None:
                next_block = self._all_blocks.get(block.next_physical)
                if next_block:
                    next_block.prev_physical = remaining_block.address

            block.next_physical = remaining_block.address

            self._all_blocks[remaining_block.address] = remaining_block

            fl, sl = self._mapping_insert(remaining_block.size)
            self._block_link(remaining_block, fl, sl)

        return block

    def _merge_prev(self, block: TLSFBlock) -> TLSFBlock:
        """
        与前一个块合并

        Args:
            block: 当前块

        Returns:
            TLSFBlock: 合并后的块
        """
        if block.prev_physical is None:
            return block

        prev_block = self._all_blocks.get(block.prev_physical)
        if prev_block is None or not prev_block.is_free:
            return block

        fl, sl = self._mapping_insert(prev_block.size)
        self._block_unlink(prev_block, fl, sl)

        prev_block.size += block.size
        prev_block.next_physical = block.next_physical

        if block.next_physical is not None:
            next_block = self._all_blocks.get(block.next_physical)
            if next_block:
                next_block.prev_physical = prev_block.address

        del self._all_blocks[block.address]

        return prev_block

    def _merge_next(self, block: TLSFBlock) -> TLSFBlock:
        """
        与后一个块合并

        Args:
            block: 当前块

        Returns:
            TLSFBlock: 合并后的块
        """
        if block.next_physical is None:
            return block

        next_block = self._all_blocks.get(block.next_physical)
        if next_block is None or not next_block.is_free:
            return block

        fl, sl = self._mapping_insert(next_block.size)
        self._block_unlink(next_block, fl, sl)

        block.size += next_block.size
        block.next_physical = next_block.next_physical

        if next_block.next_physical is not None:
            next_next = self._all_blocks.get(next_block.next_physical)
            if next_next:
                next_next.prev_physical = block.address

        del self._all_blocks[next_block.address]

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

            aligned_size = max(size, self.min_block_size)
            if alignment > self.min_block_size:
                aligned_size = max(aligned_size, alignment)

            block = self._find_suitable_block(aligned_size)

            if block is None:
                self.stats.allocation_failures += 1
                return AllocationResult(
                    success=False,
                    error_message="Out of memory"
                )

            fl, sl = self._mapping_insert(block.size)
            self._block_unlink(block, fl, sl)

            block = self._split_block(block, aligned_size)
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

            block = self._merge_prev(block)
            block = self._merge_next(block)

            fl, sl = self._mapping_insert(block.size)
            self._block_link(block, fl, sl)

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

            aligned_size = max(new_size, self.min_block_size)

            if aligned_size <= old_block.size:
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
        for addr, block in self._all_blocks.items():
            if block.is_free:
                mem_block = MemoryBlock(
                    address=block.address,
                    size=block.size,
                    state=MemoryRegionState.FREE,
                    allocator_type=self.allocator_type
                )
                blocks.append(mem_block)
        return blocks

    def get_allocated_blocks(self) -> List[MemoryBlock]:
        """
        获取已分配块列表

        Returns:
            List[MemoryBlock]: 已分配块列表
        """
        blocks = []
        for addr, block in self._allocated.items():
            mem_block = MemoryBlock(
                address=block.address,
                size=block.size,
                state=MemoryRegionState.ALLOCATED,
                allocator_type=self.allocator_type
            )
            blocks.append(mem_block)
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

        blocks_to_process = [
            block for block in self._all_blocks.values()
            if block.is_free
        ]

        for block in blocks_to_process:
            if block.is_free:
                original_size = block.size
                block = self._merge_prev(block)
                block = self._merge_next(block)

                if block.size > original_size:
                    merged_count += 1

        self.stats.defragmentation_count += 1
        return merged_count

    def get_bitmap_stats(self) -> dict:
        """
        获取位图统计

        Returns:
            dict: 位图统计信息
        """
        free_count = 0
        for fl in range(self.first_level_count):
            if self._sl_bitmap[fl]:
                free_count += bin(self._sl_bitmap[fl]).count('1')

        return {
            'fl_bitmap': bin(self._fl_bitmap),
            'free_lists_count': free_count,
            'total_blocks': len(self._all_blocks),
            'allocated_blocks': len(self._allocated)
        }
