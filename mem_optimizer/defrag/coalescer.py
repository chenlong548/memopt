"""
mem_optimizer 虚拟内存拼接器

实现虚拟内存区域的合并和拼接。
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.base import MemoryBlock, MemoryRegionState
from ..core.exceptions import DefragmentationError


class CoalesceStrategy(Enum):
    """拼接策略"""
    ADJACENT = "adjacent"
    GAP_FILLING = "gap_filling"
    AGGRESSIVE = "aggressive"


@dataclass
class CoalesceRegion:
    """可拼接区域"""
    start_address: int
    end_address: int
    total_size: int
    blocks: List[MemoryBlock]
    gaps: List[Tuple[int, int]]
    coalesce_potential: float = 0.0


class MemoryCoalescer:
    """
    虚拟内存拼接器

    检测和合并相邻的内存区域。
    """

    def __init__(self, strategy: CoalesceStrategy = CoalesceStrategy.ADJACENT):
        """
        初始化拼接器

        Args:
            strategy: 拼接策略
        """
        self.strategy = strategy
        self._coalesced_count = 0
        self._total_coalesced_size = 0

    def find_coalesce_candidates(self,
                                 blocks: List[MemoryBlock]) -> List[CoalesceRegion]:
        """
        查找可拼接的候选区域

        Args:
            blocks: 内存块列表

        Returns:
            List[CoalesceRegion]: 可拼接区域列表
        """
        if not blocks:
            return []

        sorted_blocks = sorted(blocks, key=lambda b: b.address)
        candidates = []

        current_region: Optional[CoalesceRegion] = None

        for block in sorted_blocks:
            if not block.is_free():
                if current_region:
                    candidates.append(current_region)
                    current_region = None
                continue

            if current_region is None:
                current_region = CoalesceRegion(
                    start_address=block.address,
                    end_address=block.address + block.size,
                    total_size=block.size,
                    blocks=[block],
                    gaps=[]
                )
            else:
                expected_addr = current_region.end_address

                if block.address == expected_addr:
                    current_region.end_address = block.address + block.size
                    current_region.total_size += block.size
                    current_region.blocks.append(block)
                elif self.strategy == CoalesceStrategy.AGGRESSIVE:
                    gap_size = block.address - expected_addr
                    if gap_size < 4096:
                        current_region.gaps.append((expected_addr, gap_size))
                        current_region.end_address = block.address + block.size
                        current_region.total_size += block.size + gap_size
                        current_region.blocks.append(block)
                    else:
                        candidates.append(current_region)
                        current_region = CoalesceRegion(
                            start_address=block.address,
                            end_address=block.address + block.size,
                            total_size=block.size,
                            blocks=[block],
                            gaps=[]
                        )
                else:
                    candidates.append(current_region)
                    current_region = CoalesceRegion(
                        start_address=block.address,
                        end_address=block.address + block.size,
                        total_size=block.size,
                        blocks=[block],
                        gaps=[]
                    )

        if current_region:
            candidates.append(current_region)

        for region in candidates:
            region.coalesce_potential = self._calculate_potential(region)

        return [r for r in candidates if len(r.blocks) > 1 or r.gaps]

    def _calculate_potential(self, region: CoalesceRegion) -> float:
        """
        计算拼接潜力

        Args:
            region: 区域

        Returns:
            float: 拼接潜力 (0-1)
        """
        if len(region.blocks) <= 1:
            return 0.0

        block_count = len(region.blocks)
        gap_count = len(region.gaps)

        size_efficiency = region.total_size / (region.end_address - region.start_address)

        count_benefit = 1.0 - (1.0 / block_count)

        gap_penalty = gap_count * 0.1

        return max(0.0, min(1.0, size_efficiency * count_benefit - gap_penalty))

    def coalesce(self, blocks: List[MemoryBlock]) -> List[MemoryBlock]:
        """
        执行拼接操作

        Args:
            blocks: 内存块列表

        Returns:
            List[MemoryBlock]: 拼接后的块列表
        """
        candidates = self.find_coalesce_candidates(blocks)

        result = []
        coalesced_addresses = set()

        for region in candidates:
            if len(region.blocks) > 1:
                merged_block = MemoryBlock(
                    address=region.start_address,
                    size=region.total_size,
                    state=MemoryRegionState.FREE
                )
                result.append(merged_block)

                for block in region.blocks:
                    coalesced_addresses.add(block.address)

                self._coalesced_count += 1
                self._total_coalesced_size += region.total_size

        for block in blocks:
            if block.address not in coalesced_addresses:
                result.append(block)

        return sorted(result, key=lambda b: b.address)

    def get_coalesce_stats(self) -> Dict[str, Any]:
        """
        获取拼接统计

        Returns:
            Dict: 统计信息
        """
        return {
            'coalesced_count': self._coalesced_count,
            'total_coalesced_size': self._total_coalesced_size,
            'strategy': self.strategy.value
        }

    def estimate_benefit(self, blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """
        估算拼接收益

        Args:
            blocks: 内存块列表

        Returns:
            Dict: 收益估算
        """
        candidates = self.find_coalesce_candidates(blocks)

        total_blocks = sum(len(r.blocks) for r in candidates)
        total_size = sum(r.total_size for r in candidates)
        total_gaps = sum(len(r.gaps) for r in candidates)

        return {
            'candidate_regions': len(candidates),
            'blocks_to_merge': total_blocks,
            'total_size': total_size,
            'gaps_to_fill': total_gaps,
            'estimated_block_reduction': total_blocks - len(candidates),
            'avg_potential': sum(r.coalesce_potential for r in candidates) / max(len(candidates), 1)
        }


class GapFiller:
    """
    间隙填充器

    填充内存块之间的间隙。
    """

    def __init__(self, min_gap_size: int = 64, max_gap_size: int = 4096):
        """
        初始化间隙填充器

        Args:
            min_gap_size: 最小间隙大小
            max_gap_size: 最大间隙大小
        """
        self.min_gap_size = min_gap_size
        self.max_gap_size = max_gap_size
        self._filled_gaps = 0
        self._filled_size = 0

    def find_gaps(self, blocks: List[MemoryBlock]) -> List[Tuple[int, int]]:
        """
        查找间隙

        Args:
            blocks: 内存块列表

        Returns:
            List[Tuple]: 间隙列表 (地址, 大小)
        """
        if not blocks:
            return []

        sorted_blocks = sorted(blocks, key=lambda b: b.address)
        gaps = []

        for i in range(len(sorted_blocks) - 1):
            current = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]

            gap_start = current.address + current.size
            gap_end = next_block.address

            if gap_end > gap_start:
                gap_size = gap_end - gap_start

                if self.min_gap_size <= gap_size <= self.max_gap_size:
                    gaps.append((gap_start, gap_size))

        return gaps

    def fill_gaps(self,
                 blocks: List[MemoryBlock],
                 available_blocks: Optional[List[MemoryBlock]] = None) -> List[MemoryBlock]:
        """
        填充间隙

        Args:
            blocks: 内存块列表
            available_blocks: 可用于填充的块

        Returns:
            List[MemoryBlock]: 填充后的块列表
        """
        gaps = self.find_gaps(blocks)

        if not gaps:
            return blocks

        result = blocks.copy()

        for gap_addr, gap_size in gaps:
            filler = self._find_filler(gap_size, available_blocks)

            if filler:
                filled_block = MemoryBlock(
                    address=gap_addr,
                    size=gap_size,
                    state=MemoryRegionState.FREE
                )
                result.append(filled_block)

                self._filled_gaps += 1
                self._filled_size += gap_size

        return sorted(result, key=lambda b: b.address)

    def _find_filler(self,
                    gap_size: int,
                    available_blocks: Optional[List[MemoryBlock]]) -> Optional[MemoryBlock]:
        """
        查找填充块

        Args:
            gap_size: 间隙大小
            available_blocks: 可用块

        Returns:
            MemoryBlock: 填充块
        """
        if not available_blocks:
            return MemoryBlock(address=0, size=gap_size)

        for block in available_blocks:
            if block.size >= gap_size and block.is_free():
                return block

        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'filled_gaps': self._filled_gaps,
            'filled_size': self._filled_size,
            'min_gap_size': self.min_gap_size,
            'max_gap_size': self.max_gap_size
        }
