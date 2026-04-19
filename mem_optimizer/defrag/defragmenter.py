"""
mem_optimizer 碎片整理器

实现内存碎片整理机制。
"""

import time
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.base import MemoryBlock, MemoryRegionState, DefragmenterBase
from ..core.config import DefragConfig, DefragPolicy
from ..core.exceptions import DefragmentationError


logger = logging.getLogger(__name__)


class DefragState(Enum):
    """碎片整理状态"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DefragPlan:
    """碎片整理计划"""
    moves: List[Tuple[int, int, int]]
    estimated_time: float
    estimated_fragments_reduced: int
    priority: int = 0


@dataclass
class DefragResult:
    """碎片整理结果"""
    success: bool
    blocks_moved: int
    time_elapsed: float
    fragmentation_before: float
    fragmentation_after: float
    error_message: Optional[str] = None


class Defragmenter(DefragmenterBase):
    """
    碎片整理器

    提供内存碎片检测、分析和整理功能。
    """

    def __init__(self, config: Optional[DefragConfig] = None):
        """
        初始化碎片整理器

        Args:
            config: 碎片整理配置
        """
        self.config = config or DefragConfig()

        self._state = DefragState.IDLE
        self._last_defrag_time = 0.0
        self._total_defrags = 0
        self._total_blocks_moved = 0

        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()  # 添加状态锁

        if self.config.enable_background:
            self._start_background_thread()

    def _start_background_thread(self):
        """启动后台线程"""
        self._background_thread = threading.Thread(
            target=self._background_defrag_loop,
            daemon=True
        )
        self._background_thread.start()

    def _background_defrag_loop(self):
        """后台碎片整理循环"""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.config.background_interval)

                if self._should_auto_defrag():
                    # 在后台线程中不执行实际整理，只记录日志
                    logger.debug("Auto defrag condition met, but skipping in background thread")

            except Exception as e:
                # 记录异常但继续运行
                logger.warning(f"Background defrag loop error: {e}")

    def _should_auto_defrag(self) -> bool:
        """判断是否应该自动碎片整理"""
        if self.config.policy != DefragPolicy.AUTO:
            return False

        elapsed = time.time() - self._last_defrag_time
        if elapsed < self.config.min_interval:
            return False

        return True

    def analyze(self, blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """
        分析碎片情况

        Args:
            blocks: 内存块列表

        Returns:
            Dict: 分析结果
        """
        if not blocks:
            return {
                'total_blocks': 0,
                'free_blocks': 0,
                'allocated_blocks': 0,
                'fragmentation_ratio': 0.0,
                'avg_fragment_size': 0.0,
                'max_fragment_size': 0,
                'fragments': []
            }

        free_blocks = [b for b in blocks if b.is_free()]
        allocated_blocks = [b for b in blocks if b.is_allocated()]

        total_free = sum(b.size for b in free_blocks)
        total_size = sum(b.size for b in blocks)

        fragmentation_ratio = 0.0
        if total_free > 0:
            max_free = max(b.size for b in free_blocks) if free_blocks else 0
            fragmentation_ratio = 1.0 - (max_free / total_free)

        fragments = []
        sorted_blocks = sorted(blocks, key=lambda b: b.address)

        for i in range(len(sorted_blocks) - 1):
            current = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]

            gap = next_block.address - (current.address + current.size)

            if gap > 0:
                fragments.append({
                    'address': current.address + current.size,
                    'size': gap,
                    'between': [
                        current.address,
                        next_block.address
                    ]
                })

        return {
            'total_blocks': len(blocks),
            'free_blocks': len(free_blocks),
            'allocated_blocks': len(allocated_blocks),
            'total_free': total_free,
            'total_allocated': total_size - total_free,
            'fragmentation_ratio': fragmentation_ratio,
            'avg_fragment_size': total_free / len(free_blocks) if free_blocks else 0,
            'max_fragment_size': max(b.size for b in free_blocks) if free_blocks else 0,
            'fragments': fragments,
            'fragment_count': len(fragments)
        }

    def plan(self, blocks: List[MemoryBlock]) -> List[Tuple[int, int, int]]:
        """
        制定整理计划

        Args:
            blocks: 内存块列表

        Returns:
            List[Tuple]: 移动计划 (源地址, 目标地址, 大小)
        """
        analysis = self.analyze(blocks)

        if analysis['fragmentation_ratio'] < self.config.target_fragmentation:
            return []

        moves = []
        sorted_blocks = sorted(blocks, key=lambda b: b.address)

        free_regions = []
        for block in sorted_blocks:
            if block.is_free():
                free_regions.append(block)

        allocated_regions = []
        for block in sorted_blocks:
            if block.is_allocated():
                allocated_regions.append(block)

        allocated_regions.sort(key=lambda b: b.size, reverse=True)

        for allocated in allocated_regions[:self.config.max_migrations]:
            for free_region in free_regions:
                if free_region.size >= allocated.size:

                    target_addr = free_region.address
                    moves.append((
                        allocated.address,
                        target_addr,
                        allocated.size
                    ))

                    free_region.address = target_addr + allocated.size
                    free_region.size -= allocated.size

                    if free_region.size <= 0:
                        free_regions.remove(free_region)

                    break

        return moves

    def execute(self, plan: List[Tuple[int, int, int]]) -> int:
        """
        执行整理计划

        Args:
            plan: 整理计划

        Returns:
            int: 成功移动的块数量
        """
        moved = 0

        for source, target, size in plan:
            try:
                # 这里应该有实际的内存移动逻辑
                # 目前只是模拟
                moved += 1
            except Exception as e:
                # 记录单个移动失败，但继续执行其他移动
                logger.warning(f"Failed to move block from 0x{source:x} to 0x{target:x}: {e}")
                continue

        with self._state_lock:
            self._total_blocks_moved += moved
            self._total_defrags += 1
            self._last_defrag_time = time.time()

        return moved

    def defragment(self, blocks: List[MemoryBlock]) -> int:
        """
        执行碎片整理

        Args:
            blocks: 内存块列表

        Returns:
            int: 移动的块数量
        """
        with self._state_lock:
            if self._state == DefragState.EXECUTING:
                logger.warning("Defragmentation already in progress")
                return 0
            self._state = DefragState.ANALYZING

        try:
            analysis = self.analyze(blocks)

            if analysis['fragmentation_ratio'] < self.config.threshold:
                with self._state_lock:
                    self._state = DefragState.COMPLETED
                return 0

            with self._state_lock:
                self._state = DefragState.PLANNING
            plan = self.plan(blocks)

            if not plan:
                with self._state_lock:
                    self._state = DefragState.COMPLETED
                return 0

            with self._state_lock:
                self._state = DefragState.EXECUTING
            moved = self.execute(plan)

            with self._state_lock:
                self._state = DefragState.COMPLETED
            return moved

        except Exception as e:
            with self._state_lock:
                self._state = DefragState.FAILED
            logger.error(f"Defragmentation failed: {e}")
            raise DefragmentationError(str(e))

    def get_state(self) -> DefragState:
        """
        获取当前状态

        Returns:
            DefragState: 当前状态
        """
        with self._state_lock:
            return self._state

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        with self._state_lock:
            return {
                'state': self._state.value,
                'total_defrags': self._total_defrags,
                'total_blocks_moved': self._total_blocks_moved,
                'last_defrag_time': self._last_defrag_time,
                'config': {
                    'policy': self.config.policy.value,
                    'threshold': self.config.threshold,
                    'target_fragmentation': self.config.target_fragmentation
                }
            }

    def stop(self):
        """停止碎片整理器"""
        self._stop_event.set()

        if self._background_thread:
            self._background_thread.join(timeout=5.0)

    def force_defrag(self, blocks: List[MemoryBlock]) -> DefragResult:
        """
        强制碎片整理

        Args:
            blocks: 内存块列表

        Returns:
            DefragResult: 整理结果
        """
        start_time = time.time()

        analysis_before = self.analyze(blocks)
        frag_before = analysis_before['fragmentation_ratio']

        try:
            moved = self.defragment(blocks)

            analysis_after = self.analyze(blocks)
            frag_after = analysis_after['fragmentation_ratio']

            return DefragResult(
                success=True,
                blocks_moved=moved,
                time_elapsed=time.time() - start_time,
                fragmentation_before=frag_before,
                fragmentation_after=frag_after
            )

        except Exception as e:
            return DefragResult(
                success=False,
                blocks_moved=0,
                time_elapsed=time.time() - start_time,
                fragmentation_before=frag_before,
                fragmentation_after=frag_before,
                error_message=str(e)
            )
