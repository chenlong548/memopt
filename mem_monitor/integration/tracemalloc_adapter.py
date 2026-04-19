"""
mem_monitor tracemalloc适配器模块

提供与Python tracemalloc模块的集成，追踪内存分配。
"""

import time
import os
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

# 配置模块日志记录器
logger = logging.getLogger(__name__)


@dataclass
class AllocationSnapshot:
    """
    分配快照

    记录某一时刻的内存分配状态。
    """

    timestamp: float                           # 时间戳
    total_size: int = 0                        # 总大小
    total_count: int = 0                       # 总数量

    # 按类型统计
    by_type: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # type -> (size, count)

    # 按文件统计
    by_file: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # file -> (size, count)

    # Top分配
    top_allocations: List[Tuple[str, int, int]] = field(default_factory=list)  # (trace, size, count)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'total_size': self.total_size,
            'total_count': self.total_count,
            'total_size_mb': self.total_size / (1024 * 1024),
            'type_count': len(self.by_type),
            'file_count': len(self.by_file),
        }


@dataclass
class AllocationFilter:
    """
    分配过滤器

    配置tracemalloc的过滤规则。
    """

    # 包含的文件模式
    include_patterns: List[str] = field(default_factory=list)

    # 排除的文件模式
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '<unknown>',
        '<frozen importlib._bootstrap>',
    ])

    # 最小大小
    min_size: int = 0

    # 最小数量
    min_count: int = 0

    def matches(self, traceback_str: str, size: int, count: int) -> bool:
        """
        检查是否匹配过滤条件

        Args:
            traceback_str: 回溯字符串
            size: 大小
            count: 数量

        Returns:
            bool: 是否匹配
        """
        # 检查大小
        if size < self.min_size:
            return False

        # 检查数量
        if count < self.min_count:
            return False

        # 检查排除模式
        for pattern in self.exclude_patterns:
            if pattern in traceback_str:
                return False

        # 检查包含模式
        if self.include_patterns:
            for pattern in self.include_patterns:
                if pattern in traceback_str:
                    return True
            return False

        return True


class TracemallocAdapter:
    """
    tracemalloc适配器

    提供与Python tracemalloc模块的集成。

    功能：
    1. 启动/停止内存追踪
    2. 获取内存快照
    3. 比较快照差异
    4. 分析分配热点
    """

    def __init__(self, nframe: int = 25):
        """
        初始化tracemalloc适配器

        Args:
            nframe: 追踪的栈帧数量
        """
        self._tracemalloc = None
        self._available = False
        self._nframe = nframe
        self._started = False

        # 快照历史
        self._snapshots: List[Any] = []
        self._last_snapshot: Optional[Any] = None

        # 过滤器
        self._filter = AllocationFilter()

        # 统计
        self._stats = {
            'snapshots_taken': 0,
            'total_traced': 0,
        }

        try:
            import tracemalloc
            self._tracemalloc = tracemalloc
            self._available = True
        except ImportError as e:
            logger.debug(f"tracemalloc not available: {e}")

    def is_available(self) -> bool:
        """检查tracemalloc是否可用"""
        return self._available

    def start(self) -> bool:
        """
        启动内存追踪

        Returns:
            bool: 是否成功启动
        """
        if not self._available:
            return False

        if self._started:
            return True

        try:
            if not self._tracemalloc.is_tracing():
                self._tracemalloc.start(self._nframe)

            self._started = True
            return True

        except Exception as e:
            logger.warning(f"Failed to start tracemalloc: {e}")
            return False

    def stop(self) -> bool:
        """
        停止内存追踪

        Returns:
            bool: 是否成功停止
        """
        if not self._available:
            return False

        try:
            if self._tracemalloc.is_tracing():
                self._tracemalloc.stop()

            self._started = False
            return True

        except Exception as e:
            logger.warning(f"Failed to stop tracemalloc: {e}")
            return False

    def is_tracing(self) -> bool:
        """检查是否正在追踪"""
        if not self._available:
            return False
        return self._tracemalloc.is_tracing()

    def take_snapshot(self) -> Optional[AllocationSnapshot]:
        """
        获取内存快照

        Returns:
            Optional[AllocationSnapshot]: 分配快照
        """
        if not self._available or not self._started:
            return None

        try:
            snapshot = self._tracemalloc.take_snapshot()
            self._last_snapshot = snapshot
            self._snapshots.append(snapshot)
            self._stats['snapshots_taken'] += 1

            # 限制快照数量
            if len(self._snapshots) > 100:
                self._snapshots = self._snapshots[-50:]

            return self._parse_snapshot(snapshot)

        except Exception as e:
            logger.debug(f"Failed to take snapshot: {e}")
            return None

    def _parse_snapshot(self, snapshot) -> AllocationSnapshot:
        """解析快照"""
        result = AllocationSnapshot(timestamp=time.time())

        # 获取统计信息
        stats = snapshot.statistics('traceback')

        for stat in stats:
            size = stat.size
            count = stat.count

            # 获取回溯字符串
            trace_str = str(stat.traceback) if stat.traceback else '<unknown>'

            # 应用过滤器
            if not self._filter.matches(trace_str, size, count):
                continue

            result.total_size += size
            result.total_count += count

            # 按文件统计
            if stat.traceback:
                filename = stat.traceback[0].filename
                if filename not in result.by_file:
                    result.by_file[filename] = (0, 0)
                old_size, old_count = result.by_file[filename]
                result.by_file[filename] = (old_size + size, old_count + count)

            # 添加到top分配
            result.top_allocations.append((trace_str, size, count))

        # 排序top分配
        result.top_allocations.sort(key=lambda x: x[1], reverse=True)
        result.top_allocations = result.top_allocations[:100]

        return result

    def compare_snapshots(self,
                         snapshot1: Optional[Any] = None,
                         snapshot2: Optional[Any] = None) -> List[Tuple[str, int, int]]:
        """
        比较两个快照

        Args:
            snapshot1: 第一个快照（默认为倒数第二个）
            snapshot2: 第二个快照（默认为最后一个）

        Returns:
            List[Tuple]: 差异列表 [(trace, size_diff, count_diff), ...]
        """
        if not self._available:
            return []

        try:
            if snapshot1 is None:
                if len(self._snapshots) < 2:
                    return []
                snapshot1 = self._snapshots[-2]

            if snapshot2 is None:
                if not self._snapshots:
                    return []
                snapshot2 = self._snapshots[-1]

            # 比较快照
            diff = snapshot2.compare_to(snapshot1, 'traceback')

            results = []
            for stat in diff[:50]:  # 只返回前50个差异
                trace_str = str(stat.traceback) if stat.traceback else '<unknown>'
                results.append((trace_str, stat.size_diff, stat.count_diff))

            return results

        except Exception as e:
            logger.debug(f"Failed to compare snapshots: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        if not self._available:
            return {'available': False}

        try:
            traced = self._tracemalloc.get_traced_memory()

            return {
                'available': True,
                'tracing': self.is_tracing(),
                'heap_size': traced[1],
                'heap_used': traced[0],
                'snapshots_taken': self._stats['snapshots_taken'],
                'nframe': self._nframe,
            }

        except Exception as e:
            logger.debug(f"Failed to get tracemalloc stats: {e}")
            return {'available': False}

    def get_top_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取top统计

        Args:
            limit: 返回数量

        Returns:
            List[Dict]: top统计列表
        """
        if not self._available or not self._last_snapshot:
            return []

        try:
            stats = self._last_snapshot.statistics('traceback')[:limit]

            results = []
            for stat in stats:
                trace_str = str(stat.traceback) if stat.traceback else '<unknown>'
                results.append({
                    'traceback': trace_str,
                    'size': stat.size,
                    'count': stat.count,
                    'size_mb': stat.size / (1024 * 1024),
                })

            return results

        except Exception as e:
            logger.debug(f"Failed to get top stats: {e}")
            return []

    def get_allocation_count(self) -> int:
        """获取分配数量"""
        if not self._available:
            return 0

        try:
            traced = self._tracemalloc.get_traced_memory()
            return traced[0]
        except Exception as e:
            logger.debug(f"Failed to get allocation count: {e}")
            return 0

    def get_allocation_size(self) -> int:
        """获取分配总大小"""
        if not self._available:
            return 0

        try:
            traced = self._tracemalloc.get_traced_memory()
            return traced[1]
        except Exception as e:
            logger.debug(f"Failed to get allocation size: {e}")
            return 0

    def set_filter(self, filter: AllocationFilter):
        """设置过滤器"""
        self._filter = filter

    def clear_traces(self):
        """清除追踪数据"""
        if not self._available:
            return

        try:
            # tracemalloc没有直接清除的方法
            # 可以通过停止再启动来清除
            if self._started:
                self._tracemalloc.stop()
                self._tracemalloc.start(self._nframe)

            self._snapshots.clear()
            self._last_snapshot = None

        except Exception as e:
            logger.warning(f"Failed to clear traces: {e}")

    def get_traceback(self, obj: Any) -> Optional[List[str]]:
        """
        获取对象的分配回溯

        Args:
            obj: Python对象

        Returns:
            Optional[List[str]]: 回溯列表
        """
        if not self._available:
            return None

        try:
            # 获取对象的分配回溯
            traces = self._tracemalloc.get_object_traceback(obj)

            if traces:
                return [str(frame) for frame in traces]

            return None

        except Exception as e:
            logger.debug(f"Failed to get traceback: {e}")
            return None
