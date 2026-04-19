"""
stream_processor 滚动窗口

实现固定大小、不重叠的窗口。
"""

from typing import List
import math

from .base import WindowAssigner, Window, TimeWindowSerializer, WindowSerializer
from ..core.record import Record
from .trigger import EventTimeTrigger, Trigger


class TumblingWindowAssigner(WindowAssigner):
    """
    滚动窗口分配器

    将记录分配到固定大小、不重叠的窗口中。
    """

    def __init__(self, size: float, offset: float = 0.0):
        """
        初始化滚动窗口分配器

        Args:
            size: 窗口大小（秒）
            offset: 窗口偏移量（秒）
        """
        if size <= 0:
            raise ValueError(f"Window size must be positive, got {size}")

        self._size = size
        self._offset = offset

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        timestamp = record.timestamp

        window_start = self._get_window_start(timestamp)

        window_end = window_start + self._size

        return [Window(start=window_start, end=window_end)]

    def _get_window_start(self, timestamp: float) -> float:
        """
        计算窗口起始时间

        Args:
            timestamp: 时间戳

        Returns:
            float: 窗口起始时间
        """
        adjusted_ts = timestamp - self._offset

        window_index = math.floor(adjusted_ts / self._size)

        return window_index * self._size + self._offset

    def get_default_trigger(self) -> Trigger:
        """获取默认触发器"""
        return EventTimeTrigger()

    def get_window_serializer(self) -> WindowSerializer:
        """获取窗口序列化器"""
        return TimeWindowSerializer()

    def get_size(self) -> float:
        """
        获取窗口大小

        Returns:
            float: 窗口大小（秒）
        """
        return self._size

    def get_offset(self) -> float:
        """
        获取窗口偏移量

        Returns:
            float: 窗口偏移量（秒）
        """
        return self._offset


class TumblingEventTimeWindows(TumblingWindowAssigner):
    """
    基于事件时间的滚动窗口

    使用记录的事件时间分配窗口。
    """

    def __init__(self, size: float, offset: float = 0.0):
        """
        初始化基于事件时间的滚动窗口

        Args:
            size: 窗口大小（秒）
            offset: 窗口偏移量（秒）
        """
        super().__init__(size, offset)


class TumblingProcessingTimeWindows(TumblingWindowAssigner):
    """
    基于处理时间的滚动窗口

    使用系统处理时间分配窗口。
    """

    def __init__(self, size: float, offset: float = 0.0):
        """
        初始化基于处理时间的滚动窗口

        Args:
            size: 窗口大小（秒）
            offset: 窗口偏移量（秒）
        """
        super().__init__(size, offset)

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        import time
        timestamp = time.time()

        window_start = self._get_window_start(timestamp)

        window_end = window_start + self._size

        return [Window(start=window_start, end=window_end)]
