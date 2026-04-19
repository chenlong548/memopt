"""
stream_processor 滑动窗口

实现固定大小、可重叠的窗口。
"""

from typing import List
import math

from .base import WindowAssigner, Window, TimeWindowSerializer, WindowSerializer
from ..core.record import Record
from .trigger import EventTimeTrigger, Trigger


class SlidingWindowAssigner(WindowAssigner):
    """
    滑动窗口分配器

    将记录分配到固定大小、可重叠的窗口中。
    """

    def __init__(self, size: float, slide: float, offset: float = 0.0):
        """
        初始化滑动窗口分配器

        Args:
            size: 窗口大小（秒）
            slide: 滑动步长（秒）
            offset: 窗口偏移量（秒）
        """
        if size <= 0:
            raise ValueError(f"Window size must be positive, got {size}")
        if slide <= 0:
            raise ValueError(f"Slide size must be positive, got {slide}")
        if slide > size:
            raise ValueError(f"Slide size ({slide}) cannot be larger than window size ({size})")

        self._size = size
        self._slide = slide
        self._offset = offset

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        timestamp = record.timestamp

        windows = []

        last_window_start = self._get_last_window_start(timestamp)

        window_start = last_window_start

        while window_start > timestamp - self._size:
            window_end = window_start + self._size

            if window_start <= timestamp < window_end:
                windows.append(Window(start=window_start, end=window_end))

            window_start -= self._slide

        return windows

    def _get_last_window_start(self, timestamp: float) -> float:
        """
        获取最后一个窗口的起始时间

        Args:
            timestamp: 时间戳

        Returns:
            float: 窗口起始时间
        """
        adjusted_ts = timestamp - self._offset

        window_index = math.ceil(adjusted_ts / self._slide)

        return window_index * self._slide + self._offset

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

    def get_slide(self) -> float:
        """
        获取滑动步长

        Returns:
            float: 滑动步长（秒）
        """
        return self._slide

    def get_offset(self) -> float:
        """
        获取窗口偏移量

        Returns:
            float: 窗口偏移量（秒）
        """
        return self._offset


class SlidingEventTimeWindows(SlidingWindowAssigner):
    """
    基于事件时间的滑动窗口

    使用记录的事件时间分配窗口。
    """

    def __init__(self, size: float, slide: float, offset: float = 0.0):
        """
        初始化基于事件时间的滑动窗口

        Args:
            size: 窗口大小（秒）
            slide: 滑动步长（秒）
            offset: 窗口偏移量（秒）
        """
        super().__init__(size, slide, offset)


class SlidingProcessingTimeWindows(SlidingWindowAssigner):
    """
    基于处理时间的滑动窗口

    使用系统处理时间分配窗口。
    """

    def __init__(self, size: float, slide: float, offset: float = 0.0):
        """
        初始化基于处理时间的滑动窗口

        Args:
            size: 窗口大小（秒）
            slide: 滑动步长（秒）
            offset: 窗口偏移量（秒）
        """
        super().__init__(size, slide, offset)

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        import time
        timestamp = time.time()

        windows = []

        last_window_start = self._get_last_window_start(timestamp)

        window_start = last_window_start

        while window_start > timestamp - self._size:
            window_end = window_start + self._size

            if window_start <= timestamp < window_end:
                windows.append(Window(start=window_start, end=window_end))

            window_start -= self._slide

        return windows
