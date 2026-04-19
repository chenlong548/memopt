"""
stream_processor 计数窗口

实现基于元素数量的窗口。
"""

from typing import List
from dataclasses import dataclass

from .base import WindowAssigner, Window, WindowSerializer
from ..core.record import Record
from .trigger import CountTrigger, Trigger


@dataclass
class CountWindow(Window):
    """
    计数窗口

    基于元素数量的窗口。
    """

    count: int = 0

    max_count: int = 0

    def is_full(self) -> bool:
        """窗口是否已满"""
        return self.count >= self.max_count


class CountWindowAssigner(WindowAssigner):
    """
    计数窗口分配器

    根据元素数量分配窗口。
    """

    def __init__(self, count: int):
        """
        初始化计数窗口分配器

        Args:
            count: 窗口大小（元素数量）
        """
        if count <= 0:
            raise ValueError(f"Window count must be positive, got {count}")

        self._count = count
        self._window_counter = 0

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        import time

        window_index = self._window_counter // self._count

        window_start = float(window_index * self._count)
        window_end = float((window_index + 1) * self._count)

        self._window_counter += 1

        return [Window(start=window_start, end=window_end)]

    def get_default_trigger(self) -> Trigger:
        """获取默认触发器"""
        return CountTrigger(count=self._count)

    def get_window_serializer(self) -> WindowSerializer:
        """获取窗口序列化器"""
        return CountWindowSerializer()

    def get_count(self) -> int:
        """
        获取窗口大小

        Returns:
            int: 窗口大小（元素数量）
        """
        return self._count

    def reset(self):
        """重置窗口计数器"""
        self._window_counter = 0


class SlidingCountWindowAssigner(WindowAssigner):
    """
    滑动计数窗口分配器

    实现基于元素数量的滑动窗口。
    """

    def __init__(self, count: int, slide: int):
        """
        初始化滑动计数窗口分配器

        Args:
            count: 窗口大小（元素数量）
            slide: 滑动步长（元素数量）
        """
        if count <= 0:
            raise ValueError(f"Window count must be positive, got {count}")
        if slide <= 0:
            raise ValueError(f"Slide count must be positive, got {slide}")
        if slide > count:
            raise ValueError(f"Slide ({slide}) cannot be larger than window size ({count})")

        self._count = count
        self._slide = slide
        self._element_counter = 0

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        windows = []

        window_index = self._element_counter // self._slide

        for i in range(self._count // self._slide):
            idx = window_index - i
            if idx >= 0:
                window_start = float(idx * self._slide)
                window_end = float(idx * self._slide + self._count)
                windows.append(Window(start=window_start, end=window_end))

        self._element_counter += 1

        return windows

    def get_default_trigger(self) -> Trigger:
        """获取默认触发器"""
        return CountTrigger(count=self._count)

    def get_window_serializer(self) -> WindowSerializer:
        """获取窗口序列化器"""
        return CountWindowSerializer()

    def get_count(self) -> int:
        """
        获取窗口大小

        Returns:
            int: 窗口大小（元素数量）
        """
        return self._count

    def get_slide(self) -> int:
        """
        获取滑动步长

        Returns:
            int: 滑动步长（元素数量）
        """
        return self._slide

    def reset(self):
        """重置元素计数器"""
        self._element_counter = 0


class CountWindowSerializer(WindowSerializer):
    """
    计数窗口序列化器

    序列化和反序列化计数窗口。
    """

    def serialize(self, window: Window) -> bytes:
        """序列化窗口"""
        import struct
        return struct.pack('dd', window.start, window.end)

    def deserialize(self, data: bytes) -> Window:
        """反序列化窗口"""
        import struct
        start, end = struct.unpack('dd', data)
        return Window(start=start, end=end)


class GlobalWindowAssigner(WindowAssigner):
    """
    全局窗口分配器

    将所有记录分配到同一个全局窗口。
    """

    def __init__(self):
        """初始化全局窗口分配器"""
        self._global_window = Window(start=0.0, end=float('inf'))

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        return [self._global_window]

    def get_default_trigger(self) -> Trigger:
        """获取默认触发器"""
        from .trigger import NeverTrigger
        return NeverTrigger()

    def get_window_serializer(self) -> WindowSerializer:
        """获取窗口序列化器"""
        return CountWindowSerializer()
