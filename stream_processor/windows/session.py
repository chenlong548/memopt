"""
stream_processor 会话窗口

实现基于活动间隔的窗口。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from .base import WindowAssigner, Window, TimeWindowSerializer, WindowSerializer
from ..core.record import Record
from .trigger import EventTimeTrigger, Trigger


@dataclass
class SessionWindow(Window):
    """
    会话窗口

    动态调整大小的窗口。
    """

    last_activity_time: float = 0.0

    def update_last_activity(self, timestamp: float):
        """更新最后活动时间"""
        self.last_activity_time = timestamp

    def merge_with(self, other: 'SessionWindow') -> 'SessionWindow':
        """
        合并两个会话窗口

        Args:
            other: 另一个会话窗口

        Returns:
            SessionWindow: 合并后的窗口
        """
        return SessionWindow(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            last_activity_time=max(self.last_activity_time, other.last_activity_time)
        )


class SessionWindowAssigner(WindowAssigner):
    """
    会话窗口分配器

    根据活动间隔动态分配窗口。
    """

    def __init__(self, gap: float):
        """
        初始化会话窗口分配器

        Args:
            gap: 会话间隔（秒）
        """
        if gap <= 0:
            raise ValueError(f"Session gap must be positive, got {gap}")

        self._gap = gap
        self._session_windows: Dict[str, List[SessionWindow]] = {}

    def assign_windows(self, record: Record) -> List[Window]:
        """为记录分配窗口"""
        key = record.key or record.get_key()
        timestamp = record.timestamp

        if key not in self._session_windows:
            self._session_windows[key] = []

        windows = self._session_windows[key]

        new_window = SessionWindow(
            start=timestamp,
            end=timestamp,
            last_activity_time=timestamp
        )

        windows_to_merge = []
        for i, window in enumerate(windows):
            if self._should_merge(new_window, window):
                windows_to_merge.append(i)

        if windows_to_merge:
            merged_window = new_window
            for i in reversed(windows_to_merge):
                merged_window = merged_window.merge_with(windows[i])
                windows.pop(i)

            windows.append(merged_window)
            return [Window(start=merged_window.start, end=merged_window.end)]
        else:
            windows.append(new_window)
            return [Window(start=new_window.start, end=new_window.end)]

    def _should_merge(self, window1: SessionWindow, window2: SessionWindow) -> bool:
        """
        判断两个窗口是否应该合并

        Args:
            window1: 第一个窗口
            window2: 第二个窗口

        Returns:
            bool: 是否应该合并
        """
        gap_start = min(window1.last_activity_time, window2.last_activity_time)
        gap_end = max(window1.last_activity_time, window2.last_activity_time)

        return (gap_end - gap_start) <= self._gap

    def get_default_trigger(self) -> Trigger:
        """获取默认触发器"""
        return EventTimeTrigger()

    def get_window_serializer(self) -> WindowSerializer:
        """获取窗口序列化器"""
        return TimeWindowSerializer()

    def get_gap(self) -> float:
        """
        获取会话间隔

        Returns:
            float: 会话间隔（秒）
        """
        return self._gap

    def cleanup_expired_sessions(self, watermark: float):
        """
        清理过期的会话

        Args:
            watermark: 当前watermark
        """
        for key in list(self._session_windows.keys()):
            windows = self._session_windows[key]
            self._session_windows[key] = [
                w for w in windows
                if w.end > watermark - self._gap
            ]

            if not self._session_windows[key]:
                del self._session_windows[key]


class DynamicSessionWindowAssigner(SessionWindowAssigner):
    """
    动态会话窗口分配器

    支持动态调整会话间隔。
    """

    def __init__(self,
                 initial_gap: float,
                 min_gap: float = 1.0,
                 max_gap: float = 300.0):
        """
        初始化动态会话窗口分配器

        Args:
            initial_gap: 初始会话间隔（秒）
            min_gap: 最小会话间隔（秒）
            max_gap: 最大会话间隔（秒）
        """
        super().__init__(initial_gap)
        self._min_gap = min_gap
        self._max_gap = max_gap
        self._current_gap = initial_gap

    def adjust_gap(self, activity_rate: float):
        """
        根据活动率调整会话间隔

        Args:
            activity_rate: 活动率（记录/秒）
        """
        if activity_rate > 10:
            self._current_gap = max(self._min_gap, self._gap * 0.5)
        elif activity_rate < 1:
            self._current_gap = min(self._max_gap, self._gap * 2.0)
        else:
            self._current_gap = self._gap

    def get_current_gap(self) -> float:
        """
        获取当前会话间隔

        Returns:
            float: 当前会话间隔（秒）
        """
        return self._current_gap
