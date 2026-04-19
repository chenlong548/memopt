"""
stream_processor 窗口触发器

定义窗口触发的机制。
"""

from typing import List, Any
from dataclasses import dataclass

from .base import Trigger
from ..core.record import Record
from ..core.watermark import Watermark


@dataclass
class TriggerResult:
    """
    触发结果

    表示触发器的判断结果。
    """

    fire: bool = False

    purge: bool = False

    continue_processing: bool = True


class EventTimeTrigger(Trigger):
    """
    事件时间触发器

    当watermark超过窗口结束时间时触发。
    """

    def __init__(self):
        """初始化事件时间触发器"""
        self._fired_windows = set()

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        if not records:
            return False

        if isinstance(watermark, Watermark):
            window_end = max(r.timestamp for r in records)
            return watermark.timestamp >= window_end

        return False

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return records

    def reset(self):
        """重置触发器状态"""
        self._fired_windows.clear()


class ProcessingTimeTrigger(Trigger):
    """
    处理时间触发器

    基于系统处理时间触发。
    """

    def __init__(self, interval: float = 1.0):
        """
        初始化处理时间触发器

        Args:
            interval: 触发间隔（秒）
        """
        self._interval = interval
        self._last_trigger_time = 0.0

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        import time
        current_time = time.time()

        if current_time - self._last_trigger_time >= self._interval:
            self._last_trigger_time = current_time
            return True

        return False

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return records

    def reset(self):
        """重置触发器状态"""
        self._last_trigger_time = 0.0


class CountTrigger(Trigger):
    """
    计数触发器

    当窗口中的元素数量达到阈值时触发。
    """

    def __init__(self, count: int):
        """
        初始化计数触发器

        Args:
            count: 触发阈值
        """
        if count <= 0:
            raise ValueError(f"Trigger count must be positive, got {count}")

        self._count = count

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        return len(records) >= self._count

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return records

    def reset(self):
        """重置触发器状态"""
        pass


class ContinuousEventTimeTrigger(Trigger):
    """
    连续事件时间触发器

    在watermark超过窗口结束时间后，以固定间隔持续触发。
    """

    def __init__(self, interval: float):
        """
        初始化连续事件时间触发器

        Args:
            interval: 触发间隔（秒）
        """
        self._interval = interval
        self._last_fire_time = float('-inf')

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        if not isinstance(watermark, Watermark):
            return False

        if watermark.timestamp >= self._last_fire_time + self._interval:
            self._last_fire_time = watermark.timestamp
            return True

        return False

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return records

    def reset(self):
        """重置触发器状态"""
        self._last_fire_time = float('-inf')


class NeverTrigger(Trigger):
    """
    永不触发器

    永远不会触发的触发器。
    """

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        return False

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return []

    def reset(self):
        """重置触发器状态"""
        pass


class PurgingTrigger(Trigger):
    """
    清除触发器

    触发后清除窗口状态。
    """

    def __init__(self, nested_trigger: Trigger):
        """
        初始化清除触发器

        Args:
            nested_trigger: 嵌套触发器
        """
        self._nested_trigger = nested_trigger

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        return self._nested_trigger.should_fire(records, new_record, watermark)

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        result = self._nested_trigger.on_fire(records)
        return result

    def reset(self):
        """重置触发器状态"""
        self._nested_trigger.reset()


class EarlyFiringTrigger(Trigger):
    """
    提前触发器

    在窗口结束前提前触发。
    """

    def __init__(self,
                 nested_trigger: Trigger,
                 early_fire_interval: float):
        """
        初始化提前触发器

        Args:
            nested_trigger: 嵌套触发器
            early_fire_interval: 提前触发间隔（秒）
        """
        self._nested_trigger = nested_trigger
        self._early_fire_interval = early_fire_interval
        self._last_early_fire = float('-inf')

    def should_fire(self,
                    records: List[Record],
                    new_record: Record,
                    watermark: Any) -> bool:
        """
        判断是否应该触发

        Args:
            records: 窗口中的记录
            new_record: 新到达的记录
            watermark: 当前watermark

        Returns:
            bool: 是否应该触发
        """
        if self._nested_trigger.should_fire(records, new_record, watermark):
            return True

        if isinstance(watermark, Watermark):
            if watermark.timestamp >= self._last_early_fire + self._early_fire_interval:
                self._last_early_fire = watermark.timestamp
                return True

        return False

    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        return self._nested_trigger.on_fire(records)

    def reset(self):
        """重置触发器状态"""
        self._nested_trigger.reset()
        self._last_early_fire = float('-inf')
