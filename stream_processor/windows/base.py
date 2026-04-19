"""
stream_processor 窗口基类

定义窗口的基础抽象。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any

from ..core.record import Record


@dataclass
class Window:
    """
    窗口

    表示一个时间窗口。
    """

    start: float

    end: float

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.start == other.start and self.end == other.end

    def contains(self, timestamp: float) -> bool:
        """
        判断时间戳是否在窗口内

        Args:
            timestamp: 时间戳

        Returns:
            bool: 是否在窗口内
        """
        return self.start <= timestamp < self.end

    def overlaps(self, other: 'Window') -> bool:
        """
        判断是否与另一个窗口重叠

        Args:
            other: 另一个窗口

        Returns:
            bool: 是否重叠
        """
        return self.start < other.end and self.end > other.start

    def size(self) -> float:
        """
        获取窗口大小

        Returns:
            float: 窗口大小（秒）
        """
        return self.end - self.start


class WindowAssigner(ABC):
    """
    窗口分配器基类

    定义窗口分配的标准接口。
    """

    @abstractmethod
    def assign_windows(self, record: Record) -> List[Window]:
        """
        为记录分配窗口

        Args:
            record: 数据记录

        Returns:
            List[Window]: 窗口列表
        """
        pass

    @abstractmethod
    def get_default_trigger(self) -> 'Trigger':
        """
        获取默认触发器

        Returns:
            Trigger: 默认触发器
        """
        pass

    @abstractmethod
    def get_window_serializer(self) -> 'WindowSerializer':
        """
        获取窗口序列化器

        Returns:
            WindowSerializer: 窗口序列化器
        """
        pass


class Trigger(ABC):
    """
    触发器基类

    定义窗口触发的标准接口。
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def on_fire(self, records: List[Record]) -> List[Record]:
        """
        触发时的处理

        Args:
            records: 窗口中的记录

        Returns:
            List[Record]: 要输出的记录
        """
        pass

    @abstractmethod
    def reset(self):
        """重置触发器状态"""
        pass


class WindowSerializer(ABC):
    """
    窗口序列化器基类

    定义窗口序列化的标准接口。
    """

    @abstractmethod
    def serialize(self, window: Window) -> bytes:
        """
        序列化窗口

        Args:
            window: 窗口对象

        Returns:
            bytes: 序列化后的数据
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Window:
        """
        反序列化窗口

        Args:
            data: 序列化数据

        Returns:
            Window: 窗口对象
        """
        pass


class TimeWindowSerializer(WindowSerializer):
    """
    时间窗口序列化器

    序列化和反序列化时间窗口。
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


class WindowFunction(ABC):
    """
    窗口函数基类

    定义窗口处理的标准接口。
    """

    @abstractmethod
    def apply(self, records: List[Record], window: Window) -> List[Record]:
        """
        应用窗口函数

        Args:
            records: 窗口中的记录
            window: 窗口对象

        Returns:
            List[Record]: 处理后的记录
        """
        pass


class ReduceWindowFunction(WindowFunction):
    """
    归约窗口函数

    对窗口中的记录进行归约。
    """

    def __init__(self, reduce_func):
        """
        初始化归约窗口函数

        Args:
            reduce_func: 归约函数
        """
        self._reduce_func = reduce_func

    def apply(self, records: List[Record], window: Window) -> List[Record]:
        """应用窗口函数"""
        if not records:
            return []

        result = records[0].value
        for record in records[1:]:
            result = self._reduce_func(result, record.value)

        return [Record(value=result)]


class AggregateWindowFunction(WindowFunction):
    """
    聚合窗口函数

    对窗口中的记录进行聚合。
    """

    def __init__(self, aggregate_func, initial_value=None):
        """
        初始化聚合窗口函数

        Args:
            aggregate_func: 聚合函数
            initial_value: 初始值
        """
        self._aggregate_func = aggregate_func
        self._initial_value = initial_value

    def apply(self, records: List[Record], window: Window) -> List[Record]:
        """应用窗口函数"""
        if not records:
            if self._initial_value is not None:
                return [Record(value=self._initial_value)]
            return []

        result = self._initial_value
        for record in records:
            result = self._aggregate_func(result, record.value)

        return [Record(value=result)]
