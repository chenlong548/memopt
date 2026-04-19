"""
stream_processor 流定义

定义流处理的核心抽象。
"""

from typing import Any, Callable, Optional, List, Iterator
from dataclasses import dataclass, field
from enum import Enum
import time

from .record import Record
from .watermark import Watermark


class StreamType(Enum):
    """流类型"""
    DATA = "data"
    WATERMARK = "watermark"
    CONTROL = "control"


@dataclass
class StreamRecord:
    """
    流记录

    封装数据记录和watermark。
    """

    record: Optional[Record] = None

    watermark: Optional[Watermark] = None

    stream_type: StreamType = StreamType.DATA

    timestamp: float = field(default_factory=time.time)

    def is_data(self) -> bool:
        """是否为数据记录"""
        return self.stream_type == StreamType.DATA

    def is_watermark(self) -> bool:
        """是否为watermark"""
        return self.stream_type == StreamType.WATERMARK

    def is_control(self) -> bool:
        """是否为控制记录"""
        return self.stream_type == StreamType.CONTROL

    @classmethod
    def data(cls, record: Record) -> 'StreamRecord':
        """创建数据流记录"""
        return cls(
            record=record,
            stream_type=StreamType.DATA,
            timestamp=record.timestamp
        )

    @classmethod
    def watermark_record(cls, watermark: Watermark) -> 'StreamRecord':
        """创建watermark流记录"""
        return cls(
            watermark=watermark,
            stream_type=StreamType.WATERMARK,
            timestamp=watermark.timestamp
        )

    @classmethod
    def control(cls, control_type: str, data: Any = None) -> 'StreamRecord':
        """创建控制流记录"""
        return cls(
            record=Record(value={'type': control_type, 'data': data}),
            stream_type=StreamType.CONTROL
        )


class Stream:
    """
    流

    表示数据流的核心抽象。
    """

    def __init__(self, name: str, parallelism: int = 1):
        """
        初始化流

        Args:
            name: 流名称
            parallelism: 并行度
        """
        self.name = name
        self.parallelism = parallelism
        self._records: List[StreamRecord] = []
        self._closed = False

    def emit(self, record: Record) -> 'Stream':
        """
        发射记录到流

        Args:
            record: 数据记录

        Returns:
            Stream: 当前流
        """
        if self._closed:
            raise RuntimeError(f"Stream {self.name} is closed")

        stream_record = StreamRecord.data(record)
        self._records.append(stream_record)
        return self

    def emit_watermark(self, watermark: Watermark) -> 'Stream':
        """
        发射watermark到流

        Args:
            watermark: watermark标记

        Returns:
            Stream: 当前流
        """
        if self._closed:
            raise RuntimeError(f"Stream {self.name} is closed")

        stream_record = StreamRecord.watermark_record(watermark)
        self._records.append(stream_record)
        return self

    def emit_control(self, control_type: str, data: Any = None) -> 'Stream':
        """
        发射控制记录到流

        Args:
            control_type: 控制类型
            data: 控制数据

        Returns:
            Stream: 当前流
        """
        if self._closed:
            raise RuntimeError(f"Stream {self.name} is closed")

        stream_record = StreamRecord.control(control_type, data)
        self._records.append(stream_record)
        return self

    def __iter__(self) -> Iterator[StreamRecord]:
        """迭代流记录"""
        return iter(self._records)

    def __len__(self) -> int:
        """获取流记录数量"""
        return len(self._records)

    def close(self):
        """关闭流"""
        self._closed = True

    def is_closed(self) -> bool:
        """是否已关闭"""
        return self._closed

    def clear(self):
        """清空流记录"""
        self._records.clear()

    def map(self, func: Callable[[Record], Record]) -> 'Stream':
        """
        映射转换

        Args:
            func: 转换函数

        Returns:
            Stream: 新流
        """
        new_stream = Stream(f"{self.name}_mapped", self.parallelism)

        for stream_record in self._records:
            if stream_record.is_data() and stream_record.record:
                new_record = func(stream_record.record)
                new_stream.emit(new_record)
            else:
                new_stream._records.append(stream_record)

        return new_stream

    def filter(self, predicate: Callable[[Record], bool]) -> 'Stream':
        """
        过滤

        Args:
            predicate: 过滤谓词

        Returns:
            Stream: 新流
        """
        new_stream = Stream(f"{self.name}_filtered", self.parallelism)

        for stream_record in self._records:
            if stream_record.is_data() and stream_record.record:
                if predicate(stream_record.record):
                    new_stream.emit(stream_record.record)
            else:
                new_stream._records.append(stream_record)

        return new_stream

    def flat_map(self, func: Callable[[Record], List[Record]]) -> 'Stream':
        """
        扁平映射

        Args:
            func: 转换函数

        Returns:
            Stream: 新流
        """
        new_stream = Stream(f"{self.name}_flat_mapped", self.parallelism)

        for stream_record in self._records:
            if stream_record.is_data() and stream_record.record:
                new_records = func(stream_record.record)
                for new_record in new_records:
                    new_stream.emit(new_record)
            else:
                new_stream._records.append(stream_record)

        return new_stream

    def key_by(self, key_extractor: Callable[[Record], str]) -> 'KeyedStream':
        """
        按键分组

        Args:
            key_extractor: 键提取函数

        Returns:
            KeyedStream: 键控流
        """
        return KeyedStream(self, key_extractor)


class KeyedStream(Stream):
    """
    键控流

    按键分组的流。
    """

    def __init__(self, parent: Stream, key_extractor: Callable[[Record], str]):
        """
        初始化键控流

        Args:
            parent: 父流
            key_extractor: 键提取函数
        """
        super().__init__(f"{parent.name}_keyed", parent.parallelism)
        self.parent = parent
        self.key_extractor = key_extractor
        self._keyed_records: dict[str, List[Record]] = {}

        self._partition_records()

    def _partition_records(self):
        """分区记录"""
        for stream_record in self.parent:
            if stream_record.is_data() and stream_record.record:
                key = self.key_extractor(stream_record.record)
                if key not in self._keyed_records:
                    self._keyed_records[key] = []
                self._keyed_records[key].append(stream_record.record)

    def get_keys(self) -> List[str]:
        """
        获取所有键

        Returns:
            List[str]: 键列表
        """
        return list(self._keyed_records.keys())

    def get_records_by_key(self, key: str) -> List[Record]:
        """
        获取指定键的记录

        Args:
            key: 键

        Returns:
            List[Record]: 记录列表
        """
        return self._keyed_records.get(key, [])
