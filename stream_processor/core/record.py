"""
stream_processor 数据记录

定义流处理中的数据记录结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import hashlib


@dataclass
class Record:
    """
    数据记录

    表示流处理中的单条数据记录。
    """

    value: Any

    key: Optional[str] = None

    timestamp: float = field(default_factory=time.time)

    headers: Dict[str, Any] = field(default_factory=dict)

    partition: Optional[int] = None

    offset: Optional[int] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = time.time()

    def get_key(self) -> str:
        """
        获取记录键

        Returns:
            str: 记录键，如果未设置则生成基于值的哈希键
        """
        if self.key is not None:
            return self.key

        value_str = str(self.value)
        return hashlib.md5(value_str.encode()).hexdigest()

    def get_timestamp_ms(self) -> int:
        """
        获取时间戳（毫秒）

        Returns:
            int: 毫秒时间戳
        """
        return int(self.timestamp * 1000)

    def is_late(self, watermark: float) -> bool:
        """
        判断是否为迟到数据

        Args:
            watermark: 当前watermark时间戳

        Returns:
            bool: 是否迟到
        """
        return self.timestamp < watermark

    def with_value(self, new_value: Any) -> 'Record':
        """
        创建具有新值的记录副本

        Args:
            new_value: 新值

        Returns:
            Record: 新记录
        """
        return Record(
            value=new_value,
            key=self.key,
            timestamp=self.timestamp,
            headers=self.headers.copy(),
            partition=self.partition,
            offset=self.offset
        )

    def with_timestamp(self, new_timestamp: float) -> 'Record':
        """
        创建具有新时间戳的记录副本

        Args:
            new_timestamp: 新时间戳

        Returns:
            Record: 新记录
        """
        return Record(
            value=self.value,
            key=self.key,
            timestamp=new_timestamp,
            headers=self.headers.copy(),
            partition=self.partition,
            offset=self.offset
        )

    def add_header(self, key: str, value: Any) -> 'Record':
        """
        添加头部信息

        Args:
            key: 头部键
            value: 头部值

        Returns:
            Record: 当前记录
        """
        self.headers[key] = value
        return self

    def get_header(self, key: str, default: Any = None) -> Any:
        """
        获取头部信息

        Args:
            key: 头部键
            default: 默认值

        Returns:
            Any: 头部值
        """
        return self.headers.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            Dict: 字典表示
        """
        return {
            'value': self.value,
            'key': self.key,
            'timestamp': self.timestamp,
            'headers': self.headers,
            'partition': self.partition,
            'offset': self.offset
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Record':
        """
        从字典创建记录

        Args:
            data: 字典数据

        Returns:
            Record: 记录实例
        """
        return cls(
            value=data['value'],
            key=data.get('key'),
            timestamp=data.get('timestamp', time.time()),
            headers=data.get('headers', {}),
            partition=data.get('partition'),
            offset=data.get('offset')
        )

    def __repr__(self) -> str:
        return f"Record(key={self.key}, timestamp={self.timestamp}, value={self.value})"
