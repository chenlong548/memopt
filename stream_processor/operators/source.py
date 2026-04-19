"""
stream_processor 数据源操作符

定义数据源操作符。
"""

from typing import Any, Dict, List, Optional, Callable, Iterator
import time
import threading
from abc import abstractmethod

from .base import Operator, OperatorConfig, OperatorType, OperatorState
from ..core.record import Record
from ..core.execution_context import ExecutionContext
from ..core.exceptions import SourceError


class SourceOperator(Operator):
    """
    数据源操作符基类

    从外部系统读取数据。
    """

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def read(self) -> Iterator[Record]:
        """
        读取数据

        Returns:
            Iterator[Record]: 记录迭代器
        """
        pass

    def process(self, record: Record) -> List[Record]:
        """源操作符不处理输入"""
        return []

    def open(self, context: ExecutionContext):
        """打开数据源"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        self._running = True

    def close(self):
        """关闭数据源"""
        self._running = False
        self.set_state(OperatorState.COMPLETED)

    def run(self):
        """运行数据源"""
        try:
            for record in self.read():
                if not self._running:
                    break

                for downstream in self.get_downstream_operators():
                    downstream.process(record)

        except Exception as e:
            self.set_state(OperatorState.FAILED)
            raise SourceError(f"Source operator failed: {e}") from e


class CollectionSource(SourceOperator):
    """
    集合数据源

    从内存集合读取数据。
    """

    def __init__(self,
                 name: str,
                 data: List[Any],
                 parallelism: int = 1):
        """
        初始化集合数据源

        Args:
            name: 操作符名称
            data: 数据集合
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._data = data

    def read(self) -> Iterator[Record]:
        """读取数据"""
        for item in self._data:
            yield Record(value=item)


class IteratorSource(SourceOperator):
    """
    迭代器数据源

    从迭代器读取数据。
    """

    def __init__(self,
                 name: str,
                 iterator: Iterator[Any],
                 parallelism: int = 1):
        """
        初始化迭代器数据源

        Args:
            name: 操作符名称
            iterator: 数据迭代器
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._iterator = iterator

    def read(self) -> Iterator[Record]:
        """读取数据"""
        for item in self._iterator:
            yield Record(value=item)


class FunctionSource(SourceOperator):
    """
    函数数据源

    通过函数生成数据。
    """

    def __init__(self,
                 name: str,
                 source_func: Callable[[], Iterator[Any]],
                 parallelism: int = 1):
        """
        初始化函数数据源

        Args:
            name: 操作符名称
            source_func: 数据生成函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._source_func = source_func

    def read(self) -> Iterator[Record]:
        """读取数据"""
        for item in self._source_func():
            yield Record(value=item)


class FileSource(SourceOperator):
    """
    文件数据源

    从文件读取数据。
    """

    def __init__(self,
                 name: str,
                 file_path: str,
                 parallelism: int = 1,
                 encoding: str = 'utf-8'):
        """
        初始化文件数据源

        Args:
            name: 操作符名称
            file_path: 文件路径
            parallelism: 并行度
            encoding: 文件编码
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._file_path = file_path
        self._encoding = encoding
        self._file = None

    def read(self) -> Iterator[Record]:
        """读取数据"""
        try:
            with open(self._file_path, 'r', encoding=self._encoding) as f:
                for line in f:
                    if not self._running:
                        break
                    yield Record(value=line.strip())
        except Exception as e:
            raise SourceError(f"Failed to read file {self._file_path}: {e}") from e

    def close(self):
        """关闭文件"""
        if self._file:
            self._file.close()
        super().close()


class SocketSource(SourceOperator):
    """
    Socket数据源

    从Socket读取数据。
    """

    def __init__(self,
                 name: str,
                 host: str,
                 port: int,
                 parallelism: int = 1,
                 delimiter: str = '\n'):
        """
        初始化Socket数据源

        Args:
            name: 操作符名称
            host: 主机地址
            port: 端口号
            parallelism: 并行度
            delimiter: 分隔符
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._host = host
        self._port = port
        self._delimiter = delimiter
        self._socket = None

    def read(self) -> Iterator[Record]:
        """读取数据"""
        import socket

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))

            buffer = ""
            while self._running:
                data = self._socket.recv(4096)
                if not data:
                    break

                buffer += data.decode('utf-8')

                while self._delimiter in buffer:
                    line, buffer = buffer.split(self._delimiter, 1)
                    if line:
                        yield Record(value=line)

        except Exception as e:
            raise SourceError(f"Socket source failed: {e}") from e

    def close(self):
        """关闭Socket"""
        if self._socket:
            self._socket.close()
        super().close()


class KafkaSource(SourceOperator):
    """
    Kafka数据源

    从Kafka读取数据。
    """

    def __init__(self,
                 name: str,
                 topic: str,
                 bootstrap_servers: str,
                 group_id: str,
                 parallelism: int = 1):
        """
        初始化Kafka数据源

        Args:
            name: 操作符名称
            topic: 主题名称
            bootstrap_servers: Kafka服务器地址
            group_id: 消费者组ID
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SOURCE,
            parallelism=parallelism
        )
        super().__init__(config)
        self._topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._consumer = None

    def read(self) -> Iterator[Record]:
        """读取数据"""
        try:
            from kafka import KafkaConsumer

            self._consumer = KafkaConsumer(
                self._topic,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )

            for message in self._consumer:
                if not self._running:
                    break

                yield Record(
                    value=message.value,
                    key=message.key,
                    timestamp=message.timestamp / 1000.0,
                    partition=message.partition,
                    offset=message.offset
                )

        except ImportError:
            raise SourceError("kafka-python package is required for KafkaSource")
        except Exception as e:
            raise SourceError(f"Kafka source failed: {e}") from e

    def close(self):
        """关闭Kafka消费者"""
        if self._consumer:
            self._consumer.close()
        super().close()
