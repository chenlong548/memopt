"""
stream_processor 输出操作符

定义数据输出操作符。
"""

from typing import Any, Callable, List, Optional
import time

from .base import OneInputOperator, OperatorConfig, OperatorType, OperatorState
from ..core.record import Record
from ..core.execution_context import ExecutionContext
from ..core.exceptions import SinkError


class SinkOperator(OneInputOperator):
    """
    输出操作符基类

    将数据输出到外部系统。
    """

    def __init__(self, config: OperatorConfig):
        super().__init__(config)

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        self.write(record)
        return []

    @classmethod
    def write(cls, record: Record):
        """
        写入记录

        Args:
            record: 要写入的记录
        """
        pass


class PrintSink(SinkOperator):
    """
    打印输出

    将记录打印到控制台。
    """

    def __init__(self,
                 name: str,
                 parallelism: int = 1,
                 prefix: str = ""):
        """
        初始化打印输出

        Args:
            name: 操作符名称
            parallelism: 并行度
            prefix: 打印前缀
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        self._prefix = prefix

    def write(self, record: Record):
        """写入记录"""
        if self._prefix:
            print(f"{self._prefix}{record.value}")
        else:
            print(record.value)

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class FileSink(SinkOperator):
    """
    文件输出

    将记录写入文件。
    """

    def __init__(self,
                 name: str,
                 file_path: str,
                 parallelism: int = 1,
                 mode: str = 'w',
                 encoding: str = 'utf-8'):
        """
        初始化文件输出

        Args:
            name: 操作符名称
            file_path: 文件路径
            parallelism: 并行度
            mode: 文件模式
            encoding: 文件编码
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        self._file_path = file_path
        self._mode = mode
        self._encoding = encoding
        self._file = None

    def write(self, record: Record):
        """写入记录"""
        if self._file:
            self._file.write(str(record.value) + '\n')

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        try:
            self._file = open(self._file_path, self._mode, encoding=self._encoding)
        except Exception as e:
            raise SinkError(f"Failed to open file {self._file_path}: {e}") from e

    def close(self):
        """关闭操作符"""
        if self._file:
            self._file.close()
        self.set_state(OperatorState.COMPLETED)


class FunctionSink(SinkOperator):
    """
    函数输出

    使用自定义函数处理记录。
    """

    def __init__(self,
                 name: str,
                 sink_func: Callable[[Record], None],
                 parallelism: int = 1):
        """
        初始化函数输出

        Args:
            name: 操作符名称
            sink_func: 输出函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        self._sink_func = sink_func

    def write(self, record: Record):
        """写入记录"""
        try:
            self._sink_func(record)
        except Exception as e:
            raise SinkError(f"Sink function failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class CollectionSink(SinkOperator):
    """
    集合输出

    将记录收集到内存集合。
    """

    def __init__(self,
                 name: str,
                 parallelism: int = 1):
        """
        初始化集合输出

        Args:
            name: 操作符名称
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        self._records: List[Record] = []

    def write(self, record: Record):
        """写入记录"""
        self._records.append(record)

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        self._records.clear()

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)

    def get_records(self) -> List[Record]:
        """
        获取收集的记录

        Returns:
            List[Record]: 记录列表
        """
        return self._records.copy()


class KafkaSink(SinkOperator):
    """
    Kafka输出

    将记录写入Kafka。
    """

    def __init__(self,
                 name: str,
                 topic: str,
                 bootstrap_servers: str,
                 parallelism: int = 1):
        """
        初始化Kafka输出

        Args:
            name: 操作符名称
            topic: 主题名称
            bootstrap_servers: Kafka服务器地址
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        self._topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._producer = None

    def write(self, record: Record):
        """写入记录"""
        if self._producer:
            try:
                self._producer.send(
                    self._topic,
                    key=record.key.encode() if record.key else None,
                    value=record.value if isinstance(record.value, bytes) else str(record.value).encode()
                )
            except Exception as e:
                raise SinkError(f"Failed to send to Kafka: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers
            )
        except ImportError:
            raise SinkError("kafka-python package is required for KafkaSink")
        except Exception as e:
            raise SinkError(f"Failed to create Kafka producer: {e}") from e

    def close(self):
        """关闭操作符"""
        if self._producer:
            self._producer.flush()
            self._producer.close()
        self.set_state(OperatorState.COMPLETED)


class DatabaseSink(SinkOperator):
    """
    数据库输出

    将记录写入数据库。
    """

    # SQL关键字黑名单（防止注入）
    SQL_KEYWORDS = frozenset({
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', 'JOIN', 'WHERE', 'FROM',
        'HAVING', 'GROUP', 'ORDER', 'LIMIT', 'OFFSET', '--', '/*', '*/',
        'XP_', 'SP_', 'INFORMATION_SCHEMA', 'SYSOBJECTS', 'SYSCOLUMNS'
    })

    def __init__(self,
                 name: str,
                 connection_string: str,
                 table_name: str,
                 parallelism: int = 1,
                 batch_size: int = 100,
                 allowed_columns: Optional[List[str]] = None):
        """
        初始化数据库输出

        Args:
            name: 操作符名称
            connection_string: 数据库连接字符串
            table_name: 表名
            parallelism: 并行度
            batch_size: 批量大小
            allowed_columns: 允许的列名白名单（可选，用于额外的安全控制）
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.SINK,
            parallelism=parallelism
        )
        super().__init__(config)
        
        # 验证表名
        self._table_name = self._validate_identifier(table_name, "table name")
        self._connection_string = connection_string
        self._batch_size = batch_size
        self._allowed_columns = frozenset(allowed_columns) if allowed_columns else None
        self._connection = None
        self._batch: List[Record] = []

    def _validate_identifier(self, identifier: str, identifier_type: str) -> str:
        """
        验证SQL标识符（表名、列名等）

        Args:
            identifier: 标识符
            identifier_type: 标识符类型（用于错误消息）

        Returns:
            str: 验证后的标识符

        Raises:
            SinkError: 如果标识符无效
        """
        import re
        
        if not identifier:
            raise SinkError(f"{identifier_type} cannot be empty")
        
        # 基本格式验证：只允许字母、数字、下划线，且不能以数字开头
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise SinkError(
                f"Invalid {identifier_type}: '{identifier}'. "
                f"Only alphanumeric characters and underscores are allowed, "
                f"and it must start with a letter or underscore."
            )
        
        # 长度限制
        if len(identifier) > 128:
            raise SinkError(
                f"{identifier_type} too long: '{identifier[:20]}...' "
                f"(max 128 characters)"
            )
        
        # 检查是否包含SQL关键字（不区分大小写）
        identifier_upper = identifier.upper()
        for keyword in self.SQL_KEYWORDS:
            if keyword in identifier_upper:
                raise SinkError(
                    f"Invalid {identifier_type}: '{identifier}' contains "
                    f"or resembles SQL keyword '{keyword}'"
                )
        
        return identifier

    def write(self, record: Record):
        """写入记录"""
        self._batch.append(record)

        if len(self._batch) >= self._batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """刷新批量数据"""
        if not self._batch or not self._connection:
            return

        try:
            cursor = self._connection.cursor()

            for record in self._batch:
                if isinstance(record.value, dict):
                    if not record.value:
                        continue
                    
                    # 验证并过滤列名
                    columns = []
                    values = []
                    
                    for col, val in record.value.items():
                        try:
                            validated_col = self._validate_identifier(str(col), "column name")
                            
                            # 如果设置了列名白名单，检查是否在白名单中
                            if self._allowed_columns and validated_col not in self._allowed_columns:
                                raise SinkError(
                                    f"Column '{validated_col}' is not in the allowed columns list"
                                )
                            
                            columns.append(validated_col)
                            values.append(val)
                        except SinkError:
                            raise
                    
                    if not columns:
                        continue
                    
                    # 使用参数化查询（防止SQL注入）
                    columns_str = ', '.join(columns)
                    placeholders = ', '.join(['?' for _ in columns])
                    sql = f"INSERT INTO {self._table_name} ({columns_str}) VALUES ({placeholders})"
                    
                    # 执行参数化查询
                    cursor.execute(sql, values)

            self._connection.commit()
            self._batch.clear()

        except SinkError:
            self._connection.rollback()
            raise
        except Exception as e:
            if self._connection:
                self._connection.rollback()
            raise SinkError(f"Failed to write to database: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

        try:
            import sqlite3
            self._connection = sqlite3.connect(self._connection_string)
        except Exception as e:
            raise SinkError(f"Failed to connect to database: {e}") from e

    def close(self):
        """关闭操作符"""
        if self._batch:
            try:
                self._flush_batch()
            except Exception as e:
                # 记录错误但继续关闭连接
                pass

        if self._connection:
            self._connection.close()

        self.set_state(OperatorState.COMPLETED)
