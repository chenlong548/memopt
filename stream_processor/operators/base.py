"""
stream_processor 操作符基类

定义所有操作符的基础接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Iterator
from enum import Enum
import time

from ..core.record import Record
from ..core.stream import Stream, StreamRecord
from ..core.execution_context import ExecutionContext


class OperatorType(Enum):
    """操作符类型"""
    SOURCE = "source"
    TRANSFORM = "transform"
    SINK = "sink"
    WINDOW = "window"
    COMPRESSION = "compression"


class OperatorState(Enum):
    """操作符状态"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OperatorConfig:
    """
    操作符配置

    定义操作符的配置参数。
    """

    name: str

    operator_type: OperatorType

    parallelism: int = 1

    buffer_size: int = 1000

    max_latency_ms: float = 100.0

    enable_metrics: bool = True

    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorMetrics:
    """
    操作符指标

    记录操作符的性能指标。
    """

    records_in: int = 0

    records_out: int = 0

    bytes_in: int = 0

    bytes_out: int = 0

    processing_time_ms: float = 0.0

    latency_ms: float = 0.0

    errors: int = 0

    last_process_time: float = 0.0


class Operator(ABC):
    """
    操作符基类

    定义操作符的标准接口。
    """

    def __init__(self, config: OperatorConfig):
        """
        初始化操作符

        Args:
            config: 操作符配置
        """
        self.config = config
        self._state = OperatorState.INITIALIZED
        self._metrics = OperatorMetrics()
        self._context: Optional[ExecutionContext] = None
        self._upstream_operators: List['Operator'] = []
        self._downstream_operators: List['Operator'] = []

    @abstractmethod
    def process(self, record: Record) -> List[Record]:
        """
        处理记录

        Args:
            record: 输入记录

        Returns:
            List[Record]: 输出记录列表
        """
        pass

    @abstractmethod
    def open(self, context: ExecutionContext):
        """
        打开操作符

        Args:
            context: 执行上下文
        """
        pass

    @abstractmethod
    def close(self):
        """关闭操作符"""
        pass

    def set_context(self, context: ExecutionContext):
        """
        设置执行上下文

        Args:
            context: 执行上下文
        """
        self._context = context

    def get_context(self) -> Optional[ExecutionContext]:
        """
        获取执行上下文

        Returns:
            Optional[ExecutionContext]: 执行上下文
        """
        return self._context

    def add_upstream(self, operator: 'Operator'):
        """
        添加上游操作符

        Args:
            operator: 上游操作符
        """
        if operator not in self._upstream_operators:
            self._upstream_operators.append(operator)

    def add_downstream(self, operator: 'Operator'):
        """
        添加下游操作符

        Args:
            operator: 下游操作符
        """
        if operator not in self._downstream_operators:
            self._downstream_operators.append(operator)

    def get_upstream_operators(self) -> List['Operator']:
        """
        获取上游操作符

        Returns:
            List[Operator]: 上游操作符列表
        """
        return self._upstream_operators.copy()

    def get_downstream_operators(self) -> List['Operator']:
        """
        获取下游操作符

        Returns:
            List[Operator]: 下游操作符列表
        """
        return self._downstream_operators.copy()

    def get_state(self) -> OperatorState:
        """
        获取操作符状态

        Returns:
            OperatorState: 操作符状态
        """
        return self._state

    def set_state(self, state: OperatorState):
        """
        设置操作符状态

        Args:
            state: 新状态
        """
        self._state = state

    def get_metrics(self) -> OperatorMetrics:
        """
        获取操作符指标

        Returns:
            OperatorMetrics: 操作符指标
        """
        return self._metrics

    def update_metrics(self, **kwargs):
        """
        更新操作符指标

        Args:
            **kwargs: 指标键值对
        """
        for key, value in kwargs.items():
            if hasattr(self._metrics, key):
                setattr(self._metrics, key, value)

    def _record_process_start(self):
        """记录处理开始"""
        self._metrics.last_process_time = time.time()

    def _record_process_end(self, input_size: int = 0, output_size: int = 0):
        """记录处理结束"""
        elapsed = (time.time() - self._metrics.last_process_time) * 1000
        self._metrics.processing_time_ms += elapsed
        self._metrics.records_in += 1
        self._metrics.bytes_in += input_size
        self._metrics.bytes_out += output_size

    def is_running(self) -> bool:
        """是否正在运行"""
        return self._state == OperatorState.RUNNING

    def is_completed(self) -> bool:
        """是否已完成"""
        return self._state == OperatorState.COMPLETED

    def is_failed(self) -> bool:
        """是否已失败"""
        return self._state == OperatorState.FAILED

    def get_name(self) -> str:
        """
        获取操作符名称

        Returns:
            str: 操作符名称
        """
        return self.config.name

    def get_type(self) -> OperatorType:
        """
        获取操作符类型

        Returns:
            OperatorType: 操作符类型
        """
        return self.config.operator_type

    def get_parallelism(self) -> int:
        """
        获取并行度

        Returns:
            int: 并行度
        """
        return self.config.parallelism


class OneInputOperator(Operator):
    """
    单输入操作符

    具有一个输入的操作符基类。
    """

    def __init__(self, config: OperatorConfig):
        super().__init__(config)

    @abstractmethod
    def process_element(self, record: Record) -> List[Record]:
        """
        处理单个元素

        Args:
            record: 输入记录

        Returns:
            List[Record]: 输出记录列表
        """
        pass

    def process(self, record: Record) -> List[Record]:
        """处理记录"""
        self._record_process_start()
        try:
            result = self.process_element(record)
            self._record_process_end(
                input_size=len(str(record.value)),
                output_size=sum(len(str(r.value)) for r in result)
            )
            return result
        except Exception as e:
            self._metrics.errors += 1
            raise


class TwoInputOperator(Operator):
    """
    双输入操作符

    具有两个输入的操作符基类。
    """

    def __init__(self, config: OperatorConfig):
        super().__init__(config)

    @abstractmethod
    def process_element1(self, record: Record) -> List[Record]:
        """
        处理第一个输入的元素

        Args:
            record: 输入记录

        Returns:
            List[Record]: 输出记录列表
        """
        pass

    @abstractmethod
    def process_element2(self, record: Record) -> List[Record]:
        """
        处理第二个输入的元素

        Args:
            record: 输入记录

        Returns:
            List[Record]: 输出记录列表
        """
        pass

    def process(self, record: Record, input_id: int = 0) -> List[Record]:
        """
        处理记录

        Args:
            record: 输入记录
            input_id: 输入ID（0或1）

        Returns:
            List[Record]: 输出记录列表
        """
        self._record_process_start()
        try:
            if input_id == 0:
                result = self.process_element1(record)
            else:
                result = self.process_element2(record)

            self._record_process_end(
                input_size=len(str(record.value)),
                output_size=sum(len(str(r.value)) for r in result)
            )
            return result
        except Exception as e:
            self._metrics.errors += 1
            raise
