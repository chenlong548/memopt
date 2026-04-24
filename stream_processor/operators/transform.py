"""
stream_processor 转换操作符

定义数据转换操作符。
"""

from typing import Any, Callable, List, Optional, Dict
from dataclasses import dataclass
import threading

from .base import OneInputOperator, OperatorConfig, OperatorType, OperatorState
from ..core.record import Record
from ..core.execution_context import ExecutionContext
from ..core.exceptions import TransformError


class MapOperator(OneInputOperator):
    """
    映射操作符

    对每个记录应用映射函数。
    """

    def __init__(self,
                 name: str,
                 map_func: Callable[[Any], Any],
                 parallelism: int = 1):
        """
        初始化映射操作符

        Args:
            name: 操作符名称
            map_func: 映射函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._map_func = map_func

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            new_value = self._map_func(record.value)
            return [record.with_value(new_value)]
        except Exception as e:
            raise TransformError(f"Map operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class FilterOperator(OneInputOperator):
    """
    过滤操作符

    根据谓词过滤记录。
    """

    def __init__(self,
                 name: str,
                 predicate: Callable[[Any], bool],
                 parallelism: int = 1):
        """
        初始化过滤操作符

        Args:
            name: 操作符名称
            predicate: 过滤谓词
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._predicate = predicate

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            if self._predicate(record.value):
                return [record]
            return []
        except Exception as e:
            raise TransformError(f"Filter operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class FlatMapOperator(OneInputOperator):
    """
    扁平映射操作符

    对每个记录应用函数并展平结果。
    """

    def __init__(self,
                 name: str,
                 flat_map_func: Callable[[Any], List[Any]],
                 parallelism: int = 1):
        """
        初始化扁平映射操作符

        Args:
            name: 操作符名称
            flat_map_func: 扁平映射函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._flat_map_func = flat_map_func

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            values = self._flat_map_func(record.value)
            return [record.with_value(v) for v in values]
        except Exception as e:
            raise TransformError(f"FlatMap operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class KeyByOperator(OneInputOperator):
    """
    按键分组操作符

    根据键选择器对记录分组。
    """

    def __init__(self,
                 name: str,
                 key_selector: Callable[[Any], str],
                 parallelism: int = 1):
        """
        初始化按键分组操作符

        Args:
            name: 操作符名称
            key_selector: 键选择器
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._key_selector = key_selector

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            key = self._key_selector(record.value)
            new_record = Record(
                value=record.value,
                key=key,
                timestamp=record.timestamp,
                headers=record.headers.copy(),
                partition=record.partition,
                offset=record.offset
            )
            return [new_record]
        except Exception as e:
            raise TransformError(f"KeyBy operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class ReduceOperator(OneInputOperator):
    """
    归约操作符

    对分组数据进行归约。
    """

    def __init__(self,
                 name: str,
                 reduce_func: Callable[[Any, Any], Any],
                 parallelism: int = 1):
        """
        初始化归约操作符

        Args:
            name: 操作符名称
            reduce_func: 归约函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._reduce_func = reduce_func
        self._reduce_state: Dict[str, Any] = {}
        # 使用细粒度锁：每个key一个锁，提高并发性能
        self._key_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()  # 用于保护 _key_locks 字典

    def _get_key_lock(self, key: str) -> threading.Lock:
        """获取或创建key对应的锁"""
        with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            key = record.key or record.get_key()
            
            # 获取该key的专用锁
            key_lock = self._get_key_lock(key)
            
            with key_lock:
                if key not in self._reduce_state:
                    self._reduce_state[key] = record.value
                    return [record]
                else:
                    self._reduce_state[key] = self._reduce_func(
                        self._reduce_state[key],
                        record.value
                    )
                    return [record.with_value(self._reduce_state[key])]

        except Exception as e:
            raise TransformError(f"Reduce operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        with self._global_lock:
            self._reduce_state.clear()
            self._key_locks.clear()

    def close(self):
        """关闭操作符"""
        with self._global_lock:
            self._reduce_state.clear()
            self._key_locks.clear()
        self.set_state(OperatorState.COMPLETED)


class AggregateOperator(OneInputOperator):
    """
    聚合操作符

    对分组数据进行聚合。
    """

    def __init__(self,
                 name: str,
                 aggregate_func: Callable[[Any, Any], Any],
                 initial_value: Any = None,
                 parallelism: int = 1):
        """
        初始化聚合操作符

        Args:
            name: 操作符名称
            aggregate_func: 聚合函数
            initial_value: 初始值
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._aggregate_func = aggregate_func
        self._initial_value = initial_value
        self._aggregate_state: Dict[str, Any] = {}
        # 使用细粒度锁：每个key一个锁，提高并发性能
        self._key_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()  # 用于保护 _key_locks 字典

    def _get_key_lock(self, key: str) -> threading.Lock:
        """获取或创建key对应的锁"""
        with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            key = record.key or record.get_key()
            
            # 获取该key的专用锁
            key_lock = self._get_key_lock(key)
            
            with key_lock:
                if key not in self._aggregate_state:
                    self._aggregate_state[key] = self._initial_value

                self._aggregate_state[key] = self._aggregate_func(
                    self._aggregate_state[key],
                    record.value
                )

                return [record.with_value(self._aggregate_state[key])]

        except Exception as e:
            raise TransformError(f"Aggregate operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        with self._global_lock:
            self._aggregate_state.clear()
            self._key_locks.clear()

    def close(self):
        """关闭操作符"""
        with self._global_lock:
            self._aggregate_state.clear()
            self._key_locks.clear()
        self.set_state(OperatorState.COMPLETED)


class UnionOperator(OneInputOperator):
    """
    联合操作符

    合并多个流。
    """

    def __init__(self,
                 name: str,
                 parallelism: int = 1):
        """
        初始化联合操作符

        Args:
            name: 操作符名称
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        return [record]

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

    def close(self):
        """关闭操作符"""
        self.set_state(OperatorState.COMPLETED)


class ProcessFunctionOperator(OneInputOperator):
    """
    处理函数操作符

    使用自定义处理函数处理记录。
    """

    def __init__(self,
                 name: str,
                 process_func: Callable[[Record, 'ProcessContext | None'], List[Record]],
                 parallelism: int = 1):
        """
        初始化处理函数操作符

        Args:
            name: 操作符名称
            process_func: 处理函数
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.TRANSFORM,
            parallelism=parallelism
        )
        super().__init__(config)
        self._process_func = process_func
        self._process_context: Optional['ProcessContext'] = None

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            if self._process_context:
                return self._process_func(record, self._process_context)
            return self._process_func(record, None)
        except Exception as e:
            raise TransformError(f"ProcessFunction operation failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        self._process_context = ProcessContext(context)

    def close(self):
        """关闭操作符"""
        self._process_context = None
        self.set_state(OperatorState.COMPLETED)


@dataclass
class ProcessContext:
    """
    处理上下文

    提供处理函数的上下文信息。
    """

    execution_context: ExecutionContext

    def get_current_key(self) -> Optional[str]:
        """获取当前键"""
        return self.execution_context.get_user_data('current_key')

    def get_state(self, key: str) -> Any:
        """获取状态"""
        return self.execution_context.get_user_data(key)

    def set_state(self, key: str, value: Any):
        """设置状态"""
        self.execution_context.set_user_data(key, value)
