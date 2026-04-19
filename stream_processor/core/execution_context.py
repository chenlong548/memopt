"""
stream_processor 执行上下文

管理流处理任务的执行环境。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import threading
import time
from enum import Enum


class ExecutionState(Enum):
    """执行状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionConfig:
    """
    执行配置

    定义流处理任务的执行参数。
    """

    parallelism: int = 1

    checkpoint_interval: float = 60.0

    checkpoint_timeout: float = 600.0

    max_memory_usage: int = 1024 * 1024 * 1024

    buffer_timeout: float = 100.0

    max_buffer_size: int = 10000

    enable_backpressure: bool = True

    backpressure_threshold: float = 0.8

    enable_metrics: bool = True

    metrics_report_interval: float = 10.0

    task_timeout: float = 3600.0

    retry_attempts: int = 3

    retry_delay: float = 1.0


@dataclass
class TaskMetrics:
    """
    任务指标

    记录任务执行的指标数据。
    """

    records_in: int = 0

    records_out: int = 0

    bytes_in: int = 0

    bytes_out: int = 0

    latency_ms: float = 0.0

    throughput: float = 0.0

    errors: int = 0

    start_time: float = field(default_factory=time.time)

    end_time: float = 0.0

    def update_latency(self, processing_time: float):
        """更新延迟"""
        self.latency_ms = processing_time * 1000

    def calculate_throughput(self):
        """计算吞吐量"""
        elapsed = self.end_time - self.start_time
        if elapsed > 0:
            self.throughput = self.records_out / elapsed

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'records_in': self.records_in,
            'records_out': self.records_out,
            'bytes_in': self.bytes_in,
            'bytes_out': self.bytes_out,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'errors': self.errors,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class ExecutionContext:
    """
    执行上下文

    管理流处理任务的执行环境。
    """

    def __init__(self,
                 job_name: str,
                 config: Optional[ExecutionConfig] = None):
        """
        初始化执行上下文

        Args:
            job_name: 任务名称
            config: 执行配置
        """
        self.job_name = job_name
        self.config = config or ExecutionConfig()

        self._state = ExecutionState.CREATED
        self._metrics = TaskMetrics()
        self._state_lock = threading.Lock()
        self._user_data: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._error: Optional[Exception] = None

    def get_state(self) -> ExecutionState:
        """
        获取执行状态

        Returns:
            ExecutionState: 执行状态
        """
        with self._state_lock:
            return self._state

    def set_state(self, state: ExecutionState):
        """
        设置执行状态

        Args:
            state: 新状态
        """
        with self._state_lock:
            old_state = self._state
            self._state = state

            callback_key = f"on_{state.value}"
            if callback_key in self._callbacks:
                self._callbacks[callback_key](old_state, state)

    def is_running(self) -> bool:
        """是否正在运行"""
        return self.get_state() == ExecutionState.RUNNING

    def is_completed(self) -> bool:
        """是否已完成"""
        return self.get_state() == ExecutionState.COMPLETED

    def is_failed(self) -> bool:
        """是否已失败"""
        return self.get_state() == ExecutionState.FAILED

    def is_cancelled(self) -> bool:
        """是否已取消"""
        return self.get_state() == ExecutionState.CANCELLED

    def get_metrics(self) -> TaskMetrics:
        """
        获取任务指标

        Returns:
            TaskMetrics: 任务指标
        """
        return self._metrics

    def update_metrics(self, **kwargs):
        """
        更新任务指标

        Args:
            **kwargs: 指标键值对
        """
        for key, value in kwargs.items():
            if hasattr(self._metrics, key):
                setattr(self._metrics, key, value)

    def set_user_data(self, key: str, value: Any):
        """
        设置用户数据

        Args:
            key: 键
            value: 值
        """
        self._user_data[key] = value

    def get_user_data(self, key: str, default: Any = None) -> Any:
        """
        获取用户数据

        Args:
            key: 键
            default: 默认值

        Returns:
            Any: 用户数据
        """
        return self._user_data.get(key, default)

    def register_callback(self, event: str, callback: Callable):
        """
        注册事件回调

        Args:
            event: 事件名称
            callback: 回调函数
        """
        self._callbacks[event] = callback

    def unregister_callback(self, event: str):
        """
        注销事件回调

        Args:
            event: 事件名称
        """
        self._callbacks.pop(event, None)

    def set_error(self, error: Exception):
        """
        设置错误

        Args:
            error: 异常
        """
        self._error = error
        self.set_state(ExecutionState.FAILED)

    def get_error(self) -> Optional[Exception]:
        """
        获取错误

        Returns:
            Optional[Exception]: 异常
        """
        return self._error

    def start(self):
        """启动任务"""
        self.set_state(ExecutionState.RUNNING)
        self._metrics.start_time = time.time()

    def complete(self):
        """完成任务"""
        self._metrics.end_time = time.time()
        self._metrics.calculate_throughput()
        self.set_state(ExecutionState.COMPLETED)

    def pause(self):
        """暂停任务"""
        self.set_state(ExecutionState.PAUSED)

    def resume(self):
        """恢复任务"""
        self.set_state(ExecutionState.RUNNING)

    def cancel(self):
        """取消任务"""
        self.set_state(ExecutionState.CANCELLED)

    def get_parallelism(self) -> int:
        """
        获取并行度

        Returns:
            int: 并行度
        """
        return self.config.parallelism

    def get_checkpoint_interval(self) -> float:
        """
        获取检查点间隔

        Returns:
            float: 检查点间隔（秒）
        """
        return self.config.checkpoint_interval

    def should_enable_backpressure(self) -> bool:
        """
        是否启用背压

        Returns:
            bool: 是否启用
        """
        return self.config.enable_backpressure

    def get_backpressure_threshold(self) -> float:
        """
        获取背压阈值

        Returns:
            float: 背压阈值
        """
        return self.config.backpressure_threshold
