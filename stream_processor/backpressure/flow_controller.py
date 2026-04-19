"""
stream_processor 流量控制器

实现流量控制功能。
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue

from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitStrategy


class FlowControlState(Enum):
    """流量控制状态"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FlowControlConfig:
    """
    流量控制配置

    定义流量控制的配置参数。
    """

    max_buffer_size: int = 10000

    warning_threshold: float = 0.7

    critical_threshold: float = 0.9

    enable_backpressure: bool = True

    backpressure_factor: float = 0.5

    rate_limit: Optional[float] = None


@dataclass
class FlowControlMetrics:
    """
    流量控制指标

    记录流量控制的指标数据。
    """

    buffer_size: int = 0

    buffer_utilization: float = 0.0

    input_rate: float = 0.0

    output_rate: float = 0.0

    backpressure_level: float = 0.0

    state: FlowControlState = FlowControlState.NORMAL


class FlowController:
    """
    流量控制器

    控制数据流的速率和缓冲。
    """

    def __init__(self, config: FlowControlConfig):
        """
        初始化流量控制器

        Args:
            config: 流量控制配置
        """
        self._config = config
        self._buffer = queue.Queue(maxsize=config.max_buffer_size)
        self._state = FlowControlState.NORMAL
        self._lock = threading.Lock()

        self._input_count = 0
        self._output_count = 0
        self._last_measure_time = time.time()
        self._input_rate = 0.0
        self._output_rate = 0.0

        self._rate_limiter: Optional[RateLimiter] = None
        if config.rate_limit:
            rate_config = RateLimitConfig(
                rate=config.rate_limit,
                capacity=int(config.rate_limit),
                strategy=RateLimitStrategy.TOKEN_BUCKET
            )
            self._rate_limiter = RateLimiter(rate_config)

        self._callbacks: Dict[str, Callable] = {}

    def offer(self, data: Any, timeout: Optional[float] = None) -> bool:
        """
        提供数据

        Args:
            data: 数据
            timeout: 超时时间

        Returns:
            bool: 是否成功
        """
        if self._rate_limiter:
            result = self._rate_limiter.try_acquire()
            if not result.allowed:
                return False

        try:
            if timeout is None:
                self._buffer.put_nowait(data)
            else:
                self._buffer.put(data, timeout=timeout)

            self._input_count += 1
            self._update_state()
            return True

        except queue.Full:
            return False

    def poll(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        获取数据

        Args:
            timeout: 超时时间

        Returns:
            Optional[Any]: 数据
        """
        try:
            if timeout is None:
                data = self._buffer.get_nowait()
            else:
                data = self._buffer.get(timeout=timeout)

            self._output_count += 1
            self._update_state()
            return data

        except queue.Empty:
            return None

    def _update_state(self):
        """更新状态"""
        utilization = self._buffer.qsize() / self._config.max_buffer_size

        if utilization >= self._config.critical_threshold:
            new_state = FlowControlState.CRITICAL
        elif utilization >= self._config.warning_threshold:
            new_state = FlowControlState.WARNING
        else:
            new_state = FlowControlState.NORMAL

        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._notify_state_change(old_state, new_state)

    def _notify_state_change(self, old_state: FlowControlState, new_state: FlowControlState):
        """通知状态变化"""
        callback = self._callbacks.get('on_state_change')
        if callback:
            try:
                callback(old_state, new_state)
            except Exception:
                pass

    def get_state(self) -> FlowControlState:
        """
        获取当前状态

        Returns:
            FlowControlState: 当前状态
        """
        return self._state

    def get_buffer_size(self) -> int:
        """
        获取缓冲区大小

        Returns:
            int: 缓冲区大小
        """
        return self._buffer.qsize()

    def get_metrics(self) -> FlowControlMetrics:
        """
        获取指标

        Returns:
            FlowControlMetrics: 指标数据
        """
        current_time = time.time()
        elapsed = current_time - self._last_measure_time

        if elapsed >= 1.0:
            self._input_rate = self._input_count / elapsed
            self._output_rate = self._output_count / elapsed
            self._input_count = 0
            self._output_count = 0
            self._last_measure_time = current_time

        buffer_size = self._buffer.qsize()
        utilization = buffer_size / self._config.max_buffer_size

        return FlowControlMetrics(
            buffer_size=buffer_size,
            buffer_utilization=utilization,
            input_rate=self._input_rate,
            output_rate=self._output_rate,
            backpressure_level=utilization,
            state=self._state
        )

    def should_apply_backpressure(self) -> bool:
        """
        是否应该应用背压

        Returns:
            bool: 是否应用背压
        """
        if not self._config.enable_backpressure:
            return False

        return self._state in [FlowControlState.WARNING, FlowControlState.CRITICAL]

    def get_backpressure_delay(self) -> float:
        """
        获取背压延迟

        Returns:
            float: 延迟时间（秒）
        """
        if not self.should_apply_backpressure():
            return 0.0

        utilization = self._buffer.qsize() / self._config.max_buffer_size

        if self._state == FlowControlState.CRITICAL:
            return self._config.backpressure_factor * 2.0
        else:
            return self._config.backpressure_factor * utilization

    def register_callback(self, event: str, callback: Callable):
        """
        注册回调函数

        Args:
            event: 事件名称
            callback: 回调函数
        """
        self._callbacks[event] = callback

    def clear(self):
        """清空缓冲区"""
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

        self._state = FlowControlState.NORMAL


class PriorityFlowController(FlowController):
    """
    优先级流量控制器

    支持优先级的流量控制。
    """

    def __init__(self, config: FlowControlConfig, num_priorities: int = 3):
        """
        初始化优先级流量控制器

        Args:
            config: 流量控制配置
            num_priorities: 优先级数量
        """
        super().__init__(config)
        self._num_priorities = num_priorities
        self._priority_buffers = [
            queue.Queue(maxsize=config.max_buffer_size // num_priorities)
            for _ in range(num_priorities)
        ]

    def offer_with_priority(self, data: Any, priority: int = 0, timeout: Optional[float] = None) -> bool:
        """
        提供带优先级的数据

        Args:
            data: 数据
            priority: 优先级（0最高）
            timeout: 超时时间

        Returns:
            bool: 是否成功
        """
        if priority < 0 or priority >= self._num_priorities:
            priority = self._num_priorities - 1

        try:
            if timeout is None:
                self._priority_buffers[priority].put_nowait(data)
            else:
                self._priority_buffers[priority].put(data, timeout=timeout)

            self._input_count += 1
            return True

        except queue.Full:
            return False

    def poll_priority(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        获取数据（按优先级）

        Args:
            timeout: 超时时间

        Returns:
            Optional[Any]: 数据
        """
        for buffer in self._priority_buffers:
            try:
                data = buffer.get_nowait()
                self._output_count += 1
                return data
            except queue.Empty:
                continue

        return None
