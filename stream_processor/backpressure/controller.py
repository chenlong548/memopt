"""
stream_processor 背压控制器

实现背压控制功能。
"""

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
import time
import threading
import logging

from .flow_controller import FlowController, FlowControlConfig, FlowControlState
from .rate_limiter import RateLimiter, RateLimitConfig


logger = logging.getLogger(__name__)


class BackpressureLevel(Enum):
    """背压级别"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackpressureConfig:
    """
    背压配置

    定义背压控制的配置参数。
    """

    enable: bool = True

    low_threshold: float = 0.5

    medium_threshold: float = 0.7

    high_threshold: float = 0.85

    critical_threshold: float = 0.95

    check_interval: float = 0.1

    recovery_threshold: float = 0.3

    max_delay_ms: float = 1000.0


@dataclass
class BackpressureStatus:
    """
    背压状态

    记录当前的背压状态。
    """

    level: BackpressureLevel

    utilization: float

    delay_ms: float

    input_rate: float

    output_rate: float

    timestamp: float


class BackpressureController:
    """
    背压控制器

    监控和控制背压。
    """

    def __init__(self, config: BackpressureConfig):
        """
        初始化背压控制器

        Args:
            config: 背压配置
        """
        self._config = config
        self._level = BackpressureLevel.NONE
        self._utilization = 0.0
        self._delay_ms = 0.0
        self._lock = threading.Lock()

        self._input_rate = 0.0
        self._output_rate = 0.0
        self._input_count = 0
        self._output_count = 0
        self._last_measure_time = time.time()

        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        self._callbacks: Dict[str, List[Callable]] = {}
        self._subscribers: List[Callable] = []

    def start(self):
        """启动背压控制器"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Backpressure controller started")

    def stop(self):
        """停止背压控制器"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        logger.info("Backpressure controller stopped")

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._update_metrics()
                time.sleep(self._config.check_interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def _update_metrics(self):
        """更新指标"""
        current_time = time.time()
        elapsed = current_time - self._last_measure_time

        if elapsed >= 1.0:
            self._input_rate = self._input_count / elapsed
            self._output_rate = self._output_count / elapsed
            self._input_count = 0
            self._output_count = 0
            self._last_measure_time = current_time

    def update_utilization(self, utilization: float):
        """
        更新利用率

        Args:
            utilization: 利用率（0-1）
        """
        with self._lock:
            self._utilization = utilization
            old_level = self._level
            self._level = self._calculate_level(utilization)
            self._delay_ms = self._calculate_delay(utilization)

            if old_level != self._level:
                self._notify_level_change(old_level, self._level)

    def _calculate_level(self, utilization: float) -> BackpressureLevel:
        """计算背压级别"""
        if utilization >= self._config.critical_threshold:
            return BackpressureLevel.CRITICAL
        elif utilization >= self._config.high_threshold:
            return BackpressureLevel.HIGH
        elif utilization >= self._config.medium_threshold:
            return BackpressureLevel.MEDIUM
        elif utilization >= self._config.low_threshold:
            return BackpressureLevel.LOW
        else:
            return BackpressureLevel.NONE

    def _calculate_delay(self, utilization: float) -> float:
        """计算延迟"""
        if utilization < self._config.low_threshold:
            return 0.0

        excess = utilization - self._config.low_threshold
        range_size = 1.0 - self._config.low_threshold

        delay_factor = excess / range_size
        delay_ms = delay_factor * self._config.max_delay_ms

        return min(delay_ms, self._config.max_delay_ms)

    def _notify_level_change(self, old_level: BackpressureLevel, new_level: BackpressureLevel):
        """通知级别变化"""
        callbacks = self._callbacks.get('on_level_change', [])
        for callback in callbacks:
            try:
                callback(old_level, new_level)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        for subscriber in self._subscribers:
            try:
                subscriber(self.get_status())
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    def get_level(self) -> BackpressureLevel:
        """
        获取当前级别

        Returns:
            BackpressureLevel: 背压级别
        """
        with self._lock:
            return self._level

    def get_delay(self) -> float:
        """
        获取当前延迟

        Returns:
            float: 延迟（毫秒）
        """
        with self._lock:
            return self._delay_ms

    def get_status(self) -> BackpressureStatus:
        """
        获取背压状态

        Returns:
            BackpressureStatus: 背压状态
        """
        with self._lock:
            return BackpressureStatus(
                level=self._level,
                utilization=self._utilization,
                delay_ms=self._delay_ms,
                input_rate=self._input_rate,
                output_rate=self._output_rate,
                timestamp=time.time()
            )

    def should_apply_backpressure(self) -> bool:
        """
        是否应该应用背压

        Returns:
            bool: 是否应用
        """
        if not self._config.enable:
            return False

        return self._level != BackpressureLevel.NONE

    def record_input(self, count: int = 1):
        """
        记录输入

        Args:
            count: 数量
        """
        self._input_count += count

    def record_output(self, count: int = 1):
        """
        记录输出

        Args:
            count: 数量
        """
        self._output_count += count

    def register_callback(self, event: str, callback: Callable):
        """
        注册回调函数

        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def subscribe(self, callback: Callable):
        """
        订阅状态变化

        Args:
            callback: 回调函数
        """
        self._subscribers.append(callback)

    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running


class BackpressureManager:
    """
    背压管理器

    管理多个操作符的背压。
    """

    def __init__(self, config: BackpressureConfig):
        """
        初始化背压管理器

        Args:
            config: 背压配置
        """
        self._config = config
        self._controllers: Dict[str, BackpressureController] = {}
        self._lock = threading.Lock()

    def get_or_create(self, operator_id: str) -> BackpressureController:
        """
        获取或创建背压控制器

        Args:
            operator_id: 操作符ID

        Returns:
            BackpressureController: 背压控制器
        """
        with self._lock:
            if operator_id not in self._controllers:
                controller = BackpressureController(self._config)
                self._controllers[operator_id] = controller
            return self._controllers[operator_id]

    def get_global_level(self) -> BackpressureLevel:
        """
        获取全局背压级别

        Returns:
            BackpressureLevel: 全局背压级别
        """
        with self._lock:
            if not self._controllers:
                return BackpressureLevel.NONE

            max_level = BackpressureLevel.NONE
            for controller in self._controllers.values():
                level = controller.get_level()
                if level.value > max_level.value:
                    max_level = level

            return max_level

    def start_all(self):
        """启动所有控制器"""
        with self._lock:
            for controller in self._controllers.values():
                controller.start()

    def stop_all(self):
        """停止所有控制器"""
        with self._lock:
            for controller in self._controllers.values():
                controller.stop()

    def get_all_status(self) -> Dict[str, BackpressureStatus]:
        """
        获取所有状态

        Returns:
            Dict[str, BackpressureStatus]: 状态字典
        """
        with self._lock:
            return {
                operator_id: controller.get_status()
                for operator_id, controller in self._controllers.items()
            }
