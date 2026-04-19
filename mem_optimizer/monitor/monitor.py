"""
mem_optimizer 内存监控器

实现内存使用和性能监控。
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..core.base import MemoryStatistics, AllocationResult, MemoryBlock
from ..core.config import MonitorConfig
from ..core.exceptions import MonitorError


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警"""
    level: AlertLevel
    metric: str
    value: float
    threshold: float
    timestamp: float
    message: str


@dataclass
class MetricSample:
    """指标样本"""
    timestamp: float
    metrics: Dict[str, float]


class MemoryMonitor:
    """
    内存监控器

    监控内存使用、碎片化、分配性能等指标。
    """

    def __init__(self,
                config: Optional[MonitorConfig] = None,
                memory_pool: Optional[Any] = None):
        """
        初始化内存监控器

        Args:
            config: 监控配置
            memory_pool: 内存池引用
        """
        self.config = config or MonitorConfig()
        self._memory_pool = memory_pool

        self._history: deque = deque(maxlen=self.config.history_size)
        self._alerts: List[Alert] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._allocation_times: deque = deque(maxlen=1000)
        self._allocation_sizes: deque = deque(maxlen=1000)

        self._psi_monitor = None

        if self.config.enable_psi:
            self._init_psi_monitor()

    def _init_psi_monitor(self):
        """初始化PSI监控"""
        try:
            from ..defrag.psi_metrics import PSIMonitor
            self._psi_monitor = PSIMonitor(
                psi_path=self.config.psi_path,
                sample_interval=self.config.sample_interval
            )
        except Exception:
            self._psi_monitor = None

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                sample = self._collect_sample()

                with self._lock:
                    self._history.append(sample)
                    self._check_alerts(sample)

                time.sleep(self.config.sample_interval)

            except Exception:
                time.sleep(self.config.sample_interval)

    def _collect_sample(self) -> MetricSample:
        """
        收集指标样本

        Returns:
            MetricSample: 指标样本
        """
        metrics = {}

        if self._memory_pool:
            stats = self._memory_pool.get_stats()
            metrics['total_size'] = float(stats.total_size)
            metrics['used_size'] = float(stats.used_size)
            metrics['free_size'] = float(stats.free_size)
            metrics['fragmentation_ratio'] = stats.fragmentation_ratio
            metrics['allocation_count'] = float(stats.allocation_count)
            metrics['deallocation_count'] = float(stats.deallocation_count)
            metrics['peak_usage'] = float(stats.peak_usage)
            metrics['allocation_failures'] = float(stats.allocation_failures)

            if stats.total_size > 0:
                metrics['usage_ratio'] = stats.used_size / stats.total_size
            else:
                metrics['usage_ratio'] = 0.0

        if self._allocation_times:
            metrics['avg_allocation_time'] = sum(self._allocation_times) / len(self._allocation_times)
            metrics['max_allocation_time'] = max(self._allocation_times)
            metrics['min_allocation_time'] = min(self._allocation_times)

        if self._allocation_sizes:
            metrics['avg_allocation_size'] = sum(self._allocation_sizes) / len(self._allocation_sizes)
            metrics['max_allocation_size'] = max(self._allocation_sizes)
            metrics['min_allocation_size'] = min(self._allocation_sizes)

        if self._psi_monitor:
            psi_metrics = self._psi_monitor.get_current_metrics()
            if psi_metrics:
                metrics['psi_some_avg10'] = psi_metrics.some_avg10
                metrics['psi_full_avg10'] = psi_metrics.full_avg10

        return MetricSample(
            timestamp=time.time(),
            metrics=metrics
        )

    def _check_alerts(self, sample: MetricSample):
        """
        检查告警

        Args:
            sample: 指标样本
        """
        for metric_name, threshold in self.config.alert_thresholds.items():
            value = sample.metrics.get(metric_name)

            if value is not None and value > threshold:
                level = AlertLevel.WARNING
                if value > threshold * 1.5:
                    level = AlertLevel.ERROR
                if value > threshold * 2:
                    level = AlertLevel.CRITICAL

                alert = Alert(
                    level=level,
                    metric=metric_name,
                    value=value,
                    threshold=threshold,
                    timestamp=sample.timestamp,
                    message=f"{metric_name} exceeded threshold: {value:.4f} > {threshold}"
                )

                self._alerts.append(alert)

                for handler in self._alert_handlers:
                    try:
                        handler(alert)
                    except Exception:
                        pass

    def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True

        if self._psi_monitor:
            self._psi_monitor.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def stop(self):
        """停止监控"""
        self._running = False

        if self._psi_monitor:
            self._psi_monitor.stop()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def record_allocation(self, result: AllocationResult):
        """
        记录分配

        Args:
            result: 分配结果
        """
        with self._lock:
            self._allocation_times.append(result.allocation_time)
            self._allocation_sizes.append(result.size)

    def record_deallocation(self, block: MemoryBlock):
        """
        记录释放

        Args:
            block: 内存块
        """
        pass

    def get_current_metrics(self) -> Dict[str, float]:
        """
        获取当前指标

        Returns:
            Dict: 当前指标
        """
        with self._lock:
            if self._history:
                return self._history[-1].metrics.copy()
            return {}

    def get_history(self,
                   count: Optional[int] = None,
                   metric: Optional[str] = None) -> List[Any]:
        """
        获取历史记录

        Args:
            count: 记录数量
            metric: 指标名称

        Returns:
            List: 历史记录
        """
        with self._lock:
            history = list(self._history)

            if count is not None:
                history = history[-count:]

            if metric is not None:
                return [(s.timestamp, s.metrics.get(metric)) for s in history]

            return [(s.timestamp, s.metrics) for s in history]

    def get_alerts(self,
                  level: Optional[AlertLevel] = None,
                  clear: bool = False) -> List[Alert]:
        """
        获取告警

        Args:
            level: 告警级别
            clear: 是否清除

        Returns:
            List[Alert]: 告警列表
        """
        with self._lock:
            if level is not None:
                alerts = [a for a in self._alerts if a.level == level]
            else:
                alerts = list(self._alerts)

            if clear:
                self._alerts.clear()

            return alerts

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        添加告警处理器

        Args:
            handler: 处理函数
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Alert], None]):
        """
        移除告警处理器

        Args:
            handler: 处理函数
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        with self._lock:
            stats = {
                'running': self._running,
                'history_size': len(self._history),
                'alert_count': len(self._alerts),
                'allocation_samples': len(self._allocation_times)
            }

            if self._allocation_times:
                times = list(self._allocation_times)
                stats['allocation_time_stats'] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }

            if self._allocation_sizes:
                sizes = list(self._allocation_sizes)
                stats['allocation_size_stats'] = {
                    'avg': sum(sizes) / len(sizes),
                    'min': min(sizes),
                    'max': max(sizes),
                    'count': len(sizes)
                }

            return stats

    def get_summary(self) -> Dict[str, Any]:
        """
        获取摘要

        Returns:
            Dict: 摘要信息
        """
        current = self.get_current_metrics()

        return {
            'current_metrics': current,
            'statistics': self.get_statistics(),
            'recent_alerts': [
                {
                    'level': a.level.value,
                    'metric': a.metric,
                    'value': a.value,
                    'threshold': a.threshold,
                    'message': a.message
                }
                for a in self.get_alerts()[-10:]
            ]
        }

    def export_metrics(self, format: str = 'dict') -> Any:
        """
        导出指标

        Args:
            format: 导出格式

        Returns:
            导出的数据
        """
        with self._lock:
            if format == 'dict':
                return {
                    'history': [
                        {'timestamp': s.timestamp, 'metrics': s.metrics}
                        for s in self._history
                    ],
                    'alerts': [
                        {
                            'level': a.level.value,
                            'metric': a.metric,
                            'value': a.value,
                            'threshold': a.threshold,
                            'timestamp': a.timestamp,
                            'message': a.message
                        }
                        for a in self._alerts
                    ]
                }
            elif format == 'prometheus':
                lines = []
                current = self.get_current_metrics()
                for name, value in current.items():
                    lines.append(f"mem_optimizer_{name} {value}")
                return '\n'.join(lines)
            else:
                return self.get_current_metrics()

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False
