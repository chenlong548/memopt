"""
mem_optimizer PSI指标监控

实现PSI (Pressure Stall Information) 指标监控。
"""

import os
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class PSIType(Enum):
    """PSI类型"""
    SOME = "some"
    FULL = "full"


@dataclass
class PSIMetrics:
    """PSI指标"""
    some_avg10: float = 0.0
    some_avg60: float = 0.0
    some_avg300: float = 0.0
    some_total: int = 0

    full_avg10: float = 0.0
    full_avg60: float = 0.0
    full_avg300: float = 0.0
    full_total: int = 0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'some': {
                'avg10': self.some_avg10,
                'avg60': self.some_avg60,
                'avg300': self.some_avg300,
                'total': self.some_total
            },
            'full': {
                'avg10': self.full_avg10,
                'avg60': self.full_avg60,
                'avg300': self.full_avg300,
                'total': self.full_total
            },
            'timestamp': self.timestamp
        }


@dataclass
class PSIAlert:
    """PSI告警"""
    metric_type: str
    threshold: float
    current_value: float
    timestamp: float
    message: str


class PSIMonitor:
    """
    PSI指标监控器

    监控系统内存压力指标。
    """

    def __init__(self,
                psi_path: str = "/proc/pressure/memory",
                sample_interval: float = 1.0,
                history_size: int = 3600):
        """
        初始化PSI监控器

        Args:
            psi_path: PSI文件路径
            sample_interval: 采样间隔（秒）
            history_size: 历史记录大小
        """
        self.psi_path = psi_path
        self.sample_interval = sample_interval
        self.history_size = history_size

        self._history: deque = deque(maxlen=history_size)
        self._alerts: List[PSIAlert] = []
        self._alert_thresholds: Dict[str, float] = {
            'some_avg10': 0.1,
            'full_avg10': 0.05,
            'some_avg60': 0.2,
            'full_avg60': 0.1
        }

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_metrics: Optional[PSIMetrics] = None
        self._is_available = self._check_availability()

    def _check_availability(self) -> bool:
        """检查PSI是否可用"""
        if os.name == 'nt':
            return False

        return os.path.exists(self.psi_path)

    def _parse_psi_line(self, line: str) -> Dict[str, Any]:
        """
        解析PSI行

        Args:
            line: PSI行

        Returns:
            Dict: 解析结果
        """
        result = {}

        parts = line.strip().split()
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                try:
                    if key == 'total':
                        result[key] = int(value)
                    else:
                        result[key] = float(value)
                except ValueError:
                    pass

        return result

    def read_psi(self) -> Optional[PSIMetrics]:
        """
        读取PSI指标

        Returns:
            PSIMetrics: PSI指标
        """
        if not self._is_available:
            return self._generate_mock_metrics()

        try:
            with open(self.psi_path, 'r') as f:
                content = f.read()

            lines = content.strip().split('\n')

            metrics = PSIMetrics()

            for line in lines:
                if line.startswith('some'):
                    data = self._parse_psi_line(line[5:])
                    metrics.some_avg10 = data.get('avg10', 0.0)
                    metrics.some_avg60 = data.get('avg60', 0.0)
                    metrics.some_avg300 = data.get('avg300', 0.0)
                    metrics.some_total = data.get('total', 0)

                elif line.startswith('full'):
                    data = self._parse_psi_line(line[5:])
                    metrics.full_avg10 = data.get('avg10', 0.0)
                    metrics.full_avg60 = data.get('avg60', 0.0)
                    metrics.full_avg300 = data.get('avg300', 0.0)
                    metrics.full_total = data.get('total', 0)

            metrics.timestamp = time.time()

            return metrics

        except Exception:
            return self._generate_mock_metrics()

    def _generate_mock_metrics(self) -> PSIMetrics:
        """生成模拟指标（用于不支持PSI的系统）"""
        import random

        return PSIMetrics(
            some_avg10=random.uniform(0, 0.05),
            some_avg60=random.uniform(0, 0.03),
            some_avg300=random.uniform(0, 0.02),
            some_total=int(time.time() * 1000000),
            full_avg10=random.uniform(0, 0.01),
            full_avg60=random.uniform(0, 0.005),
            full_avg300=random.uniform(0, 0.003),
            full_total=int(time.time() * 1000000),
            timestamp=time.time()
        )

    def _check_alerts(self, metrics: PSIMetrics):
        """检查告警"""
        metric_values = {
            'some_avg10': metrics.some_avg10,
            'some_avg60': metrics.some_avg60,
            'some_avg300': metrics.some_avg300,
            'full_avg10': metrics.full_avg10,
            'full_avg60': metrics.full_avg60,
            'full_avg300': metrics.full_avg300
        }

        for metric_name, value in metric_values.items():
            threshold = self._alert_thresholds.get(metric_name, float('inf'))

            if value > threshold:
                alert = PSIAlert(
                    metric_type=metric_name,
                    threshold=threshold,
                    current_value=value,
                    timestamp=time.time(),
                    message=f"PSI {metric_name} exceeded threshold: {value:.4f} > {threshold}"
                )
                self._alerts.append(alert)

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                metrics = self.read_psi()

                if metrics:
                    with self._lock:
                        self._history.append(metrics)
                        self._last_metrics = metrics
                        self._check_alerts(metrics)

                time.sleep(self.sample_interval)

            except Exception:
                time.sleep(self.sample_interval)

    def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def stop(self):
        """停止监控"""
        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def get_current_metrics(self) -> Optional[PSIMetrics]:
        """
        获取当前指标

        Returns:
            PSIMetrics: 当前指标
        """
        with self._lock:
            return self._last_metrics

    def get_history(self, count: Optional[int] = None) -> List[PSIMetrics]:
        """
        获取历史记录

        Args:
            count: 记录数量

        Returns:
            List[PSIMetrics]: 历史记录
        """
        with self._lock:
            if count is None:
                return list(self._history)
            return list(self._history)[-count:]

    def get_alerts(self, clear: bool = False) -> List[PSIAlert]:
        """
        获取告警

        Args:
            clear: 是否清除告警

        Returns:
            List[PSIAlert]: 告警列表
        """
        with self._lock:
            alerts = list(self._alerts)
            if clear:
                self._alerts.clear()
            return alerts

    def set_alert_threshold(self, metric_name: str, threshold: float):
        """
        设置告警阈值

        Args:
            metric_name: 指标名称
            threshold: 阈值
        """
        self._alert_thresholds[metric_name] = threshold

    def get_pressure_level(self) -> str:
        """
        获取压力级别

        Returns:
            str: 压力级别 (low, medium, high, critical)
        """
        metrics = self.get_current_metrics()

        if metrics is None:
            return "unknown"

        some_pressure = metrics.some_avg10
        full_pressure = metrics.full_avg10

        if full_pressure > 0.1:
            return "critical"
        elif full_pressure > 0.05 or some_pressure > 0.3:
            return "high"
        elif some_pressure > 0.1:
            return "medium"
        else:
            return "low"

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        with self._lock:
            return {
                'available': self._is_available,
                'running': self._running,
                'history_size': len(self._history),
                'alert_count': len(self._alerts),
                'current_metrics': self._last_metrics.to_dict() if self._last_metrics else None,
                'pressure_level': self.get_pressure_level(),
                'alert_thresholds': self._alert_thresholds
            }

    def is_available(self) -> bool:
        """
        检查是否可用

        Returns:
            bool: 是否可用
        """
        return self._is_available

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False
