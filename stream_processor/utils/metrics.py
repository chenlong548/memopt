"""
stream_processor 指标收集

提供性能指标收集功能。
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import threading
import json


@dataclass
class MetricValue:
    """
    指标值

    存储单个指标的数据。
    """

    name: str

    value: float

    timestamp: float = field(default_factory=time.time)

    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


class MetricType:
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    METER = "meter"


class Metric:
    """
    指标基类

    定义指标的标准接口。
    """

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        初始化指标

        Args:
            name: 指标名称
            tags: 标签
        """
        self._name = name
        self._tags = tags or {}
        self._lock = threading.Lock()

    def get_name(self) -> str:
        """获取名称"""
        return self._name

    def get_tags(self) -> Dict[str, str]:
        """获取标签"""
        return self._tags.copy()


class Counter(Metric):
    """
    计数器

    单调递增的计数器。
    """

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        super().__init__(name, tags)
        self._count = 0

    def increment(self, delta: int = 1):
        """增加计数"""
        with self._lock:
            self._count += delta

    def decrement(self, delta: int = 1):
        """减少计数"""
        with self._lock:
            self._count -= delta

    def get_count(self) -> int:
        """获取计数"""
        with self._lock:
            return self._count

    def reset(self):
        """重置计数器"""
        with self._lock:
            self._count = 0


class Gauge(Metric):
    """
    仪表

    可增可减的仪表。
    """

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        super().__init__(name, tags)
        self._value = 0.0

    def set(self, value: float):
        """设置值"""
        with self._lock:
            self._value = value

    def increment(self, delta: float = 1.0):
        """增加值"""
        with self._lock:
            self._value += delta

    def decrement(self, delta: float = 1.0):
        """减少值"""
        with self._lock:
            self._value -= delta

    def get_value(self) -> float:
        """获取值"""
        with self._lock:
            return self._value


class Histogram(Metric):
    """
    直方图

    统计数据分布。
    """

    def __init__(self,
                 name: str,
                 tags: Optional[Dict[str, str]] = None,
                 bucket_size: int = 100):
        super().__init__(name, tags)
        self._bucket_size = bucket_size
        self._values: List[float] = []
        self._sum = 0.0
        self._count = 0

    def update(self, value: float):
        """更新值"""
        with self._lock:
            self._values.append(value)
            self._sum += value
            self._count += 1

            if len(self._values) > self._bucket_size:
                self._values.pop(0)

    def get_mean(self) -> float:
        """获取平均值"""
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    def get_min(self) -> float:
        """获取最小值"""
        with self._lock:
            if not self._values:
                return 0.0
            return min(self._values)

    def get_max(self) -> float:
        """获取最大值"""
        with self._lock:
            if not self._values:
                return 0.0
            return max(self._values)

    def get_percentile(self, percentile: float) -> float:
        """
        获取百分位数

        Args:
            percentile: 百分位（0-100）

        Returns:
            float: 百分位数值
        """
        with self._lock:
            if not self._values:
                return 0.0

            sorted_values = sorted(self._values)
            index = int(len(sorted_values) * percentile / 100)
            index = min(index, len(sorted_values) - 1)
            return sorted_values[index]

    def get_count(self) -> int:
        """获取计数"""
        with self._lock:
            return self._count


class Meter(Metric):
    """
    速率计

    测量事件速率。
    """

    def __init__(self,
                 name: str,
                 tags: Optional[Dict[str, str]] = None,
                 window_size: int = 60):
        super().__init__(name, tags)
        self._window_size = window_size
        self._events: List[float] = []
        self._total_count = 0

    def mark(self, count: int = 1):
        """标记事件"""
        with self._lock:
            current_time = time.time()
            self._events.append(current_time)
            self._total_count += count

            cutoff_time = current_time - self._window_size
            self._events = [t for t in self._events if t > cutoff_time]

    def get_rate(self) -> float:
        """获取速率（事件/秒）"""
        with self._lock:
            if not self._events:
                return 0.0

            elapsed = time.time() - self._events[0]
            if elapsed == 0:
                return 0.0

            return len(self._events) / elapsed

    def get_count(self) -> int:
        """获取总计数"""
        with self._lock:
            return self._total_count


class MetricsRegistry:
    """
    指标注册表

    管理所有指标。
    """

    def __init__(self):
        """初始化指标注册表"""
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Counter:
        """
        获取或创建计数器

        Args:
            name: 指标名称
            tags: 标签

        Returns:
            Counter: 计数器
        """
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Counter(name, tags)
            return self._metrics[key]

    def gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Gauge:
        """
        获取或创建仪表

        Args:
            name: 指标名称
            tags: 标签

        Returns:
            Gauge: 仪表
        """
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Gauge(name, tags)
            return self._metrics[key]

    def histogram(self, name: str, tags: Optional[Dict[str, str]] = None) -> Histogram:
        """
        获取或创建直方图

        Args:
            name: 指标名称
            tags: 标签

        Returns:
            Histogram: 直方图
        """
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Histogram(name, tags)
            return self._metrics[key]

    def meter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Meter:
        """
        获取或创建速率计

        Args:
            name: 指标名称
            tags: 标签

        Returns:
            Meter: 速率计
        """
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Meter(name, tags)
            return self._metrics[key]

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """生成键"""
        if not tags:
            return name

        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标

        Returns:
            Dict: 指标字典
        """
        result = {}
        with self._lock:
            for key, metric in self._metrics.items():
                if isinstance(metric, Counter):
                    result[key] = {
                        'type': 'counter',
                        'value': metric.get_count()
                    }
                elif isinstance(metric, Gauge):
                    result[key] = {
                        'type': 'gauge',
                        'value': metric.get_value()
                    }
                elif isinstance(metric, Histogram):
                    result[key] = {
                        'type': 'histogram',
                        'mean': metric.get_mean(),
                        'min': metric.get_min(),
                        'max': metric.get_max(),
                        'count': metric.get_count()
                    }
                elif isinstance(metric, Meter):
                    result[key] = {
                        'type': 'meter',
                        'rate': metric.get_rate(),
                        'count': metric.get_count()
                    }

        return result

    def clear(self):
        """清空所有指标"""
        with self._lock:
            self._metrics.clear()


class MetricsCollector:
    """
    指标收集器

    收集和报告指标。
    """

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        """
        初始化指标收集器

        Args:
            registry: 指标注册表
        """
        self._registry = registry or MetricsRegistry()
        self._reporters: List[Callable] = []
        self._report_interval = 10.0
        self._report_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动收集器"""
        if self._running:
            return

        self._running = True
        self._report_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._report_thread.start()

    def stop(self):
        """停止收集器"""
        self._running = False
        if self._report_thread:
            self._report_thread.join(timeout=2.0)
            self._report_thread = None

    def _report_loop(self):
        """报告循环"""
        while self._running:
            try:
                self._report()
                time.sleep(self._report_interval)
            except Exception:
                pass

    def _report(self):
        """报告指标"""
        metrics = self._registry.get_all_metrics()
        for reporter in self._reporters:
            try:
                reporter(metrics)
            except Exception:
                pass

    def add_reporter(self, reporter: Callable):
        """添加报告器"""
        self._reporters.append(reporter)

    def get_registry(self) -> MetricsRegistry:
        """获取注册表"""
        return self._registry
