"""
mem_monitor 指标收集模块

实现内存指标的收集、聚合和导出。
"""

import time
import json
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime


class MetricType(Enum):
    """指标类型"""
    GAUGE = "gauge"           # 瞬时值
    COUNTER = "counter"       # 累计值
    HISTOGRAM = "histogram"   # 直方图
    SUMMARY = "summary"       # 摘要


@dataclass
class MetricValue:
    """
    指标值

    记录单个指标的值。
    """

    name: str                               # 指标名称
    value: float                            # 指标值
    metric_type: MetricType = MetricType.GAUGE
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'labels': self.labels,
            'description': self.description,
        }


@dataclass
class AggregatedMetrics:
    """
    聚合指标

    记录聚合后的指标统计。
    """

    name: str                               # 指标名称
    count: int = 0                          # 样本数
    sum: float = 0.0                        # 总和
    min: float = float('inf')               # 最小值
    max: float = float('-inf')              # 最大值
    avg: float = 0.0                        # 平均值
    last: float = 0.0                       # 最后值
    first: float = 0.0                      # 首次值

    # 百分位数
    p50: float = 0.0                        # 中位数
    p90: float = 0.0                        # 90分位
    p95: float = 0.0                        # 95分位
    p99: float = 0.0                        # 99分位

    # 时间范围
    start_time: float = 0.0
    end_time: float = 0.0

    def update(self, value: float, timestamp: float):
        """更新聚合"""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = self.sum / self.count
        self.last = value

        if self.first == 0:
            self.first = value

        if self.start_time == 0:
            self.start_time = timestamp
        self.end_time = timestamp

    def calculate_percentiles(self, values: List[float]):
        """计算百分位数"""
        if not values:
            return

        sorted_values = sorted(values)
        n = len(sorted_values)

        self.p50 = sorted_values[int(n * 0.5)]
        self.p90 = sorted_values[int(n * 0.9)]
        self.p95 = sorted_values[int(n * 0.95)]
        self.p99 = sorted_values[int(n * 0.99)]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'count': self.count,
            'sum': self.sum,
            'min': self.min,
            'max': self.max,
            'avg': self.avg,
            'last': self.last,
            'first': self.first,
            'p50': self.p50,
            'p90': self.p90,
            'p95': self.p95,
            'p99': self.p99,
            'duration': self.end_time - self.start_time,
        }


class MetricsCollector:
    """
    指标收集器

    收集、聚合和导出内存指标。

    功能：
    1. 收集多种类型的指标
    2. 支持滑动窗口聚合
    3. 支持多种导出格式
    4. 支持自定义指标
    """

    def __init__(self, config):
        """
        初始化指标收集器

        Args:
            config: 报告配置
        """
        self._config = config

        # 指标存储
        self._metrics: Dict[str, MetricValue] = {}
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.history_size)
        )

        # 聚合指标
        self._aggregated: Dict[str, AggregatedMetrics] = defaultdict(
            lambda: AggregatedMetrics(name="")
        )

        # 自定义指标
        self._custom_collectors: Dict[str, Callable[[], float]] = {}

        # 统计
        self._collection_count = 0
        self._last_collection_time = 0

    def register_collector(self, name: str, collector: Callable[[], float]):
        """
        注册自定义指标收集器

        Args:
            name: 指标名称
            collector: 收集函数
        """
        self._custom_collectors[name] = collector

    def collect(self, snapshot) -> Dict[str, Any]:
        """
        收集指标

        Args:
            snapshot: 内存快照

        Returns:
            Dict: 收集的指标
        """
        self._last_collection_time = time.time()
        self._collection_count += 1

        metrics = {}

        # 收集基本指标
        basic_metrics = self._collect_basic_metrics(snapshot)
        metrics.update(basic_metrics)

        # 收集自定义指标
        custom_metrics = self._collect_custom_metrics()
        metrics.update(custom_metrics)

        # 更新历史和聚合
        for name, value in metrics.items():
            self._history[name].append((self._last_collection_time, value))
            self._aggregated[name].name = name
            self._aggregated[name].update(value, self._last_collection_time)

        return metrics

    def _collect_basic_metrics(self, snapshot) -> Dict[str, float]:
        """收集基本指标"""
        metrics = {
            # 内存使用
            'memory_rss': float(snapshot.rss),
            'memory_vms': float(snapshot.vms),
            'memory_shared': float(snapshot.shared),
            'memory_available': float(snapshot.available),
            'memory_total': float(snapshot.total),
            'memory_usage_ratio': snapshot.get_usage_ratio(),

            # Python堆
            'heap_size': float(snapshot.heap_size),
            'heap_used': float(snapshot.heap_used),

            # 分配统计
            'allocation_count': float(snapshot.allocation_count),
            'deallocation_count': float(snapshot.deallocation_count),
            'allocation_rate': snapshot.allocation_rate,
            'deallocation_rate': snapshot.deallocation_rate,

            # 碎片
            'fragmentation_ratio': snapshot.fragmentation_ratio,
        }

        # GC指标
        if hasattr(snapshot, 'gc_count'):
            for i, count in enumerate(snapshot.gc_count):
                metrics[f'gc_gen{i}_count'] = float(count)

        return metrics

    def _collect_custom_metrics(self) -> Dict[str, float]:
        """收集自定义指标"""
        metrics = {}

        for name, collector in self._custom_collectors.items():
            try:
                value = collector()
                metrics[name] = value
            except Exception:
                pass

        return metrics

    def get_metric(self, name: str) -> Optional[MetricValue]:
        """获取指标"""
        return self._metrics.get(name)

    def get_history(self,
                   name: str,
                   count: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        获取指标历史

        Args:
            name: 指标名称
            count: 返回数量

        Returns:
            List[Tuple]: (timestamp, value) 列表
        """
        history = list(self._history.get(name, []))

        if count is not None:
            history = history[-count:]

        return history

    def get_aggregated(self, name: str) -> Optional[AggregatedMetrics]:
        """获取聚合指标"""
        return self._aggregated.get(name)

    def get_all_aggregated(self) -> Dict[str, AggregatedMetrics]:
        """获取所有聚合指标"""
        # 计算百分位数
        for name, agg in self._aggregated.items():
            values = [v for _, v in self._history.get(name, [])]
            agg.calculate_percentiles(values)

        return dict(self._aggregated)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'collection_count': self._collection_count,
            'last_collection_time': self._last_collection_time,
            'metrics': {name: metric.to_dict() for name, metric in self._metrics.items()},
            'aggregated': {name: agg.to_dict() for name, agg in self.get_all_aggregated().items()},
        }

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2)

    def to_prometheus(self) -> str:
        """转换为Prometheus格式"""
        lines = []

        for name, metric in self._metrics.items():
            # 添加帮助信息
            if metric.description:
                lines.append(f"# HELP mem_monitor_{name} {metric.description}")

            # 添加类型信息
            lines.append(f"# TYPE mem_monitor_{name} {metric.metric_type.value}")

            # 添加标签
            labels = ""
            if metric.labels:
                labels = "{" + ",".join(f'{k}="{v}"' for k, v in metric.labels.items()) + "}"

            # 添加值
            lines.append(f"mem_monitor_{name}{labels} {metric.value}")

        return '\n'.join(lines)

    def clear(self):
        """清除数据"""
        self._metrics.clear()
        for history in self._history.values():
            history.clear()
        self._aggregated.clear()


from typing import Tuple
