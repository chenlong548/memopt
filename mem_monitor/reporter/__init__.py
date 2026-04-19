"""
mem_monitor 报告层

提供指标收集和可视化功能。
"""

from typing import List, Dict, Any

from .metrics import (
    MetricsCollector,
    MetricType,
    MetricValue,
    AggregatedMetrics,
)
from .visualization import (
    Visualizer,
    ChartType,
    ChartConfig,
    ReportGenerator,
)

__all__ = [
    # 指标
    'MetricsCollector',
    'MetricType',
    'MetricValue',
    'AggregatedMetrics',
    # 可视化
    'Visualizer',
    'ChartType',
    'ChartConfig',
    'ReportGenerator',
]


class Reporter:
    """
    报告器

    整合指标收集和可视化功能。
    """

    def __init__(self, config):
        self._config = config
        self._metrics_collector = MetricsCollector(config)
        self._visualizer = Visualizer(config) if config.enable_visualization else None
        self._history = []

    def collect(self, snapshot) -> Dict[str, Any]:
        """收集指标"""
        metrics = self._metrics_collector.collect(snapshot)
        self._history.append(metrics)
        return metrics

    def generate_report(self, format: str = 'dict') -> Any:
        """生成报告"""
        if format == 'dict':
            return self._metrics_collector.to_dict()
        elif format == 'json':
            return self._metrics_collector.to_json()
        elif format == 'prometheus':
            return self._metrics_collector.to_prometheus()
        return self._metrics_collector.to_dict()

    def visualize(self, chart_type: str = 'line', **kwargs):
        """生成可视化"""
        if self._visualizer:
            return self._visualizer.plot(self._history, chart_type, **kwargs)
        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self._history.copy()
