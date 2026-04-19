"""
mem_monitor 可视化模块

实现内存数据的可视化展示。
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class ChartType(Enum):
    """图表类型"""
    LINE = "line"             # 折线图
    AREA = "area"             # 面积图
    BAR = "bar"               # 柱状图
    PIE = "pie"               # 饼图
    HEATMAP = "heatmap"       # 热力图
    SCATTER = "scatter"       # 散点图


@dataclass
class ChartConfig:
    """
    图表配置

    配置图表的显示选项。
    """

    title: str = ""                           # 标题
    x_label: str = "Time"                     # X轴标签
    y_label: str = "Value"                    # Y轴标签
    width: int = 800                          # 宽度
    height: int = 400                         # 高度
    show_legend: bool = True                  # 显示图例
    show_grid: bool = True                    # 显示网格
    color_scheme: str = "default"             # 配色方案
    animation: bool = True                    # 动画效果

    # 时间范围
    time_range: Optional[int] = None          # 时间范围（秒）

    # 阈值线
    thresholds: List[Dict[str, Any]] = field(default_factory=list)


class Visualizer:
    """
    可视化器

    提供内存数据的可视化功能。

    支持多种图表类型：
    - 折线图：显示指标随时间变化
    - 面积图：显示累积效果
    - 柱状图：显示对比数据
    - 饼图：显示分布比例
    - 热力图：显示访问模式
    """

    def __init__(self, config):
        """
        初始化可视化器

        Args:
            config: 报告配置
        """
        self._config = config
        self._backend = config.visualization_backend
        self._charts: Dict[str, Any] = {}

    def plot(self,
            data: List[Dict[str, Any]],
            chart_type: str = 'line',
            metrics: Optional[List[str]] = None,
            config: Optional[ChartConfig] = None) -> Optional[Any]:
        """
        绘制图表

        Args:
            data: 数据列表
            chart_type: 图表类型
            metrics: 要显示的指标列表
            config: 图表配置

        Returns:
            图表对象或None
        """
        if not data:
            return None

        config = config or ChartConfig()
        metrics = metrics or self._get_default_metrics()

        try:
            chart_type_enum = ChartType(chart_type)
        except ValueError:
            chart_type_enum = ChartType.LINE

        # 根据后端选择绘图方法
        if self._backend == 'matplotlib':
            return self._plot_matplotlib(data, chart_type_enum, metrics, config)
        elif self._backend == 'plotly':
            return self._plot_plotly(data, chart_type_enum, metrics, config)
        else:
            return self._generate_chart_data(data, chart_type_enum, metrics, config)

    def _get_default_metrics(self) -> List[str]:
        """获取默认指标"""
        return [
            'memory_usage_ratio',
            'heap_used',
            'allocation_rate',
        ]

    def _plot_matplotlib(self,
                        data: List[Dict[str, Any]],
                        chart_type: ChartType,
                        metrics: List[str],
                        config: ChartConfig) -> Optional[Any]:
        """使用matplotlib绘图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime

            fig, ax = plt.subplots(figsize=(config.width / 100, config.height / 100))

            if chart_type == ChartType.LINE:
                for metric in metrics:
                    values = [d.get(metric, 0) for d in data]
                    timestamps = [d.get('timestamp', i) for i, d in enumerate(data)]
                    ax.plot(timestamps, values, label=metric)

            elif chart_type == ChartType.AREA:
                for metric in metrics:
                    values = [d.get(metric, 0) for d in data]
                    timestamps = [d.get('timestamp', i) for i, d in enumerate(data)]
                    ax.fill_between(timestamps, values, alpha=0.3, label=metric)

            elif chart_type == ChartType.BAR:
                x = range(len(data))
                for i, metric in enumerate(metrics):
                    values = [d.get(metric, 0) for d in data]
                    ax.bar([xi + i * 0.2 for xi in x], values, width=0.2, label=metric)

            # 设置图表属性
            if config.title:
                ax.set_title(config.title)
            ax.set_xlabel(config.x_label)
            ax.set_ylabel(config.y_label)

            if config.show_legend:
                ax.legend()

            if config.show_grid:
                ax.grid(True, alpha=0.3)

            # 添加阈值线
            for threshold in config.thresholds:
                ax.axhline(y=threshold['value'], color='r', linestyle='--',
                          label=threshold.get('label', ''))

            plt.tight_layout()
            return fig

        except ImportError:
            return None

    def _plot_plotly(self,
                    data: List[Dict[str, Any]],
                    chart_type: ChartType,
                    metrics: List[str],
                    config: ChartConfig) -> Optional[Any]:
        """使用plotly绘图"""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            if chart_type == ChartType.LINE:
                for metric in metrics:
                    values = [d.get(metric, 0) for d in data]
                    timestamps = [d.get('timestamp', i) for i, d in enumerate(data)]
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name=metric
                    ))

            elif chart_type == ChartType.AREA:
                for metric in metrics:
                    values = [d.get(metric, 0) for d in data]
                    timestamps = [d.get('timestamp', i) for i, d in enumerate(data)]
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        fill='tozeroy',
                        name=metric
                    ))

            # 更新布局
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height,
                showlegend=config.show_legend,
            )

            return fig

        except ImportError:
            return None

    def _generate_chart_data(self,
                            data: List[Dict[str, Any]],
                            chart_type: ChartType,
                            metrics: List[str],
                            config: ChartConfig) -> Dict[str, Any]:
        """生成图表数据（用于前端渲染）"""
        chart_data = {
            'type': chart_type.value,
            'config': {
                'title': config.title,
                'x_label': config.x_label,
                'y_label': config.y_label,
                'width': config.width,
                'height': config.height,
                'show_legend': config.show_legend,
                'show_grid': config.show_grid,
            },
            'data': {},
        }

        for metric in metrics:
            values = [d.get(metric, 0) for d in data]
            timestamps = [d.get('timestamp', i) for i, d in enumerate(data)]
            chart_data['data'][metric] = {
                'timestamps': timestamps,
                'values': values,
            }

        return chart_data

    def create_dashboard(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        创建仪表板

        Args:
            data: 数据列表

        Returns:
            Dict: 仪表板配置
        """
        dashboard = {
            'title': 'Memory Monitor Dashboard',
            'charts': [],
            'generated_at': datetime.now().isoformat(),
        }

        # 内存使用趋势
        dashboard['charts'].append({
            'id': 'memory_trend',
            'title': 'Memory Usage Trend',
            'type': 'line',
            'metrics': ['memory_usage_ratio', 'memory_rss'],
            'position': {'row': 1, 'col': 1, 'width': 2, 'height': 1},
        })

        # 堆使用
        dashboard['charts'].append({
            'id': 'heap_usage',
            'title': 'Heap Usage',
            'type': 'area',
            'metrics': ['heap_used', 'heap_size'],
            'position': {'row': 1, 'col': 3, 'width': 1, 'height': 1},
        })

        # 分配速率
        dashboard['charts'].append({
            'id': 'allocation_rate',
            'title': 'Allocation Rate',
            'type': 'line',
            'metrics': ['allocation_rate', 'deallocation_rate'],
            'position': {'row': 2, 'col': 1, 'width': 2, 'height': 1},
        })

        # 碎片率
        dashboard['charts'].append({
            'id': 'fragmentation',
            'title': 'Fragmentation Ratio',
            'type': 'line',
            'metrics': ['fragmentation_ratio'],
            'position': {'row': 2, 'col': 3, 'width': 1, 'height': 1},
        })

        return dashboard


class ReportGenerator:
    """
    报告生成器

    生成完整的内存分析报告。
    """

    def __init__(self, config):
        self._config = config
        self._visualizer = Visualizer(config)

    def generate_html_report(self,
                            monitor_report,
                            include_charts: bool = True) -> str:
        """
        生成HTML报告

        Args:
            monitor_report: 监控报告
            include_charts: 是否包含图表

        Returns:
            str: HTML内容
        """
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="zh-CN">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <title>Memory Monitor Report</title>',
            '    <style>',
            self._get_report_css(),
            '    </style>',
            '</head>',
            '<body>',
            '    <div class="container">',
            '        <h1>Memory Monitor Report</h1>',
            f'        <p>Generated at: {datetime.now().isoformat()}</p>',
            self._generate_summary_section(monitor_report),
            self._generate_metrics_section(monitor_report),
            self._generate_alerts_section(monitor_report),
            self._generate_recommendations_section(monitor_report),
            '    </div>',
            '</body>',
            '</html>',
        ]

        return '\n'.join(html_parts)

    def _get_report_css(self) -> str:
        """获取报告CSS样式"""
        return '''
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            h2 { color: #666; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .section { margin: 20px 0; }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .metric-card {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #4CAF50;
            }
            .metric-value { font-size: 24px; font-weight: bold; color: #333; }
            .metric-label { color: #666; font-size: 14px; }
            .alert {
                padding: 10px 15px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
            .alert-error { background: #f8d7da; border-left: 4px solid #dc3545; }
            .alert-critical { background: #dc3545; color: white; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #f5f5f5; }
        '''

    def _generate_summary_section(self, report) -> str:
        """生成摘要部分"""
        summary = report.summary if hasattr(report, 'summary') else {}

        return f'''
            <div class="section">
                <h2>Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{summary.get("duration", 0):.1f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary.get("snapshots", 0)}</div>
                        <div class="metric-label">Snapshots</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary.get("peak_rss", 0) / 1024 / 1024:.1f} MB</div>
                        <div class="metric-label">Peak RSS</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary.get("alert_count", 0)}</div>
                        <div class="metric-label">Alerts</div>
                    </div>
                </div>
            </div>
        '''

    def _generate_metrics_section(self, report) -> str:
        """生成指标部分"""
        peak_metrics = report.peak_metrics if hasattr(report, 'peak_metrics') else {}
        avg_metrics = report.avg_metrics if hasattr(report, 'avg_metrics') else {}

        rows = []
        all_metrics = set(peak_metrics.keys()) | set(avg_metrics.keys())

        for metric in sorted(all_metrics):
            peak = peak_metrics.get(metric, 0)
            avg = avg_metrics.get(metric, 0)

            if isinstance(peak, float):
                peak_str = f"{peak:.2f}"
                avg_str = f"{avg:.2f}"
            else:
                peak_str = str(peak)
                avg_str = str(avg)

            rows.append(f'''
                <tr>
                    <td>{metric}</td>
                    <td>{peak_str}</td>
                    <td>{avg_str}</td>
                </tr>
            ''')

        return f'''
            <div class="section">
                <h2>Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Peak</th>
                            <th>Average</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
        '''

    def _generate_alerts_section(self, report) -> str:
        """生成告警部分"""
        alerts = report.alerts if hasattr(report, 'alerts') else []

        if not alerts:
            return '<div class="section"><h2>Alerts</h2><p>No alerts</p></div>'

        alert_items = []
        for alert in alerts[:20]:  # 限制显示数量
            level = alert.level.value if hasattr(alert, 'level') else 'warning'
            message = alert.message if hasattr(alert, 'message') else str(alert)

            alert_items.append(f'''
                <div class="alert alert-{level}">
                    <strong>{level.upper()}:</strong> {message}
                </div>
            ''')

        return f'''
            <div class="section">
                <h2>Alerts ({len(alerts)} total)</h2>
                {''.join(alert_items)}
            </div>
        '''

    def _generate_recommendations_section(self, report) -> str:
        """生成建议部分"""
        recommendations = []
        if hasattr(report, 'tiering_recommendations'):
            recommendations.extend(report.tiering_recommendations)

        if not recommendations:
            return '<div class="section"><h2>Recommendations</h2><p>No recommendations</p></div>'

        items = []
        for rec in recommendations[:10]:
            rec_type = rec.get('type', 'unknown')
            reason = rec.get('reason', '')

            items.append(f'<li><strong>{rec_type}:</strong> {reason}</li>')

        return f'''
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(items)}
                </ul>
            </div>
        '''

    def generate_markdown_report(self, monitor_report) -> str:
        """生成Markdown报告"""
        summary = monitor_report.summary if hasattr(monitor_report, 'summary') else {}

        lines = [
            '# Memory Monitor Report',
            '',
            f'Generated at: {datetime.now().isoformat()}',
            '',
            '## Summary',
            '',
            f'- Duration: {summary.get("duration", 0):.1f}s',
            f'- Snapshots: {summary.get("snapshots", 0)}',
            f'- Peak RSS: {summary.get("peak_rss", 0) / 1024 / 1024:.1f} MB',
            f'- Alerts: {summary.get("alert_count", 0)}',
            '',
            '## Peak Metrics',
            '',
        ]

        peak_metrics = monitor_report.peak_metrics if hasattr(monitor_report, 'peak_metrics') else {}
        for name, value in peak_metrics.items():
            if isinstance(value, float):
                lines.append(f'- {name}: {value:.2f}')
            else:
                lines.append(f'- {name}: {value}')

        return '\n'.join(lines)
