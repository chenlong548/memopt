"""
mem_monitor - 内存监控工具

一个专业的内存监控工具模块，提供完整的内存监控、分析和报告功能。

主要功能：
- 多种采样方式（硬件采样、软件采样、自适应采样）
- 生命周期分析
- 热点检测
- 内存泄漏检测
- NUMA感知分层管理
- 可视化报告

使用示例：
    ```python
    from mem_monitor import MemoryMonitor, MonitorConfig

    # 创建监控器
    config = MonitorConfig()
    monitor = MemoryMonitor(config)

    # 启动监控
    monitor.start()

    # 获取快照
    snapshot = monitor.get_snapshot()
    print(f"Memory usage: {snapshot.get_usage_ratio():.2%}")

    # 设置阈值
    monitor.set_threshold('memory_usage', 0.9, 'alert')

    # 注册告警回调
    def on_alert(alert):
        print(f"Alert: {alert.message}")

    monitor.add_alert_handler(on_alert)

    # 停止监控并获取报告
    report = monitor.stop()
    print(f"Peak RSS: {report.peak_metrics.get('peak_rss', 0) / 1024 / 1024:.1f} MB")
    ```

架构设计：
- 核心层（core）：MemoryMonitor主类、配置管理、异常定义
- 采样层（sampler）：硬件采样器、软件采样器、自适应采样
- 分析层（analyzer）：生命周期分析、热点分析、泄漏检测
- 分层管理层（tiering）：页面管理、NUMA感知
- 报告层（reporter）：指标收集、可视化
- 集成层（integration）：psutil、tracemalloc适配器

学术研究基础：
- PROMPT：快速可扩展内存分析框架
- Examem：低开销内存插桩
- GainSight：细粒度内存访问模式分析
- FlexMem：自适应页面分析+迁移
- NOMAD：非独占分层+事务迁移
"""

from typing import Optional

__version__ = "1.0.0"
__author__ = "Memory Monitor Team"

# 核心类
from .core import (
    MemoryMonitor,
    MemorySnapshot,
    MonitorReport,
    MonitorState,
    MonitorConfig,
    SamplerConfig,
    AnalyzerConfig,
    TieringConfig,
    ReporterConfig,
    MonitorError,
    ConfigurationError,
)

# 采样器
from .sampler import (
    SamplerBase,
    SampleData,
    SamplerState,
    create_sampler,
)

# 分析器
from .analyzer import (
    LifecycleAnalyzer,
    HotspotAnalyzer,
    LeakDetector,
    create_analyzer,
)

# 分层管理
from .tiering import (
    PageManager,
    NUMAAwareManager,
    TieringManager,
)

# 报告
from .reporter import (
    MetricsCollector,
    Visualizer,
    Reporter,
)

# 集成
from .integration import (
    PsutilAdapter,
    TracemallocAdapter,
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',

    # 核心类
    'MemoryMonitor',
    'MemorySnapshot',
    'MonitorReport',
    'MonitorState',
    'MonitorConfig',
    'SamplerConfig',
    'AnalyzerConfig',
    'TieringConfig',
    'ReporterConfig',

    # 异常
    'MonitorError',
    'ConfigurationError',

    # 采样器
    'SamplerBase',
    'SampleData',
    'SamplerState',
    'create_sampler',

    # 分析器
    'LifecycleAnalyzer',
    'HotspotAnalyzer',
    'LeakDetector',
    'create_analyzer',

    # 分层管理
    'PageManager',
    'NUMAAwareManager',
    'TieringManager',

    # 报告
    'MetricsCollector',
    'Visualizer',
    'Reporter',

    # 集成
    'PsutilAdapter',
    'TracemallocAdapter',
]


def create_monitor(config: Optional[MonitorConfig] = None) -> MemoryMonitor:
    """
    创建内存监控器的便捷函数

    Args:
        config: 监控配置，如果为None则使用默认配置

    Returns:
        MemoryMonitor: 内存监控器实例
    """
    return MemoryMonitor(config)


def quick_start(interval: float = 1.0) -> MemoryMonitor:
    """
    快速启动监控的便捷函数

    Args:
        interval: 采样间隔（秒）

    Returns:
        MemoryMonitor: 已启动的内存监控器实例
    """
    config = MonitorConfig()
    config.sampler.sample_interval = interval

    monitor = MemoryMonitor(config)
    monitor.start()

    return monitor
