"""
mem_monitor 配置管理模块

定义内存监控器的所有配置选项。
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

# 配置模块日志记录器
logger = logging.getLogger(__name__)

# 平台检测
IS_32BIT = sys.maxsize <= 2**32

# 安全常量
if IS_32BIT:
    MAX_SAFE_MEMORY = 2 * 1024 * 1024 * 1024  # 2GB
    DEFAULT_HISTORY_SIZE = 1800  # 30分钟 @ 1秒间隔
else:
    MAX_SAFE_MEMORY = 1024 * 1024 * 1024 * 1024  # 1TB
    DEFAULT_HISTORY_SIZE = 3600  # 1小时 @ 1秒间隔

# 配置验证常量
MIN_SAMPLE_INTERVAL = 0.001  # 最小采样间隔（秒）
MAX_SAMPLE_INTERVAL = 3600.0  # 最大采样间隔（秒）
MIN_BUFFER_SIZE = 100
MAX_BUFFER_SIZE = 1000000
MIN_WARMUP_SAMPLES = 1
MAX_WARMUP_SAMPLES = 10000


class SamplerType(Enum):
    """采样器类型"""
    HARDWARE_PEBS = "hardware_pebs"      # Intel PEBS
    HARDWARE_IBS = "hardware_ibs"        # AMD IBS
    SOFTWARE_EBPF = "software_ebpf"      # eBPF
    SOFTWARE_PERF = "software_perf"      # Linux perf
    SOFTWARE_TRACEMALLOC = "tracemalloc"  # Python tracemalloc
    SOFTWARE_PSUTIL = "psutil"           # psutil
    ADAPTIVE = "adaptive"                # 自适应采样
    AUTO = "auto"                        # 自动选择


class AnalyzerType(Enum):
    """分析器类型"""
    LIFECYCLE = "lifecycle"              # 生命周期分析
    HOTSPOT = "hotspot"                  # 热点分析
    LEAK_DETECTION = "leak_detection"    # 内存泄漏检测
    FRAGMENTATION = "fragmentation"      # 碎片分析
    ACCESS_PATTERN = "access_pattern"    # 访问模式分析


class TieringPolicy(Enum):
    """分层策略"""
    NONE = "none"                        # 不分层
    MANUAL = "manual"                    # 手动分层
    AUTO = "auto"                        # 自动分层
    ADAPTIVE = "adaptive"                # 自适应分层


class ReportFormat(Enum):
    """报告格式"""
    DICT = "dict"                        # 字典
    JSON = "json"                        # JSON
    PROMETHEUS = "prometheus"            # Prometheus格式
    HTML = "html"                        # HTML报告
    MARKDOWN = "markdown"                # Markdown


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ThresholdConfig:
    """
    阈值配置

    定义单个指标的阈值配置。
    """

    warning: float = 0.8          # 警告阈值
    error: float = 0.9            # 错误阈值
    critical: float = 0.95        # 严重阈值
    action: str = "alert"         # 触发动作: alert, log, callback
    callback: Optional[Callable] = None  # 回调函数


@dataclass
class SamplerConfig:
    """
    采样器配置

    配置采样行为。
    """

    sampler_type: SamplerType = SamplerType.AUTO
    sample_interval: float = 1.0           # 采样间隔（秒）
    adaptive: bool = True                   # 是否启用自适应采样
    min_interval: float = 0.1               # 最小采样间隔
    max_interval: float = 10.0              # 最大采样间隔
    warmup_samples: int = 10                # 预热样本数
    buffer_size: int = 10000                # 采样缓冲区大小

    # 硬件采样特定配置
    pebs_event: str = "mem_inst_retired.all"  # PEBS事件
    pebs_period: int = 1000                    # PEBS采样周期
    ibs_op_enable: bool = True                 # IBS操作采样
    ibs_fetch_enable: bool = False             # IBS取指采样

    # 软件采样特定配置
    tracemalloc_frames: int = 25               # tracemalloc栈帧数
    ebpf_program: str = ""                     # eBPF程序路径

    def validate(self) -> bool:
        """验证配置"""
        if self.sample_interval <= 0:
            return False
        if self.min_interval <= 0 or self.max_interval <= 0:
            return False
        if self.min_interval > self.max_interval:
            return False
        return True


@dataclass
class AnalyzerConfig:
    """
    分析器配置

    配置分析行为。
    """

    enabled_analyzers: List[AnalyzerType] = field(
        default_factory=lambda: [
            AnalyzerType.LIFECYCLE,
            AnalyzerType.HOTSPOT,
            AnalyzerType.LEAK_DETECTION
        ]
    )

    # 生命周期分析配置
    lifecycle_track_allocations: bool = True
    lifecycle_max_objects: int = 100000      # 最大追踪对象数
    lifecycle_sample_rate: float = 1.0       # 采样率 (0.0-1.0)

    # 热点分析配置
    hotspot_window_size: int = 1000          # 滑动窗口大小
    hotspot_threshold: float = 0.8           # 热点阈值
    hotspot_min_samples: int = 100           # 最小样本数

    # 泄漏检测配置
    leak_detection_interval: float = 60.0    # 检测间隔（秒）
    leak_growth_threshold: float = 0.1       # 增长阈值
    leak_min_age: float = 300.0              # 最小存活时间（秒）

    # 碎片分析配置
    fragmentation_bins: int = 16             # 碎片分析分箱数

    # 访问模式分析配置
    access_pattern_history: int = 10000      # 访问历史大小


@dataclass
class TieringConfig:
    """
    分层管理配置

    配置内存分层行为。
    """

    policy: TieringPolicy = TieringPolicy.AUTO
    enable_numa_aware: bool = True           # 启用NUMA感知
    enable_page_tracking: bool = True        # 启用页面追踪

    # 页面热度评估
    hot_page_threshold: float = 0.7          # 热页面阈值
    cold_page_threshold: float = 0.3         # 冷页面阈值
    page_scan_interval: float = 30.0         # 页面扫描间隔

    # 迁移配置
    enable_migration: bool = True            # 启用迁移
    migration_batch_size: int = 100          # 迁移批次大小
    migration_interval: float = 60.0         # 迁移间隔

    # NUMA配置
    numa_balance_threshold: float = 0.2      # NUMA平衡阈值
    numa_prefer_local: bool = True           # 优先本地节点


@dataclass
class ReporterConfig:
    """
    报告器配置

    配置报告生成行为。
    """

    enable_metrics: bool = True              # 启用指标收集
    enable_visualization: bool = True        # 启用可视化
    history_size: int = DEFAULT_HISTORY_SIZE # 历史记录大小

    # 指标配置
    metrics_interval: float = 1.0            # 指标收集间隔
    metrics_aggregation: str = "sliding"     # 聚合方式: sliding, tumbling

    # 报告格式
    default_format: ReportFormat = ReportFormat.DICT

    # 可视化配置
    visualization_backend: str = "matplotlib"  # matplotlib, plotly
    enable_realtime: bool = False             # 实时可视化

    # 导出配置
    export_path: str = ""                     # 导出路径
    auto_export: bool = False                 # 自动导出
    export_interval: float = 300.0            # 导出间隔


@dataclass
class IntegrationConfig:
    """
    集成配置

    配置与外部模块的集成。
    """

    # psutil集成
    enable_psutil: bool = True
    psutil_interval: float = 1.0

    # tracemalloc集成
    enable_tracemalloc: bool = True
    tracemalloc_frames: int = 25

    # mem_mapper集成
    enable_mem_mapper: bool = True

    # mem_optimizer集成
    enable_mem_optimizer: bool = True

    # data_compressor集成
    enable_data_compressor: bool = False

    # stream_processor集成
    enable_stream_processor: bool = False


@dataclass
class MonitorConfig:
    """
    监控器主配置

    整合所有配置选项。
    """

    # 基本信息
    name: str = "MemoryMonitor"
    version: str = "1.0.0"

    # 子配置
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    tiering: TieringConfig = field(default_factory=TieringConfig)
    reporter: ReporterConfig = field(default_factory=ReporterConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    # 阈值配置
    thresholds: Dict[str, ThresholdConfig] = field(default_factory=lambda: {
        'memory_usage': ThresholdConfig(warning=0.8, error=0.9, critical=0.95),
        'fragmentation': ThresholdConfig(warning=0.5, error=0.7, critical=0.9),
        'leak_growth': ThresholdConfig(warning=0.1, error=0.2, critical=0.3),
        'allocation_rate': ThresholdConfig(warning=1000, error=5000, critical=10000),
    })

    # 全局开关
    enable_monitoring: bool = True
    enable_alerts: bool = True
    enable_hooks: bool = True
    thread_safe: bool = True

    # 性能配置
    max_overhead: float = 0.05               # 最大允许开销 (5%)
    background_thread: bool = True           # 后台线程
    daemon_thread: bool = True               # 守护线程

    def validate(self) -> bool:
        """验证配置"""
        if not self.sampler.validate():
            return False
        if self.max_overhead < 0 or self.max_overhead > 1:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'enable_monitoring': self.enable_monitoring,
            'enable_alerts': self.enable_alerts,
            'enable_hooks': self.enable_hooks,
            'thread_safe': self.thread_safe,
            'max_overhead': self.max_overhead,
            'background_thread': self.background_thread,
            'daemon_thread': self.daemon_thread,
            'sampler': {
                'type': self.sampler.sampler_type.value,
                'interval': self.sampler.sample_interval,
                'adaptive': self.sampler.adaptive,
            },
            'analyzer': {
                'enabled': [a.value for a in self.analyzer.enabled_analyzers],
            },
            'tiering': {
                'policy': self.tiering.policy.value,
                'numa_aware': self.tiering.enable_numa_aware,
            },
            'reporter': {
                'history_size': self.reporter.history_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitorConfig':
        """
        从字典创建配置
        
        Args:
            data: 配置字典
            
        Returns:
            MonitorConfig: 配置对象
            
        Raises:
            ConfigurationError: 配置验证失败时抛出
        """
        from .exceptions import ConfigurationError
        
        # 验证输入数据类型
        if not isinstance(data, dict):
            raise ConfigurationError(
                "Configuration data must be a dictionary",
                "data"
            )
        
        config = cls()

        # 验证并设置基本配置
        if 'name' in data:
            if not isinstance(data['name'], str):
                raise ConfigurationError("name must be a string", "name")
            if len(data['name']) > 256:
                raise ConfigurationError("name must be less than 256 characters", "name")
            config.name = data['name']
            
        if 'version' in data:
            if not isinstance(data['version'], str):
                raise ConfigurationError("version must be a string", "version")
            config.version = data['version']
            
        if 'enable_monitoring' in data:
            if not isinstance(data['enable_monitoring'], bool):
                raise ConfigurationError("enable_monitoring must be a boolean", "enable_monitoring")
            config.enable_monitoring = data['enable_monitoring']
            
        if 'enable_alerts' in data:
            if not isinstance(data['enable_alerts'], bool):
                raise ConfigurationError("enable_alerts must be a boolean", "enable_alerts")
            config.enable_alerts = data['enable_alerts']
            
        if 'max_overhead' in data:
            if not isinstance(data['max_overhead'], (int, float)):
                raise ConfigurationError("max_overhead must be a number", "max_overhead")
            if not 0 <= data['max_overhead'] <= 1:
                raise ConfigurationError("max_overhead must be between 0 and 1", "max_overhead")
            config.max_overhead = data['max_overhead']

        # 解析子配置
        if 'sampler' in data:
            sd = data['sampler']
            if not isinstance(sd, dict):
                raise ConfigurationError("sampler configuration must be a dictionary", "sampler")
                
            if 'type' in sd:
                try:
                    config.sampler.sampler_type = SamplerType(sd['type'])
                except ValueError as e:
                    raise ConfigurationError(f"Invalid sampler type: {sd['type']}", "sampler.type")
                    
            if 'interval' in sd:
                if not isinstance(sd['interval'], (int, float)):
                    raise ConfigurationError("sampler.interval must be a number", "sampler.interval")
                if not MIN_SAMPLE_INTERVAL <= sd['interval'] <= MAX_SAMPLE_INTERVAL:
                    raise ConfigurationError(
                        f"sampler.interval must be between {MIN_SAMPLE_INTERVAL} and {MAX_SAMPLE_INTERVAL}",
                        "sampler.interval"
                    )
                config.sampler.sample_interval = sd['interval']
                
            if 'adaptive' in sd:
                if not isinstance(sd['adaptive'], bool):
                    raise ConfigurationError("sampler.adaptive must be a boolean", "sampler.adaptive")
                config.sampler.adaptive = sd['adaptive']

        # 验证最终配置
        if not config.validate():
            raise ConfigurationError("Configuration validation failed", "config")
            
        return config
