"""
mem_monitor 核心监控器模块

实现内存监控的主类和相关数据结构。
"""

import time
import threading
import weakref
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, Any, List, Callable,
    TypeVar, Generic, Set
)
from enum import Enum
from collections import deque
from datetime import datetime

from .config import (
    MonitorConfig,
    SamplerConfig,
    AnalyzerConfig,
    TieringConfig,
    ReporterConfig,
    AlertLevel,
    ThresholdConfig,
)
from .exceptions import (
    MonitorError,
    ConfigurationError,
    ThresholdExceededError,
)

# 配置模块日志记录器
logger = logging.getLogger(__name__)


class MonitorState(Enum):
    """监控器状态"""
    CREATED = "created"         # 已创建
    INITIALIZING = "initializing"  # 初始化中
    RUNNING = "running"         # 运行中
    PAUSED = "paused"           # 已暂停
    STOPPING = "stopping"       # 停止中
    STOPPED = "stopped"         # 已停止
    ERROR = "error"             # 错误状态


@dataclass
class MemorySnapshot:
    """
    内存快照

    记录某一时刻的内存状态。
    """

    timestamp: float                           # 时间戳
    process_id: int                            # 进程ID

    # 基本内存指标
    rss: int = 0                               # 驻留集大小 (Resident Set Size)
    vms: int = 0                               # 虚拟内存大小 (Virtual Memory Size)
    shared: int = 0                            # 共享内存
    text: int = 0                              # 代码段大小
    data: int = 0                              # 数据段大小
    available: int = 0                         # 可用内存
    total: int = 0                             # 总内存

    # Python特定指标
    heap_size: int = 0                         # Python堆大小
    heap_used: int = 0                         # Python堆使用量
    gc_count: tuple = (0, 0, 0)               # GC计数
    gc_threshold: tuple = (0, 0, 0)           # GC阈值

    # 分配统计
    allocation_count: int = 0                  # 分配次数
    deallocation_count: int = 0                # 释放次数
    allocation_size: int = 0                   # 分配总大小
    deallocation_size: int = 0                 # 释放总大小

    # 性能指标
    allocation_rate: float = 0.0               # 分配速率 (bytes/s)
    deallocation_rate: float = 0.0             # 释放速率 (bytes/s)
    fragmentation_ratio: float = 0.0           # 碎片率

    # NUMA指标
    numa_node_memory: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # 自定义指标
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def get_usage_ratio(self) -> float:
        """获取内存使用率"""
        if self.total == 0:
            return 0.0
        return self.rss / self.total

    def get_available_ratio(self) -> float:
        """获取可用内存比例"""
        if self.total == 0:
            return 0.0
        return self.available / self.total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'process_id': self.process_id,
            'rss': self.rss,
            'vms': self.vms,
            'shared': self.shared,
            'text': self.text,
            'data': self.data,
            'available': self.available,
            'total': self.total,
            'heap_size': self.heap_size,
            'heap_used': self.heap_used,
            'gc_count': self.gc_count,
            'gc_threshold': self.gc_threshold,
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'allocation_size': self.allocation_size,
            'deallocation_size': self.deallocation_size,
            'allocation_rate': self.allocation_rate,
            'deallocation_rate': self.deallocation_rate,
            'fragmentation_ratio': self.fragmentation_ratio,
            'usage_ratio': self.get_usage_ratio(),
            'available_ratio': self.get_available_ratio(),
            'numa_node_memory': self.numa_node_memory,
            'custom_metrics': self.custom_metrics,
        }


@dataclass
class Alert:
    """
    告警信息

    记录阈值触发的告警。
    """

    level: AlertLevel
    metric: str
    value: float
    threshold: float
    timestamp: float
    message: str
    action_taken: str = "none"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'level': self.level.value,
            'metric': self.metric,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'message': self.message,
            'action_taken': self.action_taken,
            'details': self.details,
        }


@dataclass
class MonitorReport:
    """
    监控报告

    包含监控期间的完整报告数据。
    """

    start_time: float
    end_time: float
    duration: float

    # 汇总统计
    summary: Dict[str, Any] = field(default_factory=dict)

    # 峰值指标
    peak_metrics: Dict[str, float] = field(default_factory=dict)

    # 平均指标
    avg_metrics: Dict[str, float] = field(default_factory=dict)

    # 告警列表
    alerts: List[Alert] = field(default_factory=list)

    # 热点分析结果
    hotspots: List[Dict[str, Any]] = field(default_factory=list)

    # 泄漏检测结果
    leaks: List[Dict[str, Any]] = field(default_factory=list)

    # 分层建议
    tiering_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # 原始快照（可选）
    snapshots: List[MemorySnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'summary': self.summary,
            'peak_metrics': self.peak_metrics,
            'avg_metrics': self.avg_metrics,
            'alerts': [a.to_dict() for a in self.alerts],
            'hotspots': self.hotspots,
            'leaks': self.leaks,
            'tiering_recommendations': self.tiering_recommendations,
        }


class HookType(Enum):
    """钩子类型"""
    PRE_ALLOCATE = "pre_allocate"         # 分配前
    POST_ALLOCATE = "post_allocate"        # 分配后
    PRE_DEALLOCATE = "pre_deallocate"      # 释放前
    POST_DEALLOCATE = "post_deallocate"    # 释放后
    THRESHOLD_EXCEEDED = "threshold_exceeded"  # 阈值超限
    SNAPSHOT_TAKEN = "snapshot_taken"      # 快照采集后
    ALERT_TRIGGERED = "alert_triggered"    # 告警触发


class MemoryMonitor:
    """
    内存监控器主类

    提供完整的内存监控功能，包括采样、分析、报告等。

    使用示例:
        ```python
        # 创建监控器
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        # 启动监控
        monitor.start()

        # 获取快照
        snapshot = monitor.get_snapshot()

        # 设置阈值
        monitor.set_threshold('memory_usage', 0.9, 'alert')

        # 注册钩子
        monitor.register_hook('threshold_exceeded', my_callback)

        # 停止监控并获取报告
        report = monitor.stop()
        ```
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        初始化内存监控器

        Args:
            config: 监控配置，如果为None则使用默认配置
        """
        self.config = config or MonitorConfig()

        # 验证配置
        if not self.config.validate():
            raise ConfigurationError("Invalid configuration")

        # 状态
        self._state = MonitorState.CREATED
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

        # 数据存储 - 使用线程安全的数据结构
        self._history: deque[MemorySnapshot] = deque(maxlen=self.config.reporter.history_size)
        self._alerts: List[Alert] = []
        self._current_snapshot: Optional[MemorySnapshot] = None
        
        # 数据访问锁 - 保护 _history, _alerts, _tracked_objects
        self._data_lock = threading.RLock()

        # 钩子
        self._hooks: Dict[HookType, List[Callable]] = {
            hook_type: [] for hook_type in HookType
        }

        # 告警处理器
        self._alert_handlers: List[Callable[[Alert], None]] = []

        # 线程控制
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.RLock() if self.config.thread_safe else None

        # 组件（延迟初始化）
        self._sampler: Optional[Any] = None
        self._analyzers: Dict[str, Any] = {}
        self._tiering_manager: Optional[Any] = None
        self._reporter: Optional[Any] = None
        self._integrations: Dict[str, Any] = {}

        # 弱引用集合（用于追踪对象）- 需要线程安全保护
        self._tracked_objects: Set[weakref.ref] = set()

        # 统计信息
        self._stats = {
            'snapshots_taken': 0,
            'alerts_triggered': 0,
            'hooks_called': 0,
            'errors': 0,
        }

    def _initialize_components(self):
        """初始化各组件"""
        self._state = MonitorState.INITIALIZING

        try:
            # 初始化采样器
            self._init_sampler()

            # 初始化分析器
            self._init_analyzers()

            # 初始化分层管理器
            self._init_tiering_manager()

            # 初始化报告器
            self._init_reporter()

            # 初始化集成
            self._init_integrations()

        except Exception as e:
            self._state = MonitorState.ERROR
            raise MonitorError(f"Failed to initialize components: {e}")

    def _init_sampler(self):
        """初始化采样器"""
        from ..sampler import create_sampler
        self._sampler = create_sampler(self.config.sampler)

    def _init_analyzers(self):
        """初始化分析器"""
        from ..analyzer import create_analyzer

        for analyzer_type in self.config.analyzer.enabled_analyzers:
            try:
                analyzer = create_analyzer(analyzer_type, self.config.analyzer)
                self._analyzers[analyzer_type.value] = analyzer
            except Exception as e:
                logger.warning(
                    f"Failed to initialize analyzer '{analyzer_type.value}': {e}",
                    exc_info=True
                )

    def _init_tiering_manager(self):
        """初始化分层管理器"""
        if self.config.tiering.policy.value != "none":
            from ..tiering import TieringManager
            self._tiering_manager = TieringManager(self.config.tiering)

    def _init_reporter(self):
        """初始化报告器"""
        from ..reporter import Reporter
        self._reporter = Reporter(self.config.reporter)

    def _init_integrations(self):
        """初始化集成模块"""
        if self.config.integration.enable_psutil:
            try:
                from ..integration import PsutilAdapter
                self._integrations['psutil'] = PsutilAdapter()
            except ImportError as e:
                logger.debug(f"psutil integration not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize psutil integration: {e}", exc_info=True)

        if self.config.integration.enable_tracemalloc:
            try:
                from ..integration import TracemallocAdapter
                self._integrations['tracemalloc'] = TracemallocAdapter(
                    self.config.integration.tracemalloc_frames
                )
            except ImportError as e:
                logger.debug(f"tracemalloc integration not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize tracemalloc integration: {e}", exc_info=True)

        if self.config.integration.enable_mem_mapper:
            try:
                from ..integration import MemMapperAdapter
                self._integrations['mem_mapper'] = MemMapperAdapter()
            except ImportError as e:
                logger.debug(f"mem_mapper integration not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize mem_mapper integration: {e}", exc_info=True)

        if self.config.integration.enable_mem_optimizer:
            try:
                from ..integration import MemOptimizerAdapter
                self._integrations['mem_optimizer'] = MemOptimizerAdapter()
            except ImportError as e:
                logger.debug(f"mem_optimizer integration not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize mem_optimizer integration: {e}", exc_info=True)

    def start(self) -> None:
        """
        启动监控

        开始后台监控线程，定期采集内存快照。
        """
        if self._state == MonitorState.RUNNING:
            return

        # 初始化组件
        if self._state == MonitorState.CREATED:
            self._initialize_components()

        self._state = MonitorState.RUNNING
        self._start_time = time.time()
        self._stop_event.clear()
        self._pause_event.clear()

        # 启动tracemalloc（如果启用）
        if 'tracemalloc' in self._integrations:
            self._integrations['tracemalloc'].start()

        # 启动后台线程
        if self.config.background_thread:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=self.config.daemon_thread,
                name="MemoryMonitorThread"
            )
            self._monitor_thread.start()

    def stop(self) -> MonitorReport:
        """
        停止监控

        停止后台监控线程并生成监控报告。

        Returns:
            MonitorReport: 监控报告
        """
        if self._state != MonitorState.RUNNING:
            return self._generate_report()

        self._state = MonitorState.STOPPING
        self._stop_event.set()

        # 等待线程结束
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        self._stop_time = time.time()
        self._state = MonitorState.STOPPED

        # 停止tracemalloc
        if 'tracemalloc' in self._integrations:
            self._integrations['tracemalloc'].stop()

        return self._generate_report()

    def pause(self) -> None:
        """暂停监控"""
        if self._state == MonitorState.RUNNING:
            self._pause_event.set()
            self._state = MonitorState.PAUSED

    def resume(self) -> None:
        """恢复监控"""
        if self._state == MonitorState.PAUSED:
            self._pause_event.clear()
            self._state = MonitorState.RUNNING

    def get_snapshot(self) -> MemorySnapshot:
        """
        获取当前内存快照

        Returns:
            MemorySnapshot: 内存快照
        """
        snapshot = self._collect_snapshot()

        with self._data_lock:
            self._current_snapshot = snapshot
            self._history.append(snapshot)
            self._stats['snapshots_taken'] += 1

        # 触发钩子
        self._trigger_hooks(HookType.SNAPSHOT_TAKEN, snapshot)

        # 检查阈值
        self._check_thresholds(snapshot)

        return snapshot

    def _collect_snapshot(self) -> MemorySnapshot:
        """
        采集内存快照

        Returns:
            MemorySnapshot: 内存快照
        """
        import os

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=os.getpid()
        )

        # 从psutil获取数据
        if 'psutil' in self._integrations:
            psutil_data = self._integrations['psutil'].get_memory_info()
            snapshot.rss = psutil_data.get('rss', 0)
            snapshot.vms = psutil_data.get('vms', 0)
            snapshot.shared = psutil_data.get('shared', 0)
            snapshot.text = psutil_data.get('text', 0)
            snapshot.data = psutil_data.get('data', 0)
            snapshot.available = psutil_data.get('available', 0)
            snapshot.total = psutil_data.get('total', 0)

        # 从tracemalloc获取数据
        if 'tracemalloc' in self._integrations:
            trace_data = self._integrations['tracemalloc'].get_stats()
            snapshot.heap_size = trace_data.get('heap_size', 0)
            snapshot.heap_used = trace_data.get('heap_used', 0)
            snapshot.allocation_count = trace_data.get('allocation_count', 0)
            snapshot.allocation_size = trace_data.get('allocation_size', 0)

        # 从GC获取数据
        import gc
        snapshot.gc_count = tuple(gc.get_count())
        snapshot.gc_threshold = tuple(gc.get_threshold())

        # 从采样器获取数据
        if self._sampler:
            sampler_data = self._sampler.get_current_data()
            snapshot.custom_metrics.update(sampler_data)

        # 从分析器获取数据
        for name, analyzer in self._analyzers.items():
            try:
                analyzer_data = analyzer.get_current_metrics()
                snapshot.custom_metrics[f'analyzer_{name}'] = analyzer_data
            except Exception as e:
                logger.debug(f"Failed to get metrics from analyzer '{name}': {e}")

        # 计算派生指标
        with self._data_lock:
            history_list = list(self._history)
        
        if len(history_list) > 0:
            prev = history_list[-1]
            time_delta = snapshot.timestamp - prev.timestamp
            if time_delta > 0:
                snapshot.allocation_rate = (
                    snapshot.allocation_size - prev.allocation_size
                ) / time_delta
                snapshot.deallocation_rate = (
                    snapshot.deallocation_size - prev.deallocation_size
                ) / time_delta

        return snapshot

    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            # 检查暂停
            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

            try:
                # 采集快照
                self.get_snapshot()

                # 运行分析器
                self._run_analyzers()

            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

            # 等待下一个采样周期
            interval = self._get_adaptive_interval()
            self._stop_event.wait(interval)

    def _get_adaptive_interval(self) -> float:
        """
        获取自适应采样间隔

        根据系统负载动态调整采样间隔。

        Returns:
            float: 采样间隔（秒）
        """
        if not self.config.sampler.adaptive:
            return self.config.sampler.sample_interval

        # 基于历史数据计算负载
        with self._data_lock:
            history_len = len(self._history)
            if history_len < self.config.sampler.warmup_samples:
                return self.config.sampler.sample_interval
            recent = list(self._history)[-10:]
        
        # 计算最近的分配速率变化
        rates = [s.allocation_rate for s in recent if s.allocation_rate > 0]

        if not rates:
            return self.config.sampler.sample_interval

        # 高负载时增加采样频率
        avg_rate = sum(rates) / len(rates)
        if avg_rate > 1000000:  # > 1MB/s
            return self.config.sampler.min_interval
        elif avg_rate < 10000:  # < 10KB/s
            return self.config.sampler.max_interval
        else:
            # 线性插值
            ratio = (avg_rate - 10000) / (1000000 - 10000)
            return self.config.sampler.max_interval - ratio * (
                self.config.sampler.max_interval - self.config.sampler.min_interval
            )

    def _run_analyzers(self):
        """运行分析器"""
        with self._data_lock:
            history_copy = list(self._history)
            
        for name, analyzer in self._analyzers.items():
            try:
                analyzer.analyze(history_copy)
            except Exception as e:
                logger.warning(f"Analyzer '{name}' failed: {e}", exc_info=True)

    def _check_thresholds(self, snapshot: MemorySnapshot):
        """
        检查阈值

        Args:
            snapshot: 当前快照
        """
        if not self.config.enable_alerts:
            return

        for metric_name, threshold_config in self.config.thresholds.items():
            value = self._get_metric_value(snapshot, metric_name)

            if value is None:
                continue

            # 确定告警级别
            level = None
            if value >= threshold_config.critical:
                level = AlertLevel.CRITICAL
            elif value >= threshold_config.error:
                level = AlertLevel.ERROR
            elif value >= threshold_config.warning:
                level = AlertLevel.WARNING

            if level:
                alert = Alert(
                    level=level,
                    metric=metric_name,
                    value=value,
                    threshold=threshold_config.warning,
                    timestamp=snapshot.timestamp,
                    message=f"Metric '{metric_name}' exceeded threshold: "
                           f"{value:.4f} >= {threshold_config.warning}",
                )

                with self._data_lock:
                    self._alerts.append(alert)
                    self._stats['alerts_triggered'] += 1

                # 触发钩子
                self._trigger_hooks(HookType.ALERT_TRIGGERED, alert)

                # 调用告警处理器
                for handler in self._alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}", exc_info=True)

                # 执行回调
                if threshold_config.callback:
                    try:
                        threshold_config.callback(alert)
                    except Exception as e:
                        logger.error(f"Threshold callback failed: {e}", exc_info=True)

    def _get_metric_value(self, snapshot: MemorySnapshot, metric_name: str) -> Optional[float]:
        """
        从快照获取指标值

        Args:
            snapshot: 内存快照
            metric_name: 指标名称

        Returns:
            Optional[float]: 指标值
        """
        # 内置指标映射
        metric_mapping = {
            'memory_usage': lambda s: s.get_usage_ratio(),
            'memory_available': lambda s: s.get_available_ratio(),
            'rss': lambda s: s.rss,
            'vms': lambda s: s.vms,
            'heap_used': lambda s: s.heap_used,
            'heap_size': lambda s: s.heap_size,
            'fragmentation': lambda s: s.fragmentation_ratio,
            'allocation_rate': lambda s: s.allocation_rate,
            'deallocation_rate': lambda s: s.deallocation_rate,
            'allocation_count': lambda s: s.allocation_count,
            'leak_growth': lambda s: s.custom_metrics.get('leak_growth_rate', 0),
        }

        if metric_name in metric_mapping:
            return metric_mapping[metric_name](snapshot)

        # 检查自定义指标
        if metric_name in snapshot.custom_metrics:
            return snapshot.custom_metrics[metric_name]

        return None

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        注册钩子函数

        Args:
            event: 事件类型
            callback: 回调函数
        """
        if not self.config.enable_hooks:
            return

        try:
            hook_type = HookType(event)
            self._hooks[hook_type].append(callback)
        except ValueError:
            raise MonitorError(f"Unknown hook type: {event}")

    def unregister_hook(self, event: str, callback: Callable) -> None:
        """
        注销钩子函数

        Args:
            event: 事件类型
            callback: 回调函数
        """
        try:
            hook_type = HookType(event)
            if callback in self._hooks[hook_type]:
                self._hooks[hook_type].remove(callback)
        except ValueError:
            pass

    def _trigger_hooks(self, hook_type: HookType, data: Any) -> None:
        """
        触发钩子

        Args:
            hook_type: 钩子类型
            data: 传递给钩子的数据
        """
        for callback in self._hooks[hook_type]:
            try:
                callback(data)
                self._stats['hooks_called'] += 1
            except Exception as e:
                logger.error(f"Hook callback failed for '{hook_type.value}': {e}", exc_info=True)

    def set_threshold(self,
                     metric: str,
                     value: float,
                     action: str = "alert",
                     callback: Optional[Callable] = None) -> None:
        """
        设置阈值

        Args:
            metric: 指标名称
            value: 阈值
            action: 触发动作
            callback: 回调函数
        """
        self.config.thresholds[metric] = ThresholdConfig(
            warning=value,
            error=value * 1.1,
            critical=value * 1.2,
            action=action,
            callback=callback
        )

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        添加告警处理器

        Args:
            handler: 处理函数
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        移除告警处理器

        Args:
            handler: 处理函数
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def track_object(self, obj: Any) -> None:
        """
        追踪对象

        Args:
            obj: 要追踪的对象
        """
        ref = weakref.ref(obj, self._on_object_collected)
        with self._data_lock:
            self._tracked_objects.add(ref)

    def _on_object_collected(self, ref: weakref.ref):
        """对象被回收时的回调"""
        with self._data_lock:
            self._tracked_objects.discard(ref)

    def get_tracked_count(self) -> int:
        """获取追踪的对象数量"""
        # 清理已回收的引用
        with self._data_lock:
            self._tracked_objects = {r for r in self._tracked_objects if r() is not None}
            return len(self._tracked_objects)

    def get_current_metrics(self) -> Dict[str, float]:
        """
        获取当前指标

        Returns:
            Dict[str, float]: 当前指标字典
        """
        if self._current_snapshot:
            return self._current_snapshot.to_dict()
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
        with self._data_lock:
            history = list(self._history)

            if count is not None:
                history = history[-count:]

            if metric is not None:
                return [(s.timestamp, self._get_metric_value(s, metric)) for s in history]

            return [(s.timestamp, s.to_dict()) for s in history]

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
        with self._data_lock:
            if level is not None:
                alerts = [a for a in self._alerts if a.level == level]
            else:
                alerts = list(self._alerts)

            if clear:
                self._alerts.clear()

            return alerts

    def get_state(self) -> MonitorState:
        """获取监控器状态"""
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._data_lock:
            stats = {
                'snapshots_taken': self._stats['snapshots_taken'],
                'alerts_triggered': self._stats['alerts_triggered'],
                'hooks_called': self._stats['hooks_called'],
                'errors': self._stats['errors'],
                'state': self._state.value,
                'history_size': len(self._history),
                'tracked_objects': self.get_tracked_count(),
                'uptime': int(time.time() - self._start_time) if self._start_time else 0,
            }
            return stats

    def _generate_report(self) -> MonitorReport:
        """
        生成监控报告

        Returns:
            MonitorReport: 监控报告
        """
        with self._data_lock:
            history = list(self._history)
            alerts_copy = list(self._alerts)

        if not history:
            return MonitorReport(
                start_time=self._start_time or 0,
                end_time=self._stop_time or time.time(),
                duration=0
            )

        start_time = history[0].timestamp
        end_time = history[-1].timestamp

        report = MonitorReport(
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time
        )

        # 计算汇总统计
        report.summary = self._calculate_summary(history)

        # 计算峰值指标
        report.peak_metrics = self._calculate_peaks(history)

        # 计算平均指标
        report.avg_metrics = self._calculate_averages(history)

        # 添加告警
        report.alerts = alerts_copy

        # 从分析器获取结果
        for name, analyzer in self._analyzers.items():
            try:
                results = analyzer.get_results()
                if name == 'hotspot':
                    report.hotspots = results
                elif name == 'leak_detection':
                    report.leaks = results
            except Exception as e:
                logger.warning(f"Failed to get results from analyzer '{name}': {e}")

        # 从分层管理器获取建议
        if self._tiering_manager:
            try:
                report.tiering_recommendations = self._tiering_manager.get_recommendations()
            except Exception as e:
                logger.warning(f"Failed to get tiering recommendations: {e}")

        return report

    def _calculate_summary(self, history: List[MemorySnapshot]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not history:
            return {}

        first = history[0]
        last = history[-1]
        
        with self._data_lock:
            alert_count = len(self._alerts)

        return {
            'duration': last.timestamp - first.timestamp,
            'snapshots': len(history),
            'start_rss': first.rss,
            'end_rss': last.rss,
            'rss_delta': last.rss - first.rss,
            'peak_rss': max(s.rss for s in history),
            'avg_rss': sum(s.rss for s in history) / len(history),
            'total_allocations': last.allocation_count - first.allocation_count,
            'total_allocated': last.allocation_size - first.allocation_size,
            'alert_count': alert_count,
        }

    def _calculate_peaks(self, history: List[MemorySnapshot]) -> Dict[str, float]:
        """计算峰值指标"""
        if not history:
            return {}

        return {
            'peak_rss': max(s.rss for s in history),
            'peak_vms': max(s.vms for s in history),
            'peak_heap': max(s.heap_used for s in history),
            'peak_allocation_rate': max(s.allocation_rate for s in history),
            'peak_fragmentation': max(s.fragmentation_ratio for s in history),
        }

    def _calculate_averages(self, history: List[MemorySnapshot]) -> Dict[str, float]:
        """计算平均指标"""
        if not history:
            return {}

        n = len(history)
        return {
            'avg_rss': sum(s.rss for s in history) / n,
            'avg_vms': sum(s.vms for s in history) / n,
            'avg_heap': sum(s.heap_used for s in history) / n,
            'avg_allocation_rate': sum(s.allocation_rate for s in history) / n,
            'avg_fragmentation': sum(s.fragmentation_ratio for s in history) / n,
            'avg_usage': sum(s.get_usage_ratio() for s in history) / n,
        }

    def _get_lock(self):
        """获取锁"""
        if self._lock:
            return self._lock
        # 返回一个空上下文管理器
        from contextlib import nullcontext
        return nullcontext()

    def export_report(self, format: str = 'dict') -> Any:
        """
        导出报告

        Args:
            format: 导出格式

        Returns:
            导出的数据
        """
        report = self._generate_report()

        if format == 'dict':
            return report.to_dict()
        elif format == 'json':
            import json
            return json.dumps(report.to_dict(), indent=2)
        elif format == 'prometheus':
            return self._export_prometheus()
        else:
            return report.to_dict()

    def _export_prometheus(self) -> str:
        """导出Prometheus格式"""
        lines = []
        metrics = self.get_current_metrics()

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"mem_monitor_{name} {value}")

        return '\n'.join(lines)

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False

    def __repr__(self) -> str:
        return f"MemoryMonitor(state={self._state.value}, snapshots={len(self._history)})"
