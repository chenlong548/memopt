"""
mem_monitor 采样器基类模块

定义采样器的标准接口和基础实现。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque


class SamplerState(Enum):
    """采样器状态"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SampleData:
    """
    采样数据

    表示一次采样的结果。
    """

    timestamp: float                           # 时间戳
    metrics: Dict[str, Any] = field(default_factory=dict)  # 指标数据
    events: List[Dict[str, Any]] = field(default_factory=list)  # 事件列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'events': self.events,
            'metadata': self.metadata,
        }


@dataclass
class MemoryAccessEvent:
    """
    内存访问事件

    表示一次内存访问的详细信息。
    """

    timestamp: float           # 时间戳
    address: int               # 内存地址
    size: int                  # 访问大小
    access_type: str           # 访问类型: read, write, execute
    latency: int               # 访问延迟（纳秒）
    numa_node: int = -1        # NUMA节点
    cpu: int = -1              # CPU核心
    pid: int = -1              # 进程ID
    tid: int = -1              # 线程ID
    call_stack: List[str] = field(default_factory=list)  # 调用栈


class SamplerBase(ABC):
    """
    采样器基类

    定义采样器的标准接口。
    """

    def __init__(self, config):
        """
        初始化采样器

        Args:
            config: 采样器配置
        """
        self._config = config
        self._state = SamplerState.CREATED
        self._buffer: deque = deque(maxlen=config.buffer_size)
        self._current_data: Dict[str, Any] = {}
        self._sample_count = 0
        self._error_count = 0
        self._start_time: Optional[float] = None

    @abstractmethod
    def start(self) -> None:
        """
        启动采样器

        开始采集内存访问数据。
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        停止采样器

        停止采集数据。
        """
        pass

    @abstractmethod
    def sample(self) -> SampleData:
        """
        执行一次采样

        Returns:
            SampleData: 采样数据
        """
        pass

    def pause(self) -> None:
        """暂停采样"""
        self._state = SamplerState.PAUSED

    def resume(self) -> None:
        """恢复采样"""
        self._state = SamplerState.RUNNING

    def get_state(self) -> SamplerState:
        """获取采样器状态"""
        return self._state

    def get_current_data(self) -> Dict[str, Any]:
        """
        获取当前采样数据

        Returns:
            Dict: 当前数据
        """
        return self._current_data.copy()

    def get_buffer(self) -> List[SampleData]:
        """
        获取缓冲区数据

        Returns:
            List[SampleData]: 缓冲区数据列表
        """
        return list(self._buffer)

    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'state': self._state.value,
            'sample_count': self._sample_count,
            'error_count': self._error_count,
            'buffer_size': len(self._buffer),
            'uptime': time.time() - self._start_time if self._start_time else 0,
        }

    def is_available(self) -> bool:
        """
        检查采样器是否可用

        Returns:
            bool: 是否可用
        """
        return True

    def _add_to_buffer(self, data: SampleData) -> None:
        """
        添加数据到缓冲区

        Args:
            data: 采样数据
        """
        self._buffer.append(data)
        self._current_data = data.metrics
        self._sample_count += 1

    def _record_error(self, error: Exception) -> None:
        """
        记录错误

        Args:
            error: 异常对象
        """
        self._error_count += 1
        self._current_data['last_error'] = str(error)


class AggregatedSampler(SamplerBase):
    """
    聚合采样器

    对采样数据进行聚合处理，减少数据量。
    """

    def __init__(self, config, window_size: int = 100):
        super().__init__(config)
        self._window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._aggregation_interval = config.sample_interval * 10

    def sample(self) -> SampleData:
        """执行采样并聚合"""
        # 获取原始采样
        raw_sample = self._raw_sample()
        self._window.append(raw_sample)

        # 定期聚合
        if len(self._window) >= self._window_size:
            return self._aggregate()

        return raw_sample

    @abstractmethod
    def _raw_sample(self) -> SampleData:
        """获取原始采样"""
        pass

    def _aggregate(self) -> SampleData:
        """聚合窗口数据"""
        if not self._window:
            return SampleData(timestamp=time.time())

        # 计算聚合指标
        aggregated_metrics = {}

        # 收集所有指标键
        all_keys = set()
        for sample in self._window:
            all_keys.update(sample.metrics.keys())

        # 对每个指标进行聚合
        for key in all_keys:
            values = [s.metrics.get(key) for s in self._window if key in s.metrics]
            if values:
                if isinstance(values[0], (int, float)):
                    aggregated_metrics[f'{key}_avg'] = sum(values) / len(values)
                    aggregated_metrics[f'{key}_min'] = min(values)
                    aggregated_metrics[f'{key}_max'] = max(values)
                    aggregated_metrics[f'{key}_sum'] = sum(values)

        return SampleData(
            timestamp=time.time(),
            metrics=aggregated_metrics,
            metadata={'aggregated': True, 'window_size': len(self._window)}
        )


class SlidingWindowSampler(SamplerBase):
    """
    滑动窗口采样器

    使用滑动窗口技术进行采样，支持时间窗口和计数窗口。
    """

    def __init__(self, config, window_type: str = 'time', window_size: float = 60.0):
        super().__init__(config)
        self._window_type = window_type
        self._window_size = window_size
        self._window_data: deque = deque()

    def sample(self) -> SampleData:
        """执行采样"""
        raw = self._raw_sample()
        self._window_data.append(raw)
        self._prune_window()
        return self._compute_window_stats()

    @abstractmethod
    def _raw_sample(self) -> SampleData:
        """获取原始采样"""
        pass

    def _prune_window(self):
        """修剪窗口"""
        if self._window_type == 'time':
            cutoff = time.time() - self._window_size
            while self._window_data and self._window_data[0].timestamp < cutoff:
                self._window_data.popleft()
        else:  # count
            while len(self._window_data) > self._window_size:
                self._window_data.popleft()

    def _compute_window_stats(self) -> SampleData:
        """计算窗口统计"""
        if not self._window_data:
            return SampleData(timestamp=time.time())

        metrics = {}
        for sample in self._window_data:
            for key, value in sample.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        stats = {}
        for key, values in metrics.items():
            if values and isinstance(values[0], (int, float)):
                stats[f'{key}_avg'] = sum(values) / len(values)
                stats[f'{key}_count'] = len(values)

        return SampleData(
            timestamp=time.time(),
            metrics=stats,
            metadata={'window_size': len(self._window_data)}
        )
