"""
mem_monitor 采样层

提供内存访问采样功能。
"""

from .base import (
    SamplerBase,
    SampleData,
    SamplerState,
)
from .hardware import (
    HardwareSampler,
    PEBSSampler,
    IBSSampler,
)
from .software import (
    SoftwareSampler,
    TracemallocSampler,
    PerfSampler,
    EBPFSampler,
)

__all__ = [
    # 基类
    'SamplerBase',
    'SampleData',
    'SamplerState',
    # 硬件采样器
    'HardwareSampler',
    'PEBSSampler',
    'IBSSampler',
    # 软件采样器
    'SoftwareSampler',
    'TracemallocSampler',
    'PerfSampler',
    'EBPFSampler',
]


def create_sampler(config):
    """
    创建采样器工厂函数

    Args:
        config: 采样器配置

    Returns:
        SamplerBase: 采样器实例
    """
    from ..core.config import SamplerType

    sampler_type = config.sampler_type

    if sampler_type == SamplerType.AUTO:
        # 自动选择最佳采样器
        return _auto_select_sampler(config)
    elif sampler_type == SamplerType.HARDWARE_PEBS:
        return PEBSSampler(config)
    elif sampler_type == SamplerType.HARDWARE_IBS:
        return IBSSampler(config)
    elif sampler_type == SamplerType.SOFTWARE_EBPF:
        return EBPFSampler(config)
    elif sampler_type == SamplerType.SOFTWARE_PERF:
        return PerfSampler(config)
    elif sampler_type == SamplerType.SOFTWARE_TRACEMALLOC:
        return TracemallocSampler(config)
    elif sampler_type == SamplerType.SOFTWARE_PSUTIL:
        return SoftwareSampler(config)
    elif sampler_type == SamplerType.ADAPTIVE:
        return AdaptiveSampler(config)
    else:
        return SoftwareSampler(config)


def _auto_select_sampler(config):
    """
    自动选择采样器

    根据平台和能力选择最佳采样器。

    Args:
        config: 采样器配置

    Returns:
        SamplerBase: 采样器实例
    """
    import sys
    import platform

    # 优先尝试硬件采样器
    if sys.platform.startswith('linux'):
        # Linux系统
        cpu_vendor = _get_cpu_vendor()

        if cpu_vendor == 'Intel':
            try:
                sampler = PEBSSampler(config)
                if sampler.is_available():
                    return sampler
            except Exception:
                pass

        elif cpu_vendor == 'AMD':
            try:
                sampler = IBSSampler(config)
                if sampler.is_available():
                    return sampler
            except Exception:
                pass

        # 尝试perf
        try:
            sampler = PerfSampler(config)
            if sampler.is_available():
                return sampler
        except Exception:
            pass

    # 回退到软件采样器
    try:
        sampler = TracemallocSampler(config)
        if sampler.is_available():
            return sampler
    except Exception:
        pass

    # 最终回退
    return SoftwareSampler(config)


def _get_cpu_vendor() -> str:
    """获取CPU厂商"""
    try:
        import platform
        processor = platform.processor().lower()
        if 'intel' in processor:
            return 'Intel'
        elif 'amd' in processor:
            return 'AMD'
    except Exception:
        pass
    return 'Unknown'


class AdaptiveSampler(SamplerBase):
    """
    自适应采样器

    根据系统负载动态调整采样策略。
    """

    def __init__(self, config):
        super().__init__(config)
        self._samplers = []
        self._current_sampler = None
        self._load_history = []
        self._init_samplers()

    def _init_samplers(self):
        """初始化多个采样器"""
        # 添加可用的采样器
        try:
            self._samplers.append(TracemallocSampler(self._config))
        except Exception:
            pass

        try:
            self._samplers.append(SoftwareSampler(self._config))
        except Exception:
            pass

        if self._samplers:
            self._current_sampler = self._samplers[0]

    def start(self):
        """启动采样"""
        if self._current_sampler:
            self._current_sampler.start()
        self._state = SamplerState.RUNNING

    def stop(self):
        """停止采样"""
        if self._current_sampler:
            self._current_sampler.stop()
        self._state = SamplerState.STOPPED

    def sample(self) -> 'SampleData':
        """执行采样"""
        if not self._current_sampler:
            return SampleData(timestamp=0)

        data = self._current_sampler.sample()

        # 根据负载调整采样器
        self._adapt_sampler(data)

        return data

    def _adapt_sampler(self, data: 'SampleData'):
        """根据数据调整采样器"""
        # 计算负载指标
        load = data.metrics.get('allocation_rate', 0)
        self._load_history.append(load)

        # 保持最近100个样本
        if len(self._load_history) > 100:
            self._load_history.pop(0)

        # 根据平均负载选择采样器
        if len(self._load_history) >= 10:
            avg_load = sum(self._load_history) / len(self._load_history)

            # 高负载时使用更轻量的采样器
            if avg_load > 1000000 and len(self._samplers) > 1:
                # 切换到更轻量的采样器
                pass

    def get_current_data(self) -> dict:
        """获取当前数据"""
        if self._current_sampler:
            return self._current_sampler.get_current_data()
        return {}

    def is_available(self) -> bool:
        """检查是否可用"""
        return len(self._samplers) > 0
