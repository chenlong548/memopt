"""
mem_optimizer 配置管理模块

定义内存分配优化器的配置选项。
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from .base import AllocatorType, AllocationStrategy
from .exceptions import ConfigurationError


# 平台检测：判断是否为32位系统
IS_32BIT = sys.maxsize <= 2**32

# 安全的内存大小常量
if IS_32BIT:
    # 32位系统：最大2GB
    MAX_SAFE_MEMORY = 2 * 1024 * 1024 * 1024
    DEFAULT_TOTAL_MEMORY = 512 * 1024 * 1024
    DEFAULT_MAX_ALLOCATION = 256 * 1024 * 1024
else:
    # 64位系统：使用更大的值
    MAX_SAFE_MEMORY = 1024 * 1024 * 1024 * 1024  # 1TB
    DEFAULT_TOTAL_MEMORY = 1024 * 1024 * 1024  # 1GB
    DEFAULT_MAX_ALLOCATION = 512 * 1024 * 1024  # 512MB


def safe_memory_size(size: int, max_size: int = MAX_SAFE_MEMORY) -> int:
    """
    安全的内存大小计算，防止整数溢出
    
    Args:
        size: 请求的内存大小
        max_size: 最大允许的内存大小
        
    Returns:
        int: 安全的内存大小
        
    Raises:
        ValueError: 当内存大小超过限制时
    """
    if size < 0:
        raise ValueError(f"Memory size cannot be negative: {size}")
    if size > max_size:
        raise ValueError(f"Memory size {size} exceeds maximum allowed {max_size}")
    return size


class DefragPolicy(Enum):
    """碎片整理策略"""
    NEVER = "never"
    MANUAL = "manual"
    AUTO = "auto"
    AGGRESSIVE = "aggressive"


class NUMAPolicy(Enum):
    """NUMA策略"""
    DEFAULT = "default"
    BIND = "bind"
    INTERLEAVE = "interleave"
    PREFERRED = "preferred"
    LOCAL = "local"


@dataclass
class AllocatorConfig:
    """
    分配器配置

    单个分配器的配置选项。
    """

    allocator_type: AllocatorType = AllocatorType.AUTO
    min_block_size: int = 64
    max_block_size: int = 1024 * 1024 * 1024
    alignment: int = 8
    strategy: AllocationStrategy = AllocationStrategy.BEST_FIT
    enable_coalescing: bool = True
    enable_splitting: bool = True
    max_splits: int = 16
    metadata_overhead: int = 16


@dataclass
class BuddyAllocatorConfig(AllocatorConfig):
    """
    Buddy分配器配置
    """

    allocator_type: AllocatorType = AllocatorType.BUDDY
    min_order: int = 6
    max_order: int = 24
    enable_fast_path: bool = True


@dataclass
class SlabAllocatorConfig(AllocatorConfig):
    """
    Slab分配器配置
    """

    allocator_type: AllocatorType = AllocatorType.SLAB
    slab_size: int = 1024 * 1024
    object_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048])
    cache_limit: int = 100
    enable_coloring: bool = True


@dataclass
class TLSFAllocatorConfig(AllocatorConfig):
    """
    TLSF分配器配置
    """

    allocator_type: AllocatorType = AllocatorType.TLSF
    first_level_bits: int = 5
    second_level_bits: int = 5
    min_block_size: int = 64
    enable_boundary_tags: bool = True


@dataclass
class DefragConfig:
    """
    碎片整理配置
    """

    policy: DefragPolicy = DefragPolicy.AUTO
    threshold: float = 0.3
    min_interval: float = 60.0
    max_migrations: int = 1000
    enable_background: bool = True
    background_interval: float = 300.0
    target_fragmentation: float = 0.1
    enable_coalescing: bool = True
    enable_compaction: bool = True


@dataclass
class NUMAConfig:
    """
    NUMA配置
    """

    policy: NUMAPolicy = NUMAPolicy.DEFAULT
    preferred_node: int = -1
    interleave_nodes: List[int] = field(default_factory=list)
    enable_migration: bool = True
    migration_threshold: float = 0.8
    balance_interval: float = 60.0
    enable_local_allocation: bool = True


@dataclass
class RLSelectorConfig:
    """
    RL策略选择器配置
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    enable_online_learning: bool = True
    history_size: int = 1000
    feature_count: int = 10


@dataclass
class BanditConfig:
    """
    多臂老虎机配置
    """

    algorithm: str = "ucb1"
    confidence_level: float = 2.0
    decay_rate: float = 0.99
    min_samples: int = 10
    enable_decay: bool = True
    enable_contextual: bool = False


@dataclass
class MonitorConfig:
    """
    监控配置
    """

    enable_monitoring: bool = True
    sample_interval: float = 1.0
    history_size: int = 3600
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'fragmentation': 0.5,
        'usage': 0.9,
        'allocation_failures': 10
    })
    enable_psi: bool = True
    psi_path: str = "/proc/pressure/memory"


@dataclass
class OptimizerConfig:
    """
    内存优化器主配置

    整合所有配置选项。
    """

    total_memory: int = field(default_factory=lambda: DEFAULT_TOTAL_MEMORY)
    base_address: int = 0

    default_allocator: AllocatorType = AllocatorType.AUTO
    allocator_configs: Dict[AllocatorType, AllocatorConfig] = field(default_factory=dict)

    defrag_config: DefragConfig = field(default_factory=DefragConfig)
    numa_config: NUMAConfig = field(default_factory=NUMAConfig)
    rl_config: RLSelectorConfig = field(default_factory=RLSelectorConfig)
    bandit_config: BanditConfig = field(default_factory=BanditConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)

    enable_numa: bool = True
    enable_defrag: bool = True
    enable_rl_selector: bool = True
    enable_monitoring: bool = True

    max_allocation_size: int = field(default_factory=lambda: DEFAULT_MAX_ALLOCATION)
    min_allocation_size: int = 16
    default_alignment: int = 8

    enable_compression: bool = False
    compression_threshold: float = 0.8

    thread_safe: bool = True
    enable_statistics: bool = True

    def __post_init__(self):
        """初始化默认分配器配置"""
        # 验证内存大小
        try:
            self.total_memory = safe_memory_size(self.total_memory)
            self.max_allocation_size = safe_memory_size(self.max_allocation_size, self.total_memory)
        except ValueError as e:
            raise ConfigurationError(str(e))
            
        if not self.allocator_configs:
            self.allocator_configs = {
                AllocatorType.BUDDY: BuddyAllocatorConfig(),
                AllocatorType.SLAB: SlabAllocatorConfig(),
                AllocatorType.TLSF: TLSFAllocatorConfig()
            }

    def get_allocator_config(self, allocator_type: AllocatorType) -> AllocatorConfig:
        """
        获取分配器配置

        Args:
            allocator_type: 分配器类型

        Returns:
            AllocatorConfig: 分配器配置
        """
        return self.allocator_configs.get(allocator_type, AllocatorConfig())

    def validate(self) -> bool:
        """
        验证配置

        Returns:
            bool: 是否有效
        """
        if self.total_memory <= 0:
            return False
        if self.max_allocation_size > self.total_memory:
            return False
        if self.min_allocation_size <= 0:
            return False
        if self.default_alignment <= 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            Dict: 配置字典
        """
        return {
            'total_memory': self.total_memory,
            'base_address': self.base_address,
            'default_allocator': self.default_allocator.value,
            'enable_numa': self.enable_numa,
            'enable_defrag': self.enable_defrag,
            'enable_rl_selector': self.enable_rl_selector,
            'enable_monitoring': self.enable_monitoring,
            'max_allocation_size': self.max_allocation_size,
            'min_allocation_size': self.min_allocation_size,
            'default_alignment': self.default_alignment,
            'thread_safe': self.thread_safe,
            'enable_statistics': self.enable_statistics
        }
