"""
mem_optimizer 内存分配优化器

提供高性能的内存分配优化功能，包括多种分配算法、
智能策略选择、碎片整理、NUMA感知等特性。

主要组件:
- MemoryPool: 内存池主类
- BuddyAllocator: Buddy分配器
- SlabAllocator: Slab分配器
- TLSFAllocator: TLSF分配器
- RLStrategySelector: 强化学习策略选择器
- Defragmenter: 碎片整理器
- NUMACoordinator: NUMA协调器
- MemoryMonitor: 内存监控器
"""

from .core import (
    AllocatorType,
    AllocationStrategy,
    MemoryRegionState,
    MemoryBlock,
    AllocationRequest,
    AllocationResult,
    MemoryStatistics,
    AllocatorBase,
    StrategySelectorBase,
    DefragmenterBase,
    NUMACoordinatorBase,
    AllocatorConfig,
    BuddyAllocatorConfig,
    SlabAllocatorConfig,
    TLSFAllocatorConfig,
    DefragConfig,
    NUMAConfig,
    RLSelectorConfig,
    BanditConfig,
    MonitorConfig,
    OptimizerConfig,
    DefragPolicy,
    NUMAPolicy,
    MemOptimizerError,
    AllocationError,
    OutOfMemoryError,
    FragmentationError,
    DefragmentationError,
    NUMAError,
    NUMANotAvailableError,
    NUMAMigrationError,
    ConfigurationError,
    AllocatorError,
    AllocatorNotInitializedError,
    InvalidBlockError,
    BlockNotFoundError,
    DoubleFreeError,
    CorruptionError,
    IntegrationError,
    MonitorError,
    PSIError,
    StrategyError,
    RLTrainingError,
    BanditError,
    MemoryPool,
    PoolSnapshot
)

from .allocators import (
    BuddyAllocator,
    SlabAllocator,
    TLSFAllocator
)

from .strategies import (
    RLStrategySelector,
    UCB1Bandit,
    ThompsonSamplingBandit,
    create_bandit
)

from .defrag import (
    Defragmenter,
    MemoryCoalescer,
    PSIMonitor
)

from .numa import (
    NUMACoordinator
)

from .monitor import (
    MemoryMonitor
)

from .integration import (
    MemMapperIntegration,
    DataCompressorIntegration
)

__version__ = "1.0.0"

__all__ = [
    'AllocatorType',
    'AllocationStrategy',
    'MemoryRegionState',
    'MemoryBlock',
    'AllocationRequest',
    'AllocationResult',
    'MemoryStatistics',
    'AllocatorBase',
    'StrategySelectorBase',
    'DefragmenterBase',
    'NUMACoordinatorBase',
    'AllocatorConfig',
    'BuddyAllocatorConfig',
    'SlabAllocatorConfig',
    'TLSFAllocatorConfig',
    'DefragConfig',
    'NUMAConfig',
    'RLSelectorConfig',
    'BanditConfig',
    'MonitorConfig',
    'OptimizerConfig',
    'DefragPolicy',
    'NUMAPolicy',
    'MemOptimizerError',
    'AllocationError',
    'OutOfMemoryError',
    'FragmentationError',
    'DefragmentationError',
    'NUMAError',
    'NUMANotAvailableError',
    'NUMAMigrationError',
    'ConfigurationError',
    'AllocatorError',
    'AllocatorNotInitializedError',
    'InvalidBlockError',
    'BlockNotFoundError',
    'DoubleFreeError',
    'CorruptionError',
    'IntegrationError',
    'MonitorError',
    'PSIError',
    'StrategyError',
    'RLTrainingError',
    'BanditError',
    'MemoryPool',
    'PoolSnapshot',
    'BuddyAllocator',
    'SlabAllocator',
    'TLSFAllocator',
    'RLStrategySelector',
    'UCB1Bandit',
    'ThompsonSamplingBandit',
    'create_bandit',
    'Defragmenter',
    'MemoryCoalescer',
    'PSIMonitor',
    'NUMACoordinator',
    'MemoryMonitor',
    'MemMapperIntegration',
    'DataCompressorIntegration'
]
