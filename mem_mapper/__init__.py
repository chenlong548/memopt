"""
mem_mapper - 高性能内存映射工具

mem_mapper是一个跨平台的高性能内存映射工具，提供NUMA感知、大页支持、
预取优化等高级功能。

主要特性：
- 跨平台支持（Linux/Windows）
- NUMA感知映射
- 大页支持
- 智能预取
- 生命周期管理
- 性能统计

使用示例：
    from mem_mapper import MemoryMapper, MapperConfig
    
    # 创建映射器
    config = MapperConfig(
        use_numa=True,
        use_huge_pages=True,
        use_prefetch=True
    )
    mapper = MemoryMapper(config)
    
    # 映射文件
    region = mapper.map_file(
        path='/path/to/file',
        mode='readonly',
        numa_node=0,
        use_huge_pages=True
    )
    
    # 使用映射区域
    # ...
    
    # 解除映射
    mapper.unmap(region)
"""

from typing import Optional

__version__ = '1.0.0'
__author__ = 'mem_mapper team'
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 核心类
    'MemoryMapper',
    'MapperConfig',
    'MappedRegion',
    
    # 注册表
    'MappingRegistry',
    'MappingRegistryView',
    
    # 异常
    'MemMapperError',
    'MMapError',
    'MUnmapError',
    'MAdviseError',
    'MProtectError',
    'MSyncError',
    'MLockError',
    'MUnlockError',
    'NUMAError',
    'NUMANotSupportedError',
    'NUMABindingError',
    'HugePageError',
    'HugePageNotAvailableError',
    'HugePagePoolExhaustedError',
    'GPUMappingError',
    'GPUNotAvailableError',
    'GPUOutOfMemoryError',
    'RegionError',
    'RegionNotFoundError',
    'RegionAlreadyExistsError',
    'ConfigError',
    'PlatformError',
    'PlatformNotSupportedError',
    'FileError',
    'FileNotFoundError',
    'FilePermissionError',
    'AlignmentError',
    'PrefetchError',
    'LifecycleError',
    
    # 数据结构
    'ProtectionFlags',
    'MappingType',
    'NUMAPolicy',
    'MappingState',
    'GPUMappingStrategy',
    'SyncState',
    'AtomicCounter',
    'AccessStatistics',
    'GPUMappingInfo',
    
    # NUMA
    'NUMANode',
    'NUMATopology',
    'NUMATopologyDetector',
    'get_numa_topology',
    'is_numa_available',
    'NUMAPolicyMode',
    'NUMAPolicyConfig',
    'NUMAPolicyManager',
    'get_numa_policy_manager',
    
    # 大页
    'HugePageInfo',
    'HugePageConfig',
    'HugePageDetector',
    'get_hugepage_detector',
    'get_hugepage_config',
    'is_hugepage_available',
    'HugePageAllocation',
    'HugePagePoolStats',
    'HugePagePool',
    'get_hugepage_pool',
    
    # 预取
    'PrefetchStrategyType',
    'PrefetchRange',
    'PrefetchStrategy',
    'SequentialPrefetchStrategy',
    'RandomPrefetchStrategy',
    'AdaptivePrefetchStrategy',
    'NoPrefetchStrategy',
    'create_prefetch_strategy',
    'PrefetchStats',
    'PrefetchExecutor',
    'prefault_region',
    
    # 生命周期
    'CleanupTask',
    'LifecycleManager',
    'RegionGuard',
    'create_region_guard',
    
    # 工具
    'PAGE_SIZE_4KB',
    'PAGE_SIZE_2MB',
    'PAGE_SIZE_1GB',
    'PGD_ALIGNMENT',
    'align_up',
    'align_down',
    'is_aligned',
    'align_to_page',
    'align_to_huge_page',
    'align_to_pgd',
    'calculate_padding',
    'calculate_pages',
    'calculate_huge_pages',
    'find_optimal_alignment',
    'align_offset_and_size',
    'get_alignment_waste',
    'get_alignment_efficiency',
    'is_power_of_two',
    'next_power_of_two',
    'previous_power_of_two',
    'align_address_to_page',
    'is_address_page_aligned',
    'calculate_page_range',
    'format_size',
    'parse_size',
    'Timer',
    'PerformanceTracker',
    'MemoryUsageTracker',
    'AccessPatternAnalyzer',
    'Benchmark',
    
    # 平台
    'PlatformBase',
    'PlatformFactory',
]

# 导入核心模块
from .core import (
    # 异常
    MemMapperError,
    MMapError,
    MUnmapError,
    MAdviseError,
    MProtectError,
    MSyncError,
    MLockError,
    MUnlockError,
    NUMAError,
    NUMANotSupportedError,
    NUMABindingError,
    HugePageError,
    HugePageNotAvailableError,
    HugePagePoolExhaustedError,
    GPUMappingError,
    GPUNotAvailableError,
    GPUOutOfMemoryError,
    RegionError,
    RegionNotFoundError,
    RegionAlreadyExistsError,
    ConfigError,
    PlatformError,
    PlatformNotSupportedError,
    FileError,
    FileNotFoundError,
    FilePermissionError,
    AlignmentError,
    PrefetchError,
    LifecycleError,
    
    # 数据结构
    ProtectionFlags,
    MappingType,
    NUMAPolicy,
    MappingState,
    GPUMappingStrategy,
    SyncState,
    AtomicCounter,
    AccessStatistics,
    GPUMappingInfo,
    MappedRegion,
    
    # 注册表
    CleanupTask,
    MappingRegistry,
    MappingRegistryView,
    
    # 映射器
    MapperConfig,
    MemoryMapper,
)

# 导入NUMA模块
from .numa import (
    NUMANode,
    NUMATopology,
    NUMATopologyDetector,
    get_numa_topology,
    is_numa_available,
    NUMAPolicyMode,
    NUMAPolicyConfig,
    NUMAPolicyManager,
    get_numa_policy_manager,
)

# 导入大页模块
from .hugepage import (
    HugePageInfo,
    HugePageConfig,
    HugePageDetector,
    get_hugepage_detector,
    get_hugepage_config,
    is_hugepage_available,
    HugePageAllocation,
    HugePagePoolStats,
    HugePagePool,
    get_hugepage_pool,
)

# 导入预取模块
from .prefetch import (
    PrefetchStrategyType,
    PrefetchRange,
    PrefetchStrategy,
    SequentialPrefetchStrategy,
    RandomPrefetchStrategy,
    AdaptivePrefetchStrategy,
    NoPrefetchStrategy,
    create_prefetch_strategy,
    PrefetchStats,
    PrefetchExecutor,
    prefault_region,
)

# 导入生命周期模块
from .lifecycle import (
    CleanupTask,
    LifecycleManager,
    RegionGuard,
    create_region_guard,
)

# 导入工具模块
from .utils import (
    # 对齐
    PAGE_SIZE_4KB,
    PAGE_SIZE_2MB,
    PAGE_SIZE_1GB,
    PGD_ALIGNMENT,
    align_up,
    align_down,
    is_aligned,
    align_to_page,
    align_to_huge_page,
    align_to_pgd,
    calculate_padding,
    calculate_pages,
    calculate_huge_pages,
    find_optimal_alignment,
    align_offset_and_size,
    get_alignment_waste,
    get_alignment_efficiency,
    is_power_of_two,
    next_power_of_two,
    previous_power_of_two,
    align_address_to_page,
    is_address_page_aligned,
    calculate_page_range,
    format_size,
    parse_size,
    
    # 统计
    Timer,
    PerformanceTracker,
    MemoryUsageTracker,
    AccessPatternAnalyzer,
    Benchmark,
)

# 导入平台模块
from .platform import (
    PlatformBase,
    PlatformFactory,
)


def get_version() -> str:
    """
    获取版本号
    
    Returns:
        版本号字符串
    """
    return __version__


def create_mapper(config: Optional[MapperConfig] = None) -> MemoryMapper:
    """
    创建内存映射器（便捷函数）
    
    Args:
        config: 映射器配置，None则使用默认配置
        
    Returns:
        内存映射器实例
    """
    return MemoryMapper(config)
