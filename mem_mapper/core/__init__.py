"""
mem_mapper 核心模块

提供核心数据结构和异常定义。
"""

from .exceptions import (
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
)

from .region import (
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
)

from .registry import (
    CleanupTask,
    MappingRegistry,
    MappingRegistryView,
)

from .mapper import (
    MapperConfig,
    MemoryMapper,
)

__all__ = [
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
    'MappedRegion',
    
    # 注册表
    'CleanupTask',
    'MappingRegistry',
    'MappingRegistryView',
    
    # 映射器
    'MapperConfig',
    'MemoryMapper',
]
