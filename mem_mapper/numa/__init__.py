"""
mem_mapper NUMA模块

提供NUMA拓扑检测和策略管理功能。
"""

from .topology import (
    NUMANode,
    NUMATopology,
    NUMATopologyDetector,
    get_numa_topology,
    is_numa_available,
)

from .policy import (
    NUMAPolicyMode,
    NUMAPolicyConfig,
    NUMAPolicyManager,
    get_numa_policy_manager,
)

__all__ = [
    # 拓扑
    'NUMANode',
    'NUMATopology',
    'NUMATopologyDetector',
    'get_numa_topology',
    'is_numa_available',
    
    # 策略
    'NUMAPolicyMode',
    'NUMAPolicyConfig',
    'NUMAPolicyManager',
    'get_numa_policy_manager',
]
