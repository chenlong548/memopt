"""
mem_optimizer NUMA模块

提供NUMA感知的内存分配功能。
"""

from .coordinator import (
    NUMACoordinator,
    NUMANode,
    NodeState,
    MemoryDistribution
)

__all__ = [
    'NUMACoordinator',
    'NUMANode',
    'NodeState',
    'MemoryDistribution'
]
