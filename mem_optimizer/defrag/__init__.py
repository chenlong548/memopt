"""
mem_optimizer 碎片整理模块

提供内存碎片整理功能。
"""

from .defragmenter import (
    Defragmenter,
    DefragState,
    DefragPlan,
    DefragResult
)
from .coalescer import (
    MemoryCoalescer,
    GapFiller,
    CoalesceRegion,
    CoalesceStrategy
)
from .psi_metrics import (
    PSIMonitor,
    PSIMetrics,
    PSIAlert,
    PSIType
)

__all__ = [
    'Defragmenter',
    'DefragState',
    'DefragPlan',
    'DefragResult',
    'MemoryCoalescer',
    'GapFiller',
    'CoalesceRegion',
    'CoalesceStrategy',
    'PSIMonitor',
    'PSIMetrics',
    'PSIAlert',
    'PSIType'
]
