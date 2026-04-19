"""
Buffer Manager - 高性能缓冲区管理器

提供缓冲区管理、缓冲池、队列、替换策略等功能。
"""

from .core.buffer import Buffer
from .core.config import BufferConfig
from .core.exceptions import (
    BufferError,
    BufferFullError,
    BufferEmptyError,
    BufferTimeoutError,
    PoolExhaustedError,
)
from .pools import BufferPool, PoolStats, RingBuffer, DoubleBuffer
from .queues import SPSCQueue, MPSCQueue, MPMCQueue
from .strategy import ARC, LRU, Prefetcher, AdaptiveStrategy

__version__ = "1.0.0"
__all__ = [
    "Buffer",
    "BufferConfig",
    "BufferError",
    "BufferFullError",
    "BufferEmptyError",
    "BufferTimeoutError",
    "PoolExhaustedError",
    "BufferPool",
    "PoolStats",
    "RingBuffer",
    "DoubleBuffer",
    "SPSCQueue",
    "MPSCQueue",
    "MPMCQueue",
    "ARC",
    "LRU",
    "Prefetcher",
    "AdaptiveStrategy",
]
