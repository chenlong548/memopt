"""
Core模块 - 核心层

提供Buffer主类、配置管理和异常定义。
"""

from .buffer import Buffer
from .config import BufferConfig
from .exceptions import (
    BufferError,
    BufferFullError,
    BufferEmptyError,
    BufferTimeoutError,
    PoolExhaustedError,
)

__all__ = [
    "Buffer",
    "BufferConfig",
    "BufferError",
    "BufferFullError",
    "BufferEmptyError",
    "BufferTimeoutError",
    "PoolExhaustedError",
]
