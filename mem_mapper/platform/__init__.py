"""
mem_mapper 平台模块

提供跨平台的内存映射操作实现。
"""

from .base import (
    PlatformBase,
    PlatformFactory,
)

__all__ = [
    'PlatformBase',
    'PlatformFactory',
]
