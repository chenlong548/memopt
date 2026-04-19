"""
sparse_array 核心模块

提供稀疏数组的核心类和接口。
"""

from .sparse_array import SparseArray
from .config import SparseArrayConfig, SparseFormat
from .exceptions import (
    SparseArrayError,
    FormatConversionError,
    UnsupportedOperationError,
    DimensionError,
    IndexOutOfBoundsError,
    GPUError,
    CompressionError
)

__all__ = [
    'SparseArray',
    'SparseArrayConfig',
    'SparseFormat',
    'SparseArrayError',
    'FormatConversionError',
    'UnsupportedOperationError',
    'DimensionError',
    'IndexOutOfBoundsError',
    'GPUError',
    'CompressionError'
]
