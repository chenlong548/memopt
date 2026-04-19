"""
sparse_array GPU加速模块

提供GPU加速的稀疏矩阵运算。
"""

from .cusparse import (
    cusparse_spmv,
    cusparse_spmm,
    is_cusparse_available,
    get_gpu_info
)
from .tensor_core import (
    tensor_core_spmm,
    optimize_for_tensor_core,
    is_tensor_core_available
)

__all__ = [
    'cusparse_spmv',
    'cusparse_spmm',
    'is_cusparse_available',
    'get_gpu_info',
    'tensor_core_spmm',
    'optimize_for_tensor_core',
    'is_tensor_core_available'
]
