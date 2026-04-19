"""
sparse_array 运算模块

提供稀疏数组的各种运算操作。
"""

from .arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    negate,
    sum,
    mean,
    max,
    min
)
from .linalg import (
    spmv,
    spmm,
    norm,
    dot
)
from .transform import (
    save_sparse,
    load_sparse,
    convert_format
)

__all__ = [
    # 算术运算
    'add',
    'subtract',
    'multiply',
    'divide',
    'negate',
    'sum',
    'mean',
    'max',
    'min',
    # 线性代数
    'spmv',
    'spmm',
    'norm',
    'dot',
    # 转换
    'save_sparse',
    'load_sparse',
    'convert_format'
]
