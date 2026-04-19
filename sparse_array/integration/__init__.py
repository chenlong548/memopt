"""
sparse_array 集成模块

提供与NumPy和SciPy的集成适配器。
"""

from .numpy_adapter import (
    to_numpy,
    from_numpy,
    numpy_compat
)
from .scipy_adapter import (
    to_scipy_sparse,
    from_scipy_sparse,
    scipy_compat
)

__all__ = [
    'to_numpy',
    'from_numpy',
    'numpy_compat',
    'to_scipy_sparse',
    'from_scipy_sparse',
    'scipy_compat'
]
