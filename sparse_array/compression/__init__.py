"""
sparse_array 压缩模块

提供稀疏矩阵的高级压缩技术。
"""

from .low_rank import (
    BlockLowRankCompressor,
    compress_low_rank,
    decompress_low_rank
)
from .hss import (
    HSSMatrix,
    hss_compress,
    hss_decompress
)

__all__ = [
    'BlockLowRankCompressor',
    'compress_low_rank',
    'decompress_low_rank',
    'HSSMatrix',
    'hss_compress',
    'hss_decompress'
]
