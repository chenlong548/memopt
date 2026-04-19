"""
块低秩压缩模块

实现稀疏矩阵的块低秩压缩技术。

块低秩压缩将稀疏矩阵划分为块，对每个块进行低秩近似（SVD/QR），
从而减少存储和计算量。
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig


class BlockLowRankCompressor:
    """
    块低秩压缩器

    将稀疏矩阵划分为块，对每个块进行低秩近似。

    存储结构:
    - U: 左奇异向量矩阵块列表
    - V: 右奇异向量矩阵块列表
    - ranks: 每个块的秩
    - block_info: 块位置信息
    """

    def __init__(self,
                 block_size: Tuple[int, int] = (32, 32),
                 rank_threshold: float = 1e-6,
                 max_rank: int = 32):
        """
        初始化压缩器

        Args:
            block_size: 块大小
            rank_threshold: 秩截断阈值
            max_rank: 最大秩
        """
        self.block_size = block_size
        self.rank_threshold = rank_threshold
        self.max_rank = max_rank

    def compress(self, arr: SparseArray) -> Dict[str, Any]:
        """
        压缩稀疏数组

        Args:
            arr: 稀疏数组

        Returns:
            Dict: 压缩数据
        """
        dense = arr.to_dense()
        n_rows, n_cols = dense.shape
        R, C = self.block_size

        U_blocks = []
        V_blocks = []
        ranks = []
        block_info = []

        # 划分块
        n_block_rows = (n_rows + R - 1) // R
        n_block_cols = (n_cols + C - 1) // C

        for br in range(n_block_rows):
            for bc in range(n_block_cols):
                row_start = br * R
                row_end = min((br + 1) * R, n_rows)
                col_start = bc * C
                col_end = min((bc + 1) * C, n_cols)

                # 提取块
                block = dense[row_start:row_end, col_start:col_end]

                # 检查是否为零块
                if np.allclose(block, 0):
                    continue

                # 低秩近似
                U, s, Vh = np.linalg.svd(block, full_matrices=False)

                # 截断
                effective_rank = min(
                    np.sum(s > self.rank_threshold * s[0]),
                    self.max_rank,
                    len(s)
                )

                if effective_rank == 0:
                    continue

                # 存储压缩后的块
                U_blocks.append(U[:, :effective_rank] * np.sqrt(s[:effective_rank]))
                V_blocks.append(np.sqrt(s[:effective_rank])[:, np.newaxis] * Vh[:effective_rank, :])
                ranks.append(effective_rank)
                block_info.append((br, bc, row_start, row_end, col_start, col_end))

        return {
            'U_blocks': U_blocks,
            'V_blocks': V_blocks,
            'ranks': ranks,
            'block_info': block_info,
            'shape': arr.shape,
            'block_size': self.block_size,
            'n_block_rows': n_block_rows,
            'n_block_cols': n_block_cols
        }

    def decompress(self, compressed: Dict[str, Any]) -> np.ndarray:
        """
        解压缩

        Args:
            compressed: 压缩数据

        Returns:
            np.ndarray: 密集数组
        """
        shape = compressed['shape']
        result = np.zeros(shape, dtype=np.float64)

        for i, (br, bc, row_start, row_end, col_start, col_end) in enumerate(compressed['block_info']):
            U = compressed['U_blocks'][i]
            V = compressed['V_blocks'][i]

            # 重构块
            block = U @ V
            result[row_start:row_end, col_start:col_end] = block[:row_end - row_start, :col_end - col_start]

        return result

    def get_compression_ratio(self, arr: SparseArray, compressed: Dict[str, Any]) -> float:
        """
        计算压缩比

        Args:
            arr: 原始稀疏数组
            compressed: 压缩数据

        Returns:
            float: 压缩比
        """
        original_size = arr.shape[0] * arr.shape[1] * 8  # 假设float64

        compressed_size = 0
        for i, rank in enumerate(compressed['ranks']):
            U = compressed['U_blocks'][i]
            V = compressed['V_blocks'][i]
            compressed_size += U.nbytes + V.nbytes

        if compressed_size == 0:
            return 1.0

        return original_size / compressed_size


def compress_low_rank(arr: SparseArray,
                      block_size: Tuple[int, int] = (32, 32),
                      rank_threshold: float = 1e-6,
                      max_rank: int = 32) -> Dict[str, Any]:
    """
    压缩稀疏数组为低秩形式

    Args:
        arr: 稀疏数组
        block_size: 块大小
        rank_threshold: 秩截断阈值
        max_rank: 最大秩

    Returns:
        Dict: 压缩数据
    """
    compressor = BlockLowRankCompressor(block_size, rank_threshold, max_rank)
    return compressor.compress(arr)


def decompress_low_rank(compressed: Dict[str, Any]) -> np.ndarray:
    """
    解压缩低秩形式

    Args:
        compressed: 压缩数据

    Returns:
        np.ndarray: 密集数组
    """
    compressor = BlockLowRankCompressor()
    return compressor.decompress(compressed)


class AdaptiveBlockCompressor(BlockLowRankCompressor):
    """
    自适应块压缩器

    根据块的特性自适应选择压缩策略。
    """

    def __init__(self,
                 block_size: Tuple[int, int] = (32, 32),
                 rank_threshold: float = 1e-6,
                 max_rank: int = 32,
                 sparsity_threshold: float = 0.3):
        """
        初始化自适应压缩器

        Args:
            block_size: 块大小
            rank_threshold: 秩截断阈值
            max_rank: 最大秩
            sparsity_threshold: 稀疏度阈值
        """
        super().__init__(block_size, rank_threshold, max_rank)
        self.sparsity_threshold = sparsity_threshold

    def compress(self, arr: SparseArray) -> Dict[str, Any]:
        """
        自适应压缩

        Args:
            arr: 稀疏数组

        Returns:
            Dict: 压缩数据
        """
        dense = arr.to_dense()
        n_rows, n_cols = dense.shape
        R, C = self.block_size

        U_blocks = []
        V_blocks = []
        ranks = []
        block_info = []
        sparse_blocks = []  # 存储稀疏块

        n_block_rows = (n_rows + R - 1) // R
        n_block_cols = (n_cols + C - 1) // C

        for br in range(n_block_rows):
            for bc in range(n_block_cols):
                row_start = br * R
                row_end = min((br + 1) * R, n_rows)
                col_start = bc * C
                col_end = min((bc + 1) * C, n_cols)

                block = dense[row_start:row_end, col_start:col_end]

                if np.allclose(block, 0):
                    continue

                # 计算稀疏度
                sparsity = 1.0 - np.count_nonzero(block) / block.size

                if sparsity > self.sparsity_threshold:
                    # 稀疏块：直接存储
                    sparse_blocks.append({
                        'data': block[block != 0],
                        'indices': np.argwhere(block != 0),
                        'position': (row_start, row_end, col_start, col_end)
                    })
                    block_info.append((br, bc, row_start, row_end, col_start, col_end, 'sparse'))
                else:
                    # 密集块：低秩压缩
                    U, s, Vh = np.linalg.svd(block, full_matrices=False)

                    effective_rank = min(
                        np.sum(s > self.rank_threshold * s[0]),
                        self.max_rank,
                        len(s)
                    )

                    if effective_rank == 0:
                        continue

                    U_blocks.append(U[:, :effective_rank] * np.sqrt(s[:effective_rank]))
                    V_blocks.append(np.sqrt(s[:effective_rank])[:, np.newaxis] * Vh[:effective_rank, :])
                    ranks.append(effective_rank)
                    block_info.append((br, bc, row_start, row_end, col_start, col_end, 'low_rank'))

        return {
            'U_blocks': U_blocks,
            'V_blocks': V_blocks,
            'ranks': ranks,
            'block_info': block_info,
            'sparse_blocks': sparse_blocks,
            'shape': arr.shape,
            'block_size': self.block_size,
            'n_block_rows': n_block_rows,
            'n_block_cols': n_block_cols
        }
