"""
HSS (Hierarchically Semi-Separable) 矩阵模块

实现层次半可分矩阵的压缩和运算。

HSS矩阵是一种层次化矩阵格式，具有O(n log n)的存储复杂度，
适合处理具有层次结构的稠密矩阵（如积分方程、协方差矩阵等）。
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig


class HSSMatrix:
    """
    HSS (Hierarchically Semi-Separable) 矩阵

    层次半可分矩阵结构:
    - D: 对角块
    - U, V: 低秩离对角块
    - R, W: 传递算子

    存储复杂度: O(n log n)
    矩阵-向量乘法复杂度: O(n log n)
    """

    def __init__(self,
                 shape: Tuple[int, int],
                 leaf_size: int = 32,
                 rank: int = 16):
        """
        初始化HSS矩阵

        Args:
            shape: 矩阵形状
            leaf_size: 叶子节点大小
            rank: 低秩近似的秩
        """
        self.shape = shape
        self.leaf_size = leaf_size
        self.rank = rank

        # HSS结构
        self.D = {}  # 对角块
        self.U = {}  # 左基
        self.V = {}  # 右基
        self.R = {}  # 传递算子（左）
        self.W = {}  # 传递算子（右）

        self._tree_depth = int(np.ceil(np.log2(max(shape) / leaf_size)))

    def from_dense(self, arr: np.ndarray) -> 'HSSMatrix':
        """
        从密集矩阵构建HSS矩阵

        Args:
            arr: 密集矩阵

        Returns:
            HSSMatrix: HSS矩阵
        """
        n_rows, n_cols = arr.shape

        # 递归构建HSS结构
        self._build_hss_recursive(arr, 0, 0, n_rows, n_cols, 0)

        return self

    def _build_hss_recursive(self, arr: np.ndarray,
                             row_start: int, col_start: int,
                             n_rows: int, n_cols: int,
                             level: int):
        """递归构建HSS结构"""
        node_key = (row_start, col_start, n_rows, n_cols)

        # 基本情况：叶子节点
        if n_rows <= self.leaf_size or n_cols <= self.leaf_size:
            self.D[node_key] = arr[row_start:row_start + n_rows,
                                   col_start:col_start + n_cols].copy()
            return

        # 分割
        mid_row = n_rows // 2
        mid_col = n_cols // 2

        # 四个子块
        # A11, A12
        # A21, A22

        # 对角块（递归处理）
        self._build_hss_recursive(arr, row_start, col_start,
                                  mid_row, mid_col, level + 1)
        self._build_hss_recursive(arr, row_start + mid_row, col_start + mid_col,
                                  n_rows - mid_row, n_cols - mid_col, level + 1)

        # 离对角块（低秩近似）
        # A12
        A12 = arr[row_start:row_start + mid_row,
                  col_start + mid_col:col_start + n_cols]
        U12, V12 = self._low_rank_approx(A12)

        # A21
        A21 = arr[row_start + mid_row:row_start + n_rows,
                  col_start:col_start + mid_col]
        U21, V21 = self._low_rank_approx(A21)

        # 存储离对角块的低秩因子
        self.U[(row_start, col_start + mid_col)] = U12
        self.V[(row_start, col_start + mid_col)] = V12
        self.U[(row_start + mid_row, col_start)] = U21
        self.V[(row_start + mid_row, col_start)] = V21

    def _low_rank_approx(self, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        低秩近似

        Args:
            block: 输入块

        Returns:
            Tuple: (U, V) 使得 block ≈ U @ V.T
        """
        if np.allclose(block, 0):
            return np.zeros((block.shape[0], 0)), np.zeros((block.shape[1], 0))

        U, s, Vh = np.linalg.svd(block, full_matrices=False)

        # 截断
        rank = min(self.rank, len(s), min(block.shape))
        rank = max(1, rank)  # 至少保留1

        # 使用阈值截断
        threshold = 1e-10 * s[0] if len(s) > 0 else 0
        effective_rank = min(np.sum(s > threshold), rank)

        if effective_rank == 0:
            effective_rank = 1

        U_truncated = U[:, :effective_rank] * np.sqrt(s[:effective_rank])
        V_truncated = (np.sqrt(s[:effective_rank])[:, np.newaxis] * Vh[:effective_rank, :]).T

        return U_truncated, V_truncated

    def to_dense(self) -> np.ndarray:
        """
        转换为密集矩阵

        Returns:
            np.ndarray: 密集矩阵
        """
        result = np.zeros(self.shape, dtype=np.float64)
        self._to_dense_recursive(result, 0, 0, self.shape[0], self.shape[1])
        return result

    def _to_dense_recursive(self, result: np.ndarray,
                            row_start: int, col_start: int,
                            n_rows: int, n_cols: int):
        """递归重建密集矩阵"""
        node_key = (row_start, col_start, n_rows, n_cols)

        # 检查是否是叶子节点
        if node_key in self.D:
            result[row_start:row_start + n_rows,
                   col_start:col_start + n_cols] = self.D[node_key]
            return

        # 分割
        mid_row = n_rows // 2
        mid_col = n_cols // 2

        # 对角块
        self._to_dense_recursive(result, row_start, col_start,
                                mid_row, mid_col)
        self._to_dense_recursive(result, row_start + mid_row, col_start + mid_col,
                                n_rows - mid_row, n_cols - mid_col)

        # 离对角块
        key12 = (row_start, col_start + mid_col)
        if key12 in self.U:
            U = self.U[key12]
            V = self.V[key12]
            if U.shape[1] > 0:
                result[row_start:row_start + mid_row,
                       col_start + mid_col:col_start + n_cols] = U @ V.T

        key21 = (row_start + mid_row, col_start)
        if key21 in self.U:
            U = self.U[key21]
            V = self.V[key21]
            if U.shape[1] > 0:
                result[row_start + mid_row:row_start + n_rows,
                       col_start:col_start + mid_col] = U @ V.T

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        矩阵-向量乘法

        Args:
            x: 输入向量

        Returns:
            np.ndarray: 结果向量
        """
        result = np.zeros(self.shape[0], dtype=np.float64)
        self._matvec_recursive(x, result, 0, 0, self.shape[0], self.shape[1])
        return result

    def _matvec_recursive(self, x: np.ndarray, result: np.ndarray,
                          row_start: int, col_start: int,
                          n_rows: int, n_cols: int):
        """递归矩阵-向量乘法"""
        node_key = (row_start, col_start, n_rows, n_cols)

        # 叶子节点
        if node_key in self.D:
            result[row_start:row_start + n_rows] += self.D[node_key] @ x[col_start:col_start + n_cols]
            return

        mid_row = n_rows // 2
        mid_col = n_cols // 2

        # 对角块
        self._matvec_recursive(x, result, row_start, col_start, mid_row, mid_col)
        self._matvec_recursive(x, result, row_start + mid_row, col_start + mid_col,
                              n_rows - mid_row, n_cols - mid_col)

        # 离对角块
        key12 = (row_start, col_start + mid_col)
        if key12 in self.U:
            U = self.U[key12]
            V = self.V[key12]
            if U.shape[1] > 0:
                result[row_start:row_start + mid_row] += U @ (V.T @ x[col_start + mid_col:col_start + n_cols])

        key21 = (row_start + mid_row, col_start)
        if key21 in self.U:
            U = self.U[key21]
            V = self.V[key21]
            if U.shape[1] > 0:
                result[row_start + mid_row:row_start + n_rows] += U @ (V.T @ x[col_start:col_start + mid_col])

    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        total = 0

        for D in self.D.values():
            total += D.nbytes

        for U in self.U.values():
            total += U.nbytes

        for V in self.V.values():
            total += V.nbytes

        return total

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        original_size = self.shape[0] * self.shape[1] * 8  # float64
        compressed_size = self.get_memory_usage()

        if compressed_size == 0:
            return 1.0

        return original_size / compressed_size


def hss_compress(arr: SparseArray,
                 leaf_size: int = 32,
                 rank: int = 16) -> HSSMatrix:
    """
    将稀疏数组压缩为HSS矩阵

    Args:
        arr: 稀疏数组
        leaf_size: 叶子节点大小
        rank: 低秩近似的秩

    Returns:
        HSSMatrix: HSS矩阵
    """
    dense = arr.to_dense()
    hss = HSSMatrix(arr.shape, leaf_size, rank)  # type: ignore
    return hss.from_dense(dense)  # type: ignore


def hss_decompress(hss: HSSMatrix) -> np.ndarray:
    """
    将HSS矩阵解压缩为密集矩阵

    Args:
        hss: HSS矩阵

    Returns:
        np.ndarray: 密集矩阵
    """
    return hss.to_dense()
