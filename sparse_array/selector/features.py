"""
特征提取模块

提取稀疏矩阵的特征，用于格式选择。
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import FormatFeatures


def extract_features(arr: np.ndarray) -> FormatFeatures:
    """
    从密集数组提取特征

    Args:
        arr: 密集数组

    Returns:
        FormatFeatures: 特征对象
    """
    features = FormatFeatures()

    # 基本统计
    total_elements = arr.size
    nnz = np.count_nonzero(arr)

    features.total_elements = total_elements
    features.sparsity = 1.0 - nnz / total_elements if total_elements > 0 else 0.0
    features.density = nnz / total_elements if total_elements > 0 else 0.0

    if arr.ndim == 2:
        n_rows, n_cols = arr.shape
        features.aspect_ratio = n_rows / n_cols if n_cols > 0 else 1.0

        # 结构特征
        features.bandwidth = _compute_bandwidth(arr)
        features.bandwidth_ratio = features.bandwidth / max(n_rows, n_cols) if max(n_rows, n_cols) > 0 else 0.0
        features.diagonal_ratio = _compute_diagonal_ratio(arr)
        features.block_structure_score = _compute_block_score(arr)

        # 分布特征
        row_nnz = np.sum(arr != 0, axis=1)
        col_nnz = np.sum(arr != 0, axis=0)

        features.row_nnz_variance = float(np.var(row_nnz))
        features.col_nnz_variance = float(np.var(col_nnz))
        features.max_row_nnz = int(np.max(row_nnz))
        features.max_col_nnz = int(np.max(col_nnz))
        
        # 判断是否规则分布
        row_cv = features.row_nnz_variance ** 0.5 / np.mean(row_nnz) if np.mean(row_nnz) > 0 else 0
        col_cv = features.col_nnz_variance ** 0.5 / np.mean(col_nnz) if np.mean(col_nnz) > 0 else 0
        features.is_regular = row_cv < 0.3 and col_cv < 0.3

    return features


def analyze_sparsity(arr: np.ndarray) -> Dict[str, Any]:
    """
    分析稀疏性

    Args:
        arr: 密集数组

    Returns:
        Dict: 稀疏性分析结果
    """
    total = arr.size
    nnz = np.count_nonzero(arr)

    # 稀疏度
    sparsity = 1.0 - nnz / total if total > 0 else 0.0

    # 稀疏模式
    if arr.ndim == 2:
        row_nnz = np.sum(arr != 0, axis=1)
        col_nnz = np.sum(arr != 0, axis=0)

        # 行稀疏度分布
        row_sparsity = 1.0 - row_nnz / arr.shape[1]
        col_sparsity = 1.0 - col_nnz / arr.shape[0]

        return {
            'total_elements': total,
            'nnz': nnz,
            'sparsity': sparsity,
            'density': 1.0 - sparsity,
            'row_sparsity_mean': float(np.mean(row_sparsity)),
            'row_sparsity_std': float(np.std(row_sparsity)),
            'col_sparsity_mean': float(np.mean(col_sparsity)),
            'col_sparsity_std': float(np.std(col_sparsity)),
            'empty_rows': int(np.sum(row_nnz == 0)),
            'empty_cols': int(np.sum(col_nnz == 0))
        }
    else:
        return {
            'total_elements': total,
            'nnz': nnz,
            'sparsity': sparsity,
            'density': 1.0 - sparsity
        }


def analyze_structure(arr: np.ndarray) -> Dict[str, Any]:
    """
    分析结构特征

    Args:
        arr: 密集数组

    Returns:
        Dict: 结构分析结果
    """
    if arr.ndim != 2:
        return {'error': 'Structure analysis only supports 2D arrays'}

    n_rows, n_cols = arr.shape

    # 带宽分析
    bandwidth = _compute_bandwidth(arr)

    # 对角线分析
    diagonal_ratio = _compute_diagonal_ratio(arr)

    # 块结构分析
    block_score = _compute_block_score(arr)

    # 三角结构
    lower_triangle = np.sum(np.tril(arr, -1) != 0)
    upper_triangle = np.sum(np.triu(arr, 1) != 0)
    diagonal = np.sum(np.diag(arr) != 0)

    return {
        'bandwidth': bandwidth,
        'bandwidth_ratio': bandwidth / max(n_rows, n_cols) if max(n_rows, n_cols) > 0 else 0,
        'diagonal_ratio': diagonal_ratio,
        'block_structure_score': block_score,
        'lower_triangle_nnz': int(lower_triangle),
        'upper_triangle_nnz': int(upper_triangle),
        'diagonal_nnz': int(diagonal),
        'is_triangular': diagonal + lower_triangle == np.count_nonzero(arr) or
                        diagonal + upper_triangle == np.count_nonzero(arr),
        'is_diagonal': diagonal_ratio > 0.9
    }


def analyze_distribution(arr: np.ndarray) -> Dict[str, Any]:
    """
    分析非零元素分布

    Args:
        arr: 密集数组

    Returns:
        Dict: 分布分析结果
    """
    if arr.ndim != 2:
        return {'error': 'Distribution analysis only supports 2D arrays'}

    row_nnz = np.sum(arr != 0, axis=1)
    col_nnz = np.sum(arr != 0, axis=0)

    # 统计量
    row_stats = {
        'mean': float(np.mean(row_nnz)),
        'std': float(np.std(row_nnz)),
        'min': int(np.min(row_nnz)),
        'max': int(np.max(row_nnz)),
        'median': float(np.median(row_nnz))
    }

    col_stats = {
        'mean': float(np.mean(col_nnz)),
        'std': float(np.std(col_nnz)),
        'min': int(np.min(col_nnz)),
        'max': int(np.max(col_nnz)),
        'median': float(np.median(col_nnz))
    }

    # 分布类型判断
    row_cv = row_stats['std'] / row_stats['mean'] if row_stats['mean'] > 0 else 0
    col_cv = col_stats['std'] / col_stats['mean'] if col_stats['mean'] > 0 else 0

    distribution_type = 'uniform'
    if row_cv > 1.0 or col_cv > 1.0:
        distribution_type = 'skewed'
    elif row_cv > 0.5 or col_cv > 0.5:
        distribution_type = 'moderate'

    return {
        'row_distribution': row_stats,
        'col_distribution': col_stats,
        'row_cv': row_cv,
        'col_cv': col_cv,
        'distribution_type': distribution_type,
        'is_regular': row_cv < 0.3 and col_cv < 0.3
    }


def _compute_bandwidth(arr: np.ndarray) -> int:
    """计算带宽"""
    rows, cols = np.nonzero(arr)
    if len(rows) == 0:
        return 0

    bandwidth = np.max(np.abs(cols - rows))
    return int(bandwidth)


def _compute_diagonal_ratio(arr: np.ndarray) -> float:
    """计算对角线元素比例"""
    nnz = np.count_nonzero(arr)
    if nnz == 0:
        return 0.0

    diagonal_nnz = np.count_nonzero(np.diag(arr))
    return diagonal_nnz / nnz


def _compute_block_score(arr: np.ndarray, block_size: int = 4) -> float:
    """
    计算块结构评分

    检查矩阵是否具有规则的块结构。
    """
    n_rows, n_cols = arr.shape

    n_block_rows = n_rows // block_size
    n_block_cols = n_cols // block_size

    if n_block_rows == 0 or n_block_cols == 0:
        return 0.0

    full_blocks = 0
    empty_blocks = 0
    total_blocks = n_block_rows * n_block_cols

    for i in range(n_block_rows):
        for j in range(n_block_cols):
            block = arr[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]

            if np.all(block != 0):
                full_blocks += 1
            elif np.all(block == 0):
                empty_blocks += 1

    return (full_blocks + empty_blocks) / total_blocks


def extract_features_from_sparse(arr: SparseArray) -> FormatFeatures:
    """
    从稀疏数组提取特征

    Args:
        arr: 稀疏数组

    Returns:
        FormatFeatures: 特征对象
    """
    features = FormatFeatures()

    # 基本统计
    features.total_elements = 1
    for dim in arr.shape:
        features.total_elements *= dim

    features.sparsity = 1.0 - arr.nnz / features.total_elements if features.total_elements > 0 else 0.0
    features.density = 1.0 - features.sparsity

    if arr.ndim == 2:
        features.aspect_ratio = arr.shape[0] / arr.shape[1] if arr.shape[1] > 0 else 1.0

        # 从CSR格式提取特征
        if arr.format == 'csr':
            csr = arr._format
            features.max_row_nnz = int(np.max(np.diff(csr.indptr)))
            features.row_nnz_variance = float(np.var(np.diff(csr.indptr)))

            # 计算带宽
            rows = np.repeat(np.arange(arr.shape[0]), np.diff(csr.indptr))
            cols = csr.indices
            if len(cols) > 0:
                features.bandwidth = int(np.max(np.abs(cols - rows)))

            # 计算对角线比例
            diagonal_mask = rows == cols
            features.diagonal_ratio = np.sum(diagonal_mask) / arr.nnz if arr.nnz > 0 else 0.0

        # 块结构评分
        features.block_structure_score = _compute_block_score(arr.to_dense())

    return features
