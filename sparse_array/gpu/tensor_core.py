"""
Tensor Core优化模块

提供针对NVIDIA Tensor Core的稀疏矩阵运算优化。

Tensor Core是NVIDIA GPU的专用矩阵计算单元，可以显著加速矩阵乘法。
对于稀疏矩阵，可以使用BCSR格式来利用Tensor Core。
"""

from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig
from ..core.exceptions import GPUError


# 缓存大小限制常量
DEFAULT_MAX_CACHE_SIZE = 100  # 默认最大缓存条目数


# 检查Tensor Core是否可用
_TENSOR_CORE_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True

    # 检查计算能力是否支持Tensor Core (>= 7.0)
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    compute_capability = props['major'] + props['minor'] * 0.1

    if compute_capability >= 7.0:
        _TENSOR_CORE_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def is_tensor_core_available() -> bool:
    """
    检查Tensor Core是否可用

    Returns:
        bool: 是否可用
    """
    return _TENSOR_CORE_AVAILABLE


def optimize_for_tensor_core(A: SparseArray,
                             block_size: Tuple[int, int] = (16, 16)) -> SparseArray:
    """
    优化稀疏矩阵以利用Tensor Core

    将稀疏矩阵转换为适合Tensor Core的BCSR格式。

    Args:
        A: 输入稀疏矩阵
        block_size: 块大小，推荐 (16, 16) 或 (8, 8)

    Returns:
        SparseArray: 优化后的稀疏矩阵
    """
    from ..formats.bcsr import BCSRFormat

    # 转换为密集数组
    dense = A.to_dense()

    # 创建BCSR格式
    bcsr = BCSRFormat.from_dense(dense, block_size)

    # 创建新的SparseArray
    result = SparseArray(
        shape=A.shape,
        format='bcsr',
        config=A._config
    )
    result._format = bcsr
    result._stats.nnz = bcsr.nnz
    result._stats.calculate_density()

    return result


def tensor_core_spmm(A: SparseArray,
                     B: np.ndarray,
                     config: Optional[SparseArrayConfig] = None) -> np.ndarray:
    """
    使用Tensor Core执行稀疏矩阵-密集矩阵乘法

    Args:
        A: 稀疏矩阵（推荐BCSR格式）
        B: 密集矩阵
        config: 配置对象

    Returns:
        np.ndarray: 结果矩阵
    """
    config = config or SparseArrayConfig()

    if not _TENSOR_CORE_AVAILABLE:
        raise GPUError(
            "Tensor Core is not available. Requires NVIDIA GPU with compute capability >= 7.0",
            device_id=config.gpu_device,
            operation="tensor_core_spmm"
        )

    # 如果不是BCSR格式，尝试转换
    if A.format != 'bcsr':
        # 分析是否适合BCSR
        block_score = _analyze_block_structure(A)
        if block_score > 0.5:
            A = optimize_for_tensor_core(A)
        else:
            # 回退到cuSPARSE
            from .cusparse import cusparse_spmm
            return cusparse_spmm(A, B, config)

    # 使用BCSR格式进行Tensor Core优化
    return _tensor_core_spmm_bcsr(A, B, config)


def _tensor_core_spmm_bcsr(A: SparseArray,
                           B: np.ndarray,
                           config: SparseArrayConfig) -> np.ndarray:
    """BCSR格式的Tensor Core SpMM"""
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

    bcsr = A._format

    # 将BCSR转换为密集矩阵进行计算
    # （实际实现中会使用专门的Tensor Core内核）
    dense_A = bcsr.to_dense()

    try:
        # 传输到GPU
        A_gpu = cp.array(dense_A)
        B_gpu = cp.array(B)

        # 使用Tensor Core加速的矩阵乘法
        # CuPy会自动使用Tensor Core
        with cp.cuda.Stream():
            # 启用Tensor Core
            C_gpu = cp.matmul(A_gpu, B_gpu)

        return cp.asnumpy(C_gpu)

    except Exception as e:
        raise GPUError(
            f"Tensor Core SpMM failed: {str(e)}",
            device_id=config.gpu_device,
            operation="tensor_core_spmm"
        )


def _analyze_block_structure(A: SparseArray) -> float:
    """
    分析稀疏矩阵的块结构

    Args:
        A: 稀疏矩阵

    Returns:
        float: 块结构评分 (0-1)
    """
    if A._format is None:
        return 0.0

    # 简单的块结构检测
    dense = A.to_dense()
    n_rows, n_cols = dense.shape

    # 检查2x2块
    block_size = 2
    n_block_rows = n_rows // block_size
    n_block_cols = n_cols // block_size

    if n_block_rows == 0 or n_block_cols == 0:
        return 0.0

    full_blocks = 0
    empty_blocks = 0
    total_blocks = n_block_rows * n_block_cols

    for i in range(n_block_rows):
        for j in range(n_block_cols):
            block = dense[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size]
            if np.all(block != 0):
                full_blocks += 1
            elif np.all(block == 0):
                empty_blocks += 1

    # 评分：全满或全空的块比例
    return (full_blocks + empty_blocks) / total_blocks


def get_optimal_block_size(A: SparseArray) -> Tuple[int, int]:
    """
    获取最优块大小

    Args:
        A: 稀疏矩阵

    Returns:
        Tuple[int, int]: 推荐的块大小
    """
    # Tensor Core最佳块大小
    tensor_core_sizes = [(16, 16), (8, 8), (4, 4)]

    best_size = (4, 4)
    best_score = 0.0

    for size in tensor_core_sizes:
        # 尝试创建BCSR
        from ..formats.bcsr import BCSRFormat
        try:
            dense = A.to_dense()
            bcsr = BCSRFormat.from_dense(dense, size)
            score = bcsr.get_average_block_fill()

            if score > best_score:
                best_score = score
                best_size = size
        except Exception:
            continue

    return best_size


class TensorCoreOptimizer:
    """
    Tensor Core优化器

    自动优化稀疏矩阵以利用Tensor Core。
    使用LRU缓存策略限制缓存大小，防止内存无限增长。
    """

    def __init__(self, config: Optional[SparseArrayConfig] = None,
                 max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        """
        初始化优化器

        Args:
            config: 配置对象
            max_cache_size: 最大缓存条目数，默认100
        """
        self.config = config or SparseArrayConfig()
        self._max_cache_size = max_cache_size
        # 使用OrderedDict实现LRU缓存
        self._cache: OrderedDict = OrderedDict()

    def optimize(self, A: SparseArray) -> SparseArray:
        """
        优化稀疏矩阵

        Args:
            A: 输入稀疏矩阵

        Returns:
            SparseArray: 优化后的稀疏矩阵
        """
        # 检查缓存
        cache_key = (id(A), A.nnz, A.format)
        if cache_key in self._cache:
            # 命中缓存，移动到末尾（最近使用）
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # 获取最优块大小
        block_size = get_optimal_block_size(A)

        # 优化
        optimized = optimize_for_tensor_core(A, block_size)

        # 缓存结果
        if self.config.enable_cache:
            # 检查缓存大小，如果超过限制则删除最旧的条目
            while len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)  # 删除最旧的条目
            self._cache[cache_key] = optimized

        return optimized

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """
        获取当前缓存大小

        Returns:
            int: 缓存中的条目数
        """
        return len(self._cache)

    def set_max_cache_size(self, size: int):
        """
        设置最大缓存大小

        Args:
            size: 新的最大缓存大小

        Raises:
            ValueError: 如果size小于等于0
        """
        if size <= 0:
            raise ValueError("max_cache_size must be positive")
        self._max_cache_size = size
        # 如果当前缓存超过新限制，删除多余的条目
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
