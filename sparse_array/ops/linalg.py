"""
线性代数运算模块

提供稀疏矩阵的线性代数运算，包括：
- SpMV: 稀疏矩阵-向量乘法
- SpMM: 稀疏矩阵-矩阵乘法
- 范数计算
"""

from typing import Union, Optional, Tuple
import numpy as np

from ..core.sparse_array import SparseArray, FormatBase
from ..core.config import SparseArrayConfig, ComputeBackend
from ..core.exceptions import DimensionError, UnsupportedOperationError


def spmv(A: SparseArray,
         x: np.ndarray,
         config: Optional[SparseArrayConfig] = None) -> np.ndarray:
    """
    稀疏矩阵-向量乘法 (SpMV)

    计算 y = A @ x

    Args:
        A: 稀疏矩阵
        x: 密集向量
        config: 配置对象

    Returns:
        np.ndarray: 结果向量 y

    Example:
        >>> A = SparseArray.random((1000, 1000), density=0.1)
        >>> x = np.random.rand(1000)
        >>> y = spmv(A, x)
    """
    config = config or SparseArrayConfig()

    if A.ndim != 2:
        raise DimensionError("SpMV requires 2D matrix", expected_dim=2, actual_dim=A.ndim)

    if x.ndim != 1:
        raise DimensionError("SpMV requires 1D vector", expected_dim=1, actual_dim=x.ndim)

    if A.shape[1] != len(x):
        raise DimensionError(
            f"Matrix-vector dimension mismatch: A.shape[1]={A.shape[1]}, x.len={len(x)}"
        )

    # 选择计算后端
    if config.backend == ComputeBackend.AUTO:
        backend = _select_backend_spmv(A, x, config)
    else:
        backend = config.backend

    # 执行计算
    if backend == ComputeBackend.GPU_CUSPARSE or backend == ComputeBackend.GPU_TENSOR_CORE:
        return _spmv_gpu(A, x, config, backend)
    else:
        return _spmv_cpu(A, x, config)


def _spmv_cpu(A: SparseArray, x: np.ndarray, config: SparseArrayConfig) -> np.ndarray:
    """CPU实现的SpMV"""
    if A._format is None:
        return np.zeros(A.shape[0], dtype=A.dtype)

    format_name = A._format.get_format_name()

    if format_name == 'csr':
        return _spmv_csr(A._format, x)
    elif format_name == 'csc':
        return _spmv_csc(A._format, x)
    elif format_name == 'coo':
        return _spmv_coo(A._format, x)
    elif format_name == 'bcsr':
        return _spmv_bcsr(A._format, x)
    else:
        # 转换为CSR后计算
        csr = A._format.to_csr()
        return _spmv_csr(csr, x)


def _spmv_csr(format_obj, x: np.ndarray) -> np.ndarray:
    """CSR格式的SpMV"""
    n_rows = format_obj.shape[0]
    y = np.zeros(n_rows, dtype=x.dtype)

    data = format_obj.data
    indices = format_obj.indices
    indptr = format_obj.indptr

    # 并行化：使用NumPy向量化
    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            y[i] = np.dot(data[start:end], x[indices[start:end]])

    return y


def _spmv_csc(format_obj, x: np.ndarray) -> np.ndarray:
    """CSC格式的SpMV"""
    n_rows = format_obj.shape[0]
    y = np.zeros(n_rows, dtype=x.dtype)

    data = format_obj.data
    indices = format_obj.indices
    indptr = format_obj.indptr

    # CSC格式按列遍历
    for j in range(format_obj.shape[1]):
        start, end = indptr[j], indptr[j + 1]
        if end > start:
            y[indices[start:end]] += data[start:end] * x[j]

    return y


def _spmv_coo(format_obj, x: np.ndarray) -> np.ndarray:
    """COO格式的SpMV"""
    n_rows = format_obj.shape[0]
    y = np.zeros(n_rows, dtype=x.dtype)

    rows = format_obj.rows
    cols = format_obj.cols
    data = format_obj.data

    # 向量化计算
    np.add.at(y, rows, data * x[cols])

    return y


def _spmv_bcsr(format_obj, x: np.ndarray) -> np.ndarray:
    """BCSR格式的SpMV"""
    n_rows = format_obj.shape[0]
    y = np.zeros(n_rows, dtype=x.dtype)

    R, C = format_obj.block_size
    row_ptr = format_obj.row_ptr
    col_indices = format_obj.col_indices
    data = format_obj.data

    for br in range(format_obj._n_block_rows):
        start, end = row_ptr[br], row_ptr[br + 1]
        row_start = br * R
        row_end = min((br + 1) * R, n_rows)

        for idx in range(start, end):
            bc = col_indices[idx]
            col_start = bc * C
            col_end = min((bc + 1) * C, len(x))

            block = data[idx]
            # 块矩阵-向量乘法
            y[row_start:row_end] += np.dot(
                block[:row_end - row_start, :col_end - col_start],
                x[col_start:col_end]
            )

    return y


def _spmv_gpu(A: SparseArray, x: np.ndarray,
              config: SparseArrayConfig, backend: ComputeBackend) -> np.ndarray:
    """GPU实现的SpMV"""
    # 延迟导入GPU模块
    try:
        from ..gpu.cusparse import cusparse_spmv
        return cusparse_spmv(A, x, config)
    except ImportError:
        # GPU不可用，回退到CPU
        return _spmv_cpu(A, x, config)


def _select_backend_spmv(A: SparseArray, x: np.ndarray,
                         config: SparseArrayConfig) -> ComputeBackend:
    """选择SpMV的计算后端"""
    # 检查GPU是否可用
    if not config.enable_gpu:
        return ComputeBackend.CPU

    # 检查矩阵大小是否值得GPU加速
    nnz = A.nnz
    if nnz < 10000:
        return ComputeBackend.CPU

    # 检查BCSR格式是否适合Tensor Core
    if A.format == 'bcsr':
        return ComputeBackend.GPU_TENSOR_CORE

    return ComputeBackend.GPU_CUSPARSE


def spmm(A: SparseArray,
         B: Union[SparseArray, np.ndarray],
         config: Optional[SparseArrayConfig] = None) -> Union[SparseArray, np.ndarray]:
    """
    稀疏矩阵-矩阵乘法 (SpMM)

    计算 C = A @ B

    Args:
        A: 稀疏矩阵
        B: 稀疏矩阵或密集矩阵
        config: 配置对象

    Returns:
        SparseArray或np.ndarray: 结果矩阵 C

    Example:
        >>> A = SparseArray.random((100, 100), density=0.1)
        >>> B = np.random.rand(100, 50)
        >>> C = spmm(A, B)
    """
    config = config or SparseArrayConfig()

    if A.ndim != 2:
        raise DimensionError("SpMM requires 2D matrix A", expected_dim=2, actual_dim=A.ndim)

    if isinstance(B, SparseArray):
        return _spmm_sparse_sparse(A, B, config)
    else:
        return _spmm_sparse_dense(A, B, config)


def _spmm_sparse_dense(A: SparseArray, B: np.ndarray,
                       config: SparseArrayConfig) -> np.ndarray:
    """稀疏-密集矩阵乘法"""
    if B.ndim != 2:
        raise DimensionError("SpMM requires 2D matrix B", expected_dim=2, actual_dim=B.ndim)

    if A.shape[1] != B.shape[0]:
        raise DimensionError(
            f"Matrix dimension mismatch: A.shape[1]={A.shape[1]}, B.shape[0]={B.shape[0]}"
        )

    # 对每列执行SpMV
    n_cols = B.shape[1]
    result = np.zeros((A.shape[0], n_cols), dtype=np.result_type(A.dtype, B.dtype))

    for j in range(n_cols):
        result[:, j] = spmv(A, B[:, j], config)

    return result


def _spmm_sparse_sparse(A: SparseArray, B: SparseArray,
                        config: SparseArrayConfig) -> SparseArray:
    """稀疏-稀疏矩阵乘法"""
    if B.ndim != 2:
        raise DimensionError("SpMM requires 2D matrix B", expected_dim=2, actual_dim=B.ndim)

    if A.shape[1] != B.shape[0]:
        raise DimensionError(
            f"Matrix dimension mismatch: A.shape[1]={A.shape[1]}, B.shape[0]={B.shape[0]}"
        )

    # 使用CSR格式进行稀疏-稀疏乘法
    A_csr = A._format.to_csr() if A.format != 'csr' else A._format
    B_csr = B._format.to_csr() if B.format != 'csr' else B._format

    # 稀疏-稀疏乘法算法
    result_data = []
    result_indices = []
    result_indptr = [0]

    for i in range(A.shape[0]):
        # 获取A的第i行
        a_start, a_end = A_csr.indptr[i], A_csr.indptr[i + 1]
        a_cols = A_csr.indices[a_start:a_end]
        a_vals = A_csr.data[a_start:a_end]

        # 计算结果行的非零列
        row_dict = {}
        for k_idx, k in enumerate(a_cols):
            b_start, b_end = B_csr.indptr[k], B_csr.indptr[k + 1]
            for j_idx in range(b_start, b_end):
                j = B_csr.indices[j_idx]
                if j not in row_dict:
                    row_dict[j] = 0.0
                row_dict[j] += a_vals[k_idx] * B_csr.data[j_idx]

        # 排序并添加到结果
        for j in sorted(row_dict.keys()):
            if row_dict[j] != 0:
                result_data.append(row_dict[j])
                result_indices.append(j)

        result_indptr.append(len(result_data))

    # 创建结果稀疏数组
    from ..formats.csr import CSRFormat
    result_format = CSRFormat(
        (A.shape[0], B.shape[1]),
        np.array(result_data, dtype=np.result_type(A.dtype, B.dtype)),
        np.array(result_indices, dtype=np.int32),
        np.array(result_indptr, dtype=np.int32)
    )

    result = SparseArray(
        shape=(A.shape[0], B.shape[1]),
        format='csr',
        config=config
    )
    result._format = result_format
    result._stats.nnz = result_format.nnz
    result._stats.calculate_density()

    return result


def norm(A: SparseArray,
         ord: Union[int, float, str] = 2,
         config: Optional[SparseArrayConfig] = None) -> float:
    """
    计算稀疏矩阵范数

    Args:
        A: 稀疏矩阵
        ord: 范数类型
            - 'fro' 或 'f': Frobenius范数
            - 1: 1-范数（列和最大值）
            - 2: 2-范数（最大奇异值，近似）
            - np.inf: 无穷范数（行和最大值）
        config: 配置对象

    Returns:
        float: 范数值
    """
    config = config or SparseArrayConfig()

    if A.ndim != 2:
        raise DimensionError("norm requires 2D matrix", expected_dim=2, actual_dim=A.ndim)

    if A._format is None:
        return 0.0

    if ord in ('fro', 'f'):
        return _norm_frobenius(A)
    elif ord == 1:
        return _norm_1(A)
    elif ord == 2:
        return _norm_2(A, config)
    elif ord == np.inf or ord == 'inf':
        return _norm_inf(A)
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def _norm_frobenius(A: SparseArray) -> float:
    """Frobenius范数"""
    if A._format is None:
        return 0.0

    data = A._format.data if hasattr(A._format, 'data') else np.array([])
    if len(data) == 0:
        return 0.0

    return np.sqrt(np.sum(data ** 2))


def _norm_1(A: SparseArray) -> float:
    """1-范数（列和最大值）"""
    if A._format is None:
        return 0.0

    # 转换为CSC计算列和
    csc = A._format.to_csc() if A.format != 'csc' else A._format

    col_sums = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        col_sums[j] = np.sum(np.abs(csc.data[start:end]))

    return np.max(col_sums)


def _norm_inf(A: SparseArray) -> float:
    """无穷范数（行和最大值）"""
    if A._format is None:
        return 0.0

    # 转换为CSR计算行和
    csr = A._format.to_csr() if A.format != 'csr' else A._format

    row_sums = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        row_sums[i] = np.sum(np.abs(csr.data[start:end]))

    return np.max(row_sums)


def _norm_2(A: SparseArray, config: SparseArrayConfig) -> float:
    """
    2-范数（最大奇异值）

    使用幂迭代法近似计算
    """
    if A._format is None:
        return 0.0

    # 幂迭代法
    n = A.shape[1]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(100):
        Av = spmv(A, v, config)
        AtAv = spmv(A.T, Av, config)
        v_new = AtAv / np.linalg.norm(AtAv)

        if np.abs(np.linalg.norm(v_new) - np.linalg.norm(v)) < 1e-10:
            break
        v = v_new

    return np.linalg.norm(spmv(A, v, config))


def dot(a: Union[SparseArray, np.ndarray],
        b: Union[SparseArray, np.ndarray],
        config: Optional[SparseArrayConfig] = None) -> Union[SparseArray, np.ndarray, float]:
    """
    点积运算

    Args:
        a: 第一个操作数
        b: 第二个操作数
        config: 配置对象

    Returns:
        点积结果
    """
    config = config or SparseArrayConfig()

    if isinstance(a, SparseArray):
        return a.dot(b)

    if isinstance(b, SparseArray):
        if a.ndim == 1 and b.ndim == 1:
            # 向量点积
            return np.dot(a, b.to_dense())
        else:
            return spmm(b.T, a, config)

    return np.dot(a, b)
