"""
SciPy适配器模块

提供与SciPy稀疏矩阵的互操作性。
"""

from typing import Optional, Union, Any
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig, SparseFormat


# 检查SciPy是否可用
_SCIPY_AVAILABLE = False
try:
    from scipy import sparse as sp_sparse
    _SCIPY_AVAILABLE = True
except ImportError:
    pass


def is_scipy_available() -> bool:
    """检查SciPy是否可用"""
    return _SCIPY_AVAILABLE


def to_scipy_sparse(arr: SparseArray, format: Optional[str] = None) -> Any:
    """
    将稀疏数组转换为SciPy稀疏矩阵

    Args:
        arr: 稀疏数组
        format: 目标格式 ('csr', 'csc', 'coo')

    Returns:
        SciPy稀疏矩阵
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    format = format or arr.format

    if format == 'csr':
        csr = arr._format.to_csr() if arr.format != 'csr' else arr._format
        return sp_sparse.csr_matrix(
            (csr.data, csr.indices, csr.indptr),
            shape=arr.shape
        )

    elif format == 'csc':
        csc = arr._format.to_csc() if arr.format != 'csc' else arr._format
        return sp_sparse.csc_matrix(
            (csc.data, csc.indices, csc.indptr),
            shape=arr.shape
        )

    elif format == 'coo':
        coo = arr._format.to_coo() if arr.format != 'coo' else arr._format
        return sp_sparse.coo_matrix(
            (coo.data, (coo.rows, coo.cols)),
            shape=arr.shape
        )

    else:
        # 默认转换为CSR
        return to_scipy_sparse(arr, 'csr')


def from_scipy_sparse(mat: Any,
                      config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    从SciPy稀疏矩阵创建稀疏数组

    Args:
        mat: SciPy稀疏矩阵
        config: 配置对象

    Returns:
        SparseArray: 稀疏数组
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    config = config or SparseArrayConfig()

    format_name = mat.format

    if format_name == 'csr':
        return SparseArray.from_csr(
            mat.shape,
            mat.data,
            mat.indices,
            mat.indptr,
            config
        )

    elif format_name == 'csc':
        # 转换为CSR
        csr = mat.tocsr()
        return SparseArray.from_csr(
            csr.shape,
            csr.data,
            csr.indices,
            csr.indptr,
            config
        )

    elif format_name == 'coo':
        return SparseArray.from_coo(
            mat.shape,
            mat.row,
            mat.col,
            mat.data,
            config
        )

    else:
        # 转换为CSR后创建
        csr = mat.tocsr()
        return SparseArray.from_csr(
            csr.shape,
            csr.data,
            csr.indices,
            csr.indptr,
            config
        )


class scipy_compat:
    """
    SciPy兼容性包装器

    使稀疏数组能够与SciPy稀疏函数无缝协作。
    """

    def __init__(self, arr: SparseArray):
        """
        初始化包装器

        Args:
            arr: 稀疏数组
        """
        self._arr = arr
        self._scipy_mat = None

    def _ensure_scipy(self):
        """确保SciPy矩阵已创建"""
        if self._scipy_mat is None:
            self._scipy_mat = to_scipy_sparse(self._arr)

    @property
    def shape(self):
        """形状"""
        return self._arr.shape

    @property
    def dtype(self):
        """数据类型"""
        return self._arr.dtype

    @property
    def nnz(self):
        """非零元素数量"""
        return self._arr.nnz

    def tocsr(self):
        """转换为CSR格式"""
        return to_scipy_sparse(self._arr, 'csr')

    def tocsc(self):
        """转换为CSC格式"""
        return to_scipy_sparse(self._arr, 'csc')

    def tocoo(self):
        """转换为COO格式"""
        return to_scipy_sparse(self._arr, 'coo')

    def toarray(self):
        """转换为密集数组"""
        return self._arr.to_dense()

    def dot(self, other):
        """矩阵乘法"""
        self._ensure_scipy()
        result = self._scipy_mat.dot(other)
        if sp_sparse.issparse(result):
            return from_scipy_sparse(result)
        return result

    def multiply(self, other):
        """元素级乘法"""
        self._ensure_scipy()
        result = self._scipy_mat.multiply(other)
        if sp_sparse.issparse(result):
            return from_scipy_sparse(result)
        return result

    def transpose(self):
        """转置"""
        return from_scipy_sparse(self._scipy_mat.transpose())

    @property
    def T(self):
        """转置属性"""
        return self.transpose()

    def __matmul__(self, other):
        """矩阵乘法运算符"""
        return self.dot(other)

    def __mul__(self, other):
        """元素级乘法运算符"""
        return self.multiply(other)

    def __repr__(self):
        """字符串表示"""
        return f"scipy_compat({self._arr})"


def linalg_solve(A: SparseArray, b: np.ndarray,
                 config: Optional[SparseArrayConfig] = None) -> np.ndarray:
    """
    使用SciPy求解稀疏线性系统 Ax = b

    Args:
        A: 系数矩阵
        b: 右端向量
        config: 配置对象

    Returns:
        np.ndarray: 解向量
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    from scipy.sparse.linalg import spsolve

    A_scipy = to_scipy_sparse(A)
    return spsolve(A_scipy, b)


def linalg_eigsh(A: SparseArray, k: int = 6,
                 config: Optional[SparseArrayConfig] = None) -> tuple:
    """
    使用SciPy计算稀疏矩阵的特征值和特征向量

    Args:
        A: 稀疏矩阵
        k: 特征值数量
        config: 配置对象

    Returns:
        Tuple: (特征值, 特征向量)
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    from scipy.sparse.linalg import eigsh

    A_scipy = to_scipy_sparse(A)
    return eigsh(A_scipy, k=k)


def linalg_svds(A: SparseArray, k: int = 6,
                config: Optional[SparseArrayConfig] = None) -> tuple:
    """
    使用SciPy计算稀疏矩阵的奇异值分解

    Args:
        A: 稀疏矩阵
        k: 奇异值数量
        config: 配置对象

    Returns:
        Tuple: (U, s, Vt)
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    from scipy.sparse.linalg import svds

    A_scipy = to_scipy_sparse(A)
    return svds(A_scipy, k=k)


def linalg_gmres(A: SparseArray, b: np.ndarray,
                 config: Optional[SparseArrayConfig] = None,
                 **kwargs) -> tuple:
    """
    使用GMRES求解稀疏线性系统

    Args:
        A: 系数矩阵
        b: 右端向量
        config: 配置对象
        **kwargs: GMRES参数

    Returns:
        Tuple: (解向量, 收敛信息)
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy is not installed")

    from scipy.sparse.linalg import gmres

    A_scipy = to_scipy_sparse(A)
    return gmres(A_scipy, b, **kwargs)
