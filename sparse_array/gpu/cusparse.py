"""
cuSPARSE封装模块

提供NVIDIA cuSPARSE库的Python封装，用于GPU加速的稀疏矩阵运算。

cuSPARSE是NVIDIA提供的高性能稀疏矩阵计算库，支持：
- SpMV: 稀疏矩阵-向量乘法
- SpMM: 稀疏矩阵-矩阵乘法
- 格式转换
- 稀疏三角求解
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig
from ..core.exceptions import GPUError


# 检查cuSPARSE是否可用
_CUSPARSE_AVAILABLE = False
_CUPY_AVAILABLE = False

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
    try:
        from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix  # type: ignore
        from cupyx.scipy.sparse.linalg import spmv as cp_spmv  # type: ignore
        _CUSPARSE_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


def is_cusparse_available() -> bool:
    """
    检查cuSPARSE是否可用

    Returns:
        bool: 是否可用
    """
    return _CUSPARSE_AVAILABLE


def get_gpu_info() -> Dict[str, Any]:
    """
    获取GPU信息

    Returns:
        Dict: GPU信息
    """
    if not _CUPY_AVAILABLE:
        return {'available': False, 'reason': 'CuPy not installed'}

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)

        return {
            'available': True,
            'device_id': device.id,
            'name': props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
            'total_memory': props['totalGlobalMem'],
            'compute_capability': (props['major'], props['minor']),
            'multiprocessor_count': props['multiProcessorCount']
        }
    except Exception as e:
        return {'available': False, 'reason': str(e)}


def cusparse_spmv(A: SparseArray,
                  x: np.ndarray,
                  config: Optional[SparseArrayConfig] = None) -> np.ndarray:
    """
    使用cuSPARSE执行稀疏矩阵-向量乘法

    Args:
        A: 稀疏矩阵
        x: 密集向量
        config: 配置对象

    Returns:
        np.ndarray: 结果向量
    """
    config = config or SparseArrayConfig()

    if not _CUSPARSE_AVAILABLE:
        raise GPUError(
            "cuSPARSE is not available. Install CuPy with CUDA support.",
            device_id=config.gpu_device,
            operation="spmv"
        )

    # 转换为CSR格式
    csr = A._format.to_csr() if A.format != 'csr' else A._format  # type: ignore

    # 传输数据到GPU
    try:
        # 创建CuPy CSR矩阵
        data_gpu = cp.array(csr.data)  # type: ignore
        indices_gpu = cp.array(csr.indices)  # type: ignore
        indptr_gpu = cp.array(csr.indptr)  # type: ignore

        cp_csr = cp_csr_matrix(
            (data_gpu, indices_gpu, indptr_gpu),
            shape=A.shape
        )

        # 传输向量到GPU
        x_gpu = cp.array(x)

        # 执行SpMV
        y_gpu = cp_csr.dot(x_gpu)

        # 传输结果回CPU
        return cp.asnumpy(y_gpu)

    except Exception as e:
        raise GPUError(
            f"cuSPARSE SpMV failed: {str(e)}",
            device_id=config.gpu_device,
            operation="spmv"
        )


def cusparse_spmm(A: SparseArray,
                  B: np.ndarray,
                  config: Optional[SparseArrayConfig] = None) -> np.ndarray:
    """
    使用cuSPARSE执行稀疏矩阵-密集矩阵乘法

    Args:
        A: 稀疏矩阵
        B: 密集矩阵
        config: 配置对象

    Returns:
        np.ndarray: 结果矩阵
    """
    config = config or SparseArrayConfig()

    if not _CUSPARSE_AVAILABLE:
        raise GPUError(
            "cuSPARSE is not available. Install CuPy with CUDA support.",
            device_id=config.gpu_device,
            operation="spmm"
        )

    # 转换为CSR格式
    csr = A._format.to_csr() if A.format != 'csr' else A._format  # type: ignore

    try:
        # 创建CuPy CSR矩阵
        data_gpu = cp.array(csr.data)  # type: ignore
        indices_gpu = cp.array(csr.indices)  # type: ignore
        indptr_gpu = cp.array(csr.indptr)  # type: ignore

        cp_csr = cp_csr_matrix(
            (data_gpu, indices_gpu, indptr_gpu),
            shape=A.shape
        )

        # 传输矩阵到GPU
        B_gpu = cp.array(B)

        # 执行SpMM
        C_gpu = cp_csr.dot(B_gpu)

        # 传输结果回CPU
        return cp.asnumpy(C_gpu)

    except Exception as e:
        raise GPUError(
            f"cuSPARSE SpMM failed: {str(e)}",
            device_id=config.gpu_device,
            operation="spmm"
        )


def cusparse_spmm_sparse(A: SparseArray,
                         B: SparseArray,
                         config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    使用cuSPARSE执行稀疏矩阵-稀疏矩阵乘法

    Args:
        A: 第一个稀疏矩阵
        B: 第二个稀疏矩阵
        config: 配置对象

    Returns:
        SparseArray: 结果稀疏矩阵
    """
    config = config or SparseArrayConfig()

    if not _CUSPARSE_AVAILABLE:
        raise GPUError(
            "cuSPARSE is not available. Install CuPy with CUDA support.",
            device_id=config.gpu_device,
            operation="spmm_sparse"
        )

    # 转换为CSR格式
    A_csr = A._format.to_csr() if A.format != 'csr' else A._format  # type: ignore
    B_csr = B._format.to_csr() if B.format != 'csr' else B._format  # type: ignore

    try:
        # 创建CuPy CSR矩阵
        A_data_gpu = cp.array(A_csr.data)  # type: ignore
        A_indices_gpu = cp.array(A_csr.indices)  # type: ignore
        A_indptr_gpu = cp.array(A_csr.indptr)  # type: ignore

        A_cp = cp_csr_matrix(
            (A_data_gpu, A_indices_gpu, A_indptr_gpu),
            shape=A.shape
        )  # type: ignore

        B_data_gpu = cp.array(B_csr.data)  # type: ignore
        B_indices_gpu = cp.array(B_csr.indices)  # type: ignore
        B_indptr_gpu = cp.array(B_csr.indptr)  # type: ignore

        B_cp = cp_csr_matrix(
            (B_data_gpu, B_indices_gpu, B_indptr_gpu),
            shape=B.shape
        )  # type: ignore

        # 执行稀疏-稀疏乘法
        C_cp = A_cp.dot(B_cp)

        # 转换回CPU
        C_cp = C_cp.tocsr()

        result = SparseArray.from_csr(
            C_cp.shape,
            cp.asnumpy(C_cp.data),
            cp.asnumpy(C_cp.indices),
            cp.asnumpy(C_cp.indptr),
            config
        )

        return result

    except Exception as e:
        raise GPUError(
            f"cuSPARSE sparse SpMM failed: {str(e)}",
            device_id=config.gpu_device,
            operation="spmm_sparse"
        )


class CuSparseContext:
    """
    cuSPARSE上下文管理器

    管理GPU资源和流。
    """

    def __init__(self, device_id: int = 0):
        """
        初始化上下文

        Args:
            device_id: GPU设备ID
        """
        self.device_id = device_id
        self.stream = None

    def __enter__(self):
        """进入上下文"""
        if not _CUPY_AVAILABLE:
            raise GPUError("CuPy not available")

        cp.cuda.Device(self.device_id).use()
        self.stream = cp.cuda.Stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.stream is not None:
            self.stream.synchronize()
        return False
