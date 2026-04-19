"""
sparse_array - 高性能稀疏数组模块

提供多种稀疏存储格式、GPU加速、压缩技术和自动格式选择功能。

特性:
- 多格式支持: CSR, CSC, COO, BCSR, Bitmap
- 自动格式选择: 基于特征和操作类型
- GPU加速: cuSPARSE和Tensor Core优化
- 压缩支持: 块低秩压缩和HSS矩阵
- NumPy/SciPy兼容: 无缝集成现有生态

Example:
    >>> from sparse_array import SparseArray
    >>>
    >>> # 从密集数组创建
    >>> dense = np.random.rand(1000, 1000)
    >>> dense[dense < 0.9] = 0
    >>> sparse = SparseArray.from_dense(dense)
    >>>
    >>> # 稀疏矩阵-向量乘法
    >>> x = np.random.rand(1000)
    >>> y = sparse.dot(x)
    >>>
    >>> # 格式转换
    >>> csr = sparse.to_csr()
    >>> csc = sparse.to_csc()
"""

__version__ = '1.0.0'
__author__ = 'Algorithm & Architecture Optimization Expert'

# 核心类
from .core import (
    SparseArray,
    SparseArrayConfig,
    SparseFormat,
    SparseArrayError,
    FormatConversionError,
    UnsupportedOperationError,
    DimensionError,
    IndexOutOfBoundsError,
    GPUError,
    CompressionError
)

# 存储格式
from .formats import (
    CSRFormat,
    CSCFormat,
    COOFormat,
    BCSRFormat,
    BitmapFormat
)

# 运算
from .ops import (
    spmv,
    spmm,
    norm,
    dot,
    add,
    subtract,
    multiply,
    divide,
    sum,
    mean,
    max,
    min,
    save_sparse,
    load_sparse
)

# GPU加速
from .gpu import (
    is_cusparse_available,
    is_tensor_core_available,
    get_gpu_info
)

# 压缩
from .compression import (
    compress_low_rank,
    decompress_low_rank,
    hss_compress,
    hss_decompress
)

# 格式选择
from .selector import (
    extract_features,
    select_format,
    recommend_format
)

# 集成
from .integration import (
    to_numpy,
    from_numpy,
    to_scipy_sparse,
    from_scipy_sparse
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',

    # 核心类
    'SparseArray',
    'SparseArrayConfig',
    'SparseFormat',

    # 异常
    'SparseArrayError',
    'FormatConversionError',
    'UnsupportedOperationError',
    'DimensionError',
    'IndexOutOfBoundsError',
    'GPUError',
    'CompressionError',

    # 存储格式
    'CSRFormat',
    'CSCFormat',
    'COOFormat',
    'BCSRFormat',
    'BitmapFormat',

    # 线性代数
    'spmv',
    'spmm',
    'norm',
    'dot',

    # 算术运算
    'add',
    'subtract',
    'multiply',
    'divide',
    'sum',
    'mean',
    'max',
    'min',

    # 序列化
    'save_sparse',
    'load_sparse',

    # GPU
    'is_cusparse_available',
    'is_tensor_core_available',
    'get_gpu_info',

    # 压缩
    'compress_low_rank',
    'decompress_low_rank',
    'hss_compress',
    'hss_decompress',

    # 格式选择
    'extract_features',
    'select_format',
    'recommend_format',

    # 集成
    'to_numpy',
    'from_numpy',
    'to_scipy_sparse',
    'from_scipy_sparse'
]
