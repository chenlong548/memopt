"""
算术运算模块

提供稀疏数组的元素级运算和聚合运算。
"""

from typing import Union, Optional, Tuple
import numpy as np

from ..core.sparse_array import SparseArray, FormatBase
from ..core.config import SparseArrayConfig
from ..core.exceptions import DimensionError


def add(a: Union[SparseArray, np.ndarray, float, int],
        b: Union[SparseArray, np.ndarray, float, int],
        config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    稀疏数组加法

    Args:
        a: 第一个操作数
        b: 第二个操作数
        config: 配置对象

    Returns:
        SparseArray: 加法结果
    """
    config = config or SparseArrayConfig()

    # 处理标量
    if np.isscalar(a):
        return _add_scalar(b, a, config)
    if np.isscalar(b):
        return _add_scalar(a, b, config)

    # 处理数组
    if isinstance(a, SparseArray) and isinstance(b, SparseArray):
        return _add_sparse_sparse(a, b, config)
    elif isinstance(a, SparseArray):
        return _add_sparse_dense(a, b, config)
    elif isinstance(b, SparseArray):
        return _add_sparse_dense(b, a, config)
    else:
        # 两个都是密集数组
        result = SparseArray.from_dense(a + b, config=config)
        return result


def _add_scalar(arr: Union[SparseArray, np.ndarray],
                scalar: Union[float, int],
                config: SparseArrayConfig) -> SparseArray:
    """稀疏数组加标量"""
    if isinstance(arr, SparseArray):
        # 加标量会使所有零元素变为非零
        # 结果变为密集数组
        dense = arr.to_dense() + scalar
        return SparseArray.from_dense(dense, config=config)
    else:
        return SparseArray.from_dense(arr + scalar, config=config)


def _add_sparse_sparse(a: SparseArray, b: SparseArray,
                       config: SparseArrayConfig) -> SparseArray:
    """稀疏-稀疏加法"""
    if a.shape != b.shape:
        raise DimensionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # 转换为COO格式
    a_coo = a._format.to_coo() if a.format != 'coo' else a._format
    b_coo = b._format.to_coo() if b.format != 'coo' else b._format

    # 合并坐标
    rows = np.concatenate([a_coo.rows, b_coo.rows])
    cols = np.concatenate([a_coo.cols, b_coo.cols])
    data = np.concatenate([a_coo.data, b_coo.data])

    # 创建结果
    result = SparseArray.from_coo(a.shape, rows, cols, data, config)
    result._format.sum_duplicates()  # 合并重复坐标

    return result


def _add_sparse_dense(a: SparseArray, b: np.ndarray,
                      config: SparseArrayConfig) -> SparseArray:
    """稀疏-密集加法"""
    if a.shape != b.shape:
        raise DimensionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    result = a.to_dense() + b
    return SparseArray.from_dense(result, config=config)


def subtract(a: Union[SparseArray, np.ndarray, float, int],
             b: Union[SparseArray, np.ndarray, float, int],
             config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    稀疏数组减法

    Args:
        a: 第一个操作数
        b: 第二个操作数
        config: 配置对象

    Returns:
        SparseArray: 减法结果
    """
    config = config or SparseArrayConfig()

    if np.isscalar(b):
        return add(a, -b, config)

    if np.isscalar(a):
        if isinstance(b, SparseArray):
            result = a - b.to_dense()
        else:
            result = a - b
        return SparseArray.from_dense(result, config=config)

    # a - b = a + (-b)
    neg_b = negate(b, config) if isinstance(b, SparseArray) else -b
    return add(a, neg_b, config)


def multiply(a: Union[SparseArray, np.ndarray, float, int],
             b: Union[SparseArray, np.ndarray, float, int],
             config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    元素级乘法

    Args:
        a: 第一个操作数
        b: 第二个操作数
        config: 配置对象

    Returns:
        SparseArray: 乘法结果
    """
    config = config or SparseArrayConfig()

    # 处理标量
    if np.isscalar(a):
        return _multiply_scalar(b, a, config)
    if np.isscalar(b):
        return _multiply_scalar(a, b, config)

    # 处理数组
    if isinstance(a, SparseArray) and isinstance(b, SparseArray):
        return _multiply_sparse_sparse(a, b, config)
    elif isinstance(a, SparseArray):
        return _multiply_sparse_dense(a, b, config)
    elif isinstance(b, SparseArray):
        return _multiply_sparse_dense(b, a, config)
    else:
        result = SparseArray.from_dense(a * b, config=config)
        return result


def _multiply_scalar(arr: Union[SparseArray, np.ndarray],
                     scalar: Union[float, int],
                     config: SparseArrayConfig) -> SparseArray:
    """稀疏数组乘标量"""
    if isinstance(arr, SparseArray):
        # 标量乘法保持稀疏结构
        if arr._format is None:
            return arr.copy()

        # 复制并缩放数据
        new_format = arr._format.copy()
        if hasattr(new_format, '_data'):
            new_format._data = new_format._data * scalar

        result = SparseArray(shape=arr.shape, format=arr.format, config=config)
        result._format = new_format
        result._stats.nnz = new_format.nnz
        result._stats.calculate_density()

        return result
    else:
        return SparseArray.from_dense(arr * scalar, config=config)


def _multiply_sparse_sparse(a: SparseArray, b: SparseArray,
                            config: SparseArrayConfig) -> SparseArray:
    """稀疏-稀疏元素级乘法"""
    if a.shape != b.shape:
        raise DimensionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # 转换为COO格式
    a_coo = a._format.to_coo() if a.format != 'coo' else a._format
    b_coo = b._format.to_coo() if b.format != 'coo' else b._format

    # 构建字典查找
    b_dict = {}
    for i in range(b_coo.nnz):
        key = (b_coo.rows[i], b_coo.cols[i])
        b_dict[key] = b_coo.data[i]

    # 计算乘积
    result_rows = []
    result_cols = []
    result_data = []

    for i in range(a_coo.nnz):
        key = (a_coo.rows[i], a_coo.cols[i])
        if key in b_dict:
            val = a_coo.data[i] * b_dict[key]
            if val != 0:
                result_rows.append(key[0])
                result_cols.append(key[1])
                result_data.append(val)

    return SparseArray.from_coo(
        a.shape,
        np.array(result_rows, dtype=np.int32),
        np.array(result_cols, dtype=np.int32),
        np.array(result_data, dtype=np.result_type(a.dtype, b.dtype)),
        config
    )


def _multiply_sparse_dense(a: SparseArray, b: np.ndarray,
                           config: SparseArrayConfig) -> SparseArray:
    """稀疏-密集元素级乘法"""
    if a.shape != b.shape:
        raise DimensionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # 转换为COO格式
    a_coo = a._format.to_coo() if a.format != 'coo' else a._format

    # 计算乘积
    result_data = a_coo.data * b[a_coo.rows, a_coo.cols]

    # 过滤零元素
    mask = result_data != 0

    return SparseArray.from_coo(
        a.shape,
        a_coo.rows[mask],
        a_coo.cols[mask],
        result_data[mask],
        config
    )


def divide(a: Union[SparseArray, np.ndarray, float, int],
           b: Union[SparseArray, np.ndarray, float, int],
           config: Optional[SparseArrayConfig] = None,
           fill_value: Optional[float] = None) -> SparseArray:
    """
    元素级除法

    Args:
        a: 第一个操作数
        b: 第二个操作数
        config: 配置对象
        fill_value: 除零时的填充值，如果为None则抛出异常

    Returns:
        SparseArray: 除法结果

    Raises:
        ZeroDivisionError: 当除数为零且fill_value为None时
    """
    config = config or SparseArrayConfig()

    # 处理标量除数
    if np.isscalar(b):
        if b == 0:
            if fill_value is not None:
                # 返回与a相同形状的全fill_value数组
                if isinstance(a, SparseArray):
                    result = np.full(a.shape, fill_value, dtype=a.dtype)
                else:
                    result = np.full(np.shape(a), fill_value, dtype=np.result_type(a))
                return SparseArray.from_dense(result, config=config)
            raise ZeroDivisionError("division by zero")
        return multiply(a, 1.0 / b, config)

    # 处理标量被除数
    if np.isscalar(a):
        if isinstance(b, SparseArray):
            dense_b = b.to_dense()
            if fill_value is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(dense_b != 0, a / dense_b, fill_value)
            else:
                if np.any(dense_b == 0):
                    raise ZeroDivisionError("division by zero: divisor contains zero elements")
                result = a / dense_b
        else:
            if fill_value is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(b != 0, a / b, fill_value)
            else:
                if np.any(b == 0):
                    raise ZeroDivisionError("division by zero: divisor contains zero elements")
                result = a / b
        return SparseArray.from_dense(result, config=config)

    # 数组除法
    if isinstance(a, SparseArray):
        dense_a = a.to_dense()
    else:
        dense_a = a

    if isinstance(b, SparseArray):
        dense_b = b.to_dense()
    else:
        dense_b = b

    # 检查除零
    if fill_value is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(dense_b != 0, dense_a / dense_b, fill_value)
    else:
        if np.any(dense_b == 0):
            raise ZeroDivisionError("division by zero: divisor contains zero elements")
        result = dense_a / dense_b

    return SparseArray.from_dense(result, config=config)


def negate(arr: Union[SparseArray, np.ndarray],
           config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    取负

    Args:
        arr: 输入数组
        config: 配置对象

    Returns:
        SparseArray: 取负结果
    """
    config = config or SparseArrayConfig()

    if isinstance(arr, SparseArray):
        return multiply(arr, -1, config)
    else:
        return SparseArray.from_dense(-arr, config=config)


def sum(arr: SparseArray,
        axis: Optional[int] = None,
        config: Optional[SparseArrayConfig] = None) -> Union[float, np.ndarray, SparseArray]:
    """
    求和

    Args:
        arr: 稀疏数组
        axis: 求和轴，None表示全部求和
        config: 配置对象

    Returns:
        求和结果
    """
    config = config or SparseArrayConfig()

    if arr._format is None:
        if axis is None:
            return 0.0
        return np.zeros(arr.shape[axis] if axis == 0 else arr.shape[0])

    if axis is None:
        # 全部求和
        if hasattr(arr._format, 'data'):
            return np.sum(arr._format.data)
        return np.sum(arr.to_dense())

    if arr.ndim != 2:
        raise DimensionError("sum with axis only supports 2D arrays", expected_dim=2, actual_dim=arr.ndim)

    if axis == 0:
        # 按列求和
        return _sum_axis_0(arr)
    elif axis == 1:
        # 按行求和
        return _sum_axis_1(arr)
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _sum_axis_0(arr: SparseArray) -> np.ndarray:
    """按列求和"""
    # 转换为CSC格式
    csc = arr._format.to_csc() if arr.format != 'csc' else arr._format

    result = np.zeros(arr.shape[1], dtype=arr.dtype)
    for j in range(arr.shape[1]):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        result[j] = np.sum(csc.data[start:end])

    return result


def _sum_axis_1(arr: SparseArray) -> np.ndarray:
    """按行求和"""
    # 转换为CSR格式
    csr = arr._format.to_csr() if arr.format != 'csr' else arr._format

    result = np.zeros(arr.shape[0], dtype=arr.dtype)
    for i in range(arr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        result[i] = np.sum(csr.data[start:end])

    return result


def mean(arr: SparseArray,
         axis: Optional[int] = None,
         config: Optional[SparseArrayConfig] = None) -> Union[float, np.ndarray]:
    """
    求均值

    Args:
        arr: 稀疏数组
        axis: 求均值轴
        config: 配置对象

    Returns:
        均值结果
    """
    config = config or SparseArrayConfig()

    if axis is None:
        # 全部均值
        total = sum(arr, None, config)
        return total / arr._shape[0] / arr._shape[1] if arr.ndim == 2 else total / arr._shape[0]

    if arr.ndim != 2:
        raise DimensionError("mean with axis only supports 2D arrays", expected_dim=2, actual_dim=arr.ndim)

    if axis == 0:
        # 按列均值
        col_sums = _sum_axis_0(arr)
        return col_sums / arr.shape[0]
    elif axis == 1:
        # 按行均值
        row_sums = _sum_axis_1(arr)
        return row_sums / arr.shape[1]
    else:
        raise ValueError(f"Invalid axis: {axis}")


def max(arr: SparseArray,
        axis: Optional[int] = None,
        config: Optional[SparseArrayConfig] = None) -> Union[float, np.ndarray]:
    """
    求最大值

    Args:
        arr: 稀疏数组
        axis: 求最大值轴
        config: 配置对象

    Returns:
        最大值结果
    """
    config = config or SparseArrayConfig()

    if arr._format is None:
        return 0.0

    if axis is None:
        # 全部最大值
        if hasattr(arr._format, 'data') and len(arr._format.data) > 0:
            return np.max(arr._format.data)
        return 0.0

    if arr.ndim != 2:
        raise DimensionError("max with axis only supports 2D arrays", expected_dim=2, actual_dim=arr.ndim)

    # 转换为密集数组计算
    dense = arr.to_dense()
    return np.max(dense, axis=axis)


def min(arr: SparseArray,
        axis: Optional[int] = None,
        config: Optional[SparseArrayConfig] = None) -> Union[float, np.ndarray]:
    """
    求最小值

    Args:
        arr: 稀疏数组
        axis: 求最小值轴
        config: 配置对象

    Returns:
        最小值结果
    """
    config = config or SparseArrayConfig()

    if arr._format is None:
        return 0.0

    if axis is None:
        # 全部最小值
        if hasattr(arr._format, 'data') and len(arr._format.data) > 0:
            return np.min(arr._format.data)
        return 0.0

    if arr.ndim != 2:
        raise DimensionError("min with axis only supports 2D arrays", expected_dim=2, actual_dim=arr.ndim)

    # 转换为密集数组计算
    dense = arr.to_dense()
    return np.min(dense, axis=axis)
