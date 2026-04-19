"""
NumPy适配器模块

提供与NumPy的互操作性。
"""

from typing import Optional, Union, Tuple, Any
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig


def to_numpy(arr: SparseArray) -> np.ndarray:
    """
    将稀疏数组转换为NumPy数组

    Args:
        arr: 稀疏数组

    Returns:
        np.ndarray: NumPy数组
    """
    return arr.to_dense()


def from_numpy(arr: np.ndarray,
               threshold: float = 0.0,
               config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    从NumPy数组创建稀疏数组

    Args:
        arr: NumPy数组
        threshold: 稀疏化阈值
        config: 配置对象

    Returns:
        SparseArray: 稀疏数组
    """
    return SparseArray.from_dense(arr, threshold, config)


class numpy_compat:
    """
    NumPy兼容性包装器

    使稀疏数组能够与NumPy函数无缝协作。
    """

    def __init__(self, arr: SparseArray):
        """
        初始化包装器

        Args:
            arr: 稀疏数组
        """
        self._arr = arr

    def __array__(self) -> np.ndarray:
        """转换为NumPy数组"""
        return self._arr.to_dense()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """支持NumPy通用函数"""
        # 将稀疏数组转换为密集数组
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, SparseArray):
                new_inputs.append(inp.to_dense())
            elif isinstance(inp, numpy_compat):
                new_inputs.append(inp._arr.to_dense())
            else:
                new_inputs.append(inp)

        # 调用原始ufunc
        result = getattr(ufunc, method)(*new_inputs, **kwargs)

        # 如果结果是数组，尝试转换回稀疏
        if isinstance(result, np.ndarray):
            return SparseArray.from_dense(result)

        return result

    def __array_function__(self, func, types, args, kwargs):
        """支持NumPy数组函数"""
        # 转换参数
        new_args = []
        for arg in args:
            if isinstance(arg, SparseArray):
                new_args.append(arg.to_dense())
            elif isinstance(arg, numpy_compat):
                new_args.append(arg._arr.to_dense())
            else:
                new_args.append(arg)

        # 调用原始函数
        result = func(*new_args, **kwargs)

        # 如果结果是数组，尝试转换回稀疏
        if isinstance(result, np.ndarray):
            return SparseArray.from_dense(result)

        return result

    @property
    def shape(self) -> Tuple[int, ...]:
        """形状"""
        return self._arr.shape

    @property
    def dtype(self) -> np.dtype:
        """数据类型"""
        return self._arr.dtype

    @property
    def ndim(self) -> int:
        """维度数"""
        return self._arr.ndim

    def __len__(self) -> int:
        """长度"""
        return len(self._arr)

    def __getitem__(self, key) -> Any:
        """索引访问"""
        return self._arr[key]

    def __setitem__(self, key, value):
        """索引设置"""
        self._arr[key] = value

    def __repr__(self) -> str:
        """字符串表示"""
        return f"numpy_compat({self._arr})"


def asarray(arr: Union[SparseArray, np.ndarray, Any],
            config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    将输入转换为稀疏数组

    Args:
        arr: 输入数组
        config: 配置对象

    Returns:
        SparseArray: 稀疏数组
    """
    if isinstance(arr, SparseArray):
        return arr

    if isinstance(arr, np.ndarray):
        return from_numpy(arr, config=config)

    # 尝试转换为NumPy数组
    try:
        np_arr = np.asarray(arr)
        return from_numpy(np_arr, config=config)
    except Exception:
        raise TypeError(f"Cannot convert {type(arr)} to SparseArray")


def concatenate(arrays: list, axis: int = 0,
                config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    连接稀疏数组

    Args:
        arrays: 稀疏数组列表
        axis: 连接轴
        config: 配置对象

    Returns:
        SparseArray: 连接后的稀疏数组
    """
    # 转换为密集数组连接
    dense_arrays = [arr.to_dense() if isinstance(arr, SparseArray) else arr
                   for arr in arrays]

    result = np.concatenate(dense_arrays, axis=axis)

    return SparseArray.from_dense(result, config=config)


def stack(arrays: list, axis: int = 0,
          config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """
    堆叠稀疏数组

    Args:
        arrays: 稀疏数组列表
        axis: 堆叠轴
        config: 配置对象

    Returns:
        SparseArray: 堆叠后的稀疏数组
    """
    dense_arrays = [arr.to_dense() if isinstance(arr, SparseArray) else arr
                   for arr in arrays]

    result = np.stack(dense_arrays, axis=axis)

    return SparseArray.from_dense(result, config=config)


def vstack(arrays: list,
           config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """垂直堆叠"""
    return concatenate(arrays, axis=0, config=config)


def hstack(arrays: list,
           config: Optional[SparseArrayConfig] = None) -> SparseArray:
    """水平堆叠"""
    return concatenate(arrays, axis=1, config=config)
