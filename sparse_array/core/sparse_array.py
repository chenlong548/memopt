"""
sparse_array 主类模块

提供稀疏数组的核心实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Any, Dict
import numpy as np

from .config import (
    SparseArrayConfig,
    SparseArrayStats,
    SparseFormat,
    ComputeBackend,
    CompressionType
)
from .exceptions import (
    SparseArrayError,
    FormatConversionError,
    UnsupportedOperationError,
    DimensionError,
    IndexOutOfBoundsError
)


class FormatBase(ABC):
    """
    存储格式基类

    定义所有稀疏存储格式的标准接口。
    """

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64):
        """
        初始化格式

        Args:
            shape: 数组形状
            dtype: 数据类型
        """
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._nnz = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取形状"""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """获取数据类型"""
        return self._dtype

    @property
    def ndim(self) -> int:
        """获取维度数"""
        return len(self._shape)

    @property
    def nnz(self) -> int:
        """获取非零元素数量"""
        return self._nnz

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        """
        转换为密集数组

        Returns:
            np.ndarray: 密集数组
        """
        pass

    @abstractmethod
    def to_coo(self) -> 'FormatBase':
        """
        转换为COO格式

        Returns:
            FormatBase: COO格式
        """
        pass

    @abstractmethod
    def to_csr(self) -> 'FormatBase':
        """
        转换为CSR格式

        Returns:
            FormatBase: CSR格式
        """
        pass

    @abstractmethod
    def to_csc(self) -> 'FormatBase':
        """
        转换为CSC格式

        Returns:
            FormatBase: CSC格式
        """
        pass

    @abstractmethod
    def get_item(self, key) -> Any:
        """
        获取元素或切片

        Args:
            key: 索引键

        Returns:
            元素或子数组
        """
        pass

    @abstractmethod
    def set_item(self, key, value):
        """
        设置元素或切片

        Args:
            key: 索引键
            value: 值
        """
        pass

    @abstractmethod
    def copy(self) -> 'FormatBase':
        """
        深拷贝

        Returns:
            FormatBase: 拷贝
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        获取内存使用量

        Returns:
            int: 字节数
        """
        pass

    def get_format_name(self) -> str:
        """获取格式名称"""
        return self.__class__.__name__.replace('Format', '').lower()


class SparseArray:
    """
    稀疏数组主类

    提供统一的稀疏数组接口，支持多种存储格式和计算后端。

    特性:
    - 多格式支持: CSR, CSC, COO, BCSR, Bitmap
    - 自动格式选择: 基于特征和操作类型
    - GPU加速: cuSPARSE和Tensor Core优化
    - 压缩支持: 块低秩压缩和HSS矩阵
    - NumPy/SciPy兼容: 无缝集成现有生态

    Example:
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
        >>> csr_array = sparse.to_format('csr')
        >>> csc_array = sparse.to_format('csc')
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 data: Optional[np.ndarray] = None,
                 indices: Optional[np.ndarray] = None,
                 indptr: Optional[np.ndarray] = None,
                 format: Union[str, SparseFormat] = SparseFormat.AUTO,
                 dtype: np.dtype = np.float64,
                 config: Optional[SparseArrayConfig] = None):
        """
        初始化稀疏数组

        Args:
            shape: 数组形状
            data: 非零元素值数组
            indices: 列/行索引数组
            indptr: 行/列指针数组（CSR/CSC格式）
            format: 存储格式
            dtype: 数据类型
            config: 配置对象

        Raises:
            ValueError: 如果shape参数无效
        """
        # 验证shape参数
        self._shape = self._validate_shape(shape)
        self._dtype = np.dtype(dtype)
        self._config = config or SparseArrayConfig()

        # 解析格式
        if isinstance(format, str):
            format = SparseFormat(format.lower())
        self._format_type = format

        # 内部存储
        self._format: Optional[FormatBase] = None
        self._stats = SparseArrayStats(shape=self._shape)

        # 延迟初始化
        if data is not None:
            self._initialize_from_arrays(data, indices, indptr)

    @classmethod
    def from_dense(cls,
                   arr: np.ndarray,
                   threshold: float = 0.0,
                   config: Optional[SparseArrayConfig] = None) -> 'SparseArray':
        """
        从密集数组创建稀疏数组

        Args:
            arr: 密集数组
            threshold: 稀疏化阈值，绝对值小于此值的元素设为零
            config: 配置对象

        Returns:
            SparseArray: 稀疏数组
        """
        config = config or SparseArrayConfig()

        # 应用阈值
        if threshold > 0:
            arr = arr.copy()
            arr[np.abs(arr) < threshold] = 0

        # 计算稀疏度
        nnz = np.count_nonzero(arr)
        total = arr.size
        sparsity = 1.0 - nnz / total

        # 决定是否使用稀疏格式
        if sparsity < config.sparsity_threshold:
            # 密度太高，不适合稀疏存储
            result = cls(shape=arr.shape, config=config)
            result._format = _DenseWrapper(arr)
            result._stats.nnz = nnz
            result._stats.calculate_density()
            return result

        # 自动选择格式
        if config.format == SparseFormat.AUTO:
            format_type = cls._auto_select_format(arr, config)
        else:
            format_type = config.format

        # 创建对应格式
        result = cls(shape=arr.shape, format=format_type, config=config)
        result._format = cls._create_format_from_dense(arr, format_type)
        result._stats.nnz = result._format.nnz
        result._stats.calculate_density()

        return result

    @classmethod
    def from_coo(cls,
                 shape: Tuple[int, ...],
                 rows: np.ndarray,
                 cols: np.ndarray,
                 data: np.ndarray,
                 config: Optional[SparseArrayConfig] = None) -> 'SparseArray':
        """
        从COO坐标创建稀疏数组

        Args:
            shape: 数组形状
            rows: 行索引数组
            cols: 列索引数组
            data: 数据数组
            config: 配置对象

        Returns:
            SparseArray: 稀疏数组
        """
        config = config or SparseArrayConfig()
        result = cls(shape=shape, format=SparseFormat.COO, config=config)

        # 延迟导入避免循环依赖
        from ..formats.coo import COOFormat
        result._format = COOFormat(shape, rows, cols, data)
        result._stats.nnz = len(data)
        result._stats.calculate_density()

        return result

    @classmethod
    def from_csr(cls,
                 shape: Tuple[int, ...],
                 data: np.ndarray,
                 indices: np.ndarray,
                 indptr: np.ndarray,
                 config: Optional[SparseArrayConfig] = None) -> 'SparseArray':
        """
        从CSR格式创建稀疏数组

        Args:
            shape: 数组形状
            data: 非零元素值数组
            indices: 列索引数组
            indptr: 行指针数组
            config: 配置对象

        Returns:
            SparseArray: 稀疏数组
        """
        config = config or SparseArrayConfig()
        result = cls(shape=shape, format=SparseFormat.CSR, config=config)

        from ..formats.csr import CSRFormat
        result._format = CSRFormat(shape, data, indices, indptr)
        result._stats.nnz = len(data)
        result._stats.calculate_density()

        return result

    @classmethod
    def zeros(cls,
              shape: Tuple[int, ...],
              dtype: np.dtype = np.float64,
              config: Optional[SparseArrayConfig] = None) -> 'SparseArray':
        """
        创建全零稀疏数组

        Args:
            shape: 数组形状
            dtype: 数据类型
            config: 配置对象

        Returns:
            SparseArray: 全零稀疏数组
        """
        config = config or SparseArrayConfig()
        result = cls(shape=shape, format=SparseFormat.CSR, dtype=dtype, config=config)

        from ..formats.csr import CSRFormat
        result._format = CSRFormat(shape, np.array([], dtype=dtype),
                                   np.array([], dtype=np.int32),
                                   np.zeros(shape[0] + 1, dtype=np.int32))
        result._stats.nnz = 0
        result._stats.calculate_density()

        return result

    @classmethod
    def identity(cls,
                 n: int,
                 dtype: np.dtype = np.float64,
                 config: Optional[SparseArrayConfig] = None) -> 'SparseArray':
        """
        创建单位稀疏矩阵

        Args:
            n: 矩阵维度
            dtype: 数据类型
            config: 配置对象

        Returns:
            SparseArray: 单位稀疏矩阵
        """
        config = config or SparseArrayConfig()
        result = cls(shape=(n, n), format=SparseFormat.CSR, dtype=dtype, config=config)

        from ..formats.csr import CSRFormat
        data = np.ones(n, dtype=dtype)
        indices = np.arange(n, dtype=np.int32)
        indptr = np.arange(n + 1, dtype=np.int32)

        result._format = CSRFormat((n, n), data, indices, indptr)
        result._stats.nnz = n
        result._stats.calculate_density()

        return result

    @classmethod
    def random(cls,
               shape: Tuple[int, ...],
               density: float = 0.1,
               dtype: np.dtype = np.float64,
               config: Optional[SparseArrayConfig] = None,
               random_state: Optional[int] = None) -> 'SparseArray':
        """
        创建随机稀疏矩阵

        Args:
            shape: 数组形状
            density: 非零元素密度
            dtype: 数据类型
            config: 配置对象
            random_state: 随机种子（用于创建局部随机生成器，线程安全）

        Returns:
            SparseArray: 随机稀疏矩阵
        """
        config = config or SparseArrayConfig()

        # 使用局部随机生成器，确保线程安全
        rng = np.random.default_rng(random_state)

        # 计算非零元素数量
        total = 1
        for dim in shape:
            total *= dim
        nnz = int(total * density)

        # 生成随机坐标
        if len(shape) == 2:
            rows = rng.integers(0, shape[0], nnz)
            cols = rng.integers(0, shape[1], nnz)
            data = rng.random(nnz).astype(dtype)

            return cls.from_coo(shape, rows, cols, data, config)
        else:
            raise DimensionError("Random sparse array only supports 2D matrices",
                                expected_dim=2, actual_dim=len(shape))

    # ==================== 属性 ====================

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取形状"""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """获取数据类型"""
        return self._dtype

    @property
    def ndim(self) -> int:
        """获取维度数"""
        return len(self._shape)

    @property
    def nnz(self) -> int:
        """获取非零元素数量"""
        return self._format.nnz if self._format else 0

    @property
    def format(self) -> str:
        """获取当前格式名称"""
        return self._format.get_format_name() if self._format else 'none'

    @property
    def T(self) -> 'SparseArray':
        """转置"""
        return self.transpose()

    # ==================== 转换方法 ====================

    def to_dense(self) -> np.ndarray:
        """
        转换为密集数组

        Returns:
            np.ndarray: 密集数组
        """
        if self._format is None:
            return np.zeros(self._shape, dtype=self._dtype)
        return self._format.to_dense()

    def to_format(self, format: Union[str, SparseFormat]) -> 'SparseArray':
        """
        转换为指定格式

        Args:
            format: 目标格式

        Returns:
            SparseArray: 转换后的稀疏数组
        """
        if isinstance(format, str):
            format = SparseFormat(format.lower())

        if self._format is None:
            raise SparseArrayError("Cannot convert empty sparse array")

        # 执行转换
        format_map = {
            SparseFormat.COO: self._format.to_coo,
            SparseFormat.CSR: self._format.to_csr,
            SparseFormat.CSC: self._format.to_csc,
        }

        if format not in format_map:
            raise FormatConversionError(
                f"Unsupported format conversion: {self.format} -> {format.value}",
                source_format=self.format,
                target_format=format.value
            )

        new_format = format_map[format]()

        result = SparseArray(shape=self._shape, format=format, config=self._config)
        result._format = new_format
        result._stats.nnz = new_format.nnz
        result._stats.calculate_density()

        return result

    def to_csr(self) -> 'SparseArray':
        """转换为CSR格式"""
        return self.to_format(SparseFormat.CSR)

    def to_csc(self) -> 'SparseArray':
        """转换为CSC格式"""
        return self.to_format(SparseFormat.CSC)

    def to_coo(self) -> 'SparseArray':
        """转换为COO格式"""
        return self.to_format(SparseFormat.COO)

    # ==================== 算术运算 ====================

    def dot(self, other: Union['SparseArray', np.ndarray]) -> Union['SparseArray', np.ndarray]:
        """
        矩阵乘法

        Args:
            other: 另一个矩阵或向量

        Returns:
            乘积结果
        """
        if self._format is None:
            if isinstance(other, SparseArray):
                return SparseArray.zeros(self._shape[:-1] + other._shape[1:])
            return np.zeros(self._shape[:-1])

        # 延迟导入运算模块
        from ..ops.linalg import spmv, spmm

        if isinstance(other, SparseArray):
            # SpMM: 稀疏矩阵-矩阵乘法
            return spmm(self, other, self._config)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                # SpMV: 稀疏矩阵-向量乘法
                return spmv(self, other, self._config)
            else:
                # SpMM: 稀疏矩阵-密集矩阵乘法
                return spmm(self, other, self._config)
        else:
            raise TypeError(f"Unsupported type for dot product: {type(other)}")

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'SparseArray':
        """
        转置

        Args:
            axes: 轴顺序（仅支持二维）

        Returns:
            转置后的稀疏数组
        """
        if self._format is None:
            result = SparseArray(shape=self._shape[::-1], config=self._config)
            return result

        if self.ndim != 2:
            raise UnsupportedOperationError(
                "Transpose only supports 2D arrays",
                operation="transpose",
                format_type=self.format
            )

        # 转置：交换行和列
        # CSR转置后变为CSC，CSC转置后变为CSR
        if self._format_type == SparseFormat.CSR:
            # CSR转置：先转COO，交换行列，再转CSC
            coo = self._format.to_coo()
            transposed_coo_rows = coo.cols.copy()
            transposed_coo_cols = coo.rows.copy()
            from ..formats.coo import COOFormat
            transposed_coo = COOFormat(
                (self._shape[1], self._shape[0]),
                transposed_coo_rows,
                transposed_coo_cols,
                coo.data.copy(),
                self._dtype
            )
            transposed = transposed_coo.to_csc()
            result = SparseArray(shape=self._shape[::-1], format=SparseFormat.CSC, config=self._config)
        elif self._format_type == SparseFormat.CSC:
            # CSC转置：先转COO，交换行列，再转CSR
            coo = self._format.to_coo()
            transposed_coo_rows = coo.cols.copy()
            transposed_coo_cols = coo.rows.copy()
            from ..formats.coo import COOFormat
            transposed_coo = COOFormat(
                (self._shape[1], self._shape[0]),
                transposed_coo_rows,
                transposed_coo_cols,
                coo.data.copy(),
                self._dtype
            )
            transposed = transposed_coo.to_csr()
            result = SparseArray(shape=self._shape[::-1], format=SparseFormat.CSR, config=self._config)
        else:
            # 其他格式先转COO再转置
            coo = self._format.to_coo()
            transposed_coo_rows = coo.cols.copy()
            transposed_coo_cols = coo.rows.copy()
            from ..formats.coo import COOFormat
            transposed_coo = COOFormat(
                (self._shape[1], self._shape[0]),
                transposed_coo_rows,
                transposed_coo_cols,
                coo.data.copy(),
                self._dtype
            )
            transposed = transposed_coo.to_csr()
            result = SparseArray(shape=self._shape[::-1], format=SparseFormat.CSR, config=self._config)

        result._format = transposed
        result._stats.nnz = transposed.nnz
        result._stats.calculate_density()

        return result

    def __matmul__(self, other) -> Union['SparseArray', np.ndarray]:
        """矩阵乘法运算符 @"""
        return self.dot(other)

    def __add__(self, other: Union['SparseArray', np.ndarray, float, int]) -> 'SparseArray':
        """加法"""
        from ..ops.arithmetic import add
        return add(self, other, self._config)

    def __radd__(self, other) -> 'SparseArray':
        """右加法"""
        return self.__add__(other)

    def __sub__(self, other: Union['SparseArray', np.ndarray, float, int]) -> 'SparseArray':
        """减法"""
        from ..ops.arithmetic import subtract
        return subtract(self, other, self._config)

    def __rsub__(self, other) -> 'SparseArray':
        """右减法"""
        from ..ops.arithmetic import subtract
        return subtract(other, self, self._config)

    def __mul__(self, other: Union['SparseArray', np.ndarray, float, int]) -> 'SparseArray':
        """元素级乘法"""
        from ..ops.arithmetic import multiply
        return multiply(self, other, self._config)

    def __rmul__(self, other) -> 'SparseArray':
        """右乘法"""
        return self.__mul__(other)

    def __truediv__(self, other: Union['SparseArray', np.ndarray, float, int]) -> 'SparseArray':
        """元素级除法"""
        from ..ops.arithmetic import divide
        return divide(self, other, self._config)

    def __neg__(self) -> 'SparseArray':
        """取负"""
        from ..ops.arithmetic import negate
        return negate(self, self._config)

    # ==================== 索引操作 ====================

    def __getitem__(self, key) -> Any:
        """
        获取元素或切片

        Args:
            key: 索引键

        Returns:
            元素或子数组
        """
        if self._format is None:
            return 0

        return self._format.get_item(key)

    def __setitem__(self, key, value):
        """
        设置元素或切片

        Args:
            key: 索引键
            value: 值
        """
        if self._format is None:
            raise SparseArrayError("Cannot set item on empty sparse array")

        self._format.set_item(key, value)
        self._stats.nnz = self._format.nnz
        self._stats.calculate_density()

    # ==================== 聚合运算 ====================

    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray, 'SparseArray']:
        """
        求和

        Args:
            axis: 求和轴

        Returns:
            求和结果
        """
        from ..ops.arithmetic import sum as sparse_sum
        return sparse_sum(self, axis, self._config)

    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        求均值

        Args:
            axis: 求均值轴

        Returns:
            均值结果
        """
        from ..ops.arithmetic import mean as sparse_mean
        return sparse_mean(self, axis, self._config)

    def max(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        求最大值

        Args:
            axis: 求最大值轴

        Returns:
            最大值结果
        """
        from ..ops.arithmetic import max as sparse_max
        return sparse_max(self, axis, self._config)

    def min(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        求最小值

        Args:
            axis: 求最小值轴

        Returns:
            最小值结果
        """
        from ..ops.arithmetic import min as sparse_min
        return sparse_min(self, axis, self._config)

    def norm(self, ord: Union[int, float, str] = 2) -> float:
        """
        计算范数

        Args:
            ord: 范数类型

        Returns:
            范数值
        """
        from ..ops.linalg import norm as sparse_norm
        return sparse_norm(self, ord, self._config)

    # ==================== 序列化 ====================

    def save(self, path: str, format: str = 'npz'):
        """
        保存到文件

        Args:
            path: 文件路径
            format: 文件格式 ('npz', 'mtx', 'bin')
        """
        from ..ops.transform import save_sparse
        save_sparse(self, path, format)

    @classmethod
    def load(cls, path: str, format: str = 'auto') -> 'SparseArray':
        """
        从文件加载

        Args:
            path: 文件路径
            format: 文件格式 ('npz', 'mtx', 'bin', 'auto')

        Returns:
            SparseArray: 加载的稀疏数组
        """
        from ..ops.transform import load_sparse
        return load_sparse(path, format)

    # ==================== 工具方法 ====================

    def copy(self) -> 'SparseArray':
        """
        深拷贝

        Returns:
            SparseArray: 拷贝
        """
        result = SparseArray(shape=self._shape, format=self._format_type, config=self._config)
        if self._format is not None:
            result._format = self._format.copy()
        result._stats = SparseArrayStats(
            shape=self._stats.shape,
            nnz=self._stats.nnz,
            density=self._stats.density,
            sparsity=self._stats.sparsity
        )
        return result

    def get_memory_usage(self) -> int:
        """
        获取内存使用量

        Returns:
            int: 字节数
        """
        if self._format is None:
            return 0
        return self._format.get_memory_usage()

    def get_stats(self) -> SparseArrayStats:
        """
        获取统计信息

        Returns:
            SparseArrayStats: 统计信息
        """
        self._stats.memory_usage = self.get_memory_usage()
        return self._stats

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"SparseArray(shape={self._shape}, format={self.format}, "
                f"nnz={self.nnz}, dtype={self._dtype})")

    def __len__(self) -> int:
        """长度"""
        return self._shape[0]

    def __bool__(self) -> bool:
        """布尔值"""
        return self.nnz > 0

    # ==================== 私有方法 ====================

    @staticmethod
    def _validate_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        验证shape参数

        Args:
            shape: 形状元组

        Returns:
            Tuple[int, ...]: 验证后的形状元组

        Raises:
            ValueError: 如果shape无效
        """
        if shape is None:
            raise ValueError("shape cannot be None")

        try:
            shape_tuple = tuple(shape)
        except TypeError:
            raise ValueError(f"shape must be iterable, got {type(shape)}")

        if len(shape_tuple) == 0:
            raise ValueError("shape cannot be empty")

        for i, dim in enumerate(shape_tuple):
            if not isinstance(dim, (int, np.integer)):
                raise ValueError(f"shape[{i}] must be an integer, got {type(dim)}")
            if dim <= 0:
                raise ValueError(f"shape[{i}] must be positive, got {dim}")

        return shape_tuple

    def _initialize_from_arrays(self, data, indices, indptr):
        """从数组初始化"""
        if self._format_type == SparseFormat.CSR:
            from ..formats.csr import CSRFormat
            self._format = CSRFormat(self._shape, data, indices, indptr)
        elif self._format_type == SparseFormat.CSC:
            from ..formats.csc import CSCFormat
            self._format = CSCFormat(self._shape, data, indices, indptr)
        else:
            raise SparseArrayError(f"Unsupported format for array initialization: {self._format_type}")

        self._stats.nnz = len(data)
        self._stats.calculate_density()

    @staticmethod
    def _auto_select_format(arr: np.ndarray, config: SparseArrayConfig) -> SparseFormat:
        """
        自动选择最优格式

        Args:
            arr: 密集数组
            config: 配置

        Returns:
            SparseFormat: 选择的格式
        """
        from ..selector.features import extract_features
        from ..selector.auto_select import select_format

        features = extract_features(arr)
        return select_format(features, config)

    @staticmethod
    def _create_format_from_dense(arr: np.ndarray, format_type: SparseFormat) -> FormatBase:
        """
        从密集数组创建格式

        Args:
            arr: 密集数组
            format_type: 格式类型

        Returns:
            FormatBase: 格式对象
        """
        if format_type == SparseFormat.CSR:
            from ..formats.csr import CSRFormat
            return CSRFormat.from_dense(arr)
        elif format_type == SparseFormat.CSC:
            from ..formats.csc import CSCFormat
            return CSCFormat.from_dense(arr)
        elif format_type == SparseFormat.COO:
            from ..formats.coo import COOFormat
            return COOFormat.from_dense(arr)
        elif format_type == SparseFormat.BCSR:
            from ..formats.bcsr import BCSRFormat
            return BCSRFormat.from_dense(arr)
        elif format_type == SparseFormat.BITMAP:
            from ..formats.bitmap import BitmapFormat
            return BitmapFormat.from_dense(arr)
        else:
            # 默认使用CSR
            from ..formats.csr import CSRFormat
            return CSRFormat.from_dense(arr)


class _DenseWrapper(FormatBase):
    """
    密集数组包装器

    用于密度太高不适合稀疏存储的情况。
    """

    def __init__(self, arr: np.ndarray):
        super().__init__(arr.shape, arr.dtype)
        self._data = arr
        self._nnz = np.count_nonzero(arr)

    def to_dense(self) -> np.ndarray:
        return self._data.copy()

    def to_coo(self):
        from ..formats.coo import COOFormat
        return COOFormat.from_dense(self._data)

    def to_csr(self):
        from ..formats.csr import CSRFormat
        return CSRFormat.from_dense(self._data)

    def to_csc(self):
        from ..formats.csc import CSCFormat
        return CSCFormat.from_dense(self._data)

    def get_item(self, key):
        return self._data[key]

    def set_item(self, key, value):
        self._data[key] = value
        self._nnz = np.count_nonzero(self._data)

    def copy(self):
        return _DenseWrapper(self._data.copy())

    def get_memory_usage(self) -> int:
        return self._data.nbytes
