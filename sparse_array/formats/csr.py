"""
CSR (Compressed Sparse Row) 存储格式实现

CSR格式是科学计算中最常用的稀疏存储格式，使用三个数组：
- data: 非零元素值，形状 (nnz,)
- indices: 列索引，形状 (nnz,)
- indptr: 行指针，形状 (n_rows + 1,)

优点:
- 行切片高效
- SpMV运算高效
- 内存效率高

缺点:
- 列切片效率低
- 增量构建复杂
"""

from typing import Tuple, Optional, Union, Any, List, TYPE_CHECKING
import numpy as np

from ..core.sparse_array import FormatBase
from ..core.exceptions import IndexOutOfBoundsError

if TYPE_CHECKING:
    from .coo import COOFormat
    from .csc import CSCFormat


class CSRFormat(FormatBase):
    """
    CSR (Compressed Sparse Row) 存储格式

    使用行压缩方式存储稀疏矩阵。

    存储结构:
    - data: 非零元素值数组，形状 (nnz,)
    - indices: 列索引数组，形状 (nnz,)
    - indptr: 行指针数组，形状 (n_rows + 1,)
      第i行的非零元素在 data[indptr[i]:indptr[i+1]]

    Example:
        >>> # 创建一个 3x3 对角矩阵
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> indices = np.array([0, 1, 2])
        >>> indptr = np.array([0, 1, 2, 3])
        >>> csr = CSRFormat((3, 3), data, indices, indptr)
        >>> # 第0行: data[0:1] = [1.0], indices[0:1] = [0]
        >>> # 第1行: data[1:2] = [2.0], indices[1:2] = [1]
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 data: np.ndarray,
                 indices: np.ndarray,
                 indptr: np.ndarray,
                 dtype: Optional[np.dtype] = None):
        """
        初始化CSR格式

        Args:
            shape: 矩阵形状
            data: 非零元素值数组
            indices: 列索引数组
            indptr: 行指针数组
            dtype: 数据类型
        """
        super().__init__(shape, dtype or data.dtype)

        self._data = np.asarray(data, dtype=self._dtype)
        self._indices = np.asarray(indices, dtype=np.int32)
        self._indptr = np.asarray(indptr, dtype=np.int32)

        self._nnz = len(self._data)

        # 验证数据
        self._validate()

    def _validate(self):
        """验证数据一致性"""
        if len(self._shape) != 2:
            raise ValueError("CSR format only supports 2D arrays")

        if len(self._indptr) != self._shape[0] + 1:
            raise ValueError(f"indptr length must be {self._shape[0] + 1}")

        if len(self._indices) != len(self._data):
            raise ValueError("indices and data must have the same length")

        if self._nnz > 0:
            if np.any(self._indices < 0) or np.any(self._indices >= self._shape[1]):
                raise ValueError("Column indices out of bounds")

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> 'CSRFormat':
        """
        从密集数组创建CSR格式

        Args:
            arr: 密集数组

        Returns:
            CSRFormat: CSR格式对象
        """
        rows, cols = np.nonzero(arr)
        data = arr[rows, cols]

        # 构建indptr
        indptr = np.zeros(arr.shape[0] + 1, dtype=np.int32)
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)

        return cls(arr.shape, data, cols, indptr, arr.dtype)

    @property
    def data(self) -> np.ndarray:
        """获取数据数组"""
        return self._data

    @property
    def indices(self) -> np.ndarray:
        """获取列索引数组"""
        return self._indices

    @property
    def indptr(self) -> np.ndarray:
        """获取行指针数组"""
        return self._indptr

    def to_dense(self) -> np.ndarray:
        """转换为密集数组（向量化实现）"""
        result = np.zeros(self._shape, dtype=self._dtype)

        if self._nnz == 0:
            return result

        # 向量化赋值：使用高级索引
        # 展开行索引
        rows = np.repeat(np.arange(self._shape[0], dtype=np.int32),
                        np.diff(self._indptr))
        result[rows, self._indices] = self._data

        return result

    def to_coo(self) -> 'COOFormat':
        """转换为COO格式"""
        from .coo import COOFormat

        # 展开行索引
        rows = np.repeat(np.arange(self._shape[0], dtype=np.int32),
                        np.diff(self._indptr))

        return COOFormat(self._shape, rows, self._indices.copy(),
                        self._data.copy(), self._dtype)

    def to_csr(self) -> 'CSRFormat':
        """转换为CSR格式（返回自身）"""
        return self

    def to_csc(self) -> 'CSCFormat':
        """转换为CSC格式"""
        from .csc import CSCFormat

        if self._nnz == 0:
            indptr = np.zeros(self._shape[1] + 1, dtype=np.int32)
            return CSCFormat(self._shape, np.array([], dtype=self._dtype),
                           np.array([], dtype=np.int32), indptr)

        # 转置操作：CSR转置为CSC
        # 使用计数排序
        col_counts = np.zeros(self._shape[1], dtype=np.int32)
        np.add.at(col_counts, self._indices, 1)

        csc_indptr = np.zeros(self._shape[1] + 1, dtype=np.int32)
        csc_indptr[1:] = np.cumsum(col_counts)

        # 分配空间
        csc_indices = np.zeros(self._nnz, dtype=np.int32)
        csc_data = np.zeros(self._nnz, dtype=self._dtype)

        # 填充数据
        col_pos = csc_indptr.copy()
        for i in range(self._shape[0]):
            for j in range(self._indptr[i], self._indptr[i + 1]):
                col = self._indices[j]
                pos = col_pos[col]
                csc_indices[pos] = i
                csc_data[pos] = self._data[j]
                col_pos[col] += 1

        return CSCFormat(self._shape, csc_data, csc_indices, csc_indptr)

    def get_row(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取一行的数据

        Args:
            row: 行索引

        Returns:
            Tuple: (列索引数组, 数据数组)
        """
        if row < 0 or row >= self._shape[0]:
            raise IndexOutOfBoundsError(
                f"Row index {row} out of bounds",
                index=(row,),
                shape=self._shape
            )

        start, end = self._indptr[row], self._indptr[row + 1]
        return self._indices[start:end].copy(), self._data[start:end].copy()

    def get_item(self, key) -> Any:
        """获取元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            # 处理负索引
            if isinstance(row, int):
                if row < 0:
                    row += self._shape[0]
                if row < 0 or row >= self._shape[0]:
                    raise IndexOutOfBoundsError(
                        f"Row index {row} out of bounds",
                        index=key,
                        shape=self._shape
                    )

            if isinstance(col, int):
                if col < 0:
                    col += self._shape[1]
                if col < 0 or col >= self._shape[1]:
                    raise IndexOutOfBoundsError(
                        f"Column index {col} out of bounds",
                        index=key,
                        shape=self._shape
                    )

            # 单个元素访问
            if isinstance(row, int) and isinstance(col, int):
                start, end = self._indptr[row], self._indptr[row + 1]
                idx = np.searchsorted(self._indices[start:end], col)
                if idx < end - start and self._indices[start + idx] == col:
                    return self._data[start + idx]
                return self._dtype.type(0)

            # 切片操作
            return self._slice(row, col)

        # 单索引：获取行
        if isinstance(key, int):
            if key < 0:
                key += self._shape[0]
            indices, data = self.get_row(key)
            result = np.zeros(self._shape[1], dtype=self._dtype)
            result[indices] = data
            return result

        raise TypeError(f"Invalid index type: {type(key)}")

    def _slice(self, row_key, col_key) -> 'CSRFormat':
        """切片操作"""
        # 处理行切片
        if isinstance(row_key, slice):
            row_start, row_stop, row_step = row_key.indices(self._shape[0])
            row_indices = list(range(row_start, row_stop, row_step))
        else:
            row_indices = [row_key] if isinstance(row_key, int) else list(row_key)

        # 处理列切片
        if isinstance(col_key, slice):
            col_start, col_stop, col_step = col_key.indices(self._shape[1])
            col_indices = list(range(col_start, col_stop, col_step))
            col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(col_indices)}
        elif isinstance(col_key, int):
            col_indices = [col_key]
            col_map = {col_key: 0}
        else:
            col_indices = list(col_key)
            col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(col_indices)}

        # 构建新CSR
        new_data = []
        new_indices = []
        new_indptr = [0]

        for row in row_indices:
            start, end = self._indptr[row], self._indptr[row + 1]
            for j in range(start, end):
                col = self._indices[j]
                if col in col_map:
                    new_data.append(self._data[j])
                    new_indices.append(col_map[col])

            new_indptr.append(len(new_data))

        new_shape = (len(row_indices), len(col_indices))

        return CSRFormat(
            new_shape,
            np.array(new_data, dtype=self._dtype),
            np.array(new_indices, dtype=np.int32),
            np.array(new_indptr, dtype=np.int32)
        )

    def set_item(self, key, value):
        """设置元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            if isinstance(row, int) and isinstance(col, int):
                self._set_single_element(row, col, value)
                return

        raise TypeError("Only single element assignment is supported for CSR format")

    def _set_single_element(self, row: int, col: int, value):
        """设置单个元素"""
        start, end = self._indptr[row], self._indptr[row + 1]

        # 查找插入位置
        idx = np.searchsorted(self._indices[start:end], col)
        actual_idx = start + idx

        if idx < end - start and self._indices[actual_idx] == col:
            # 元素已存在
            if value == 0:
                # 删除元素
                self._data = np.delete(self._data, actual_idx)
                self._indices = np.delete(self._indices, actual_idx)
                self._indptr[row + 1:] -= 1
                self._nnz -= 1
            else:
                # 更新元素
                self._data[actual_idx] = value
        elif value != 0:
            # 插入新元素
            self._data = np.insert(self._data, actual_idx, value)
            self._indices = np.insert(self._indices, actual_idx, col)
            self._indptr[row + 1:] += 1
            self._nnz += 1

    def copy(self) -> 'CSRFormat':
        """深拷贝"""
        return CSRFormat(
            self._shape,
            self._data.copy(),
            self._indices.copy(),
            self._indptr.copy(),
            self._dtype
        )

    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        return (self._data.nbytes + self._indices.nbytes + self._indptr.nbytes)

    def eliminate_zeros(self):
        """消除零元素"""
        mask = self._data != 0
        if np.all(mask):
            return

        # 重建数组
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(self._shape[0]):
            start, end = self._indptr[i], self._indptr[i + 1]
            row_mask = mask[start:end]
            new_data.extend(self._data[start:end][row_mask])
            new_indices.extend(self._indices[start:end][row_mask])
            new_indptr.append(len(new_data))

        self._data = np.array(new_data, dtype=self._dtype)
        self._indices = np.array(new_indices, dtype=np.int32)
        self._indptr = np.array(new_indptr, dtype=np.int32)
        self._nnz = len(self._data)

    def sort_indices(self):
        """对每行的列索引排序"""
        for i in range(self._shape[0]):
            start, end = self._indptr[i], self._indptr[i + 1]
            if end - start > 1:
                order = np.argsort(self._indices[start:end])
                self._indices[start:end] = self._indices[start:end][order]
                self._data[start:end] = self._data[start:end][order]

    def get_nnz_per_row(self) -> np.ndarray:
        """获取每行的非零元素数量"""
        return np.diff(self._indptr)

    def get_row_bandwidth(self) -> int:
        """获取矩阵带宽"""
        if self._nnz == 0:
            return 0

        bandwidth = 0
        for i in range(self._shape[0]):
            start, end = self._indptr[i], self._indptr[i + 1]
            if end > start:
                row_bandwidth = np.max(np.abs(self._indices[start:end] - i))
                bandwidth = max(bandwidth, row_bandwidth)

        return bandwidth
