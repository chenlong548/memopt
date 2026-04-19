"""
CSC (Compressed Sparse Column) 存储格式实现

CSC格式是CSR格式的转置版本，使用三个数组：
- data: 非零元素值，形状 (nnz,)
- indices: 行索引，形状 (nnz,)
- indptr: 列指针，形状 (n_cols + 1,)

优点:
- 列切片高效
- SpMV运算（转置）高效
- 适合列操作

缺点:
- 行切片效率低
- 增量构建复杂
"""

from typing import Tuple, Optional, Union, Any, TYPE_CHECKING
import numpy as np

from ..core.sparse_array import FormatBase
from ..core.exceptions import IndexOutOfBoundsError

if TYPE_CHECKING:
    from .coo import COOFormat
    from .csr import CSRFormat


class CSCFormat(FormatBase):
    """
    CSC (Compressed Sparse Column) 存储格式

    使用列压缩方式存储稀疏矩阵。

    存储结构:
    - data: 非零元素值数组，形状 (nnz,)
    - indices: 行索引数组，形状 (nnz,)
    - indptr: 列指针数组，形状 (n_cols + 1,)
      第j列的非零元素在 data[indptr[j]:indptr[j+1]]

    Example:
        >>> # 创建一个 3x3 对角矩阵
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> indices = np.array([0, 1, 2])
        >>> indptr = np.array([0, 1, 2, 3])
        >>> csc = CSCFormat((3, 3), data, indices, indptr)
        >>> # 第0列: data[0:1] = [1.0], indices[0:1] = [0]
        >>> # 第1列: data[1:2] = [2.0], indices[1:2] = [1]
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 data: np.ndarray,
                 indices: np.ndarray,
                 indptr: np.ndarray,
                 dtype: Optional[np.dtype] = None):
        """
        初始化CSC格式

        Args:
            shape: 矩阵形状
            data: 非零元素值数组
            indices: 行索引数组
            indptr: 列指针数组
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
            raise ValueError("CSC format only supports 2D arrays")

        if len(self._indptr) != self._shape[1] + 1:
            raise ValueError(f"indptr length must be {self._shape[1] + 1}")

        if len(self._indices) != len(self._data):
            raise ValueError("indices and data must have the same length")

        if self._nnz > 0:
            if np.any(self._indices < 0) or np.any(self._indices >= self._shape[0]):
                raise ValueError("Row indices out of bounds")

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> 'CSCFormat':
        """
        从密集数组创建CSC格式

        Args:
            arr: 密集数组

        Returns:
            CSCFormat: CSC格式对象
        """
        rows, cols = np.nonzero(arr)
        data = arr[rows, cols]

        # 按列排序
        order = np.lexsort((rows, cols))
        rows = rows[order]
        cols = cols[order]
        data = data[order]

        # 构建indptr
        indptr = np.zeros(arr.shape[1] + 1, dtype=np.int32)
        np.add.at(indptr, cols + 1, 1)
        np.cumsum(indptr, out=indptr)

        return cls(arr.shape, data, rows, indptr, arr.dtype)

    @property
    def data(self) -> np.ndarray:
        """获取数据数组"""
        return self._data

    @property
    def indices(self) -> np.ndarray:
        """获取行索引数组"""
        return self._indices

    @property
    def indptr(self) -> np.ndarray:
        """获取列指针数组"""
        return self._indptr

    def to_dense(self) -> np.ndarray:
        """转换为密集数组（向量化实现）"""
        result = np.zeros(self._shape, dtype=self._dtype)

        if self._nnz == 0:
            return result

        # 向量化赋值：使用高级索引
        # 展开列索引
        cols = np.repeat(np.arange(self._shape[1], dtype=np.int32),
                        np.diff(self._indptr))
        result[self._indices, cols] = self._data

        return result

    def to_coo(self) -> 'COOFormat':
        """转换为COO格式"""
        from .coo import COOFormat

        # 展开列索引
        cols = np.repeat(np.arange(self._shape[1], dtype=np.int32),
                        np.diff(self._indptr))

        return COOFormat(self._shape, self._indices.copy(), cols,
                        self._data.copy(), self._dtype)

    def to_csr(self) -> 'CSRFormat':
        """转换为CSR格式"""
        # CSC转置为CSR
        from .csr import CSRFormat

        if self._nnz == 0:
            indptr = np.zeros(self._shape[0] + 1, dtype=np.int32)
            return CSRFormat(self._shape, np.array([], dtype=self._dtype),
                           np.array([], dtype=np.int32), indptr)

        # 使用计数排序
        row_counts = np.zeros(self._shape[0], dtype=np.int32)
        np.add.at(row_counts, self._indices, 1)

        csr_indptr = np.zeros(self._shape[0] + 1, dtype=np.int32)
        csr_indptr[1:] = np.cumsum(row_counts)

        # 分配空间
        csr_indices = np.zeros(self._nnz, dtype=np.int32)
        csr_data = np.zeros(self._nnz, dtype=self._dtype)

        # 填充数据
        row_pos = csr_indptr.copy()
        for j in range(self._shape[1]):
            for k in range(self._indptr[j], self._indptr[j + 1]):
                row = self._indices[k]
                pos = row_pos[row]
                csr_indices[pos] = j
                csr_data[pos] = self._data[k]
                row_pos[row] += 1

        return CSRFormat(self._shape, csr_data, csr_indices, csr_indptr)

    def to_csc(self) -> 'CSCFormat':
        """转换为CSC格式（返回自身）"""
        return self

    def get_col(self, col: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取一列的数据

        Args:
            col: 列索引

        Returns:
            Tuple: (行索引数组, 数据数组)
        """
        if col < 0 or col >= self._shape[1]:
            raise IndexOutOfBoundsError(
                f"Column index {col} out of bounds",
                index=(col,),
                shape=self._shape
            )

        start, end = self._indptr[col], self._indptr[col + 1]
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
                start, end = self._indptr[col], self._indptr[col + 1]
                idx = np.searchsorted(self._indices[start:end], row)
                if idx < end - start and self._indices[start + idx] == row:
                    return self._data[start + idx]
                return self._dtype.type(0)

            # 处理单列访问（返回数组）
            if isinstance(col, int) and isinstance(row, slice):
                if row == slice(None):
                    # sparse[:, 50] - 返回整列
                    indices, data = self.get_col(col)
                    result = np.zeros(self._shape[0], dtype=self._dtype)
                    result[indices] = data
                    return result

            # 切片操作
            return self._slice(row, col)

        # 单索引：获取列
        if isinstance(key, int):
            if key < 0:
                key += self._shape[1]
            indices, data = self.get_col(key)
            result = np.zeros(self._shape[0], dtype=self._dtype)
            result[indices] = data
            return result

        raise TypeError(f"Invalid index type: {type(key)}")

    def _slice(self, row_key, col_key) -> 'CSCFormat':
        """切片操作"""
        # 处理列切片
        if isinstance(col_key, slice):
            col_start, col_stop, col_step = col_key.indices(self._shape[1])
            col_indices = list(range(col_start, col_stop, col_step))
        else:
            col_indices = [col_key] if isinstance(col_key, int) else list(col_key)

        # 处理行切片
        if isinstance(row_key, slice):
            row_start, row_stop, row_step = row_key.indices(self._shape[0])
            row_indices = set(range(row_start, row_stop, row_step))
        elif isinstance(row_key, int):
            row_indices = {row_key}
        else:
            row_indices = set(row_key)

        # 构建新CSC
        new_data = []
        new_indices = []
        new_indptr = [0]

        for col in col_indices:
            start, end = self._indptr[col], self._indptr[col + 1]
            for k in range(start, end):
                row = self._indices[k]
                if row in row_indices:
                    new_data.append(self._data[k])
                    new_indices.append(row)

            new_indptr.append(len(new_data))

        new_shape = (len(row_indices) if isinstance(row_key, (slice, list)) else 1,
                    len(col_indices))

        return CSCFormat(
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

        raise TypeError("Only single element assignment is supported for CSC format")

    def _set_single_element(self, row: int, col: int, value):
        """设置单个元素"""
        start, end = self._indptr[col], self._indptr[col + 1]

        # 查找插入位置
        idx = np.searchsorted(self._indices[start:end], row)
        actual_idx = start + idx

        if idx < end - start and self._indices[actual_idx] == row:
            # 元素已存在
            if value == 0:
                # 删除元素
                self._data = np.delete(self._data, actual_idx)
                self._indices = np.delete(self._indices, actual_idx)
                self._indptr[col + 1:] -= 1
                self._nnz -= 1
            else:
                # 更新元素
                self._data[actual_idx] = value
        elif value != 0:
            # 插入新元素
            self._data = np.insert(self._data, actual_idx, value)
            self._indices = np.insert(self._indices, actual_idx, row)
            self._indptr[col + 1:] += 1
            self._nnz += 1

    def copy(self) -> 'CSCFormat':
        """深拷贝"""
        return CSCFormat(
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

        for j in range(self._shape[1]):
            start, end = self._indptr[j], self._indptr[j + 1]
            col_mask = mask[start:end]
            new_data.extend(self._data[start:end][col_mask])
            new_indices.extend(self._indices[start:end][col_mask])
            new_indptr.append(len(new_data))

        self._data = np.array(new_data, dtype=self._dtype)
        self._indices = np.array(new_indices, dtype=np.int32)
        self._indptr = np.array(new_indptr, dtype=np.int32)
        self._nnz = len(self._data)

    def sort_indices(self):
        """对每列的行索引排序"""
        for j in range(self._shape[1]):
            start, end = self._indptr[j], self._indptr[j + 1]
            if end - start > 1:
                order = np.argsort(self._indices[start:end])
                self._indices[start:end] = self._indices[start:end][order]
                self._data[start:end] = self._data[start:end][order]

    def get_nnz_per_col(self) -> np.ndarray:
        """获取每列的非零元素数量"""
        return np.diff(self._indptr)
