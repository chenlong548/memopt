"""
COO (Coordinate) 存储格式实现

COO格式是最直观的稀疏存储格式，使用三个数组分别存储：
- 行索引 (rows)
- 列索引 (cols)
- 数据值 (data)

优点:
- 格式简单，易于理解和构建
- 支持高效的增量构建
- 适合矩阵组装和格式转换

缺点:
- 随机访问效率低
- 算术运算需要排序
"""

from typing import Tuple, Optional, Union, Any, TYPE_CHECKING
import numpy as np

from ..core.sparse_array import FormatBase

if TYPE_CHECKING:
    from .csr import CSRFormat
    from .csc import CSCFormat


class COOFormat(FormatBase):
    """
    COO (Coordinate) 存储格式

    使用坐标列表存储非零元素的位置和值。

    存储结构:
    - rows: 行索引数组，形状 (nnz,)
    - cols: 列索引数组，形状 (nnz,)
    - data: 数据值数组，形状 (nnz,)

    Example:
        >>> # 创建一个 3x3 稀疏矩阵
        >>> rows = np.array([0, 1, 2])
        >>> cols = np.array([0, 1, 2])
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> coo = COOFormat((3, 3), rows, cols, data)
        >>> dense = coo.to_dense()
        >>> # dense = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 rows: np.ndarray,
                 cols: np.ndarray,
                 data: np.ndarray,
                 dtype: Optional[np.dtype] = None):
        """
        初始化COO格式

        Args:
            shape: 矩阵形状
            rows: 行索引数组
            cols: 列索引数组
            data: 数据值数组
            dtype: 数据类型
        """
        super().__init__(shape, dtype or data.dtype)

        self._rows = np.asarray(rows, dtype=np.int32)
        self._cols = np.asarray(cols, dtype=np.int32)
        self._data = np.asarray(data, dtype=self._dtype)

        self._nnz = len(self._data)
        self._sorted = False

        # 验证数据
        self._validate()

    def _validate(self):
        """验证数据一致性"""
        if len(self._rows) != len(self._cols) or len(self._rows) != len(self._data):
            raise ValueError("rows, cols, and data must have the same length")

        if len(self._shape) != 2:
            raise ValueError("COO format only supports 2D arrays")

        # 检查索引范围
        if self._nnz > 0:
            if np.any(self._rows < 0) or np.any(self._rows >= self._shape[0]):
                raise ValueError("Row indices out of bounds")
            if np.any(self._cols < 0) or np.any(self._cols >= self._shape[1]):
                raise ValueError("Column indices out of bounds")

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> 'COOFormat':
        """
        从密集数组创建COO格式

        Args:
            arr: 密集数组

        Returns:
            COOFormat: COO格式对象
        """
        rows, cols = np.nonzero(arr)
        data = arr[rows, cols]
        return cls(arr.shape, rows, cols, data, arr.dtype)

    @property
    def rows(self) -> np.ndarray:
        """获取行索引数组"""
        return self._rows

    @property
    def cols(self) -> np.ndarray:
        """获取列索引数组"""
        return self._cols

    @property
    def data(self) -> np.ndarray:
        """获取数据数组"""
        return self._data

    def to_dense(self) -> np.ndarray:
        """转换为密集数组"""
        result = np.zeros(self._shape, dtype=self._dtype)
        if self._nnz > 0:
            result[self._rows, self._cols] = self._data
        return result

    def to_coo(self) -> 'COOFormat':
        """转换为COO格式（返回自身）"""
        return self

    def to_csr(self) -> 'CSRFormat':
        """转换为CSR格式"""
        from .csr import CSRFormat

        if self._nnz == 0:
            indptr = np.zeros(self._shape[0] + 1, dtype=np.int32)
            indices = np.array([], dtype=np.int32)
            data = np.array([], dtype=self._dtype)
            return CSRFormat(self._shape, data, indices, indptr)

        # 按行排序
        self._sort()

        # 构建CSR格式
        indptr = np.zeros(self._shape[0] + 1, dtype=np.int32)
        np.add.at(indptr, self._rows + 1, 1)
        np.cumsum(indptr, out=indptr)

        return CSRFormat(self._shape, self._data.copy(), self._cols.copy(), indptr)

    def to_csc(self) -> 'CSCFormat':
        """转换为CSC格式"""
        from .csc import CSCFormat

        if self._nnz == 0:
            indptr = np.zeros(self._shape[1] + 1, dtype=np.int32)
            indices = np.array([], dtype=np.int32)
            data = np.array([], dtype=self._dtype)
            return CSCFormat(self._shape, data, indices, indptr)

        # 按列排序
        self._sort(by_col=True)

        # 构建CSC格式
        indptr = np.zeros(self._shape[1] + 1, dtype=np.int32)
        np.add.at(indptr, self._cols + 1, 1)
        np.cumsum(indptr, out=indptr)

        return CSCFormat(self._shape, self._data.copy(), self._rows.copy(), indptr)

    def _sort(self, by_col: bool = False):
        """
        排序坐标

        Args:
            by_col: 是否按列排序
        """
        if self._sorted:
            return

        if by_col:
            # 先按列，再按行排序
            order = np.lexsort((self._rows, self._cols))
        else:
            # 先按行，再按列排序
            order = np.lexsort((self._cols, self._rows))

        self._rows = self._rows[order]
        self._cols = self._cols[order]
        self._data = self._data[order]
        self._sorted = True

    def get_item(self, key) -> Any:
        """获取元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            # 查找对应坐标
            mask = (self._rows == row) & (self._cols == col)
            if np.any(mask):
                return self._data[mask][0]
            return 0

        # 切片操作
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        # 转换为COO切片
        return self._slice_coo(row_key, col_key)

    def _slice_coo(self, row_key, col_key) -> 'COOFormat':
        """切片操作"""
        # 处理行切片
        if isinstance(row_key, slice):
            row_start, row_stop, row_step = row_key.indices(self._shape[0])
            row_mask = (self._rows >= row_start) & (self._rows < row_stop)
            if row_step != 1:
                row_mask &= ((self._rows - row_start) % row_step == 0)
            new_rows = (self._rows[row_mask] - row_start) // row_step if row_step != 1 else self._rows[row_mask] - row_start
        else:
            row_mask = self._rows == row_key
            new_rows = np.zeros(np.sum(row_mask), dtype=np.int32)

        # 处理列切片
        if isinstance(col_key, slice):
            col_start, col_stop, col_step = col_key.indices(self._shape[1])
            col_mask = (self._cols >= col_start) & (self._cols < col_stop)
            if col_step != 1:
                col_mask &= ((self._cols - col_start) % col_step == 0)
            new_cols = (self._cols[col_mask] - col_start) // col_step if col_step != 1 else self._cols[col_mask] - col_start
        else:
            col_mask = self._cols == col_key
            new_cols = np.zeros(np.sum(col_mask), dtype=np.int32)

        mask = row_mask & col_mask

        # 计算新形状
        if isinstance(row_key, slice):
            new_nrows = len(range(*row_key.indices(self._shape[0])))
        else:
            new_nrows = 1

        if isinstance(col_key, slice):
            new_ncols = len(range(*col_key.indices(self._shape[1])))
        else:
            new_ncols = 1

        return COOFormat(
            (new_nrows, new_ncols),
            new_rows if isinstance(row_key, slice) else self._rows[mask] - row_key,
            new_cols if isinstance(col_key, slice) else self._cols[mask] - col_key,
            self._data[mask].copy()
        )

    def set_item(self, key, value):
        """设置元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            # 查找是否已存在
            mask = (self._rows == row) & (self._cols == col)
            if np.any(mask):
                if value == 0:
                    # 删除元素
                    keep_mask = ~mask
                    self._rows = self._rows[keep_mask]
                    self._cols = self._cols[keep_mask]
                    self._data = self._data[keep_mask]
                    self._nnz -= 1
                else:
                    # 更新元素
                    self._data[mask] = value
            elif value != 0:
                # 添加新元素
                self._rows = np.append(self._rows, row)
                self._cols = np.append(self._cols, col)
                self._data = np.append(self._data, value)
                self._nnz += 1
                self._sorted = False

    def copy(self) -> 'COOFormat':
        """深拷贝"""
        return COOFormat(
            self._shape,
            self._rows.copy(),
            self._cols.copy(),
            self._data.copy(),
            self._dtype
        )

    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        return (self._rows.nbytes + self._cols.nbytes + self._data.nbytes)

    def eliminate_zeros(self):
        """消除零元素"""
        mask = self._data != 0
        self._rows = self._rows[mask]
        self._cols = self._cols[mask]
        self._data = self._data[mask]
        self._nnz = len(self._data)
        self._sorted = False

    def sum_duplicates(self):
        """合并重复坐标"""
        self._sort()

        if self._nnz <= 1:
            return

        # 找出唯一坐标
        unique_mask = np.ones(self._nnz, dtype=bool)
        for i in range(1, self._nnz):
            if (self._rows[i] == self._rows[i-1] and
                self._cols[i] == self._cols[i-1]):
                self._data[i-1] += self._data[i]
                unique_mask[i] = False

        self._rows = self._rows[unique_mask]
        self._cols = self._cols[unique_mask]
        self._data = self._data[unique_mask]
        self._nnz = len(self._data)
