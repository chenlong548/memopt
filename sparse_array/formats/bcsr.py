"""
BCSR (Block Compressed Sparse Row) 存储格式实现

BCSR格式是CSR的块版本，将矩阵划分为固定大小的块，只存储非零块。
适合具有块结构的稀疏矩阵，特别是GPU和Tensor Core优化。

存储结构:
- data: 非零块数据，形状 (nnb, block_size[0], block_size[1])
- col_indices: 块列索引，形状 (nnb,)
- row_ptr: 块行指针，形状 (n_block_rows + 1,)
- block_size: 块大小 (R, C)

优点:
- GPU和Tensor Core友好
- 块操作高效
- 适合结构化稀疏

缺点:
- 需要块结构
- 小块效率低
"""

from typing import Tuple, Optional, Union, Any
import numpy as np

from ..core.sparse_array import FormatBase


class BCSRFormat(FormatBase):
    """
    BCSR (Block Compressed Sparse Row) 存储格式

    使用块压缩方式存储稀疏矩阵，适合GPU加速。

    存储结构:
    - data: 非零块数据数组，形状 (nnb, R, C)
    - col_indices: 块列索引数组，形状 (nnb,)
    - row_ptr: 块行指针数组，形状 (n_block_rows + 1,)
    - block_size: 块大小 (R, C)

    Example:
        >>> # 创建一个 4x4 矩阵，块大小 2x2
        >>> data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> col_indices = np.array([0, 1])
        >>> row_ptr = np.array([0, 1, 2])
        >>> bcsr = BCSRFormat((4, 4), data, col_indices, row_ptr, (2, 2))
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 data: np.ndarray,
                 col_indices: np.ndarray,
                 row_ptr: np.ndarray,
                 block_size: Tuple[int, int] = (4, 4),
                 dtype: Optional[np.dtype] = None):
        """
        初始化BCSR格式

        Args:
            shape: 矩阵形状
            data: 非零块数据数组
            col_indices: 块列索引数组
            row_ptr: 块行指针数组
            block_size: 块大小 (R, C)
            dtype: 数据类型
        """
        super().__init__(shape, dtype or data.dtype if len(data) > 0 else np.float64)

        self._block_size = tuple(block_size)
        self._data = np.asarray(data, dtype=self._dtype)
        self._col_indices = np.asarray(col_indices, dtype=np.int32)
        self._row_ptr = np.asarray(row_ptr, dtype=np.int32)

        # 计算块维度
        self._n_block_rows = (self._shape[0] + self._block_size[0] - 1) // self._block_size[0]
        self._n_block_cols = (self._shape[1] + self._block_size[1] - 1) // self._block_size[1]

        # 非零块数量
        self._nnb = len(self._col_indices)

        # 计算非零元素数量
        self._nnz = np.count_nonzero(self._data) if len(self._data) > 0 else 0

        self._validate()

    def _validate(self):
        """验证数据一致性"""
        if len(self._shape) != 2:
            raise ValueError("BCSR format only supports 2D arrays")

        if len(self._row_ptr) != self._n_block_rows + 1:
            raise ValueError(f"row_ptr length must be {self._n_block_rows + 1}")

        if len(self._col_indices) != len(self._data):
            raise ValueError("col_indices and data must have the same length")

        if self._nnb > 0:
            if self._data.ndim != 3 or self._data.shape[1:] != self._block_size:
                raise ValueError(f"data shape must be (nnb, {self._block_size[0]}, {self._block_size[1]})")

    @classmethod
    def from_dense(cls,
                   arr: np.ndarray,
                   block_size: Tuple[int, int] = (4, 4),
                   threshold: float = 0.0) -> 'BCSRFormat':
        """
        从密集数组创建BCSR格式

        Args:
            arr: 密集数组
            block_size: 块大小
            threshold: 块稀疏化阈值

        Returns:
            BCSRFormat: BCSR格式对象
        """
        R, C = block_size
        n_rows, n_cols = arr.shape

        n_block_rows = (n_rows + R - 1) // R
        n_block_cols = (n_cols + C - 1) // C

        data_list = []
        col_indices_list = []
        row_ptr = [0]

        for br in range(n_block_rows):
            row_start = br * R
            row_end = min((br + 1) * R, n_rows)

            for bc in range(n_block_cols):
                col_start = bc * C
                col_end = min((bc + 1) * C, n_cols)

                # 提取块
                block = arr[row_start:row_end, col_start:col_end]

                # 检查块是否为非零
                if threshold > 0:
                    block_density = np.count_nonzero(block) / block.size
                    if block_density < threshold:
                        continue

                if np.any(block != 0):
                    # 填充块到完整大小
                    if block.shape != (R, C):
                        padded_block = np.zeros((R, C), dtype=arr.dtype)
                        padded_block[:row_end - row_start, :col_end - col_start] = block
                        block = padded_block

                    data_list.append(block)
                    col_indices_list.append(bc)

            row_ptr.append(len(col_indices_list))

        if data_list:
            data = np.stack(data_list, axis=0)
            col_indices = np.array(col_indices_list, dtype=np.int32)
        else:
            data = np.array([], dtype=arr.dtype).reshape(0, R, C)
            col_indices = np.array([], dtype=np.int32)

        return cls(arr.shape, data, col_indices, np.array(row_ptr, dtype=np.int32),
                  block_size, arr.dtype)

    @property
    def data(self) -> np.ndarray:
        """获取块数据数组"""
        return self._data

    @property
    def col_indices(self) -> np.ndarray:
        """获取块列索引数组"""
        return self._col_indices

    @property
    def row_ptr(self) -> np.ndarray:
        """获取块行指针数组"""
        return self._row_ptr

    @property
    def block_size(self) -> Tuple[int, int]:
        """获取块大小"""
        return self._block_size

    @property
    def nnb(self) -> int:
        """获取非零块数量"""
        return self._nnb

    def to_dense(self) -> np.ndarray:
        """转换为密集数组"""
        result = np.zeros(self._shape, dtype=self._dtype)
        R, C = self._block_size

        for br in range(self._n_block_rows):
            start, end = self._row_ptr[br], self._row_ptr[br + 1]
            row_start = br * R
            row_end = min((br + 1) * R, self._shape[0])

            for idx in range(start, end):
                bc = self._col_indices[idx]
                col_start = bc * C
                col_end = min((bc + 1) * C, self._shape[1])

                block = self._data[idx]
                result[row_start:row_end, col_start:col_end] = block[:row_end - row_start, :col_end - col_start]

        return result

    def to_coo(self):
        """转换为COO格式"""
        from .coo import COOFormat

        rows_list = []
        cols_list = []
        data_list = []
        R, C = self._block_size

        for br in range(self._n_block_rows):
            start, end = self._row_ptr[br], self._row_ptr[br + 1]

            for idx in range(start, end):
                bc = self._col_indices[idx]
                block = self._data[idx]

                # 展开块
                for i in range(R):
                    for j in range(C):
                        if block[i, j] != 0:
                            rows_list.append(br * R + i)
                            cols_list.append(bc * C + j)
                            data_list.append(block[i, j])

        return COOFormat(
            self._shape,
            np.array(rows_list, dtype=np.int32),
            np.array(cols_list, dtype=np.int32),
            np.array(data_list, dtype=self._dtype)
        )

    def to_csr(self):
        """转换为CSR格式"""
        coo = self.to_coo()
        return coo.to_csr()

    def to_csc(self):
        """转换为CSC格式"""
        coo = self.to_coo()
        return coo.to_csc()

    def get_item(self, key) -> Any:
        """获取元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            # 处理负索引
            if isinstance(row, int):
                if row < 0:
                    row += self._shape[0]
            if isinstance(col, int):
                if col < 0:
                    col += self._shape[1]

            # 单个元素访问
            if isinstance(row, int) and isinstance(col, int):
                R, C = self._block_size
                br = row // R
                bc = col // C

                # 查找块
                start, end = self._row_ptr[br], self._row_ptr[br + 1]
                idx = np.searchsorted(self._col_indices[start:end], bc)

                if idx < end - start and self._col_indices[start + idx] == bc:
                    block = self._data[start + idx]
                    local_row = row % R
                    local_col = col % C
                    return block[local_row, local_col]

                return self._dtype.type(0)

        # 其他情况转换为密集数组
        dense = self.to_dense()
        return dense[key]

    def set_item(self, key, value):
        """设置元素"""
        raise NotImplementedError("BCSR format does not support incremental element assignment")

    def copy(self) -> 'BCSRFormat':
        """深拷贝"""
        return BCSRFormat(
            self._shape,
            self._data.copy(),
            self._col_indices.copy(),
            self._row_ptr.copy(),
            self._block_size,
            self._dtype
        )

    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        return (self._data.nbytes + self._col_indices.nbytes + self._row_ptr.nbytes)

    def get_block_density(self) -> float:
        """获取块密度"""
        if self._nnb == 0:
            return 0.0
        total_blocks = self._n_block_rows * self._n_block_cols
        return self._nnb / total_blocks

    def get_average_block_fill(self) -> float:
        """获取平均块填充率"""
        if self._nnb == 0:
            return 0.0
        return self._nnz / (self._nnb * self._block_size[0] * self._block_size[1])

    def optimize_block_size(self) -> Tuple[int, int]:
        """
        优化块大小

        Returns:
            Tuple[int, int]: 推荐的块大小
        """
        # 分析块填充模式
        if self._nnb == 0:
            return self._block_size

        # 计算平均块填充
        avg_fill = self.get_average_block_fill()

        # 根据填充率推荐块大小
        if avg_fill > 0.8:
            # 高填充率，使用大块
            return (8, 8)
        elif avg_fill > 0.5:
            return (4, 4)
        elif avg_fill > 0.3:
            return (2, 2)
        else:
            # 低填充率，使用小块或转为CSR
            return (1, 1)  # 等效于CSR


# 延迟导入
from .coo import COOFormat
