"""
Bitmap 存储格式实现

Bitmap格式使用位图标记非零元素位置，适合结构化稀疏矩阵。
特别适合具有规则模式的稀疏矩阵，如对角矩阵、带状矩阵等。

存储结构:
- bitmap: 位图数组，标记非零元素位置
- data: 非零元素值数组
- shape: 矩阵形状

优点:
- 快速定位非零元素
- 适合结构化稀疏
- 内存效率高（对于特定模式）

缺点:
- 不适合随机稀疏
- 位操作开销
"""

from typing import Tuple, Optional, Union, Any, TYPE_CHECKING
import numpy as np

from ..core.sparse_array import FormatBase

if TYPE_CHECKING:
    from .coo import COOFormat


class BitmapFormat(FormatBase):
    """
    Bitmap 存储格式

    使用位图标记非零元素位置。

    存储结构:
    - bitmap: 位图数组，形状 (n_rows, n_cols + 7) // 8
    - data: 非零元素值数组，形状 (nnz,)
    - 按行优先顺序存储

    Example:
        >>> # 创建一个 4x4 对角矩阵
        >>> dense = np.eye(4)
        >>> bitmap = BitmapFormat.from_dense(dense)
        >>> # bitmap存储: 位图标记对角线位置
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 bitmap: np.ndarray,
                 data: np.ndarray,
                 dtype: Optional[np.dtype] = None):
        """
        初始化Bitmap格式

        Args:
            shape: 矩阵形状
            bitmap: 位图数组
            data: 非零元素值数组
            dtype: 数据类型
        """
        super().__init__(shape, dtype or data.dtype)

        self._bitmap = np.asarray(bitmap, dtype=np.uint8)
        self._data = np.asarray(data, dtype=self._dtype)

        # 计算位图每行的字节数
        self._bytes_per_row = (self._shape[1] + 7) // 8

        self._nnz = len(self._data)

        self._validate()

    def _validate(self):
        """验证数据一致性"""
        if len(self._shape) != 2:
            raise ValueError("Bitmap format only supports 2D arrays")

        expected_bitmap_size = self._shape[0] * self._bytes_per_row
        if len(self._bitmap) != expected_bitmap_size:
            raise ValueError(f"bitmap size must be {expected_bitmap_size}")

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> 'BitmapFormat':
        """
        从密集数组创建Bitmap格式

        Args:
            arr: 密集数组

        Returns:
            BitmapFormat: Bitmap格式对象
        """
        n_rows, n_cols = arr.shape
        bytes_per_row = (n_cols + 7) // 8

        # 创建位图
        bitmap = np.zeros(n_rows * bytes_per_row, dtype=np.uint8)

        # 收集非零元素
        data_list = []

        for i in range(n_rows):
            for j in range(n_cols):
                if arr[i, j] != 0:
                    # 设置位
                    byte_idx = i * bytes_per_row + j // 8
                    bit_idx = j % 8
                    bitmap[byte_idx] |= (1 << bit_idx)
                    data_list.append(arr[i, j])

        data = np.array(data_list, dtype=arr.dtype) if data_list else np.array([], dtype=arr.dtype)

        return cls(arr.shape, bitmap, data, arr.dtype)

    @property
    def bitmap(self) -> np.ndarray:
        """获取位图数组"""
        return self._bitmap

    @property
    def data(self) -> np.ndarray:
        """获取数据数组"""
        return self._data

    def _get_bit(self, row: int, col: int) -> bool:
        """
        获取位值

        Args:
            row: 行索引
            col: 列索引

        Returns:
            bool: 位值
        """
        byte_idx = row * self._bytes_per_row + col // 8
        bit_idx = col % 8
        return bool(self._bitmap[byte_idx] & (1 << bit_idx))

    def _set_bit(self, row: int, col: int, value: bool):
        """
        设置位值

        Args:
            row: 行索引
            col: 列索引
            value: 位值
        """
        byte_idx = row * self._bytes_per_row + col // 8
        bit_idx = col % 8
        if value:
            self._bitmap[byte_idx] |= (1 << bit_idx)
        else:
            self._bitmap[byte_idx] &= ~(1 << bit_idx)

    def to_dense(self) -> np.ndarray:
        """转换为密集数组（向量化实现）"""
        result = np.zeros(self._shape, dtype=self._dtype)

        if self._nnz == 0:
            return result

        # 向量化实现：批量处理位图
        # 将位图转换为布尔数组
        n_rows, n_cols = self._shape

        # 创建行和列索引数组
        rows_list = []
        cols_list = []

        # 批量处理每行的位图
        for i in range(n_rows):
            row_start = i * self._bytes_per_row
            row_bitmap = self._bitmap[row_start:row_start + self._bytes_per_row]

            # 解码这一行的位图
            for byte_idx, byte_val in enumerate(row_bitmap):
                if byte_val == 0:
                    continue

                # 检查每个位
                for bit_idx in range(8):
                    col = byte_idx * 8 + bit_idx
                    if col >= n_cols:
                        break
                    if byte_val & (1 << bit_idx):
                        rows_list.append(i)
                        cols_list.append(col)

        # 向量化赋值
        if rows_list:
            result[np.array(rows_list), np.array(cols_list)] = self._data

        return result

    def to_coo(self):
        """转换为COO格式"""
        from .coo import COOFormat

        rows_list = []
        cols_list = []

        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._get_bit(i, j):
                    rows_list.append(i)
                    cols_list.append(j)

        return COOFormat(
            self._shape,
            np.array(rows_list, dtype=np.int32),
            np.array(cols_list, dtype=np.int32),
            self._data.copy(),
            self._dtype
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
                # 边界检查
                if row < 0 or row >= self._shape[0]:
                    raise IndexError(f"Row index {row} out of bounds for shape {self._shape}")

            if isinstance(col, int):
                if col < 0:
                    col += self._shape[1]
                # 边界检查
                if col < 0 or col >= self._shape[1]:
                    raise IndexError(f"Column index {col} out of bounds for shape {self._shape}")

            # 单个元素访问
            if isinstance(row, int) and isinstance(col, int):
                if not self._get_bit(row, col):
                    return self._dtype.type(0)

                # 计算数据索引
                data_idx = 0
                for i in range(row):
                    for j in range(self._shape[1]):
                        if self._get_bit(i, j):
                            data_idx += 1

                for j in range(col):
                    if self._get_bit(row, j):
                        data_idx += 1

                # 边界检查：确保数据索引不越界
                if data_idx >= len(self._data):
                    raise IndexError(
                        f"Data index {data_idx} out of bounds. "
                        f"Bitmap and data array are inconsistent. "
                        f"Expected nnz: {len(self._data)}, computed index: {data_idx}"
                    )

                return self._data[data_idx]

        # 其他情况转换为密集数组
        dense = self.to_dense()
        return dense[key]

    def set_item(self, key, value):
        """设置元素"""
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            if isinstance(row, int) and isinstance(col, int):
                # Bitmap格式不支持高效的元素修改
                # 需要重建数据数组
                raise NotImplementedError(
                    "Bitmap format does not support efficient element assignment. "
                    "Convert to CSR or COO for modification."
                )

        raise TypeError("Only single element assignment is supported")

    def copy(self) -> 'BitmapFormat':
        """深拷贝"""
        return BitmapFormat(
            self._shape,
            self._bitmap.copy(),
            self._data.copy(),
            self._dtype
        )

    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        return self._bitmap.nbytes + self._data.nbytes

    def count_bits_in_row(self, row: int) -> int:
        """
        计算一行中的非零元素数量

        Args:
            row: 行索引

        Returns:
            int: 非零元素数量
        """
        count = 0
        for j in range(self._shape[1]):
            if self._get_bit(row, j):
                count += 1
        return count

    def count_bits_in_col(self, col: int) -> int:
        """
        计算一列中的非零元素数量

        Args:
            col: 列索引

        Returns:
            int: 非零元素数量
        """
        count = 0
        for i in range(self._shape[0]):
            if self._get_bit(i, col):
                count += 1
        return count

    def get_row_indices(self, row: int) -> np.ndarray:
        """
        获取一行的非零列索引

        Args:
            row: 行索引

        Returns:
            np.ndarray: 列索引数组
        """
        indices = []
        for j in range(self._shape[1]):
            if self._get_bit(row, j):
                indices.append(j)
        return np.array(indices, dtype=np.int32)

    def get_col_indices(self, col: int) -> np.ndarray:
        """
        获取一列的非零行索引

        Args:
            col: 列索引

        Returns:
            np.ndarray: 行索引数组
        """
        indices = []
        for i in range(self._shape[0]):
            if self._get_bit(i, col):
                indices.append(i)
        return np.array(indices, dtype=np.int32)

    def detect_pattern(self) -> str:
        """
        检测稀疏模式

        Returns:
            str: 模式类型 ('diagonal', 'banded', 'block', 'random')
        """
        if self._nnz == 0:
            return 'empty'

        # 检测对角线模式
        diagonal_count = 0
        for i in range(min(self._shape)):
            if self._get_bit(i, i):
                diagonal_count += 1

        if diagonal_count == self._nnz:
            return 'diagonal'

        # 检测带状模式
        bandwidth = self._estimate_bandwidth()
        if bandwidth < min(self._shape) * 0.1:
            return 'banded'

        # 检测块模式
        block_score = self._estimate_block_score()
        if block_score > 0.7:
            return 'block'

        return 'random'

    def _estimate_bandwidth(self) -> int:
        """估计带宽"""
        max_band = 0
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._get_bit(i, j):
                    band = abs(j - i)
                    max_band = max(max_band, band)
        return max_band

    def _estimate_block_score(self) -> float:
        """估计块结构评分"""
        # 简单的块检测：检查2x2块
        block_count = 0
        total_blocks = 0

        for i in range(0, self._shape[0] - 1, 2):
            for j in range(0, self._shape[1] - 1, 2):
                total_blocks += 1
                # 检查2x2块是否全非零或全零
                bits = [
                    self._get_bit(i, j),
                    self._get_bit(i, j + 1),
                    self._get_bit(i + 1, j),
                    self._get_bit(i + 1, j + 1)
                ]
                if all(bits) or not any(bits):
                    block_count += 1

        if total_blocks == 0:
            return 0.0
        return block_count / total_blocks
