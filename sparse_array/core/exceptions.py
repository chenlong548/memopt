"""
sparse_array 异常定义

定义稀疏数组相关的异常类。
"""


class SparseArrayError(Exception):
    """sparse_array基础异常"""
    pass


class FormatConversionError(SparseArrayError):
    """格式转换错误"""

    def __init__(self, message: str, source_format: str = "", target_format: str = ""):
        self.source_format = source_format
        self.target_format = target_format
        super().__init__(message)


class UnsupportedOperationError(SparseArrayError):
    """不支持的操作错误"""

    def __init__(self, message: str, operation: str = "", format_type: str = ""):
        self.operation = operation
        self.format_type = format_type
        super().__init__(message)


class DimensionError(SparseArrayError):
    """维度错误"""

    def __init__(self, message: str, expected_dim: int = 0, actual_dim: int = 0):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(message)


class IndexOutOfBoundsError(SparseArrayError):
    """索引越界错误"""

    def __init__(self, message: str, index: tuple = (), shape: tuple = ()):
        self.index = index
        self.shape = shape
        super().__init__(message)


class GPUError(SparseArrayError):
    """GPU加速错误"""

    def __init__(self, message: str, device_id: int = 0, operation: str = ""):
        self.device_id = device_id
        self.operation = operation
        super().__init__(message)


class CompressionError(SparseArrayError):
    """压缩错误"""

    def __init__(self, message: str, compression_type: str = ""):
        self.compression_type = compression_type
        super().__init__(message)


class FormatSelectionError(SparseArrayError):
    """格式选择错误"""

    def __init__(self, message: str, available_formats: list = []):
        self.available_formats = available_formats
        super().__init__(message)


class MemoryLimitError(SparseArrayError):
    """内存限制错误"""

    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Memory limit exceeded: required {required} bytes, "
            f"available {available} bytes"
        )


class InvalidSparseDataError(SparseArrayError):
    """无效稀疏数据错误"""

    def __init__(self, message: str, data_info: dict = {}):
        self.data_info = data_info
        super().__init__(message)
