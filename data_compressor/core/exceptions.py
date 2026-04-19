"""
data_compressor 异常定义

定义压缩相关的异常类。
"""


class DataCompressorError(Exception):
    """data_compressor基础异常"""
    pass


class CompressionError(DataCompressorError):
    """压缩错误"""

    def __init__(self, message: str, algorithm: str = None, original_size: int = None):
        self.algorithm = algorithm
        self.original_size = original_size
        super().__init__(message)


class DecompressionError(DataCompressorError):
    """解压错误"""

    def __init__(self, message: str, algorithm: str = None, compressed_size: int = None):
        self.algorithm = algorithm
        self.compressed_size = compressed_size
        super().__init__(message)


class AlgorithmNotFoundError(DataCompressorError):
    """算法未找到错误"""

    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        super().__init__(f"Algorithm not found: {algorithm}")


class UnsupportedDataTypeError(DataCompressorError):
    """不支持的数据类型错误"""

    def __init__(self, data_type: str):
        self.data_type = data_type
        super().__init__(f"Unsupported data type: {data_type}")


class ConfigurationError(DataCompressorError):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message)


class ValidationError(DataCompressorError):
    """数据验证错误"""

    def __init__(self, message: str, data_info: dict = None):
        self.data_info = data_info or {}
        super().__init__(message)


class MemoryLimitError(DataCompressorError):
    """内存限制错误"""

    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Memory limit exceeded: required {required} bytes, "
            f"available {available} bytes"
        )


class StreamError(DataCompressorError):
    """流处理错误"""

    def __init__(self, message: str, stream_state: str = None):
        self.stream_state = stream_state
        super().__init__(message)


class GPUError(DataCompressorError):
    """GPU加速错误"""

    def __init__(self, message: str, device_id: int = None):
        self.device_id = device_id
        super().__init__(message)
