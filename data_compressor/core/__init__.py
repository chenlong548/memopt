"""
核心模块
"""

from .base import (
    CompressionAlgorithm,
    CompressionLevel,
    CompressionConfig,
    CompressionStats,
    CompressedData,
    DataType,
    CompressorBase,
    StreamCompressorBase,
    DataTypeDetectorBase,
    AlgorithmSelectorBase,
)

from .compressor import DataCompressor
from .exceptions import (
    DataCompressorError,
    CompressionError,
    DecompressionError,
    AlgorithmNotFoundError,
    UnsupportedDataTypeError,
    ConfigurationError,
    ValidationError,
    MemoryLimitError,
    StreamError,
    GPUError,
)

__all__ = [
    'CompressionAlgorithm',
    'CompressionLevel',
    'CompressionConfig',
    'CompressionStats',
    'CompressedData',
    'DataType',
    'CompressorBase',
    'StreamCompressorBase',
    'DataTypeDetectorBase',
    'AlgorithmSelectorBase',
    'DataCompressor',
    'DataCompressorError',
    'CompressionError',
    'DecompressionError',
    'AlgorithmNotFoundError',
    'UnsupportedDataTypeError',
    'ConfigurationError',
    'ValidationError',
    'MemoryLimitError',
    'StreamError',
    'GPUError',
]
