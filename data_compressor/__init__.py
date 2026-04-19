"""
data_compressor 模块

提供高性能的数据压缩功能，支持多种压缩算法和自适应选择。
"""

from .core.base import (
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

from .core.compressor import DataCompressor
from .core.exceptions import (
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

from .detection.type_detector import DataTypeDetector
from .algorithms.adaptive.selector import AdaptiveAlgorithmSelector
from .algorithms.adaptive.feature_extractor import FeatureExtractor
from .stream.stream_compressor import StreamCompressor
from .integration.mem_mapper_integration import (
    MemMapperIntegration,
    CompressedMemoryMapper,
)

__version__ = "1.0.0"

__all__ = [
    # 核心类
    'DataCompressor',
    'StreamCompressor',
    'MemMapperIntegration',
    'CompressedMemoryMapper',

    # 检测器
    'DataTypeDetector',
    'AdaptiveAlgorithmSelector',
    'FeatureExtractor',

    # 枚举类型
    'CompressionAlgorithm',
    'CompressionLevel',
    'CompressionConfig',
    'CompressionStats',
    'CompressedData',
    'DataType',

    # 基类
    'CompressorBase',
    'StreamCompressorBase',
    'DataTypeDetectorBase',
    'AlgorithmSelectorBase',

    # 异常
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


def compress(data: bytes,
            algorithm: str = 'auto',
            level: int = -1,
            **kwargs) -> CompressedData:
    """
    便捷压缩函数

    Args:
        data: 待压缩数据
        algorithm: 压缩算法 ('auto' | 'zstd' | 'lz4' | 'brotli')
        level: 压缩级别 (-1表示自动)
        **kwargs: 其他配置参数

    Returns:
        CompressedData: 压缩后的数据

    Example:
        >>> compressed = compress(b'hello world', algorithm='zstd')
        >>> print(f"Compressed: {len(compressed.data)} bytes")
    """
    # 创建配置
    config = CompressionConfig(
        algorithm=CompressionAlgorithm(algorithm),
        level=CompressionLevel(level) if level > 0 else CompressionLevel.AUTO,
        **kwargs
    )

    # 创建压缩器
    compressor = DataCompressor(config)

    # 执行压缩
    return compressor.compress(data, config)


def decompress(compressed: CompressedData) -> bytes:
    """
    便捷解压函数

    Args:
        compressed: 压缩数据

    Returns:
        bytes: 解压后的数据

    Example:
        >>> data = b'hello world'
        >>> compressed = compress(data)
        >>> decompressed = decompress(compressed)
        >>> assert data == decompressed
    """
    compressor = DataCompressor()
    return compressor.decompress(compressed)


def analyze(data: bytes) -> dict:
    """
    分析数据特征

    Args:
        data: 待分析数据

    Returns:
        dict: 分析结果

    Example:
        >>> analysis = analyze(b'hello world')
        >>> print(f"Data type: {analysis['data_type']}")
        >>> print(f"Recommended algorithm: {analysis['recommended_algorithm']}")
    """
    compressor = DataCompressor()
    return compressor.analyze(data)


def benchmark(data: bytes,
             algorithms: list = None) -> dict:
    """
    基准测试多个算法

    Args:
        data: 测试数据
        algorithms: 要测试的算法列表

    Returns:
        dict: 测试结果

    Example:
        >>> results = benchmark(b'hello world')
        >>> for alg, stats in results.items():
        ...     print(f"{alg}: {stats.compression_ratio:.2f}x")
    """
    compressor = DataCompressor()
    return compressor.benchmark(data, algorithms)


# 添加便捷函数到导出列表
__all__.extend(['compress', 'decompress', 'analyze', 'benchmark'])
