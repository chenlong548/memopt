"""
ZSTD压缩算法包装器

封装Zstandard压缩库，提供高性能压缩。
"""

import logging
from typing import Union, Optional

from ..core.base import (
    CompressorBase,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressionStats,
    CompressedData,
    DataType
)
from ..core.exceptions import CompressionError, DecompressionError

logger = logging.getLogger(__name__)

# 尝试导入zstandard库
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard library not available, ZSTD compression disabled")


class ZstdCompressor(CompressorBase):
    """
    ZSTD压缩器

    使用Zstandard算法进行压缩，提供良好的压缩比和速度平衡。
    """

    def __init__(self):
        """初始化ZSTD压缩器"""
        if not ZSTD_AVAILABLE:
            raise ImportError("zstandard library is required for ZSTD compression")

        # 级别映射
        self._level_map = {
            CompressionLevel.FASTEST: 1,
            CompressionLevel.FAST: 3,
            CompressionLevel.BALANCED: 10,  # ZipNN推荐的平衡级别
            CompressionLevel.HIGH: 15,
            CompressionLevel.MAXIMUM: 19,   # ZipNN推荐的最大级别
            CompressionLevel.AUTO: 10,
        }

    def compress(self,
                data: Union[bytes, bytearray, memoryview],
                config: Optional[CompressionConfig] = None) -> CompressedData:
        """
        压缩数据

        Args:
            data: 待压缩数据
            config: 压缩配置

        Returns:
            CompressedData: 压缩后的数据容器

        Raises:
            CompressionError: 压缩失败时抛出
        """
        if config is None:
            config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD)

        # 获取压缩级别
        level = self._level_map.get(config.level, 10)

        try:
            # 创建压缩器
            compressor = zstd.ZstdCompressor(level=level)

            # 执行压缩
            compressed_data = compressor.compress(bytes(data))

            # 创建压缩数据容器
            compressed = CompressedData(
                data=compressed_data,
                algorithm=CompressionAlgorithm.ZSTD,
                level=config.level,
                original_size=len(data),
                compressed_size=len(compressed_data),
                data_type=config.data_type
            )

            logger.debug(
                f"ZSTD compression: {len(data)} -> {len(compressed_data)} bytes "
                f"(ratio: {len(data)/len(compressed_data):.2f}x, level: {level})"
            )

            return compressed

        except Exception as e:
            logger.error(f"ZSTD compression failed: {e}")
            raise CompressionError(f"ZSTD compression failed: {e}", algorithm="zstd") from e

    def decompress(self, compressed: CompressedData) -> bytes:
        """
        解压数据

        Args:
            compressed: 压缩数据容器

        Returns:
            bytes: 解压后的数据

        Raises:
            DecompressionError: 解压失败时抛出
        """
        try:
            # 创建解压器
            decompressor = zstd.ZstdDecompressor()

            # 执行解压
            decompressed = decompressor.decompress(compressed.data)

            # 验证大小
            if len(decompressed) != compressed.original_size:
                logger.warning(
                    f"Decompressed size mismatch: expected {compressed.original_size}, "
                    f"got {len(decompressed)}"
                )

            logger.debug(
                f"ZSTD decompression: {len(compressed.data)} -> {len(decompressed)} bytes"
            )

            return decompressed

        except Exception as e:
            logger.error(f"ZSTD decompression failed: {e}")
            raise DecompressionError(f"ZSTD decompression failed: {e}", algorithm="zstd") from e

    def get_algorithm(self) -> CompressionAlgorithm:
        """
        获取算法类型

        Returns:
            CompressionAlgorithm: 算法类型
        """
        return CompressionAlgorithm.ZSTD

    def get_capabilities(self) -> dict:
        """
        获取算法能力

        Returns:
            Dict: 算法能力描述
        """
        return {
            'algorithm': 'zstd',
            'compression_ratio': '2-5x',
            'speed': 'fast',
            'memory_usage': 'medium',
            'supported_levels': [1, 3, 10, 15, 19],
            'features': {
                'streaming': True,
                'dictionary': True,
                'multi_threading': True,
            },
            'best_for': [
                'general_purpose',
                'model_weights',
                'numpy_arrays',
                'text_data',
            ]
        }

    def compress_stream(self,
                       input_stream,
                       output_stream,
                       config: Optional[CompressionConfig] = None):
        """
        流式压缩

        Args:
            input_stream: 输入流
            output_stream: 输出流
            config: 压缩配置
        """
        if config is None:
            config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD)

        level = self._level_map.get(config.level, 10)

        try:
            compressor = zstd.ZstdCompressor(level=level)
            compressor.copy_stream(input_stream, output_stream)

        except Exception as e:
            logger.error(f"ZSTD stream compression failed: {e}")
            raise CompressionError(f"ZSTD stream compression failed: {e}", algorithm="zstd") from e

    def decompress_stream(self, input_stream, output_stream):
        """
        流式解压

        Args:
            input_stream: 输入流
            output_stream: 输出流
        """
        try:
            decompressor = zstd.ZstdDecompressor()
            decompressor.copy_stream(input_stream, output_stream)

        except Exception as e:
            logger.error(f"ZSTD stream decompression failed: {e}")
            raise DecompressionError(f"ZSTD stream decompression failed: {e}", algorithm="zstd") from e
