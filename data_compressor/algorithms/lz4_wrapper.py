"""
LZ4压缩算法包装器

封装LZ4压缩库，提供极速压缩。
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

# 尝试导入lz4库
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 library not available, LZ4 compression disabled")


class LZ4Compressor(CompressorBase):
    """
    LZ4压缩器

    使用LZ4算法进行压缩，提供最快的压缩速度。
    """

    def __init__(self):
        """初始化LZ4压缩器"""
        if not LZ4_AVAILABLE:
            raise ImportError("lz4 library is required for LZ4 compression")

        # 级别映射
        self._level_map = {
            CompressionLevel.FASTEST: 0,    # 最快
            CompressionLevel.FAST: 1,
            CompressionLevel.BALANCED: 3,
            CompressionLevel.HIGH: 6,
            CompressionLevel.MAXIMUM: 9,    # 最高压缩
            CompressionLevel.AUTO: 3,
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
            config = CompressionConfig(algorithm=CompressionAlgorithm.LZ4)

        # 获取压缩级别
        level = self._level_map.get(config.level, 3)

        try:
            # 执行压缩
            compressed_data = lz4.frame.compress(
                bytes(data),
                compression_level=level,
                block_size=lz4.frame.BLOCKSIZE_DEFAULT,
                content_checksum=True,
            )

            # 创建压缩数据容器
            compressed = CompressedData(
                data=compressed_data,
                algorithm=CompressionAlgorithm.LZ4,
                level=config.level,
                original_size=len(data),
                compressed_size=len(compressed_data),
                data_type=config.data_type
            )

            logger.debug(
                f"LZ4 compression: {len(data)} -> {len(compressed_data)} bytes "
                f"(ratio: {len(data)/len(compressed_data):.2f}x, level: {level})"
            )

            return compressed

        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            raise CompressionError(f"LZ4 compression failed: {e}", algorithm="lz4") from e

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
            # 执行解压
            decompressed = lz4.frame.decompress(compressed.data)

            # 验证大小
            if len(decompressed) != compressed.original_size:
                logger.warning(
                    f"Decompressed size mismatch: expected {compressed.original_size}, "
                    f"got {len(decompressed)}"
                )

            logger.debug(
                f"LZ4 decompression: {len(compressed.data)} -> {len(decompressed)} bytes"
            )

            return decompressed

        except Exception as e:
            logger.error(f"LZ4 decompression failed: {e}")
            raise DecompressionError(f"LZ4 decompression failed: {e}", algorithm="lz4") from e

    def get_algorithm(self) -> CompressionAlgorithm:
        """
        获取算法类型

        Returns:
            CompressionAlgorithm: 算法类型
        """
        return CompressionAlgorithm.LZ4

    def get_capabilities(self) -> dict:
        """
        获取算法能力

        Returns:
            Dict: 算法能力描述
        """
        return {
            'algorithm': 'lz4',
            'compression_ratio': '1.5-3x',
            'speed': 'very_fast',
            'memory_usage': 'low',
            'supported_levels': [0, 1, 3, 6, 9],
            'features': {
                'streaming': True,
                'block_mode': True,
                'content_checksum': True,
            },
            'best_for': [
                'real_time_compression',
                'high_throughput',
                'large_files',
                'network_transfer',
            ]
        }
