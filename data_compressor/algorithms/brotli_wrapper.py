"""
Brotli压缩算法包装器

封装Brotli压缩库，提供高压缩比。
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

# 尝试导入brotli库
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logger.warning("brotli library not available, Brotli compression disabled")


class BrotliCompressor(CompressorBase):
    """
    Brotli压缩器

    使用Brotli算法进行压缩，提供最佳的压缩比。
    特别适合文本数据和Web内容。
    """

    def __init__(self):
        """初始化Brotli压缩器"""
        if not BROTLI_AVAILABLE:
            raise ImportError("brotli library is required for Brotli compression")

        # 级别映射
        self._level_map = {
            CompressionLevel.FASTEST: 0,
            CompressionLevel.FAST: 2,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.HIGH: 9,
            CompressionLevel.MAXIMUM: 11,   # Brotli最高级别
            CompressionLevel.AUTO: 6,
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
            config = CompressionConfig(algorithm=CompressionAlgorithm.BROTLI)

        # 获取压缩级别
        level = self._level_map.get(config.level, 6)

        try:
            # 执行压缩
            compressed_data = brotli.compress(
                bytes(data),
                quality=level,
                mode=brotli.MODE_GENERIC,  # 通用模式
            )

            # 创建压缩数据容器
            compressed = CompressedData(
                data=compressed_data,
                algorithm=CompressionAlgorithm.BROTLI,
                level=config.level,
                original_size=len(data),
                compressed_size=len(compressed_data),
                data_type=config.data_type
            )

            logger.debug(
                f"Brotli compression: {len(data)} -> {len(compressed_data)} bytes "
                f"(ratio: {len(data)/len(compressed_data):.2f}x, level: {level})"
            )

            return compressed

        except Exception as e:
            logger.error(f"Brotli compression failed: {e}")
            raise CompressionError(f"Brotli compression failed: {e}", algorithm="brotli") from e

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
            decompressed = brotli.decompress(compressed.data)

            # 验证大小
            if len(decompressed) != compressed.original_size:
                logger.warning(
                    f"Decompressed size mismatch: expected {compressed.original_size}, "
                    f"got {len(decompressed)}"
                )

            logger.debug(
                f"Brotli decompression: {len(compressed.data)} -> {len(decompressed)} bytes"
            )

            return decompressed

        except Exception as e:
            logger.error(f"Brotli decompression failed: {e}")
            raise DecompressionError(f"Brotli decompression failed: {e}", algorithm="brotli") from e

    def get_algorithm(self) -> CompressionAlgorithm:
        """
        获取算法类型

        Returns:
            CompressionAlgorithm: 算法类型
        """
        return CompressionAlgorithm.BROTLI

    def get_capabilities(self) -> dict:
        """
        获取算法能力

        Returns:
            Dict: 算法能力描述
        """
        return {
            'algorithm': 'brotli',
            'compression_ratio': '2-6x',
            'speed': 'slow',
            'memory_usage': 'high',
            'supported_levels': list(range(12)),  # 0-11
            'features': {
                'streaming': True,
                'text_mode': True,
                'window_size': True,
            },
            'best_for': [
                'text_data',
                'web_content',
                'json',
                'static_assets',
            ]
        }
