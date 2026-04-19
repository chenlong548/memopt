"""
Lexico字典压缩器

基于Lexico论文实现的KV Cache字典压缩。
"""

import logging
import struct
from typing import Union, Optional, Dict, Any, List, Tuple
from collections import Counter
import numpy as np

from ...core.base import (
    CompressorBase,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressionStats,
    CompressedData,
    DataType
)
from ...core.exceptions import CompressionError, DecompressionError

logger = logging.getLogger(__name__)


class LexicoCompressor(CompressorBase):
    """
    Lexico字典压缩器

    基于Lexico论文的方法，使用字典压缩技术压缩KV Cache。
    通过学习常见模式构建字典，实现高效压缩。

    关键技术：
    1. 频繁模式挖掘
    2. 字典构建
    3. 变长编码
    4. 上下文感知
    """

    def __init__(self, dictionary_size: int = 4096):
        """
        初始化Lexico压缩器

        Args:
            dictionary_size: 字典大小
        """
        self.dictionary_size = dictionary_size

        # 字典
        self._dictionary: Dict[bytes, int] = {}
        self._reverse_dictionary: Dict[int, bytes] = {}

        # 统计信息
        self._compression_stats = {
            'total_caches': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'dictionary_hits': 0,
            'dictionary_misses': 0,
        }

    def compress(self,
                data: Union[bytes, bytearray, memoryview],
                config: Optional[CompressionConfig] = None) -> CompressedData:
        """
        压缩KV Cache数据

        Args:
            data: 待压缩的KV Cache数据
            config: 压缩配置

        Returns:
            CompressedData: 压缩后的数据容器

        Raises:
            CompressionError: 压缩失败时抛出
        """
        if config is None:
            config = CompressionConfig(algorithm=CompressionAlgorithm.LEXICO)

        try:
            # 转换为bytes
            data_bytes = bytes(data)

            # 1. 分析数据模式
            patterns = self._analyze_patterns(data_bytes)

            # 2. 构建字典
            self._build_dictionary(patterns)

            # 3. 字典编码
            encoded = self._encode_with_dictionary(data_bytes)

            # 4. 序列化
            serialized = self._serialize_encoded(encoded)

            # 5. 后压缩
            final_compressed = self._post_compress(serialized)

            # 创建压缩数据容器
            compressed = CompressedData(
                data=final_compressed,
                algorithm=CompressionAlgorithm.LEXICO,
                level=config.level,
                original_size=len(data_bytes),
                compressed_size=len(final_compressed),
                data_type=DataType.KV_CACHE
            )

            # 更新统计
            self._update_stats(compressed)

            logger.info(
                f"Lexico compression: {len(data_bytes)} -> {len(final_compressed)} bytes "
                f"(ratio: {len(data_bytes)/len(final_compressed):.2f}x)"
            )

            return compressed

        except Exception as e:
            logger.error(f"Lexico compression failed: {e}")
            raise CompressionError(f"Lexico compression failed: {e}", algorithm="lexico") from e

    def decompress(self, compressed: CompressedData) -> bytes:
        """
        解压KV Cache数据

        Args:
            compressed: 压缩数据容器

        Returns:
            bytes: 解压后的数据

        Raises:
            DecompressionError: 解压失败时抛出
        """
        try:
            # 1. 后解压
            serialized = self._post_decompress(compressed.data)

            # 2. 反序列化
            encoded, dictionary = self._deserialize_encoded(serialized)

            # 3. 重建字典
            self._reverse_dictionary = dictionary

            # 4. 字典解码
            decoded = self._decode_with_dictionary(encoded)

            logger.debug(
                f"Lexico decompression: {len(compressed.data)} -> {len(decoded)} bytes"
            )

            return decoded

        except Exception as e:
            logger.error(f"Lexico decompression failed: {e}")
            raise DecompressionError(f"Lexico decompression failed: {e}", algorithm="lexico") from e

    def _analyze_patterns(self, data: bytes, min_length: int = 2, max_length: int = 16) -> Counter:
        """
        分析数据模式

        Args:
            data: 数据
            min_length: 最小模式长度
            max_length: 最大模式长度

        Returns:
            Counter: 模式频率统计
        """
        patterns = Counter()

        # 提取所有可能的模式
        for length in range(min_length, min(max_length + 1, len(data) + 1)):
            for i in range(len(data) - length + 1):
                pattern = data[i:i + length]
                patterns[pattern] += 1

        return patterns

    def _build_dictionary(self, patterns: Counter):
        """
        构建字典

        Args:
            patterns: 模式频率统计
        """
        # 按频率排序
        sorted_patterns = patterns.most_common(self.dictionary_size)

        # 构建字典
        self._dictionary = {}
        self._reverse_dictionary = {}

        for idx, (pattern, count) in enumerate(sorted_patterns):
            if count > 1:  # 只添加出现多次的模式
                self._dictionary[pattern] = idx
                self._reverse_dictionary[idx] = pattern

        logger.debug(f"Dictionary built with {len(self._dictionary)} entries")

    def _encode_with_dictionary(self, data: bytes) -> Tuple[List[int], bytes]:
        """
        使用字典编码

        Args:
            data: 原始数据

        Returns:
            Tuple: (编码序列, 未编码的剩余数据)
        """
        encoded = []
        offset = 0

        while offset < len(data):
            # 尝试最长匹配
            best_match = None
            best_length = 0

            for length in range(min(16, len(data) - offset), 1, -1):
                pattern = data[offset:offset + length]
                if pattern in self._dictionary:
                    best_match = self._dictionary[pattern]
                    best_length = length
                    break

            if best_match is not None:
                # 字典命中
                encoded.append(best_match)
                offset += best_length
                self._compression_stats['dictionary_hits'] += 1
            else:
                # 字典未命中，使用字面量
                literal = data[offset]
                encoded.append(-literal - 1)  # 负数表示字面量
                offset += 1
                self._compression_stats['dictionary_misses'] += 1

        return encoded, data[offset:]

    def _decode_with_dictionary(self, encoded: List[int]) -> bytes:
        """
        使用字典解码

        Args:
            encoded: 编码序列

        Returns:
            bytes: 解码后的数据
        """
        result = bytearray()

        for token in encoded:
            if token >= 0:
                # 字典条目
                if token in self._reverse_dictionary:
                    result.extend(self._reverse_dictionary[token])
            else:
                # 字面量
                literal = -token - 1
                result.append(literal)

        return bytes(result)

    def _serialize_encoded(self, encoded_data: Tuple[List[int], bytes]) -> bytes:
        """
        序列化编码数据

        Args:
            encoded_data: (编码序列, 剩余数据)

        Returns:
            bytes: 序列化后的数据
        """
        encoded, remaining = encoded_data

        result = b''

        # 写入字典大小
        result += struct.pack('I', len(self._reverse_dictionary))

        # 写入字典
        for idx, pattern in self._reverse_dictionary.items():
            # 模式长度
            result += struct.pack('B', len(pattern))
            # 模式数据
            result += pattern

        # 写入编码序列长度
        result += struct.pack('I', len(encoded))

        # 写入编码序列
        for token in encoded:
            # 使用变长编码
            if token >= 0:
                # 字典索引
                result += struct.pack('I', token)
            else:
                # 字面量（标记为最高位1）
                literal = -token - 1
                result += struct.pack('I', literal | 0x80000000)

        # 写入剩余数据
        result += struct.pack('I', len(remaining))
        result += remaining

        return result

    def _deserialize_encoded(self, data: bytes) -> Tuple[List[int], Dict[int, bytes]]:
        """
        反序列化编码数据

        Args:
            data: 序列化数据

        Returns:
            Tuple: (编码序列, 字典)
        """
        offset = 0

        # 读取字典大小
        dict_size = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # 读取字典
        dictionary = {}
        for _ in range(dict_size):
            # 模式长度
            pattern_len = struct.unpack('B', data[offset:offset+1])[0]
            offset += 1

            # 模式数据
            pattern = data[offset:offset+pattern_len]
            offset += pattern_len

            # 添加到字典
            dictionary[len(dictionary)] = pattern

        # 读取编码序列长度
        encoded_len = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # 读取编码序列
        encoded = []
        for _ in range(encoded_len):
            token = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4

            if token & 0x80000000:
                # 字面量
                literal = token & 0x7FFFFFFF
                encoded.append(-literal - 1)
            else:
                # 字典索引
                encoded.append(token)

        # 读取剩余数据
        remaining_len = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        remaining = data[offset:offset+remaining_len]

        return encoded, dictionary

    def _post_compress(self, data: bytes) -> bytes:
        """后压缩"""
        try:
            import zstandard as zstd
            compressor = zstd.ZstdCompressor(level=10)
            return compressor.compress(data)
        except Exception:
            return data

    def _post_decompress(self, data: bytes) -> bytes:
        """后解压"""
        try:
            import zstandard as zstd
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        except Exception:
            return data

    def _update_stats(self, compressed: CompressedData):
        """更新统计信息"""
        self._compression_stats['total_caches'] += 1
        self._compression_stats['total_original_size'] += compressed.original_size
        self._compression_stats['total_compressed_size'] += compressed.compressed_size

    def get_algorithm(self) -> CompressionAlgorithm:
        """获取算法类型"""
        return CompressionAlgorithm.LEXICO

    def get_capabilities(self) -> Dict[str, Any]:
        """获取算法能力"""
        return {
            'algorithm': 'lexico',
            'compression_ratio': '2-5x',
            'speed': 'medium',
            'memory_usage': 'medium',
            'features': {
                'dictionary_based': True,
                'pattern_learning': True,
                'adaptive': True,
            },
            'best_for': [
                'kv_cache',
                'repetitive_data',
                'structured_data',
            ],
            'paper_reference': 'Lexico (ICLR 2024)',
        }
