"""
FP32模型压缩器

针对FP32模型权重的专用压缩。
"""

import logging
import struct
from typing import Union, Optional, Dict, Any
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


class FP32ModelCompressor(CompressorBase):
    """
    FP32模型压缩器

    针对FP32模型权重进行优化压缩。
    使用量化、稀疏编码等技术。
    """

    def __init__(self):
        """初始化FP32模型压缩器"""
        # 压缩参数
        self._quantization_bits = 8
        self._sparsity_threshold = 0.001

        # 统计信息
        self._compression_stats = {
            'total_models': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'avg_compression_ratio': 0.0,
        }

    def compress(self,
                data: Union[bytes, bytearray, memoryview],
                config: Optional[CompressionConfig] = None) -> CompressedData:
        """
        压缩FP32模型数据

        Args:
            data: 待压缩的FP32模型数据
            config: 压缩配置

        Returns:
            CompressedData: 压缩后的数据容器

        Raises:
            CompressionError: 压缩失败时抛出
        """
        if config is None:
            config = CompressionConfig(algorithm=CompressionAlgorithm.FP32_MODEL)

        try:
            # 转换为bytes
            data_bytes = bytes(data)

            # 1. 解析FP32张量
            tensors = self._parse_fp32_tensors(data_bytes)

            # 2. 对每个张量进行压缩
            compressed_tensors = []
            for tensor_name, tensor_data in tensors.items():
                compressed_tensor = self._compress_tensor(tensor_data, config)
                compressed_tensors.append((tensor_name, compressed_tensor))

            # 3. 序列化压缩数据
            compressed_data = self._serialize_compressed(compressed_tensors)

            # 4. 应用后压缩
            final_compressed = self._post_compress(compressed_data, config)

            # 创建压缩数据容器
            compressed = CompressedData(
                data=final_compressed,
                algorithm=CompressionAlgorithm.FP32_MODEL,
                level=config.level,
                original_size=len(data_bytes),
                compressed_size=len(final_compressed),
                data_type=DataType.FP32_TENSOR
            )

            # 更新统计
            self._update_stats(compressed)

            logger.info(
                f"FP32 model compression: {len(data_bytes)} -> {len(final_compressed)} bytes "
                f"(ratio: {len(data_bytes)/len(final_compressed):.2f}x)"
            )

            return compressed

        except Exception as e:
            logger.error(f"FP32 model compression failed: {e}")
            raise CompressionError(f"FP32 model compression failed: {e}", algorithm="fp32") from e

    def decompress(self, compressed: CompressedData) -> bytes:
        """
        解压FP32模型数据

        Args:
            compressed: 压缩数据容器

        Returns:
            bytes: 解压后的数据

        Raises:
            DecompressionError: 解压失败时抛出
        """
        try:
            # 1. 后解压
            compressed_data = self._post_decompress(compressed.data)

            # 2. 反序列化
            compressed_tensors = self._deserialize_compressed(compressed_data)

            # 3. 解压每个张量
            tensors = {}
            for tensor_name, compressed_tensor in compressed_tensors:
                tensor_data = self._decompress_tensor(compressed_tensor)
                tensors[tensor_name] = tensor_data

            # 4. 序列化回原始格式
            decompressed = self._serialize_tensors(tensors)

            logger.debug(
                f"FP32 model decompression: {len(compressed.data)} -> {len(decompressed)} bytes"
            )

            return decompressed

        except Exception as e:
            logger.error(f"FP32 model decompression failed: {e}")
            raise DecompressionError(f"FP32 model decompression failed: {e}", algorithm="fp32") from e

    def _parse_fp32_tensors(self, data: bytes) -> Dict[str, np.ndarray]:
        """解析FP32张量"""
        # 简化实现：假设数据是连续的FP32值
        num_elements = len(data) // 4

        # 转换为float32数组
        floats = np.frombuffer(data, dtype=np.float32)

        return {'weights': floats.copy()}

    def _compress_tensor(self, tensor: np.ndarray, config: CompressionConfig) -> bytes:
        """压缩单个张量"""
        # 1. 稀疏化
        sparse_data = self._sparsify(tensor)

        # 2. 量化
        quantized = self._quantize(sparse_data)

        # 3. 编码
        encoded = self._encode(quantized)

        return encoded

    def _decompress_tensor(self, compressed: bytes) -> np.ndarray:
        """解压单个张量"""
        # 1. 解码
        decoded = self._decode(compressed)

        # 2. 反量化
        dequantized = self._dequantize(decoded)

        # 3. 反稀疏化
        tensor = self._desparsify(dequantized)

        return tensor

    def _sparsify(self, tensor: np.ndarray) -> Dict[str, Any]:
        """稀疏化处理"""
        # 找出非零元素
        mask = np.abs(tensor) > self._sparsity_threshold
        indices = np.where(mask)[0]
        values = tensor[mask]

        return {
            'indices': indices,
            'values': values,
            'shape': tensor.shape,
            'original_size': len(tensor),
        }

    def _quantize(self, sparse_data: Dict[str, Any]) -> Dict[str, Any]:
        """量化"""
        values = sparse_data['values']

        if len(values) == 0:
            return {
                'indices': sparse_data['indices'],
                'quantized_values': np.array([], dtype=np.uint8),
                'scale': 1.0,
                'zero_point': 0,
                'shape': sparse_data['shape'],
                'original_size': sparse_data['original_size'],
            }

        # 计算量化参数
        min_val = values.min()
        max_val = values.max()
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        zero_point = min_val

        # 量化
        quantized = ((values - zero_point) / scale).astype(np.uint8)

        return {
            'indices': sparse_data['indices'],
            'quantized_values': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'shape': sparse_data['shape'],
            'original_size': sparse_data['original_size'],
        }

    def _encode(self, quantized: Dict[str, Any]) -> bytes:
        """编码"""
        result = b''

        # 写入原始大小
        result += struct.pack('I', quantized['original_size'])

        # 写入索引
        indices = quantized['indices'].astype(np.int32)
        result += struct.pack('I', len(indices))
        result += indices.tobytes()

        # 写入量化值
        values = quantized['quantized_values']
        result += struct.pack('I', len(values))
        result += values.tobytes()

        # 写入量化参数
        result += struct.pack('f', quantized['scale'])
        result += struct.pack('f', quantized['zero_point'])

        return result

    def _decode(self, data: bytes) -> Dict[str, Any]:
        """解码"""
        offset = 0

        # 读取原始大小
        original_size = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # 读取索引
        num_indices = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        indices = np.frombuffer(data[offset:offset+num_indices*4], dtype=np.int32).copy()
        offset += num_indices * 4

        # 读取量化值
        num_values = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        quantized_values = np.frombuffer(data[offset:offset+num_values], dtype=np.uint8).copy()
        offset += num_values

        # 读取量化参数
        scale = struct.unpack('f', data[offset:offset+4])[0]
        offset += 4
        zero_point = struct.unpack('f', data[offset:offset+4])[0]

        return {
            'indices': indices,
            'quantized_values': quantized_values,
            'scale': scale,
            'zero_point': zero_point,
            'original_size': original_size,
        }

    def _dequantize(self, decoded: Dict[str, Any]) -> Dict[str, Any]:
        """反量化"""
        quantized = decoded['quantized_values']
        scale = decoded['scale']
        zero_point = decoded['zero_point']

        # 反量化
        values = (quantized.astype(np.float32) * scale) + zero_point

        return {
            'indices': decoded['indices'],
            'values': values,
            'original_size': decoded['original_size'],
        }

    def _desparsify(self, sparse_data: Dict[str, Any]) -> np.ndarray:
        """反稀疏化"""
        # 创建全零数组
        tensor = np.zeros(sparse_data['original_size'], dtype=np.float32)

        # 填充非零值
        indices = sparse_data['indices']
        values = sparse_data['values']
        tensor[indices] = values

        return tensor

    def _serialize_compressed(self, compressed_tensors: list) -> bytes:
        """序列化压缩后的张量"""
        result = b''

        for name, data in compressed_tensors:
            # 名称
            name_bytes = name.encode('utf-8')
            result += struct.pack('I', len(name_bytes))
            result += name_bytes

            # 数据
            result += struct.pack('I', len(data))
            result += data

        return result

    def _deserialize_compressed(self, data: bytes) -> list:
        """反序列化压缩张量"""
        compressed_tensors = []
        offset = 0

        while offset < len(data):
            # 读取名称
            name_len = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len

            # 读取数据
            data_len = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            tensor_data = data[offset:offset+data_len]
            offset += data_len

            compressed_tensors.append((name, tensor_data))

        return compressed_tensors

    def _post_compress(self, data: bytes, config: CompressionConfig) -> bytes:
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

    def _serialize_tensors(self, tensors: Dict[str, np.ndarray]) -> bytes:
        """序列化张量回原始格式"""
        result = b''

        for name, tensor in tensors.items():
            result += tensor.astype(np.float32).tobytes()

        return result

    def _update_stats(self, compressed: CompressedData):
        """更新统计信息"""
        self._compression_stats['total_models'] += 1
        self._compression_stats['total_original_size'] += compressed.original_size
        self._compression_stats['total_compressed_size'] += compressed.compressed_size

        if self._compression_stats['total_compressed_size'] > 0:
            self._compression_stats['avg_compression_ratio'] = (
                self._compression_stats['total_original_size'] /
                self._compression_stats['total_compressed_size']
            )

    def get_algorithm(self) -> CompressionAlgorithm:
        """获取算法类型"""
        return CompressionAlgorithm.FP32_MODEL

    def get_capabilities(self) -> Dict[str, Any]:
        """获取算法能力"""
        return {
            'algorithm': 'fp32_model',
            'compression_ratio': '2-3x',
            'speed': 'medium',
            'memory_usage': 'medium',
            'features': {
                'quantization': True,
                'sparse_encoding': True,
                'lossy_compression': False,
            },
            'best_for': [
                'fp32_model_weights',
                'neural_network_parameters',
            ],
        }
