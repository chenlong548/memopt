"""
特征提取器

提取数据特征用于算法选择。
"""

import logging
from typing import Dict, Any, Union
import math
from collections import Counter

from ...core.base import DataType

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    数据特征提取器

    提取多种特征用于算法选择和性能预测。
    """

    def __init__(self, sample_size: int = 8192):
        """
        初始化特征提取器

        Args:
            sample_size: 采样大小（字节）
        """
        self.sample_size = sample_size

    def extract(self,
               data: Union[bytes, bytearray, memoryview],
               data_type: DataType) -> Dict[str, Any]:
        """
        提取数据特征

        Args:
            data: 待提取特征的数据
            data_type: 数据类型

        Returns:
            Dict: 特征字典
        """
        features = {
            'size': len(data),
            'data_type': data_type.value,
        }

        # 采样
        sample = self._get_sample(data)

        # 基本特征
        features.update(self._extract_basic_features(sample))

        # 统计特征
        features.update(self._extract_statistical_features(sample))

        # 模式特征
        features.update(self._extract_pattern_features(sample))

        # 根据数据类型提取特定特征
        if data_type == DataType.NUMPY_ARRAY:
            features.update(self._extract_numpy_features(data))

        elif data_type in [DataType.BF16_TENSOR, DataType.FP32_TENSOR]:
            features.update(self._extract_tensor_features(data, data_type))

        elif data_type == DataType.KV_CACHE:
            features.update(self._extract_kv_cache_features(data))

        elif data_type == DataType.TEXT:
            features.update(self._extract_text_features(sample))

        # 估算压缩比
        features['estimated_ratio'] = self._estimate_compression_ratio(features)

        return features

    def _get_sample(self, data: Union[bytes, bytearray, memoryview]) -> bytes:
        """获取数据样本"""
        if isinstance(data, memoryview):
            data = bytes(data)

        if len(data) <= self.sample_size:
            return bytes(data)

        # 均匀采样
        step = len(data) // self.sample_size
        sample = bytes(data[::step][:self.sample_size])

        return sample

    def _extract_basic_features(self, sample: bytes) -> Dict[str, Any]:
        """提取基本特征"""
        return {
            'entropy': self._calculate_entropy(sample),
            'redundancy': 0.0,  # 后面计算
            'unique_bytes': len(set(sample)),
            'zero_ratio': sample.count(0) / len(sample) if len(sample) > 0 else 0.0,
        }

    def _extract_statistical_features(self, sample: bytes) -> Dict[str, Any]:
        """提取统计特征"""
        if len(sample) == 0:
            return {}

        # 字节频率分布
        counter = Counter(sample)
        frequencies = list(counter.values())
        total = sum(frequencies)

        # 计算统计量
        mean_freq = total / len(frequencies) if frequencies else 0
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies) if frequencies else 0
        std_dev = math.sqrt(variance)

        # 归一化
        normalized_freqs = [f / total for f in frequencies]

        return {
            'byte_frequency_mean': mean_freq,
            'byte_frequency_std': std_dev,
            'byte_frequency_max': max(frequencies) if frequencies else 0,
            'byte_frequency_min': min(frequencies) if frequencies else 0,
            'frequency_concentration': max(normalized_freqs) if normalized_freqs else 0,
        }

    def _extract_pattern_features(self, sample: bytes) -> Dict[str, Any]:
        """提取模式特征"""
        if len(sample) < 4:
            return {
                'has_repeated_patterns': False,
                'run_length_avg': 0.0,
            }

        # 检测重复模式
        patterns = self._find_repeated_patterns(sample)

        # 计算游程长度
        run_lengths = self._calculate_run_lengths(sample)
        avg_run_length = sum(run_lengths) / len(run_lengths) if run_lengths else 0

        return {
            'has_repeated_patterns': len(patterns) > 0,
            'pattern_count': len(patterns),
            'run_length_avg': avg_run_length,
            'run_length_max': max(run_lengths) if run_lengths else 0,
        }

    def _extract_numpy_features(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """提取NumPy数组特征"""
        try:
            import io
            import numpy as np

            arr = np.load(io.BytesIO(bytes(data)))

            return {
                'array_shape': arr.shape,
                'array_dtype': str(arr.dtype),
                'array_size': arr.size,
                'array_sparsity': 1.0 - (np.count_nonzero(arr) / arr.size),
                'array_value_range': (float(arr.min()), float(arr.max())),
                'array_mean': float(arr.mean()),
                'array_std': float(arr.std()),
            }
        except Exception as e:
            logger.warning(f"Failed to extract NumPy features: {e}")
            return {}

    def _extract_tensor_features(self,
                                data: Union[bytes, bytearray, memoryview],
                                data_type: DataType) -> Dict[str, Any]:
        """提取张量特征"""
        try:
            import struct
            import numpy as np

            data_bytes = bytes(data)

            if data_type == DataType.FP32_TENSOR:
                # FP32: 4字节浮点数
                num_floats = len(data_bytes) // 4
                floats = struct.unpack(f'{num_floats}f', data_bytes[:num_floats*4])

                return {
                    'tensor_dtype': 'float32',
                    'tensor_size': num_floats,
                    'tensor_sparsity': 1.0 - (np.count_nonzero(floats) / num_floats),
                    'tensor_value_range': (float(min(floats)), float(max(floats))),
                    'tensor_mean': float(np.mean(floats)),
                    'tensor_std': float(np.std(floats)),
                }

            elif data_type == DataType.BF16_TENSOR:
                # BF16: 2字节（简化处理）
                return {
                    'tensor_dtype': 'bfloat16',
                    'tensor_size': len(data_bytes) // 2,
                    'tensor_sparsity': self._estimate_bf16_sparsity(data_bytes),
                }

        except Exception as e:
            logger.warning(f"Failed to extract tensor features: {e}")

        return {}

    def _extract_kv_cache_features(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """提取KV Cache特征"""
        data_bytes = bytes(data)

        # 估算层数
        estimated_layers = len(data_bytes) // 1024

        # 估算稀疏度
        sparsity = self._estimate_sparsity(data_bytes)

        return {
            'kv_cache_size': len(data_bytes),
            'estimated_layers': estimated_layers,
            'kv_cache_sparsity': sparsity,
            'compression_potential': 'high' if sparsity > 0.3 else 'medium',
        }

    def _extract_text_features(self, sample: bytes) -> Dict[str, Any]:
        """提取文本特征"""
        try:
            text = sample.decode('utf-8', errors='ignore')

            # 分词
            words = text.split()

            # 字符频率
            char_freq = Counter(text)

            return {
                'text_length': len(text),
                'word_count': len(words),
                'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
                'unique_chars': len(char_freq),
                'char_diversity': len(char_freq) / len(text) if text else 0,
            }
        except Exception:
            return {}

    def _calculate_entropy(self, data: bytes) -> float:
        """计算数据熵"""
        if len(data) == 0:
            return 0.0

        counter = Counter(data)
        total = len(data)

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy

    def _find_repeated_patterns(self, sample: bytes, min_length: int = 4) -> list:
        """查找重复模式"""
        patterns = []
        seen = {}

        for i in range(len(sample) - min_length + 1):
            pattern = sample[i:i + min_length]

            if pattern in seen:
                if pattern not in [p[0] for p in patterns]:
                    patterns.append((pattern, seen[pattern], i))
            else:
                seen[pattern] = i

        return patterns

    def _calculate_run_lengths(self, sample: bytes) -> list:
        """计算游程长度"""
        if len(sample) == 0:
            return []

        run_lengths = []
        current_run = 1

        for i in range(1, len(sample)):
            if sample[i] == sample[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1

        run_lengths.append(current_run)

        return run_lengths

    def _estimate_sparsity(self, data: bytes) -> float:
        """估计数据稀疏度"""
        if len(data) == 0:
            return 0.0

        # 统计零值比例
        zero_count = data.count(0)
        return zero_count / len(data)

    def _estimate_bf16_sparsity(self, data: bytes) -> float:
        """估计BF16张量稀疏度"""
        # 简化实现：检查零字节比例
        zero_count = data.count(0)
        return zero_count / len(data) if len(data) > 0 else 0.0

    def _estimate_compression_ratio(self, features: Dict[str, Any]) -> float:
        """估算压缩比"""
        # 基于特征的简单估算
        entropy = features.get('entropy', 0.0)
        redundancy = features.get('redundancy', 0.0)

        # 计算冗余度
        if entropy > 0:
            redundancy = 1.0 - (entropy / 8.0)

        # 估算压缩比
        # 高冗余度 -> 高压缩比
        # 低冗余度 -> 低压缩比
        estimated_ratio = 1.0 + redundancy * 5.0

        # 根据其他特征调整
        if features.get('has_repeated_patterns', False):
            estimated_ratio *= 1.2

        if features.get('zero_ratio', 0.0) > 0.3:
            estimated_ratio *= 1.3

        return estimated_ratio
