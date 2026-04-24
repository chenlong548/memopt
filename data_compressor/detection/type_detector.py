"""
数据类型检测器

自动检测数据类型并分析数据特征。
"""

import struct
import logging
from typing import Union, Dict, Any, Optional
import numpy as np

from ..core.base import DataType

logger = logging.getLogger(__name__)


class DataTypeDetector:
    """
    数据类型检测器

    通过多种启发式方法自动检测数据类型。
    """

    def __init__(self):
        """初始化检测器"""
        # 数据类型签名
        self._signatures = {
            # NumPy数组签名
            b'\x93NUMPY': DataType.NUMPY_ARRAY,
            # JSON签名
            b'{': DataType.JSON,
            b'[': DataType.JSON,
            # 文本签名（UTF-8 BOM）
            b'\xef\xbb\xbf': DataType.TEXT,
        }

        # 模型权重签名（常见框架）
        self._model_signatures = {
            b'PK': 'zip',           # ZIP格式（常见于模型文件）
            b'\x89HDF': 'hdf5',     # HDF5格式
            b'\x1f\x8b': 'gzip',    # GZIP格式
            b'PyTorch': 'pytorch',  # PyTorch模型
            b'TensorFlow': 'tf',    # TensorFlow模型
        }

    def detect(self, data: Union[bytes, bytearray, memoryview]) -> DataType:
        """
        检测数据类型

        Args:
            data: 待检测数据

        Returns:
            DataType: 检测到的数据类型
        """
        # 转换为bytes以便检查
        if isinstance(data, memoryview):
            data_bytes = bytes(data[:1024])  # 只检查前1KB
        else:
            data_bytes = bytes(data[:1024])

        # 1. 检查签名
        detected_type = self._check_signature(data_bytes)
        if detected_type != DataType.GENERIC:
            return detected_type

        # 2. 检查是否为NumPy数组
        if self._is_numpy_array(data):
            return DataType.NUMPY_ARRAY

        # 3. 检查是否为模型权重
        if self._is_model_weights(data_bytes):
            return DataType.MODEL_WEIGHTS

        # 4. 检查是否为KV Cache
        if self._is_kv_cache(data):
            return DataType.KV_CACHE

        # 5. 检查是否为稀疏矩阵
        if self._is_sparse_matrix(data):
            return DataType.SPARSE_MATRIX

        # 6. 检查是否为文本
        if self._is_text(data_bytes):
            return DataType.TEXT

        # 7. 检查是否为JSON
        if self._is_json(data_bytes):
            return DataType.JSON

        # 8. 检查是否为BF16/FP32张量
        tensor_type = self._detect_tensor_type(data)
        if tensor_type != DataType.GENERIC:
            return tensor_type

        # 默认返回二进制类型（而非通用类型）
        return DataType.BINARY

    def analyze(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """
        分析数据特征

        Args:
            data: 待分析数据

        Returns:
            Dict: 数据特征分析结果
        """
        data_type = self.detect(data)

        analysis = {
            'data_type': data_type,
            'size': len(data),
            'entropy': self._calculate_entropy(data),
            'redundancy': 0.0,
            'patterns': [],
            'recommendations': []
        }

        # 计算冗余度
        if analysis['entropy'] > 0:
            max_entropy = 8.0  # 最大熵（字节）
            analysis['redundancy'] = 1.0 - (analysis['entropy'] / max_entropy)

        # 根据数据类型添加特定分析
        if data_type == DataType.NUMPY_ARRAY:
            analysis.update(self._analyze_numpy_array(data))

        elif data_type == DataType.MODEL_WEIGHTS:
            analysis.update(self._analyze_model_weights(data))

        elif data_type == DataType.KV_CACHE:
            analysis.update(self._analyze_kv_cache(data))

        elif data_type == DataType.TEXT:
            analysis.update(self._analyze_text(data))

        return analysis

    def _check_signature(self, data: bytes) -> DataType:
        """检查数据签名"""
        for signature, data_type in self._signatures.items():
            if data.startswith(signature):
                return data_type
        return DataType.GENERIC

    def _is_numpy_array(self, data: Union[bytes, bytearray, memoryview]) -> bool:
        """检查是否为NumPy数组"""
        try:
            if isinstance(data, (bytes, bytearray)):
                return data[:6] == b'\x93NUMPY'
            return False
        except Exception:
            return False

    def _is_model_weights(self, data: bytes) -> bool:
        """检查是否为模型权重"""
        for signature in self._model_signatures.values():
            if data.startswith(signature.encode() if isinstance(signature, str) else signature):
                return True

        # 检查是否包含模型相关的魔数
        # PyTorch模型通常以PK开头（ZIP格式）
        if data[:2] == b'PK':
            return True

        return False

    def _is_kv_cache(self, data: Union[bytes, bytearray, memoryview]) -> bool:
        """
        检查是否为KV Cache

        KV Cache通常具有特定的结构：
        - 包含key和value对
        - 数据按层组织
        - 通常有特定的头部信息
        - 需要更严格的特征检测
        """
        try:
            if len(data) < 1024:
                return False

            sample_size = min(len(data), 4096)
            sample = bytes(data[:sample_size])

            # KV Cache特征检测需要多个条件同时满足
            # 1. 数据长度必须是特定块大小的倍数
            chunk_size = 256
            if len(data) % chunk_size != 0:
                return False

            # 2. 检查是否有层结构的特征（重复模式）
            # KV Cache通常有明显的层边界
            num_chunks = len(data) // chunk_size
            if num_chunks < 2:
                return False

            # 3. 检查数据是否有浮点数特征（KV Cache通常是FP16/BF16）
            # 采样检查是否为合理的浮点数范围
            try:
                import struct
                import math
                num_floats = min(sample_size // 2, 64)
                half_floats = struct.unpack(f'{num_floats}e', sample[:num_floats * 2])

                # 检查是否为合理的浮点数（不是全零或极端值）
                non_zero_floats = [f for f in half_floats if abs(f) > 1e-10]
                if len(non_zero_floats) < num_floats * 0.1:
                    return False

                # 检查数值范围是否合理（模型权重通常在-100到100之间）
                if not all(-1e6 < f < 1e6 for f in half_floats if not math.isnan(f)):
                    return False
            except Exception:
                return False

            # 4. 检查熵值（KV Cache通常有中等熵值）
            entropy = self._calculate_entropy(sample)
            if entropy < 4.0 or entropy > 7.5:
                return False

            return True

        except Exception:
            return False

    def _is_sparse_matrix(self, data: Union[bytes, bytearray, memoryview]) -> bool:
        """检查是否为稀疏矩阵"""
        try:
            # 检查是否为CSR/CSC格式的特征
            # 稀疏矩阵通常有indices, indptr, data三个数组

            # 简化检查：检查数据密度
            if len(data) < 1024:
                return False

            # 统计零值比例
            sample = bytes(data[:4096])
            zero_count = sample.count(b'\x00')
            zero_ratio = zero_count / len(sample)

            # 如果零值比例很高，可能是稀疏矩阵
            return zero_ratio > 0.5

        except Exception:
            return False

    def _is_text(self, data: bytes) -> bool:
        """检查是否为文本数据"""
        try:
            # 尝试解码为UTF-8
            decoded = data.decode('utf-8')

            # 检查是否包含常见文本字符
            text_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t')
            if not text_chars:
                return False

            # 计算可打印字符比例
            printable_ratio = sum(1 for c in decoded if c in text_chars) / len(decoded)
            return printable_ratio > 0.7

        except Exception:
            return False

    def _is_json(self, data: bytes) -> bool:
        """检查是否为JSON数据"""
        try:
            import json
            json.loads(data)
            return True
        except Exception:
            return False

    def _detect_tensor_type(self, data: Union[bytes, bytearray, memoryview]) -> DataType:
        """检测张量类型（BF16/FP32）"""
        try:
            data_len = len(data)
            
            if data_len < 64:
                return DataType.GENERIC

            sample_size = min(data_len, 1024)
            sample = bytes(data[:sample_size])

            if data_len % 4 == 0:
                num_floats = sample_size // 4
                floats = struct.unpack(f'{num_floats}f', sample[:num_floats * 4])

                import math
                valid_floats = [f for f in floats if not math.isnan(f) and not math.isinf(f)]
                if len(valid_floats) > 0:
                    if all(-1e10 < f < 1e10 for f in valid_floats):
                        non_zero = [abs(f) for f in valid_floats if abs(f) > 1e-10]
                        if len(non_zero) > len(valid_floats) * 0.3:
                            mean_val = sum(non_zero) / len(non_zero) if non_zero else 0
                            variance = sum((abs(f) - mean_val) ** 2 for f in non_zero) / len(non_zero)
                            if variance < 1e20:
                                return DataType.FP32_TENSOR

            if data_len % 2 == 0:
                import math
                num_halfs = sample_size // 2
                half_floats = struct.unpack(f'{num_halfs}e', sample[:num_halfs * 2])
                
                valid_halfs = [f for f in half_floats if not math.isnan(f) and not math.isinf(f)]
                if len(valid_halfs) > 0:
                    if all(-1e10 < f < 1e10 for f in valid_halfs):
                        non_zero = [abs(f) for f in valid_halfs if abs(f) > 1e-10]
                        if len(non_zero) > len(valid_halfs) * 0.3:
                            entropy = self._calculate_entropy(sample)
                            if 4.0 < entropy < 7.5:
                                mean_val = sum(non_zero) / len(non_zero) if non_zero else 0
                                variance = sum((abs(f) - mean_val) ** 2 for f in non_zero) / len(non_zero)
                                if variance < 1e20:
                                    return DataType.BF16_TENSOR

            return DataType.GENERIC

        except Exception:
            return DataType.GENERIC

    def _calculate_entropy(self, data: Union[bytes, bytearray, memoryview]) -> float:
        """计算数据熵"""
        try:
            import math
            from collections import Counter

            # 统计字节频率
            if isinstance(data, memoryview):
                byte_data = bytes(data)
            else:
                byte_data = data

            counter = Counter(byte_data)
            total = len(byte_data)

            # 计算熵
            entropy = 0.0
            for count in counter.values():
                if count > 0:
                    probability = count / total
                    entropy -= probability * math.log2(probability)

            return entropy

        except Exception:
            return 0.0

    def _analyze_numpy_array(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """分析NumPy数组特征"""
        try:
            import io
            arr = np.load(io.BytesIO(bytes(data)))

            return {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'size': arr.size,
                'is_sparse': self._check_sparsity(arr),
                'value_range': (float(arr.min()), float(arr.max())),
            }
        except Exception:
            return {}

    def _analyze_model_weights(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """分析模型权重特征"""
        # 简化实现
        return {
            'format': 'unknown',
            'estimated_layers': 0,
            'compression_potential': 'high'
        }

    def _analyze_kv_cache(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """分析KV Cache特征"""
        return {
            'estimated_layers': len(data) // 1024,  # 粗略估计
            'compression_potential': 'high',
            'sparsity': self._estimate_sparsity(data)
        }

    def _analyze_text(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """分析文本特征"""
        try:
            text = bytes(data).decode('utf-8')
            words = text.split()

            return {
                'length': len(text),
                'word_count': len(words),
                'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
                'compression_potential': 'medium'
            }
        except Exception:
            return {}

    def _check_sparsity(self, arr: np.ndarray, threshold: float = 0.1) -> bool:
        """检查数组稀疏性"""
        if arr.size == 0:
            return False

        zero_count = np.count_nonzero(arr == 0)
        sparsity = zero_count / arr.size

        return bool(sparsity > threshold)

    def _estimate_sparsity(self, data: Union[bytes, bytearray, memoryview]) -> float:
        """估计数据稀疏度"""
        try:
            sample = bytes(data[:4096])
            zero_count = sample.count(b'\x00')
            return zero_count / len(sample)
        except Exception:
            return 0.0
