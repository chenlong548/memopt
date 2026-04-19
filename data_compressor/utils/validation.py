"""
数据验证器

验证输入数据的合法性和完整性。
"""

import logging
from typing import Union, Optional, Dict, Any
import struct

from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器

    验证输入数据的合法性和完整性。
    """

    def __init__(self,
                 max_size: int = 1024 * 1024 * 1024,  # 1GB
                 min_size: int = 0,
                 check_patterns: bool = True):
        """
        初始化数据验证器

        Args:
            max_size: 最大数据大小（字节）
            min_size: 最小数据大小（字节）
            check_patterns: 是否检查数据模式
        """
        self.max_size = max_size
        self.min_size = min_size
        self.check_patterns = check_patterns

        # 危险模式（用于安全检查）
        self._dangerous_patterns = [
            b'\x00\x00\x00\x00',  # 空字节序列
        ]

    def validate(self,
                data: Union[bytes, bytearray, memoryview],
                strict: bool = False) -> bool:
        """
        验证数据

        Args:
            data: 待验证数据
            strict: 是否严格模式

        Returns:
            bool: 验证是否通过

        Raises:
            ValidationError: 验证失败时抛出
        """
        # 检查数据类型
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValidationError(
                f"Invalid data type: {type(data).__name__}, expected bytes-like object",
                {'type': type(data).__name__}
            )

        # 检查数据大小
        data_size = len(data)

        if data_size < self.min_size:
            raise ValidationError(
                f"Data size too small: {data_size} bytes, minimum {self.min_size}",
                {'size': data_size, 'min_size': self.min_size}
            )

        if data_size > self.max_size:
            raise ValidationError(
                f"Data size too large: {data_size} bytes, maximum {self.max_size}",
                {'size': data_size, 'max_size': self.max_size}
            )

        # 严格模式下的额外检查
        if strict:
            self._strict_validate(data)

        return True

    def _strict_validate(self, data: Union[bytes, bytearray, memoryview]):
        """严格验证"""
        # 检查数据完整性
        if isinstance(data, memoryview):
            # 检查memoryview是否有效
            try:
                _ = bytes(data[:1024])
            except Exception as e:
                raise ValidationError(
                    f"Invalid memoryview: {e}",
                    {'error': str(e)}
                )

        # 检查数据模式
        if self.check_patterns:
            self._check_patterns(data)

    def _check_patterns(self, data: Union[bytes, bytearray, memoryview]):
        """检查数据模式"""
        # 检查是否全为零
        if len(data) > 0:
            sample = bytes(data[:min(len(data), 4096)])
            if all(b == 0 for b in sample):
                logger.warning("Data appears to be all zeros")

    def validate_compressed_data(self, compressed: Any) -> bool:
        """
        验证压缩数据

        Args:
            compressed: 压缩数据对象

        Returns:
            bool: 验证是否通过

        Raises:
            ValidationError: 验证失败时抛出
        """
        # 检查必要属性
        required_attrs = ['data', 'algorithm', 'original_size', 'compressed_size']

        for attr in required_attrs:
            if not hasattr(compressed, attr):
                raise ValidationError(
                    f"Missing required attribute: {attr}",
                    {'missing_attr': attr}
                )

        # 检查数据大小一致性
        if len(compressed.data) != compressed.compressed_size:
            raise ValidationError(
                f"Data size mismatch: {len(compressed.data)} != {compressed.compressed_size}",
                {
                    'actual_size': len(compressed.data),
                    'declared_size': compressed.compressed_size
                }
            )

        # 检查压缩比合理性
        if compressed.original_size > 0:
            ratio = compressed.original_size / compressed.compressed_size
            if ratio < 0.01 or ratio > 1000:
                logger.warning(f"Suspicious compression ratio: {ratio:.2f}x")

        return True

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        验证文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 文件信息

        Raises:
            ValidationError: 验证失败时抛出
        """
        import os

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise ValidationError(
                f"File not found: {file_path}",
                {'file_path': file_path}
            )

        # 检查是否为文件
        if not os.path.isfile(file_path):
            raise ValidationError(
                f"Not a file: {file_path}",
                {'file_path': file_path}
            )

        # 获取文件信息
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size

        # 检查文件大小
        if file_size > self.max_size:
            raise ValidationError(
                f"File too large: {file_size} bytes, maximum {self.max_size}",
                {'file_size': file_size, 'max_size': self.max_size}
            )

        return {
            'path': file_path,
            'size': file_size,
            'readable': os.access(file_path, os.R_OK),
            'writable': os.access(file_path, os.W_OK),
        }

    def check_memory_limit(self, required: int, available: Optional[int] = None) -> bool:
        """
        检查内存限制

        Args:
            required: 需要的内存大小（字节）
            available: 可用内存大小（字节），None则自动检测

        Returns:
            bool: 是否满足内存限制

        Raises:
            ValidationError: 内存不足时抛出
        """
        if available is None:
            # 尝试获取可用内存
            try:
                import psutil
                available = psutil.virtual_memory().available
            except ImportError:
                # psutil不可用，跳过检查
                return True

        if required > available:
            raise ValidationError(
                f"Insufficient memory: required {required} bytes, available {available} bytes",
                {
                    'required': required,
                    'available': available,
                    'deficit': required - available
                }
            )

        return True

    def estimate_memory_usage(self, data_size: int, algorithm: str) -> int:
        """
        估算内存使用量

        Args:
            data_size: 数据大小（字节）
            algorithm: 算法名称

        Returns:
            int: 估算的内存使用量（字节）
        """
        # 不同算法的内存开销估算
        memory_overhead = {
            'zstd': 1.5,    # ZSTD大约需要1.5倍内存
            'lz4': 1.2,     # LZ4内存开销较小
            'brotli': 2.0,  # Brotli内存开销较大
            'bf16': 1.3,    # BF16压缩
            'kv_cache': 1.5, # KV Cache压缩
        }

        multiplier = memory_overhead.get(algorithm, 2.0)
        return int(data_size * multiplier)

    def get_data_info(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """
        获取数据信息

        Args:
            data: 数据

        Returns:
            Dict: 数据信息
        """
        data_size = len(data)

        info = {
            'size': data_size,
            'size_mb': data_size / (1024 * 1024),
            'type': type(data).__name__,
        }

        # 采样分析
        if data_size > 0:
            sample_size = min(data_size, 4096)
            sample = bytes(data[:sample_size])

            # 统计字节分布
            unique_bytes = len(set(sample))
            zero_count = sample.count(0)

            info['unique_bytes'] = unique_bytes
            info['zero_ratio'] = zero_count / sample_size
            info['entropy'] = self._calculate_entropy(sample)

        return info

    def _calculate_entropy(self, data: bytes) -> float:
        """计算数据熵"""
        if len(data) == 0:
            return 0.0

        from collections import Counter
        import math

        counter = Counter(data)
        total = len(data)

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy
