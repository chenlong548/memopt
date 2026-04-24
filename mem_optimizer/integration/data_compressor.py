"""
mem_optimizer data_compressor集成模块

实现与data_compressor模块的集成。
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from ..core.base import AllocatorType, AllocationResult
from ..core.exceptions import IntegrationError


@dataclass
class CompressedMemoryBlock:
    """压缩内存块"""
    original_address: int
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    is_compressed: bool


class DataCompressorIntegration:
    """
    data_compressor集成器

    提供与data_compressor模块的无缝集成。
    """

    def __init__(self, memory_pool: Optional[Any] = None):
        """
        初始化集成器

        Args:
            memory_pool: 内存池引用
        """
        self._memory_pool = memory_pool
        self._compressor: Optional[Any] = None
        self._compressed_blocks: Dict[int, CompressedMemoryBlock] = {}
        self._integration_enabled = False

        self._compression_threshold = 0.8
        self._auto_compress = False
        self._compression_stats = {
            'total_compressed': 0,
            'total_decompressed': 0,
            'bytes_saved': 0
        }

        self._init_compressor()

    def _init_compressor(self):
        """初始化data_compressor"""
        try:
            import sys
            import os

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from data_compressor.core.compressor import DataCompressor
            from data_compressor.core.base import CompressionConfig

            self._compressor = DataCompressor()
            self._integration_enabled = True

        except ImportError:
            self._compressor = None
            self._integration_enabled = False
        except Exception:
            self._compressor = None
            self._integration_enabled = False

    def is_available(self) -> bool:
        """
        检查data_compressor是否可用

        Returns:
            bool: 是否可用
        """
        return self._integration_enabled and self._compressor is not None

    def compress_block(self,
                      address: int,
                      data: bytes,
                      algorithm: Optional[str] = None) -> Optional[CompressedMemoryBlock]:
        """
        压缩内存块

        Args:
            address: 内存地址
            data: 数据
            algorithm: 压缩算法

        Returns:
            CompressedMemoryBlock: 压缩后的块信息
        """
        if not self.is_available():
            return None

        try:
            from data_compressor.core.base import CompressionConfig, CompressionAlgorithm

            if algorithm:
                alg = CompressionAlgorithm(algorithm)
            else:
                alg = CompressionAlgorithm.AUTO

            config = CompressionConfig(algorithm=alg)

            if self._compressor:
                compressed = self._compressor.compress(data, config)
            else:
                raise IntegrationError("Compressor not available", module="data_compressor")

            block = CompressedMemoryBlock(
                original_address=address,
                original_size=len(data),
                compressed_size=len(compressed.data),
                compression_ratio=compressed.stats.compression_ratio if compressed.stats else 1.0,
                algorithm=compressed.algorithm.value if compressed.algorithm else "unknown",
                is_compressed=True
            )

            self._compressed_blocks[address] = block

            self._compression_stats['total_compressed'] += 1
            self._compression_stats['bytes_saved'] += (block.original_size - block.compressed_size)

            return block

        except Exception as e:
            raise IntegrationError(f"Compression failed: {e}", module="data_compressor")

    def decompress_block(self, address: int, compressed_data: bytes) -> Optional[bytes]:
        """
        解压内存块

        Args:
            address: 内存地址
            compressed_data: 压缩数据

        Returns:
            bytes: 解压后的数据
        """
        if not self.is_available():
            return None

        try:
            from data_compressor.core.base import CompressedData, CompressionAlgorithm

            block_info = self._compressed_blocks.get(address)

            if block_info:
                algorithm = CompressionAlgorithm(block_info.algorithm)
                original_size = block_info.original_size
            else:
                algorithm = CompressionAlgorithm.ZSTD
                original_size = len(compressed_data) * 2
            
            compressed = CompressedData(
                data=compressed_data,
                algorithm=algorithm,
                original_size=original_size,
                compressed_size=len(compressed_data),
                level=None  # type: ignore
            )  # type: ignore

            if self._compressor:
                decompressed = self._compressor.decompress(compressed)
            else:
                raise IntegrationError("Compressor not available", module="data_compressor")

            self._compression_stats['total_decompressed'] += 1

            return decompressed

        except Exception as e:
            raise IntegrationError(f"Decompression failed: {e}", module="data_compressor")

    def should_compress(self, size: int, pool_usage: float) -> bool:
        """
        判断是否应该压缩

        Args:
            size: 数据大小
            pool_usage: 内存池使用率

        Returns:
            bool: 是否应该压缩
        """
        if not self._auto_compress:
            return False

        if pool_usage < self._compression_threshold:
            return False

        return size > 1024

    def set_compression_threshold(self, threshold: float):
        """
        设置压缩阈值

        Args:
            threshold: 阈值 (0-1)
        """
        self._compression_threshold = max(0.0, min(1.0, threshold))

    def enable_auto_compress(self, enable: bool = True):
        """
        启用/禁用自动压缩

        Args:
            enable: 是否启用
        """
        self._auto_compress = enable

    def get_compressed_block_info(self, address: int) -> Optional[CompressedMemoryBlock]:
        """
        获取压缩块信息

        Args:
            address: 内存地址

        Returns:
            CompressedMemoryBlock: 块信息
        """
        return self._compressed_blocks.get(address)

    def get_all_compressed_blocks(self) -> Dict[int, CompressedMemoryBlock]:
        """
        获取所有压缩块

        Returns:
            Dict: 地址到块信息的映射
        """
        return self._compressed_blocks.copy()

    def analyze_data(self, data: bytes) -> Dict[str, Any]:
        """
        分析数据特征

        Args:
            data: 数据

        Returns:
            Dict: 分析结果
        """
        if not self.is_available():
            return {'available': False}

        try:
            if self._compressor:
                analysis = self._compressor.analyze(data)
            else:
                return {'available': False}
            return {
                'available': True,
                'data_type': analysis.get('data_type'),
                'recommended_algorithm': analysis.get('recommended_algorithm'),
                'estimated_ratio': analysis.get('estimated_compression_ratio', 1.0),
                'features': analysis.get('features', {})
            }
        except Exception:
            return {'available': False}

    def benchmark_algorithms(self,
                            data: bytes,
                            algorithms: Optional[list] = None) -> Dict[str, Any]:
        """
        基准测试压缩算法

        Args:
            data: 测试数据
            algorithms: 算法列表

        Returns:
            Dict: 基准测试结果
        """
        if not self.is_available():
            return {'available': False}

        try:
            if self._compressor:
                results = self._compressor.benchmark(data, algorithms)
            else:
                return {'available': False}

            return {
                'available': True,
                'results': {
                    alg: {
                        'compression_ratio': stats.compression_ratio if stats else 0,
                        'compression_time': stats.compression_time if stats else 0,
                        'throughput': stats.throughput_mbps if stats else 0
                    }
                    for alg, stats in results.items()
                }
            }
        except Exception:
            return {'available': False}

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        获取压缩统计

        Returns:
            Dict: 统计信息
        """
        stats = {
            'available': self.is_available(),
            'auto_compress': self._auto_compress,
            'compression_threshold': self._compression_threshold,
            'compressed_blocks_count': len(self._compressed_blocks),
            **self._compression_stats
        }

        if self.is_available():
            try:
                if self._compressor:
                    summary = self._compressor.get_stats_summary()
                else:
                    summary = {}
                stats['compressor_summary'] = summary
            except Exception:
                pass

        return stats

    def get_memory_savings(self) -> Dict[str, Any]:
        """
        获取内存节省

        Returns:
            Dict: 节省信息
        """
        total_original = sum(b.original_size for b in self._compressed_blocks.values())
        total_compressed = sum(b.compressed_size for b in self._compressed_blocks.values())

        return {
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'bytes_saved': total_original - total_compressed,
            'average_ratio': total_original / total_compressed if total_compressed > 0 else 1.0,
            'blocks_count': len(self._compressed_blocks)
        }

    def clear_compressed_blocks(self):
        """清除压缩块记录"""
        self._compressed_blocks.clear()

    def cleanup(self):
        """清理资源"""
        self.clear_compressed_blocks()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
        return False
