"""
Data Compressor适配器模块

提供与data_compressor模块的集成适配器。
"""

from typing import Any, Callable, Optional, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ..core.lazy import Lazy
from ..thunk.memo_thunk import Memothunk


class CompressorAdapter:
    """
    Data Compressor适配器

    该类提供了lazy_evaluator与data_compressor模块的集成适配器，
    允许在惰性计算中使用数据压缩功能。

    Example:
        >>> adapter = CompressorAdapter()
        >>> lazy_compressed = adapter.lazy_compress(data)
        >>> result = lazy_compressed.get()
    """

    def __init__(self):
        """初始化CompressorAdapter"""
        self._compressor = None
        try:
            # 尝试导入data_compressor模块
            from data_compressor import DataCompressor
            self._compressor = DataCompressor
        except ImportError:
            pass

    def is_available(self) -> bool:
        """
        检查data_compressor模块是否可用

        Returns:
            bool: 如果可用返回True，否则返回False
        """
        return self._compressor is not None

    def lazy_compress(self, data: Any, algorithm: str = 'auto') -> Memothunk:
        """
        创建惰性压缩

        Args:
            data: 要压缩的数据
            algorithm: 压缩算法

        Returns:
            Memothunk: 带记忆化的thunk
        """
        def compress():
            if not self.is_available():
                # 如果compressor不可用，返回原始数据
                return data

            if self._compressor:
                compressor = self._compressor()
                if hasattr(compressor, 'compress'):
                    from data_compressor.core.base import CompressionConfig, CompressionAlgorithm
                    config = CompressionConfig(
                        algorithm=CompressionAlgorithm(algorithm)
                    )
                    return compressor.compress(data, config)
                else:
                    return data
            else:
                return data

        return Memothunk(compress)

    def lazy_decompress(self, compressed_data: Any) -> Memothunk:
        """
        创建惰性解压

        Args:
            compressed_data: 压缩的数据

        Returns:
            Memothunk: 带记忆化的thunk
        """
        def decompress():
            if not self.is_available():
                # 如果compressor不可用，返回原始数据
                return compressed_data

            if self._compressor:
                compressor = self._compressor()
                if hasattr(compressor, 'decompress'):
                    return compressor.decompress(compressed_data)
                else:
                    return compressed_data
            else:
                return compressed_data

        return Memothunk(decompress)

    def compress_on_demand(self, data: Any, algorithm: str = 'auto') -> Lazy:
        """
        按需压缩

        Args:
            data: 要压缩的数据
            algorithm: 压缩算法

        Returns:
            Lazy: 惰性值
        """
        return Lazy(lambda: self.lazy_compress(data, algorithm).get())

    def create_compressed_cache(self, max_size: int = 100) -> Any:
        """
        创建压缩缓存

        Args:
            max_size: 最大缓存大小

        Returns:
            Any: 压缩缓存对象
        """
        from ..memoization.lru_cache import LRUCache

        cache = LRUCache(max_size=max_size)

        class CompressedCache:
            def __init__(self, cache_instance, adapter):
                self._cache = cache_instance
                self._adapter = adapter

            def get(self, key: str) -> Optional[Any]:
                compressed = self._cache.get(key)
                if compressed is not None:
                    return self._adapter.lazy_decompress(compressed).get()
                return None

            def put(self, key: str, value: Any) -> None:
                compressed = self._adapter.lazy_compress(value).get()
                self._cache.put(key, compressed)

            def invalidate(self, key: str) -> bool:
                return self._cache.invalidate(key)

            def clear(self) -> None:
                self._cache.clear()

        return CompressedCache(cache, self)

    def batch_compress(self, data_list: list, algorithm: str = 'auto') -> list:
        """
        批量压缩

        Args:
            data_list: 数据列表
            algorithm: 压缩算法

        Returns:
            list: 压缩后的数据列表
        """
        return [
            self.lazy_compress(data, algorithm).get()
            for data in data_list
        ]

    def batch_decompress(self, compressed_list: list) -> list:
        """
        批量解压

        Args:
            compressed_list: 压缩数据列表

        Returns:
            list: 解压后的数据列表
        """
        return [
            self.lazy_decompress(data).get()
            for data in compressed_list
        ]

    def __repr__(self) -> str:
        """字符串表示"""
        available = "available" if self.is_available() else "not available"
        return f"CompressorAdapter(status={available})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
