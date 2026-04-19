"""
mem_mapper集成模块

实现data_compressor与mem_mapper的无缝集成。
"""

import logging
from typing import Optional, Dict, Any
import os

from ..core.base import (
    CompressionConfig,
    CompressionStats,
    CompressedData,
    DataType
)
from ..core.compressor import DataCompressor
from ..core.exceptions import CompressionError, DecompressionError

logger = logging.getLogger(__name__)


class MemMapperIntegration:
    """
    mem_mapper集成类

    提供压缩内存映射文件的功能。
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        初始化集成模块

        Args:
            config: 压缩配置
        """
        self.config = config or CompressionConfig()
        self.compressor = DataCompressor(self.config)

        # mem_mapper实例（延迟导入）
        self._mem_mapper = None

    def compress_mapped_region(self,
                               region: Any,
                               output_path: str,
                               config: Optional[CompressionConfig] = None) -> CompressionStats:
        """
        压缩内存映射区域

        Args:
            region: MappedRegion对象
            output_path: 输出文件路径
            config: 压缩配置

        Returns:
            CompressionStats: 压缩统计信息

        Raises:
            CompressionError: 压缩失败时抛出
        """
        actual_config = config or self.config

        logger.info(f"Compressing mapped region to {output_path}")

        try:
            # 获取映射数据
            # 注意：这里需要通过ctypes或其他方式访问映射的内存
            import ctypes

            # 读取映射区域的数据
            data = self._read_mapped_region(region)

            # 压缩数据
            compressed = self.compressor.compress(data, actual_config)

            # 写入文件
            self._write_compressed_file(output_path, compressed)

            logger.info(
                f"Mapped region compressed: {compressed.stats.original_size} -> "
                f"{compressed.stats.compressed_size} bytes "
                f"({compressed.stats.compression_ratio:.2f}x)"
            )

            return compressed.stats

        except Exception as e:
            logger.error(f"Failed to compress mapped region: {e}")
            raise CompressionError(f"Failed to compress mapped region: {e}") from e

    def decompress_to_mapping(self,
                             compressed_path: str,
                             mapper: Any,
                             mode: str = 'readonly',
                             numa_node: int = -1) -> Any:
        """
        解压文件到内存映射

        Args:
            compressed_path: 压缩文件路径
            mapper: MemoryMapper实例
            mode: 映射模式
            numa_node: NUMA节点

        Returns:
            MappedRegion: 映射区域对象

        Raises:
            DecompressionError: 解压失败时抛出
        """
        logger.info(f"Decompressing {compressed_path} to memory mapping")

        try:
            # 读取压缩文件
            compressed = self._read_compressed_file(compressed_path)

            # 解压数据
            decompressed = self.compressor.decompress(compressed)

            # 创建临时文件
            temp_path = self._create_temp_file(decompressed)

            # 映射到内存
            region = mapper.map_file(
                path=temp_path,
                mode=mode,
                numa_node=numa_node
            )

            logger.info(
                f"Decompressed to mapping: {compressed.compressed_size} -> "
                f"{len(decompressed)} bytes"
            )

            return region

        except Exception as e:
            logger.error(f"Failed to decompress to mapping: {e}")
            raise DecompressionError(f"Failed to decompress to mapping: {e}") from e

    def create_compressed_mapping(self,
                                 file_path: str,
                                 mapper: Any,
                                 config: Optional[CompressionConfig] = None,
                                 cache_compressed: bool = True) -> Dict[str, Any]:
        """
        创建压缩映射

        Args:
            file_path: 原始文件路径
            mapper: MemoryMapper实例
            config: 压缩配置
            cache_compressed: 是否缓存压缩文件

        Returns:
            Dict: 包含映射信息和压缩统计
        """
        actual_config = config or self.config

        logger.info(f"Creating compressed mapping for {file_path}")

        result = {
            'original_path': file_path,
            'compressed_path': None,
            'mapping': None,
            'stats': None
        }

        try:
            # 检查是否已有压缩缓存
            compressed_path = self._get_cached_compressed_path(file_path)

            if compressed_path and os.path.exists(compressed_path):
                # 使用缓存的压缩文件
                logger.info(f"Using cached compressed file: {compressed_path}")
            else:
                # 压缩文件
                compressed_path = self._compress_file(file_path, actual_config)

                if cache_compressed:
                    self._cache_compressed_file(file_path, compressed_path)

            # 映射压缩文件
            region = mapper.map_file(
                path=compressed_path,
                mode='readonly',
                numa_node=-1
            )

            result['compressed_path'] = compressed_path
            result['mapping'] = region

            logger.info(f"Compressed mapping created successfully")

            return result

        except Exception as e:
            logger.error(f"Failed to create compressed mapping: {e}")
            raise CompressionError(f"Failed to create compressed mapping: {e}") from e

    def _read_mapped_region(self, region: Any) -> bytes:
        """读取映射区域数据"""
        import ctypes

        # 获取基地址和大小
        base_addr = region.base_address
        size = region.size

        # 创建缓冲区
        buffer = (ctypes.c_ubyte * size)()

        # 从映射内存复制数据
        ctypes.memmove(buffer, base_addr, size)

        return bytes(buffer)

    def _write_compressed_file(self, path: str, compressed: CompressedData):
        """写入压缩文件"""
        import struct

        with open(path, 'wb') as f:
            # 写入魔数
            f.write(b'DCMAP')

            # 写入版本
            f.write(struct.pack('I', 1))

            # 写入元数据
            metadata = compressed.get_metadata()
            import json
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            f.write(struct.pack('I', len(metadata_bytes)))
            f.write(metadata_bytes)

            # 写入数据
            f.write(compressed.data)

    def _read_compressed_file(self, path: str) -> CompressedData:
        """读取压缩文件"""
        import struct
        import json

        with open(path, 'rb') as f:
            # 读取魔数
            magic = f.read(5)
            if magic != b'DCMAP':
                raise DecompressionError("Invalid compressed file format")

            # 读取版本
            version = struct.unpack('I', f.read(4))[0]

            # 读取元数据
            metadata_len = struct.unpack('I', f.read(4))[0]
            metadata_bytes = f.read(metadata_len)
            metadata = json.loads(metadata_bytes.decode('utf-8'))

            # 读取数据
            data = f.read()

            # 创建压缩数据容器
            from ..core.base import CompressionAlgorithm, CompressionLevel

            compressed = CompressedData(
                data=data,
                algorithm=CompressionAlgorithm(metadata['algorithm']),
                level=CompressionLevel(metadata['level']),
                original_size=metadata['original_size'],
                compressed_size=metadata['compressed_size'],
                data_type=DataType(metadata['data_type'])
            )

            return compressed

    def _create_temp_file(self, data: bytes) -> str:
        """创建临时文件"""
        import tempfile

        fd, temp_path = tempfile.mkstemp(suffix='.dcmp')

        try:
            os.write(fd, data)
        finally:
            os.close(fd)

        return temp_path

    def _compress_file(self, file_path: str, config: CompressionConfig) -> str:
        """压缩文件"""
        # 读取文件
        with open(file_path, 'rb') as f:
            data = f.read()

        # 压缩
        compressed = self.compressor.compress(data, config)

        # 创建压缩文件路径
        compressed_path = file_path + '.dcmp'

        # 写入
        self._write_compressed_file(compressed_path, compressed)

        return compressed_path

    def _get_cached_compressed_path(self, original_path: str) -> Optional[str]:
        """获取缓存的压缩文件路径"""
        # 简化实现：检查同目录下是否有压缩文件
        compressed_path = original_path + '.dcmp'

        if os.path.exists(compressed_path):
            # 检查是否过期（原始文件更新）
            if os.path.getmtime(compressed_path) >= os.path.getmtime(original_path):
                return compressed_path

        return None

    def _cache_compressed_file(self, original_path: str, compressed_path: str):
        """缓存压缩文件"""
        # 简化实现：压缩文件已经在正确位置
        pass

    def get_compression_stats(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件压缩统计信息

        Args:
            file_path: 文件路径

        Returns:
            Dict: 压缩统计信息
        """
        try:
            # 读取文件
            with open(file_path, 'rb') as f:
                data = f.read()

            # 分析数据
            analysis = self.compressor.analyze(data)

            return {
                'file_path': file_path,
                'file_size': len(data),
                'data_type': analysis['data_type'],
                'recommended_algorithm': analysis['recommended_algorithm'],
                'estimated_ratio': analysis['estimated_compression_ratio'],
                'features': analysis['features']
            }

        except Exception as e:
            logger.error(f"Failed to get compression stats: {e}")
            return {}


class CompressedMemoryMapper:
    """
    压缩内存映射器

    提供透明的压缩内存映射功能。
    """

    def __init__(self, mapper: Any, config: Optional[CompressionConfig] = None):
        """
        初始化压缩内存映射器

        Args:
            mapper: MemoryMapper实例
            config: 压缩配置
        """
        self.mapper = mapper
        self.config = config or CompressionConfig()
        self.integration = MemMapperIntegration(self.config)

        # 缓存
        self._compressed_cache: Dict[str, Any] = {}

    def map_file_compressed(self,
                           path: str,
                           mode: str = 'readonly',
                           numa_node: int = -1,
                           auto_compress: bool = True) -> Any:
        """
        映射文件（自动压缩）

        Args:
            path: 文件路径
            mode: 映射模式
            numa_node: NUMA节点
            auto_compress: 是否自动压缩

        Returns:
            MappedRegion: 映射区域
        """
        if auto_compress:
            # 检查缓存
            if path in self._compressed_cache:
                return self._compressed_cache[path]['mapping']

            # 创建压缩映射
            result = self.integration.create_compressed_mapping(
                file_path=path,
                mapper=self.mapper,
                config=self.config
            )

            # 缓存
            self._compressed_cache[path] = result

            return result['mapping']
        else:
            # 直接映射
            return self.mapper.map_file(path, mode, numa_node=numa_node)

    def get_compression_stats(self, path: str) -> Optional[CompressionStats]:
        """获取压缩统计信息"""
        if path in self._compressed_cache:
            return self._compressed_cache[path]['stats']
        return None

    def clear_cache(self):
        """清除缓存"""
        self._compressed_cache.clear()
