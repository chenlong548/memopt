"""
数据压缩器适配器模块

提供与数据压缩器的集成适配。
"""

import zlib
import gzip
from typing import Optional, Callable, Dict, Any
from enum import Enum
from ..core.buffer import Buffer
from ..pools.buffer_pool import BufferPool


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"


class CompressorAdapter:
    """
    数据压缩器适配器
    
    将缓冲区管理器与数据压缩功能集成。
    支持多种压缩算法。
    
    注意：
    - compress/decompress: 单次压缩，每次调用独立，不影响流式压缩状态
    - compress_stream/decompress_stream: 流式压缩，保持压缩器状态
    """
    
    def __init__(
        self,
        compression_type: CompressionType = CompressionType.ZLIB,
        compression_level: int = 6,
        buffer_pool: Optional[BufferPool] = None,
    ):
        """
        初始化压缩器适配器
        
        Args:
            compression_type: 压缩类型
            compression_level: 压缩级别（0-9）
            buffer_pool: 缓冲池实例（可选）
        """
        self._compression_type = compression_type
        self._compression_level = compression_level
        self._buffer_pool = buffer_pool
        
        # 流式压缩器实例（用于流式压缩，保持状态）
        self._stream_compressor: Optional[Any] = None
        self._stream_decompressor: Optional[Any] = None
        
        # 初始化流式压缩器
        self._init_stream_compressors()
        
        # 统计信息
        self._total_compressed = 0
        self._total_decompressed = 0
        self._original_size = 0
        self._compressed_size = 0
    
    def _init_stream_compressors(self) -> None:
        """初始化流式压缩器"""
        if self._compression_type == CompressionType.ZLIB:
            self._stream_compressor = zlib.compressobj(self._compression_level)
            self._stream_decompressor = zlib.decompressobj()
        elif self._compression_type == CompressionType.GZIP:
            # GZIP 不支持流式状态，每次创建新实例
            self._stream_compressor = None
            self._stream_decompressor = None
    
    def compress(self, data: bytes) -> bytes:
        """
        单次压缩数据（独立压缩，不影响流式压缩状态）
        
        每次调用都会创建新的压缩器实例，确保压缩结果独立。
        
        Args:
            data: 原始数据
        
        Returns:
            压缩后的数据
        """
        if self._compression_type == CompressionType.NONE:
            return data
        
        self._original_size += len(data)
        
        if self._compression_type == CompressionType.ZLIB:
            # 创建独立的压缩器实例进行单次压缩
            compressor = zlib.compressobj(self._compression_level)
            compressed = compressor.compress(data) + compressor.flush()
        elif self._compression_type == CompressionType.GZIP:
            compressed = gzip.compress(data, self._compression_level)
        else:
            compressed = data
        
        self._compressed_size += len(compressed)
        self._total_compressed += 1
        
        return compressed
    
    def compress_to_buffer(self, data: bytes, buffer: Buffer) -> int:
        """
        压缩数据并写入缓冲区
        
        Args:
            data: 原始数据
            buffer: 目标缓冲区
        
        Returns:
            写入的字节数
        """
        compressed = self.compress(data)
        return buffer.write(compressed)
    
    def decompress(self, data: bytes) -> bytes:
        """
        单次解压数据（独立解压，不影响流式解压状态）
        
        每次调用都会创建新的解压器实例，确保解压结果独立。
        
        Args:
            data: 压缩数据
        
        Returns:
            解压后的数据
        """
        if self._compression_type == CompressionType.NONE:
            return data
        
        if self._compression_type == CompressionType.ZLIB:
            # 创建独立的解压器实例进行单次解压
            decompressor = zlib.decompressobj()
            decompressed = decompressor.decompress(data) + decompressor.flush()
        elif self._compression_type == CompressionType.GZIP:
            decompressed = gzip.decompress(data)
        else:
            decompressed = data
        
        self._total_decompressed += 1
        
        return decompressed
    
    def decompress_to_buffer(self, data: bytes, buffer: Buffer) -> int:
        """
        解压数据并写入缓冲区
        
        Args:
            data: 压缩数据
            buffer: 目标缓冲区
        
        Returns:
            写入的字节数
        """
        decompressed = self.decompress(data)
        return buffer.write(decompressed)
    
    def compress_stream(
        self,
        data: bytes,
        chunk_size: int = 4096,
        flush: bool = False,
    ) -> bytes:
        """
        流式压缩数据（保持压缩器状态）
        
        Args:
            data: 原始数据
            chunk_size: 处理块大小
            flush: 是否刷新压缩器（完成压缩后调用）
        
        Returns:
            压缩后的数据
        """
        if self._compression_type == CompressionType.NONE:
            return data
        
        if self._compression_type == CompressionType.GZIP:
            # GZIP 不支持流式压缩，回退到单次压缩
            return self.compress(data)
        
        result = bytearray()
        offset = 0
        
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            
            if self._compression_type == CompressionType.ZLIB:
                compressed_chunk = self._stream_compressor.compress(chunk)
                result.extend(compressed_chunk)
            
            offset += chunk_size
        
        # 根据参数决定是否刷新
        if flush and self._compression_type == CompressionType.ZLIB:
            result.extend(self._stream_compressor.flush())
            # 刷新后重新初始化压缩器
            self._stream_compressor = zlib.compressobj(self._compression_level)
        
        self._original_size += len(data)
        self._compressed_size += len(result)
        self._total_compressed += 1
        
        return bytes(result)
    
    def decompress_stream(self, data: bytes, flush: bool = False) -> bytes:
        """
        流式解压数据（保持解压器状态）
        
        Args:
            data: 压缩数据
            flush: 是否刷新解压器
        
        Returns:
            解压后的数据
        """
        if self._compression_type == CompressionType.NONE:
            return data
        
        if self._compression_type == CompressionType.GZIP:
            # GZIP 不支持流式解压，回退到单次解压
            return self.decompress(data)
        
        if self._compression_type == CompressionType.ZLIB:
            decompressed = self._stream_decompressor.decompress(data)
            
            if flush:
                decompressed += self._stream_decompressor.flush()
                # 刷新后重新初始化解压器
                self._stream_decompressor = zlib.decompressobj()
            
            self._total_decompressed += 1
            return decompressed
        
        return data
    
    @property
    def compression_ratio(self) -> float:
        """获取压缩率"""
        if self._original_size == 0:
            return 0.0
        return self._compressed_size / self._original_size
    
    @property
    def space_saved(self) -> float:
        """获取节省的空间比例"""
        if self._original_size == 0:
            return 0.0
        return 1.0 - self.compression_ratio
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "compression_type": self._compression_type.value,
            "compression_level": self._compression_level,
            "total_compressed": self._total_compressed,
            "total_decompressed": self._total_decompressed,
            "original_size": self._original_size,
            "compressed_size": self._compressed_size,
            "compression_ratio": self.compression_ratio,
            "space_saved": self.space_saved,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_compressed = 0
        self._total_decompressed = 0
        self._original_size = 0
        self._compressed_size = 0
    
    def reset(self) -> None:
        """重置压缩器状态"""
        self._init_stream_compressors()
        self.reset_stats()
    
    def __repr__(self) -> str:
        return (
            f"CompressorAdapter(type={self._compression_type.value}, "
            f"level={self._compression_level}, ratio={self.compression_ratio:.2%})"
        )
