"""
流处理器适配器模块

提供与流处理器的集成适配。
"""

import io
from typing import Optional, Iterator, BinaryIO, Callable
from ..core.buffer import Buffer
from ..pools.buffer_pool import BufferPool
from ..pools.ring_buffer import RingBuffer


class StreamAdapter:
    """
    流处理器适配器
    
    将缓冲区管理器与流式数据处理集成。
    支持文件流、网络流等场景。
    """
    
    def __init__(
        self,
        buffer_pool: Optional[BufferPool] = None,
        ring_buffer: Optional[RingBuffer] = None,
        chunk_size: int = 4096,
    ):
        """
        初始化流适配器
        
        Args:
            buffer_pool: 缓冲池实例（可选）
            ring_buffer: 环形缓冲区实例（可选）
            chunk_size: 处理块大小
        """
        self._buffer_pool = buffer_pool
        self._ring_buffer = ring_buffer
        self._chunk_size = chunk_size
        
        # 统计信息
        self._total_read = 0
        self._total_written = 0
        self._chunks_processed = 0
    
    def read_from_stream(
        self,
        stream: BinaryIO,
        size: Optional[int] = None,
    ) -> Iterator[bytes]:
        """
        从流中读取数据
        
        Args:
            stream: 输入流
            size: 要读取的总大小，None表示读取到EOF
        
        Yields:
            数据块
        """
        remaining = size
        
        while True:
            chunk_size = self._chunk_size
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
                if chunk_size <= 0:
                    break
            
            data = stream.read(chunk_size)
            if not data:
                break
            
            self._total_read += len(data)
            self._chunks_processed += 1
            
            if remaining is not None:
                remaining -= len(data)
            
            yield data
    
    def write_to_stream(
        self,
        stream: BinaryIO,
        data: bytes,
    ) -> int:
        """
        将数据写入流
        
        Args:
            stream: 输出流
            data: 要写入的数据
        
        Returns:
            写入的字节数
        """
        written = stream.write(data)
        self._total_written += written
        self._chunks_processed += 1
        return written
    
    def read_to_buffer(
        self,
        stream: BinaryIO,
        buffer: Buffer,
    ) -> int:
        """
        从流读取数据到缓冲区
        
        Args:
            stream: 输入流
            buffer: 目标缓冲区
        
        Returns:
            读取的字节数
        """
        data = stream.read(buffer.available)
        if data:
            written = buffer.write(data)
            self._total_read += written
            return written
        return 0
    
    def write_from_buffer(
        self,
        stream: BinaryIO,
        buffer: Buffer,
    ) -> int:
        """
        将缓冲区数据写入流
        
        Args:
            stream: 输出流
            buffer: 源缓冲区
        
        Returns:
            写入的字节数
        """
        data = buffer.data
        if data:
            written = stream.write(data)
            self._total_written += written
            buffer.clear()
            return written
        return 0
    
    def read_to_ring_buffer(
        self,
        stream: BinaryIO,
        size: Optional[int] = None,
    ) -> int:
        """
        从流读取数据到环形缓冲区
        
        Args:
            stream: 输入流
            size: 要读取的大小
        
        Returns:
            读取的字节数
        """
        if self._ring_buffer is None:
            raise ValueError("Ring buffer not configured")
        
        read_size = size or self._ring_buffer.writable
        read_size = min(read_size, self._ring_buffer.writable)
        
        data = stream.read(read_size)
        if data:
            written = self._ring_buffer.write(data)
            self._total_read += written
            return written
        return 0
    
    def write_from_ring_buffer(
        self,
        stream: BinaryIO,
        size: Optional[int] = None,
    ) -> int:
        """
        将环形缓冲区数据写入流
        
        Args:
            stream: 输出流
            size: 要写入的大小
        
        Returns:
            写入的字节数
        """
        if self._ring_buffer is None:
            raise ValueError("Ring buffer not configured")
        
        read_size = size or self._ring_buffer.readable
        read_size = min(read_size, self._ring_buffer.readable)
        
        if read_size > 0:
            data = self._ring_buffer.read(read_size)
            written = stream.write(data)
            self._total_written += written
            return written
        return 0
    
    def process_stream(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        processor: Callable[[bytes], Optional[bytes]],
    ) -> dict:
        """
        处理流数据
        
        Args:
            input_stream: 输入流
            output_stream: 输出流
            processor: 数据处理函数，接收字节数据，返回处理后的字节数据或None
        
        Returns:
            处理统计信息
        """
        for chunk in self.read_from_stream(input_stream):
            processed = processor(chunk)
            if processed:
                self.write_to_stream(output_stream, processed)
        
        return self.stats
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_read": self._total_read,
            "total_written": self._total_written,
            "chunks_processed": self._chunks_processed,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_read = 0
        self._total_written = 0
        self._chunks_processed = 0
    
    def __repr__(self) -> str:
        return (
            f"StreamAdapter(chunk_size={self._chunk_size}, "
            f"total_read={self._total_read}, total_written={self._total_written})"
        )
