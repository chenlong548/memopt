"""
环形缓冲区模块

实现高效的环形缓冲区，容量必须是2的幂以支持位运算优化。
"""

import threading
from typing import Optional
from ..core.exceptions import (
    BufferFullError,
    BufferEmptyError,
    RingBufferCapacityError,
)


class RingBuffer:
    """
    环形缓冲区
    
    高效的FIFO缓冲区实现，容量必须是2的幂。
    使用位运算优化取模操作，提高性能。
    线程安全的实现。
    """
    
    def __init__(self, capacity: int):
        """
        初始化环形缓冲区
        
        Args:
            capacity: 缓冲区容量，必须是2的幂
        
        Raises:
            RingBufferCapacityError: 容量不是2的幂
        """
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise RingBufferCapacityError(capacity)
        
        self._capacity = capacity
        self._mask = capacity - 1  # 用于位运算优化
        self._buffer = bytearray(capacity)
        self._read_pos = 0
        self._write_pos = 0
        self._count = 0  # 数据计数器
        self._lock = threading.RLock()
    
    def write(self, data: bytes) -> int:
        """
        写入数据到环形缓冲区
        
        Args:
            data: 要写入的字节数据
        
        Returns:
            实际写入的字节数
        
        Raises:
            BufferFullError: 缓冲区已满，无法写入全部数据
        """
        if not data:
            return 0
        
        with self._lock:
            writable = self.writable
            write_size = len(data)
            
            if write_size > writable:
                raise BufferFullError(
                    f"Not enough space: required {write_size}, available {writable}"
                )
            
            # 分两段写入（处理环形边界）
            first_chunk = min(write_size, self._capacity - self._write_pos)
            self._buffer[self._write_pos:self._write_pos + first_chunk] = data[:first_chunk]
            
            if first_chunk < write_size:
                # 写入第二段（从缓冲区开头）
                second_chunk = write_size - first_chunk
                self._buffer[0:second_chunk] = data[first_chunk:]
            
            # 更新写位置（使用位运算优化）
            self._write_pos = (self._write_pos + write_size) & self._mask
            self._count += write_size
            
            return write_size
    
    def try_write(self, data: bytes) -> int:
        """
        尝试写入数据，如果空间不足则写入部分数据
        
        Args:
            data: 要写入的字节数据
        
        Returns:
            实际写入的字节数
        """
        if not data:
            return 0
        
        with self._lock:
            writable = self.writable
            write_size = min(len(data), writable)
            
            if write_size == 0:
                return 0
            
            # 分两段写入
            first_chunk = min(write_size, self._capacity - self._write_pos)
            self._buffer[self._write_pos:self._write_pos + first_chunk] = data[:first_chunk]
            
            if first_chunk < write_size:
                second_chunk = write_size - first_chunk
                self._buffer[0:second_chunk] = data[first_chunk:first_chunk + second_chunk]
            
            self._write_pos = (self._write_pos + write_size) & self._mask
            self._count += write_size
            
            return write_size
    
    def read(self, size: int) -> bytes:
        """
        从环形缓冲区读取数据
        
        Args:
            size: 要读取的字节数
        
        Returns:
            读取的字节数据
        
        Raises:
            BufferEmptyError: 缓冲区数据不足
        """
        if size <= 0:
            return b""
        
        with self._lock:
            readable = self.readable
            
            if size > readable:
                raise BufferEmptyError(
                    f"Not enough data: requested {size}, available {readable}"
                )
            
            # 分两段读取（处理环形边界）
            result = bytearray(size)
            first_chunk = min(size, self._capacity - self._read_pos)
            result[:first_chunk] = self._buffer[self._read_pos:self._read_pos + first_chunk]
            
            if first_chunk < size:
                # 读取第二段（从缓冲区开头）
                second_chunk = size - first_chunk
                result[first_chunk:] = self._buffer[0:second_chunk]
            
            # 更新读位置
            self._read_pos = (self._read_pos + size) & self._mask
            self._count -= size
            
            return bytes(result)
    
    def try_read(self, size: int) -> bytes:
        """
        尝试读取数据，如果数据不足则读取部分数据
        
        Args:
            size: 要读取的字节数
        
        Returns:
            读取的字节数据
        """
        if size <= 0:
            return b""
        
        with self._lock:
            readable = self.readable
            read_size = min(size, readable)
            
            if read_size == 0:
                return b""
            
            result = bytearray(read_size)
            first_chunk = min(read_size, self._capacity - self._read_pos)
            result[:first_chunk] = self._buffer[self._read_pos:self._read_pos + first_chunk]
            
            if first_chunk < read_size:
                second_chunk = read_size - first_chunk
                result[first_chunk:] = self._buffer[0:second_chunk]
            
            self._read_pos = (self._read_pos + read_size) & self._mask
            self._count -= read_size
            
            return bytes(result)
    
    def peek(self, size: int) -> bytes:
        """
        查看数据但不移动读指针
        
        Args:
            size: 要查看的字节数
        
        Returns:
            查看的字节数据
        
        Raises:
            BufferEmptyError: 数据不足
        """
        if size <= 0:
            return b""
        
        with self._lock:
            readable = self.readable
            
            if size > readable:
                raise BufferEmptyError(
                    f"Not enough data: requested {size}, available {readable}"
                )
            
            result = bytearray(size)
            first_chunk = min(size, self._capacity - self._read_pos)
            result[:first_chunk] = self._buffer[self._read_pos:self._read_pos + first_chunk]
            
            if first_chunk < size:
                second_chunk = size - first_chunk
                result[first_chunk:] = self._buffer[0:second_chunk]
            
            return bytes(result)
    
    def skip(self, size: int) -> None:
        """
        跳过指定字节数的数据
        
        Args:
            size: 要跳过的字节数
        
        Raises:
            BufferEmptyError: 数据不足
        """
        if size <= 0:
            return
        
        with self._lock:
            if size > self.readable:
                raise BufferEmptyError(
                    f"Not enough data to skip: requested {size}, available {self.readable}"
                )
            self._read_pos = (self._read_pos + size) & self._mask
            self._count -= size
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._count = 0
    
    @property
    def writable(self) -> int:
        """获取可写入空间"""
        return self._capacity - self._count
    
    @property
    def readable(self) -> int:
        """获取可读取数据量"""
        return self._count
    
    @property
    def capacity(self) -> int:
        """获取缓冲区容量"""
        return self._capacity
    
    @property
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self._count == 0
    
    @property
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.writable == 0
    
    def __len__(self) -> int:
        """返回缓冲区中数据量"""
        return self.readable
    
    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self._capacity}, "
            f"readable={self.readable}, writable={self.writable})"
        )
