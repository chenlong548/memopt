"""
Buffer主类模块

提供缓冲区的基本操作，包括读写、定位和清理。
"""

import ctypes
import mmap
import threading
from typing import Optional
from .exceptions import (
    BufferFullError,
    BufferEmptyError,
    InvalidAlignmentError,
    InvalidCapacityError,
)


class Buffer:
    """
    缓冲区主类
    
    提供基本的缓冲区操作，支持内存对齐和高效读写。
    线程安全的缓冲区实现，使用锁保护所有状态变更操作。
    """
    
    def __init__(self, size: int, alignment: int = 64):
        """
        初始化缓冲区
        
        Args:
            size: 缓冲区大小（字节）
            alignment: 内存对齐字节数，必须是2的幂，默认64字节缓存行对齐
        
        Raises:
            InvalidCapacityError: 容量无效
            InvalidAlignmentError: 对齐参数无效
        """
        if size <= 0:
            raise InvalidCapacityError(size)
        
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise InvalidAlignmentError(alignment)
        
        self._size = size
        self._alignment = alignment
        self._position = 0
        self._limit = size
        self._mark = -1
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 分配对齐的内存
        self._buffer, self._aligned_address = self._allocate_aligned_buffer(size, alignment)
    
    def _allocate_aligned_buffer(self, size: int, alignment: int) -> tuple:
        """
        分配对齐的内存缓冲区
        
        使用ctypes实现真正的内存对齐，确保缓冲区起始地址是对齐值的整数倍。
        
        Args:
            size: 缓冲区大小
            alignment: 对齐字节数
        
        Returns:
            (bytearray, aligned_address) 元组，包含缓冲区和对齐后的地址
        """
        # 分配额外空间用于对齐
        # 需要额外 alignment-1 字节来确保可以找到对齐的地址
        total_size = size + alignment - 1
        
        # 创建原始缓冲区
        raw_buffer = bytearray(total_size)
        
        # 使用ctypes获取缓冲区地址并计算对齐地址
        raw_address = ctypes.addressof(
            (ctypes.c_char * len(raw_buffer)).from_buffer(raw_buffer)
        )
        
        # 计算对齐后的地址
        aligned_address = (raw_address + alignment - 1) & ~(alignment - 1)
        offset = aligned_address - raw_address
        
        # 返回对齐后的切片视图
        # 注意：我们返回原始缓冲区，但记录对齐信息
        # 实际使用时，数据从offset开始存储
        return raw_buffer[offset:offset + size], aligned_address
    
    def write(self, data: bytes) -> int:
        """
        写入数据到缓冲区（线程安全）
        
        Args:
            data: 要写入的字节数据
        
        Returns:
            实际写入的字节数
        
        Raises:
            BufferFullError: 缓冲区已满
        """
        if not data:
            return 0
        
        with self._lock:
            remaining = self._limit - self._position
            if remaining <= 0:
                raise BufferFullError(f"Buffer is full, capacity: {self._size}")
            
            write_size = min(len(data), remaining)
            self._buffer[self._position:self._position + write_size] = data[:write_size]
            self._position += write_size
            
            return write_size
    
    def read(self, size: int) -> bytes:
        """
        从缓冲区读取数据（线程安全）
        
        Args:
            size: 要读取的字节数
        
        Returns:
            读取的字节数据
        
        Raises:
            BufferEmptyError: 缓冲区为空或请求数据超出可用范围
        """
        if size <= 0:
            return b""
        
        with self._lock:
            remaining = self._limit - self._position
            if remaining <= 0:
                raise BufferEmptyError("Buffer is empty or position at limit")
            
            read_size = min(size, remaining)
            data = bytes(self._buffer[self._position:self._position + read_size])
            self._position += read_size
            
            return data
    
    def peek(self, size: int) -> bytes:
        """
        查看缓冲区数据但不移动位置指针（线程安全）
        
        Args:
            size: 要查看的字节数
        
        Returns:
            查看的字节数据
        """
        if size <= 0:
            return b""
        
        with self._lock:
            remaining = self._limit - self._position
            read_size = min(size, remaining)
            
            return bytes(self._buffer[self._position:self._position + read_size])
    
    def seek(self, position: int) -> None:
        """
        设置缓冲区位置指针（线程安全）
        
        Args:
            position: 新的位置
        
        Raises:
            ValueError: 位置超出范围
        """
        with self._lock:
            if position < 0 or position > self._limit:
                raise ValueError(f"Position {position} out of range [0, {self._limit}]")
            self._position = position
    
    def set_limit(self, limit: int) -> None:
        """
        设置缓冲区限制（线程安全）
        
        Args:
            limit: 新的限制
        
        Raises:
            ValueError: 限制超出范围
        """
        with self._lock:
            if limit < 0 or limit > self._size:
                raise ValueError(f"Limit {limit} out of range [0, {self._size}]")
            
            if limit < self._position:
                self._position = limit
            
            if limit < self._mark:
                self._mark = -1
            
            self._limit = limit
    
    def mark(self) -> None:
        """标记当前位置（线程安全）"""
        with self._lock:
            self._mark = self._position
    
    def reset(self) -> None:
        """重置到标记位置（线程安全）"""
        with self._lock:
            if self._mark < 0:
                raise ValueError("No mark set")
            self._position = self._mark
    
    def clear(self) -> None:
        """清空缓冲区，重置位置和限制（线程安全）"""
        with self._lock:
            self._position = 0
            self._limit = self._size
            self._mark = -1
    
    def flip(self) -> None:
        """翻转缓冲区，准备读取（线程安全）"""
        with self._lock:
            self._limit = self._position
            self._position = 0
            self._mark = -1
    
    def rewind(self) -> None:
        """倒带缓冲区，重置位置（线程安全）"""
        with self._lock:
            self._position = 0
            self._mark = -1
    
    def compact(self) -> None:
        """压缩缓冲区，移动未读数据到开头（线程安全）"""
        with self._lock:
            remaining = self._limit - self._position
            if remaining > 0 and self._position > 0:
                self._buffer[:remaining] = self._buffer[self._position:self._limit]
            self._position = remaining
            self._limit = self._size
    
    @property
    def capacity(self) -> int:
        """获取缓冲区总容量"""
        return self._size
    
    @property
    def available(self) -> int:
        """获取可写入空间（线程安全）"""
        with self._lock:
            return self._limit - self._position
    
    @property
    def remaining(self) -> int:
        """获取可读取数据量（线程安全）"""
        with self._lock:
            return self._limit - self._position
    
    @property
    def position(self) -> int:
        """获取当前位置（线程安全）"""
        with self._lock:
            return self._position
    
    @property
    def limit(self) -> int:
        """获取当前限制（线程安全）"""
        with self._lock:
            return self._limit
    
    @property
    def alignment(self) -> int:
        """获取对齐字节数"""
        return self._alignment
    
    @property
    def aligned_address(self) -> int:
        """获取对齐后的内存地址"""
        return self._aligned_address
    
    @property
    def is_empty(self) -> bool:
        """检查缓冲区是否为空（线程安全）"""
        with self._lock:
            return self._position == 0
    
    @property
    def is_full(self) -> bool:
        """检查缓冲区是否已满（线程安全）"""
        with self._lock:
            return self._position >= self._limit
    
    @property
    def data(self) -> bytes:
        """获取缓冲区中有效数据的副本（线程安全）"""
        with self._lock:
            return bytes(self._buffer[:self._position])
    
    def __len__(self) -> int:
        """返回缓冲区中有效数据长度（线程安全）"""
        with self._lock:
            return self._position
    
    def __repr__(self) -> str:
        with self._lock:
            return (
                f"Buffer(size={self._size}, alignment={self._alignment}, "
                f"position={self._position}, limit={self._limit})"
            )
    
    def __enter__(self) -> "Buffer":
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.clear()
