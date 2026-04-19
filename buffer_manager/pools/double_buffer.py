"""
双缓冲模块

实现双缓冲机制，支持前后缓冲区交换。
"""

import threading
from typing import Optional, Generator
from contextlib import contextmanager
from ..core.buffer import Buffer


class DoubleBuffer:
    """
    双缓冲实现
    
    提供前后两个缓冲区，支持原子交换操作。
    常用于生产者-消费者场景。
    线程安全的实现，提供上下文管理器模式确保原子操作。
    """
    
    def __init__(self, buffer_size: int, alignment: int = 64):
        """
        初始化双缓冲
        
        Args:
            buffer_size: 每个缓冲区的大小（字节）
            alignment: 内存对齐字节数
        """
        self._buffer_size = buffer_size
        self._alignment = alignment
        
        # 创建两个缓冲区
        self._buffers = [
            Buffer(buffer_size, alignment),
            Buffer(buffer_size, alignment),
        ]
        
        # 前后缓冲区索引
        self._front_index = 0
        self._back_index = 1
        
        # 同步原语
        self._lock = threading.Lock()
        self._swap_count = 0
    
    def swap(self) -> None:
        """
        交换前后缓冲区
        
        原子操作，将后缓冲区变为前缓冲区，前缓冲区变为后缓冲区。
        交换后，新的前缓冲区会被flip准备读取，新的后缓冲区会被清空。
        """
        with self._lock:
            # 交换索引
            self._front_index, self._back_index = self._back_index, self._front_index
            
            # Flip新的前缓冲区，准备读取
            self._buffers[self._front_index].flip()
            
            # 清空新的后缓冲区
            self._buffers[self._back_index].clear()
            
            self._swap_count += 1
    
    @contextmanager
    def get_front(self) -> Generator[Buffer, None, None]:
        """
        获取前缓冲区的上下文管理器（线程安全）
        
        在上下文管理器内，锁会被持有，确保对前缓冲区的操作是原子的。
        
        Yields:
            前缓冲区Buffer对象
        
        Example:
            with double_buffer.get_front() as front:
                data = front.read(100)
        """
        with self._lock:
            yield self._buffers[self._front_index]
    
    @contextmanager
    def get_back(self) -> Generator[Buffer, None, None]:
        """
        获取后缓冲区的上下文管理器（线程安全）
        
        在上下文管理器内，锁会被持有，确保对后缓冲区的操作是原子的。
        
        Yields:
            后缓冲区Buffer对象
        
        Example:
            with double_buffer.get_back() as back:
                back.write(data)
        """
        with self._lock:
            yield self._buffers[self._back_index]
    
    @property
    def front(self) -> Buffer:
        """
        获取前缓冲区（用于读取）
        
        警告：此属性返回 Buffer 对象后锁立即释放，存在线程安全风险。
        推荐使用 get_front() 上下文管理器进行原子操作。
        
        Returns:
            前缓冲区Buffer对象
        """
        with self._lock:
            return self._buffers[self._front_index]
    
    @property
    def back(self) -> Buffer:
        """
        获取后缓冲区（用于写入）
        
        警告：此属性返回 Buffer 对象后锁立即释放，存在线程安全风险。
        推荐使用 get_back() 上下文管理器进行原子操作。
        
        Returns:
            后缓冲区Buffer对象
        """
        with self._lock:
            return self._buffers[self._back_index]
    
    def write_to_back(self, data: bytes) -> int:
        """
        向后缓冲区写入数据
        
        Args:
            data: 要写入的数据
        
        Returns:
            实际写入的字节数
        """
        with self._lock:
            return self._buffers[self._back_index].write(data)
    
    def read_from_front(self, size: int) -> bytes:
        """
        从前缓冲区读取数据
        
        Args:
            size: 要读取的字节数
        
        Returns:
            读取的字节数据
        """
        with self._lock:
            return self._buffers[self._front_index].read(size)
    
    def peek_front(self, size: int) -> bytes:
        """
        查看前缓冲区数据但不移动位置指针
        
        Args:
            size: 要查看的字节数
        
        Returns:
            查看的字节数据
        """
        with self._lock:
            return self._buffers[self._front_index].peek(size)
    
    def clear_front(self) -> None:
        """清空前缓冲区"""
        with self._lock:
            self._buffers[self._front_index].clear()
    
    def clear_back(self) -> None:
        """清空后缓冲区"""
        with self._lock:
            self._buffers[self._back_index].clear()
    
    def clear_all(self) -> None:
        """清空所有缓冲区"""
        with self._lock:
            self._buffers[0].clear()
            self._buffers[1].clear()
    
    @property
    def buffer_size(self) -> int:
        """获取缓冲区大小"""
        return self._buffer_size
    
    @property
    def alignment(self) -> int:
        """获取对齐字节数"""
        return self._alignment
    
    @property
    def swap_count(self) -> int:
        """获取交换次数"""
        return self._swap_count
    
    @property
    def front_available(self) -> int:
        """获取前缓冲区可读数据量"""
        with self._lock:
            return self._buffers[self._front_index].remaining
    
    @property
    def back_available(self) -> int:
        """获取后缓冲区可写空间"""
        with self._lock:
            return self._buffers[self._back_index].available
    
    def __repr__(self) -> str:
        return (
            f"DoubleBuffer(buffer_size={self._buffer_size}, "
            f"swaps={self._swap_count})"
        )
