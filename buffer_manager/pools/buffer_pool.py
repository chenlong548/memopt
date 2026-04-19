"""
缓冲池管理模块

提供缓冲区的池化管理，支持高效获取和释放缓冲区。
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque
from ..core.buffer import Buffer
from ..core.exceptions import PoolExhaustedError, BufferTimeoutError


@dataclass
class PoolStats:
    """缓冲池统计信息"""
    total_buffers: int = 0  # 总缓冲区数
    available_buffers: int = 0  # 可用缓冲区数
    total_acquires: int = 0  # 总获取次数
    total_releases: int = 0  # 总释放次数
    total_timeouts: int = 0  # 超时次数
    avg_acquire_time: float = 0.0  # 平均获取时间（毫秒）
    peak_usage: int = 0  # 峰值使用量
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "total_buffers": self.total_buffers,
            "available_buffers": self.available_buffers,
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_timeouts": self.total_timeouts,
            "avg_acquire_time": self.avg_acquire_time,
            "peak_usage": self.peak_usage,
        }


class BufferPool:
    """
    缓冲池管理类
    
    管理一组预分配的缓冲区，支持高效的获取和释放操作。
    线程安全的实现。
    """
    
    def __init__(self, buffer_size: int, num_buffers: int, alignment: int = 64):
        """
        初始化缓冲池
        
        Args:
            buffer_size: 每个缓冲区的大小（字节）
            num_buffers: 缓冲区数量
            alignment: 内存对齐字节数
        """
        self._buffer_size = buffer_size
        self._num_buffers = num_buffers
        self._alignment = alignment
        
        # 预分配缓冲区
        self._buffers: List[Buffer] = [
            Buffer(buffer_size, alignment) for _ in range(num_buffers)
        ]
        
        # 可用缓冲区队列
        self._available: List[Buffer] = self._buffers.copy()
        
        # 使用中的缓冲区集合
        self._in_use: set = set()
        
        # 同步原语
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # 统计信息
        self._stats = PoolStats(
            total_buffers=num_buffers,
            available_buffers=num_buffers,
        )
        # 使用 deque 替代列表，自动限制大小，避免高并发下的内存压力
        self._acquire_times: deque = deque(maxlen=1000)
    
    def acquire(self, timeout: Optional[float] = None) -> Buffer:
        """
        从池中获取一个缓冲区
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
        
        Returns:
            Buffer对象
        
        Raises:
            BufferTimeoutError: 获取超时
            PoolExhaustedError: 池已耗尽
        """
        start_time = time.time()
        
        with self._condition:
            while not self._available:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self._stats.total_timeouts += 1
                        raise BufferTimeoutError(
                            f"Failed to acquire buffer within {timeout} seconds"
                        )
                    self._condition.wait(timeout - elapsed)
                else:
                    self._condition.wait()
            
            if not self._available:
                raise PoolExhaustedError("No available buffers in pool")
            
            buffer = self._available.pop()
            self._in_use.add(buffer)
            
            # 更新统计
            self._stats.available_buffers = len(self._available)
            self._stats.total_acquires += 1
            self._stats.peak_usage = max(
                self._stats.peak_usage, 
                len(self._in_use)
            )
            
            # 记录获取时间（deque 自动限制大小，无需手动截断）
            acquire_time = (time.time() - start_time) * 1000
            self._acquire_times.append(acquire_time)
            self._stats.avg_acquire_time = (
                sum(self._acquire_times) / len(self._acquire_times)
            )
            
            return buffer
    
    def release(self, buffer: Buffer) -> None:
        """
        释放缓冲区回池中
        
        Args:
            buffer: 要释放的缓冲区
        
        Raises:
            ValueError: 缓冲区不属于此池
        """
        with self._condition:
            if buffer not in self._in_use:
                raise ValueError("Buffer does not belong to this pool or already released")
            
            # 清空缓冲区
            buffer.clear()
            
            self._in_use.remove(buffer)
            self._available.append(buffer)
            
            # 更新统计
            self._stats.available_buffers = len(self._available)
            self._stats.total_releases += 1
            
            # 通知等待的线程
            self._condition.notify()
    
    def resize(self, new_size: int) -> None:
        """
        调整池大小
        
        Args:
            new_size: 新的缓冲区数量
        """
        with self._lock:
            if new_size > self._num_buffers:
                # 扩展池
                additional = new_size - self._num_buffers
                for _ in range(additional):
                    buffer = Buffer(self._buffer_size, self._alignment)
                    self._buffers.append(buffer)
                    self._available.append(buffer)
            elif new_size < self._num_buffers:
                # 收缩池（只能移除空闲缓冲区）
                to_remove = self._num_buffers - new_size
                removed = 0
                new_available = []
                for buffer in self._available:
                    if removed < to_remove:
                        self._buffers.remove(buffer)
                        removed += 1
                    else:
                        new_available.append(buffer)
                self._available = new_available
            
            self._num_buffers = new_size
            self._stats.total_buffers = new_size
            self._stats.available_buffers = len(self._available)
    
    def get_stats(self) -> PoolStats:
        """
        获取池统计信息
        
        Returns:
            PoolStats对象
        """
        with self._lock:
            return PoolStats(
                total_buffers=self._stats.total_buffers,
                available_buffers=self._stats.available_buffers,
                total_acquires=self._stats.total_acquires,
                total_releases=self._stats.total_releases,
                total_timeouts=self._stats.total_timeouts,
                avg_acquire_time=self._stats.avg_acquire_time,
                peak_usage=self._stats.peak_usage,
            )
    
    @property
    def stats(self) -> PoolStats:
        """获取统计信息"""
        return self.get_stats()
    
    @property
    def available_count(self) -> int:
        """获取可用缓冲区数量"""
        with self._lock:
            return len(self._available)
    
    @property
    def in_use_count(self) -> int:
        """获取使用中缓冲区数量"""
        with self._lock:
            return len(self._in_use)
    
    @property
    def buffer_size(self) -> int:
        """获取缓冲区大小"""
        return self._buffer_size
    
    @property
    def capacity(self) -> int:
        """获取池容量"""
        return self._num_buffers
    
    def __len__(self) -> int:
        """返回池中缓冲区总数"""
        return self._num_buffers
    
    def __repr__(self) -> str:
        return (
            f"BufferPool(buffer_size={self._buffer_size}, "
            f"capacity={self._num_buffers}, "
            f"available={len(self._available)})"
        )
