"""
MPSC队列模块

实现多生产者单消费者队列。
"""

import threading
import time
from typing import Any, Optional
from ..core.exceptions import BufferFullError


class MPSCQueue:
    """
    多生产者单消费者队列
    
    支持多个生产者同时入队，单个消费者出队。
    使用锁保护写入端，读取端无锁。
    使用 Condition 实现高效的阻塞等待。
    """
    
    def __init__(self, capacity: int):
        """
        初始化MPSC队列
        
        Args:
            capacity: 队列容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._capacity = capacity
        self._buffer = [None] * capacity
        
        # 读写指针
        self._head = 0  # 读指针（消费者）
        self._tail = 0  # 写指针（生产者）
        
        # 生产者锁和条件变量
        self._producer_lock = threading.Lock()
        self._not_full = threading.Condition(self._producer_lock)
        
        # 统计信息
        self._enqueue_count = 0
        self._dequeue_count = 0
    
    def enqueue(self, item: Any) -> bool:
        """
        入队操作（线程安全）
        
        Args:
            item: 要入队的元素
        
        Returns:
            是否成功入队
        """
        with self._producer_lock:
            next_tail = (self._tail + 1) % self._capacity
            
            if next_tail == self._head:
                # 队列已满
                return False
            
            self._buffer[self._tail] = item
            self._tail = next_tail
            self._enqueue_count += 1
            
            return True
    
    def enqueue_blocking(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        阻塞入队操作（使用 Condition 实现高效等待）
        
        Args:
            item: 要入队的元素
            timeout: 超时时间（秒）
        
        Returns:
            是否成功入队
        """
        with self._not_full:
            start_time = time.time()
            
            while self._is_full_unlocked():
                if timeout is None:
                    self._not_full.wait()
                else:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    if not self._not_full.wait(remaining):
                        return False
            
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._enqueue_count += 1
            
            return True
    
    def _is_full_unlocked(self) -> bool:
        """
        检查队列是否已满（内部方法，不加锁）
        
        注意：调用此方法前必须已持有锁
        
        Returns:
            队列是否已满
        """
        return (self._tail + 1) % self._capacity == self._head
    
    def dequeue(self) -> Optional[Any]:
        """
        出队操作（单消费者，无需锁）
        
        Returns:
            出队的元素，如果队列为空返回None
        """
        if self._head == self._tail:
            return None
        
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._dequeue_count += 1
        
        # 通知等待的生产者（队列有空位了）
        with self._not_full:
            self._not_full.notify()
        
        return item
    
    def peek(self) -> Optional[Any]:
        """
        查看队首元素但不移除
        
        Returns:
            队首元素，如果队列为空返回None
        """
        if self._head == self._tail:
            return None
        return self._buffer[self._head]
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self._head == self._tail
    
    def is_full(self) -> bool:
        """检查队列是否已满"""
        return (self._tail + 1) % self._capacity == self._head
    
    def size(self) -> int:
        """获取队列当前大小"""
        if self._tail >= self._head:
            return self._tail - self._head
        return self._capacity - self._head + self._tail
    
    def clear(self) -> None:
        """清空队列"""
        with self._producer_lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._tail = 0
    
    @property
    def capacity(self) -> int:
        """获取队列容量"""
        return self._capacity
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "capacity": self._capacity,
            "size": self.size(),
            "enqueue_count": self._enqueue_count,
            "dequeue_count": self._dequeue_count,
        }
    
    def __len__(self) -> int:
        """返回队列当前大小"""
        return self.size()
    
    def __repr__(self) -> str:
        return f"MPSCQueue(capacity={self._capacity}, size={self.size()})"
