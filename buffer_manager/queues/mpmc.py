"""
MPMC队列模块

实现多生产者多消费者队列。
"""

import threading
from typing import Any, Optional
from collections import deque


class MPMCQueue:
    """
    多生产者多消费者队列
    
    支持多个生产者和多个消费者同时操作。
    使用锁保护队列操作，确保线程安全。
    """
    
    def __init__(self, capacity: int):
        """
        初始化MPMC队列
        
        Args:
            capacity: 队列容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._capacity = capacity
        self._buffer = [None] * capacity
        
        # 读写指针
        self._head = 0
        self._tail = 0
        
        # 同步原语
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        # 统计信息
        self._enqueue_count = 0
        self._dequeue_count = 0
    
    def enqueue(self, item: Any) -> bool:
        """
        入队操作
        
        Args:
            item: 要入队的元素
        
        Returns:
            是否成功入队
        """
        with self._lock:
            next_tail = (self._tail + 1) % self._capacity
            
            if next_tail == self._head:
                # 队列已满
                return False
            
            self._buffer[self._tail] = item
            self._tail = next_tail
            self._enqueue_count += 1
            
            # 通知等待的消费者
            self._not_empty.notify()
            
            return True
    
    def enqueue_blocking(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        阻塞入队操作
        
        Args:
            item: 要入队的元素
            timeout: 超时时间（秒）
        
        Returns:
            是否成功入队
        """
        with self._not_full:
            while self._is_full_unlocked():
                if timeout is None:
                    self._not_full.wait()
                else:
                    if not self._not_full.wait(timeout):
                        return False
            
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._enqueue_count += 1
            
            # 通知等待的消费者
            self._not_empty.notify()
            
            return True
    
    def dequeue(self) -> Optional[Any]:
        """
        出队操作
        
        Returns:
            出队的元素，如果队列为空返回None
        """
        with self._lock:
            if self._head == self._tail:
                return None
            
            item = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._dequeue_count += 1
            
            # 通知等待的生产者
            self._not_full.notify()
            
            return item
    
    def dequeue_blocking(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        阻塞出队操作
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            出队的元素，如果超时返回None
        """
        with self._not_empty:
            while self._is_empty_unlocked():
                if timeout is None:
                    self._not_empty.wait()
                else:
                    if not self._not_empty.wait(timeout):
                        return None
            
            item = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._dequeue_count += 1
            
            # 通知等待的生产者
            self._not_full.notify()
            
            return item
    
    def peek(self) -> Optional[Any]:
        """
        查看队首元素但不移除
        
        Returns:
            队首元素，如果队列为空返回None
        """
        with self._lock:
            if self._head == self._tail:
                return None
            return self._buffer[self._head]
    
    def _is_empty_unlocked(self) -> bool:
        """
        检查队列是否为空（内部方法，不加锁）
        
        注意：调用此方法前必须已持有锁
        
        Returns:
            队列是否为空
        """
        return self._head == self._tail
    
    def _is_full_unlocked(self) -> bool:
        """
        检查队列是否已满（内部方法，不加锁）
        
        注意：调用此方法前必须已持有锁
        
        Returns:
            队列是否已满
        """
        return (self._tail + 1) % self._capacity == self._head
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        with self._lock:
            return self._head == self._tail
    
    def is_full(self) -> bool:
        """检查队列是否已满"""
        with self._lock:
            return (self._tail + 1) % self._capacity == self._head
    
    def size(self) -> int:
        """获取队列当前大小"""
        with self._lock:
            if self._tail >= self._head:
                return self._tail - self._head
            return self._capacity - self._head + self._tail
    
    def clear(self) -> None:
        """清空队列"""
        with self._lock:
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
        with self._lock:
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
        return f"MPMCQueue(capacity={self._capacity}, size={self.size()})"
