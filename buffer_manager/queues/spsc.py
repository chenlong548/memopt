"""
SPSC无锁队列模块

实现单生产者单消费者无锁队列。
"""

import threading
from typing import Any, Optional
from collections import deque


class SPSCQueue:
    """
    单生产者单消费者无锁队列
    
    高性能的FIFO队列，适用于单生产者单消费者场景。
    使用简化的无锁设计，在Python中利用GIL实现线程安全。
    当容量为2的幂时，使用位运算优化取模操作。
    """
    
    def __init__(self, capacity: int):
        """
        初始化SPSC队列
        
        Args:
            capacity: 队列容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._capacity = capacity
        self._buffer = [None] * capacity
        
        # 检查容量是否为2的幂，用于位运算优化
        self._is_power_of_two = (capacity & (capacity - 1)) == 0
        self._mask = capacity - 1 if self._is_power_of_two else None
        
        # 读写指针（使用原子操作）
        self._head = 0  # 读指针
        self._tail = 0  # 写指针
        
        # 统计信息
        self._enqueue_count = 0
        self._dequeue_count = 0
    
    def _next_index(self, index: int) -> int:
        """
        计算下一个索引位置（使用位运算优化）
        
        Args:
            index: 当前索引
        
        Returns:
            下一个索引
        """
        if self._is_power_of_two and self._mask is not None:
            return (index + 1) & self._mask
        return (index + 1) % self._capacity
    
    def enqueue(self, item: Any) -> bool:
        """
        入队操作
        
        Args:
            item: 要入队的元素
        
        Returns:
            是否成功入队
        """
        tail = self._tail
        next_tail = self._next_index(tail)
        
        if next_tail == self._head:
            # 队列已满
            return False
        
        self._buffer[tail] = item
        self._tail = next_tail
        self._enqueue_count += 1
        
        return True
    
    def dequeue(self) -> Optional[Any]:
        """
        出队操作
        
        Returns:
            出队的元素，如果队列为空返回None
        """
        if self._head == self._tail:
            # 队列为空
            return None
        
        item = self._buffer[self._head]
        self._buffer[self._head] = None  # 帮助GC
        self._head = self._next_index(self._head)
        self._dequeue_count += 1
        
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
        return self._next_index(self._tail) == self._head
    
    def size(self) -> int:
        """获取队列当前大小"""
        if self._tail >= self._head:
            return self._tail - self._head
        return self._capacity - self._head + self._tail
    
    def clear(self) -> None:
        """清空队列"""
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
        return f"SPSCQueue(capacity={self._capacity}, size={self.size()})"
    
    def __iter__(self):
        """迭代器"""
        head = self._head
        while head != self._tail:
            yield self._buffer[head]
            head = self._next_index(head)
