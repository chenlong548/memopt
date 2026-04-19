"""
替换策略模块

实现LRU和ARC（自适应替换缓存）替换策略。
"""

import threading
from typing import Any, Optional
from collections import OrderedDict


class LRU:
    """
    LRU（最近最少使用）缓存
    
    线程安全的LRU缓存实现。
    """
    
    def __init__(self, capacity: int):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值，如果不存在返回None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # 移动到末尾（最近使用）
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            return value
    
    def put(self, key: Any, value: Any) -> Optional[Any]:
        """
        放入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        
        Returns:
            被淘汰的值，如果没有淘汰返回None
        """
        with self._lock:
            evicted = None
            
            if key in self._cache:
                # 更新已存在的键
                self._cache.pop(key)
            elif len(self._cache) >= self._capacity:
                # 淘汰最久未使用的
                evicted_key, evicted = self._cache.popitem(last=False)
            
            self._cache[key] = value
            return evicted
    
    def remove(self, key: Any) -> Optional[Any]:
        """
        移除缓存项
        
        Args:
            key: 缓存键
        
        Returns:
            被移除的值，如果不存在返回None
        """
        with self._lock:
            if key in self._cache:
                return self._cache.pop(key)
            return None
    
    def contains(self, key: Any) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """获取当前缓存大小"""
        with self._lock:
            return len(self._cache)
    
    @property
    def capacity(self) -> int:
        """获取缓存容量"""
        return self._capacity
    
    @property
    def hit_rate(self) -> float:
        """获取命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
    
    def __repr__(self) -> str:
        return f"LRU(capacity={self._capacity}, size={self.size()})"


class ARC:
    """
    ARC（自适应替换缓存）
    
    结合LRU和LFU的优点，自适应调整缓存策略。
    """
    
    def __init__(self, capacity: int):
        """
        初始化ARC缓存
        
        Args:
            capacity: 缓存容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._capacity = capacity
        self._p = 0  # 目标大小调整参数
        
        # 四个链表
        self._t1: OrderedDict = OrderedDict()  # 最近访问一次的缓存
        self._t2: OrderedDict = OrderedDict()  # 最近访问多次的缓存
        self._b1: OrderedDict = OrderedDict()  # t1的幽灵列表
        self._b2: OrderedDict = OrderedDict()  # t2的幽灵列表
        
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值，如果不存在返回None
        """
        with self._lock:
            # 在t1或t2中找到
            if key in self._t1:
                self._hits += 1
                value = self._t1.pop(key)
                self._t2[key] = value  # 移动到t2
                return value
            
            if key in self._t2:
                self._hits += 1
                value = self._t2.pop(key)
                self._t2[key] = value  # 移动到末尾
                return value
            
            self._misses += 1
            return None
    
    def put(self, key: Any, value: Any) -> None:
        """
        放入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 如果已在缓存中，更新并移动
            if key in self._t1:
                self._t1.pop(key)
                self._t2[key] = value
                return
            
            if key in self._t2:
                self._t2.pop(key)
                self._t2[key] = value
                return
            
            # 在幽灵列表中
            if key in self._b1:
                # 调整p
                self._p = min(
                    self._capacity,
                    self._p + max(1, len(self._b2) // len(self._b1))
                )
                self._replace(key)
                self._b1.pop(key)
                self._t2[key] = value
                return
            
            if key in self._b2:
                # 调整p
                self._p = max(
                    0,
                    self._p - max(1, len(self._b1) // len(self._b2))
                )
                self._replace(key)
                self._b2.pop(key)
                self._t2[key] = value
                return
            
            # 新条目
            if len(self._t1) + len(self._b1) >= self._capacity:
                if len(self._t1) < self._capacity:
                    if self._b1:
                        self._b1.popitem(last=False)
                    self._replace(key)
                else:
                    if self._t1:
                        self._t1.popitem(last=False)
            
            if (len(self._t1) + len(self._b1) + 
                len(self._t2) + len(self._b2) >= 2 * self._capacity):
                if len(self._t2) + len(self._b2) >= self._capacity:
                    if self._b2:
                        self._b2.popitem(last=False)
                else:
                    if self._b1:
                        self._b1.popitem(last=False)
            
            self._t1[key] = value
    
    def _replace(self, key: Any) -> None:
        """替换策略"""
        t1_size = len(self._t1)
        
        if (t1_size > 0 and 
            (t1_size > self._p or 
             (key in self._b2 and t1_size == self._p))):
            # 从t1淘汰到b1
            k, v = self._t1.popitem(last=False)
            self._b1[k] = v
        else:
            # 从t2淘汰到b2
            if self._t2:
                k, v = self._t2.popitem(last=False)
                self._b2[k] = v
    
    def contains(self, key: Any) -> bool:
        """检查键是否在缓存中"""
        with self._lock:
            return key in self._t1 or key in self._t2
    
    def remove(self, key: Any) -> Optional[Any]:
        """移除缓存项"""
        with self._lock:
            if key in self._t1:
                return self._t1.pop(key)
            if key in self._t2:
                return self._t2.pop(key)
            if key in self._b1:
                return self._b1.pop(key)
            if key in self._b2:
                return self._b2.pop(key)
            return None
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._t1.clear()
            self._t2.clear()
            self._b1.clear()
            self._b2.clear()
            self._p = 0
    
    def size(self) -> int:
        """获取当前缓存大小"""
        with self._lock:
            return len(self._t1) + len(self._t2)
    
    @property
    def capacity(self) -> int:
        """获取缓存容量"""
        return self._capacity
    
    @property
    def hit_rate(self) -> float:
        """获取命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": self.size(),
                "t1_size": len(self._t1),
                "t2_size": len(self._t2),
                "b1_size": len(self._b1),
                "b2_size": len(self._b2),
                "p": self._p,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
    
    def __repr__(self) -> str:
        return f"ARC(capacity={self._capacity}, size={self.size()})"
