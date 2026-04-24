"""
LRU缓存实现模块

实现线程安全的LRU（最近最少使用）缓存。
"""

from typing import TypeVar, Generic, Optional, Dict, Callable
from collections import OrderedDict
import threading
import time

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    线程安全的LRU缓存

    该类实现了LRU（Least Recently Used）缓存淘汰策略，
    当缓存达到最大容量时，淘汰最久未使用的项。

    Attributes:
        _max_size: 最大容量
        _cache: 有序字典，维护访问顺序
        _lock: 线程锁
        _ttl: 可选的过期时间（秒）
        _timestamps: 键的访问时间戳

    Example:
        >>> cache = LRUCache(max_size=100)
        >>> cache.put("key1", "value1")
        >>> value = cache.get("key1")
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        初始化LRU缓存

        Args:
            max_size: 最大容量，默认1000
            ttl: 可选的过期时间（秒），None表示不过期
        """
        self._max_size = max_size
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._ttl = ttl
        self._timestamps: Dict[K, float] = {}

    def get(self, key: K) -> Optional[V]:
        """
        获取缓存值

        如果键存在且未过期，返回值并将其移到最近使用位置。
        如果键不存在或已过期，返回None。

        Args:
            key: 缓存键

        Returns:
            Optional[V]: 缓存值，如果不存在或已过期返回None
        """
        with self._lock:
            if key not in self._cache:
                return None

            # 检查是否过期
            if self._ttl is not None:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp > self._ttl:
                    # 过期，删除
                    del self._cache[key]
                    del self._timestamps[key]
                    return None

            # 移到最近使用位置
            value = self._cache.pop(key)
            self._cache[key] = value
            self._timestamps[key] = time.time()
            return value

    def put(self, key: K, value: V) -> None:
        """
        添加缓存项

        如果键已存在，更新值并移到最近使用位置。
        如果缓存已满，淘汰最久未使用的项。

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 如果键已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 如果缓存已满，删除最久未使用的项
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]

            # 添加新项
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def invalidate(self, key: K) -> bool:
        """
        使指定键失效

        Args:
            key: 缓存键

        Returns:
            bool: 如果键存在并删除返回True，否则返回False
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """
        清空缓存
        """
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """
        获取当前缓存大小

        Returns:
            int: 当前缓存项数量
        """
        with self._lock:
            return len(self._cache)

    def max_size(self) -> int:
        """
        获取最大容量

        Returns:
            int: 最大容量
        """
        return self._max_size

    def contains(self, key: K) -> bool:
        """
        检查是否包含指定键

        Args:
            key: 缓存键

        Returns:
            bool: 如果包含返回True，否则返回False
        """
        with self._lock:
            if key not in self._cache:
                return False

            # 检查是否过期
            if self._ttl is not None:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp > self._ttl:
                    return False

            return True

    def get_or_compute(self, key: K, compute_func: Callable[[], V]) -> V:
        """
        获取缓存值，如果不存在则计算并缓存

        Args:
            key: 缓存键
            compute_func: 计算函数，无参数，返回类型V的值

        Returns:
            V: 缓存值或计算结果
        """
        value = self.get(key)
        if value is not None:
            return value

        value = compute_func()
        self.put(key, value)
        return value

    def cleanup_expired(self) -> int:
        """
        清理过期项

        Returns:
            int: 清理的项数量
        """
        if self._ttl is None:
            return 0

        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp > self._ttl
            ]

            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                del self._timestamps[key]

            return len(expired_keys)

    def __len__(self) -> int:
        """返回当前缓存大小"""
        return self.size()

    def __contains__(self, key: K) -> bool:
        """检查是否包含指定键"""
        return self.contains(key)

    def __repr__(self) -> str:
        """字符串表示"""
        return f"LRUCache(size={self.size()}/{self._max_size}, ttl={self._ttl})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
