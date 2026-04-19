"""
多级缓存实现模块

实现多级缓存系统，支持不同速度和容量的缓存层级。
"""

from typing import TypeVar, Generic, Optional, List, Callable
import threading

K = TypeVar('K')
V = TypeVar('V')


class CacheLevel:
    """缓存级别定义"""
    L1 = "L1"  # 一级缓存（最快，容量最小）
    L2 = "L2"  # 二级缓存
    L3 = "L3"  # 三级缓存（最慢，容量最大）


class MultiLevelCache(Generic[K, V]):
    """
    多级缓存系统

    该类实现了多级缓存架构，支持不同速度和容量的缓存层级。
    数据首先在L1缓存中查找，如果未命中则查找L2，以此类推。
    写入时会同时写入所有层级。

    Attributes:
        _levels: 缓存层级列表
        _lock: 线程锁
        _stats: 统计信息

    Example:
        >>> cache = MultiLevelCache()
        >>> cache.add_level(CacheLevel.L1, LRUCache(max_size=100))
        >>> cache.add_level(CacheLevel.L2, LRUCache(max_size=1000))
        >>> cache.put("key1", "value1")
        >>> value = cache.get("key1")
    """

    def __init__(self):
        """初始化多级缓存"""
        self._levels: List[tuple] = []  # [(level_name, cache_instance), ...]
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'level_hits': {},  # {level_name: hit_count}
        }

    def add_level(self, level_name: str, cache_instance: Generic[K, V]) -> None:
        """
        添加缓存层级

        Args:
            level_name: 层级名称
            cache_instance: 缓存实例（需支持get/put/invalidate方法）
        """
        with self._lock:
            self._levels.append((level_name, cache_instance))
            self._stats['level_hits'][level_name] = 0

    def get(self, key: K) -> Optional[V]:
        """
        从多级缓存获取值

        按顺序从各级缓存查找，找到后会将值提升到更快的层级。

        Args:
            key: 缓存键

        Returns:
            Optional[V]: 缓存值，如果所有层级都未命中返回None
        """
        with self._lock:
            for i, (level_name, cache) in enumerate(self._levels):
                value = cache.get(key)
                if value is not None:
                    # 命中，更新统计
                    self._stats['hits'] += 1
                    self._stats['level_hits'][level_name] += 1

                    # 提升到更快的层级
                    for j in range(i):
                        faster_level_name, faster_cache = self._levels[j]
                        faster_cache.put(key, value)

                    return value

            # 未命中
            self._stats['misses'] += 1
            return None

    def put(self, key: K, value: V) -> None:
        """
        将值写入所有缓存层级

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            for level_name, cache in self._levels:
                cache.put(key, value)

    def invalidate(self, key: K) -> None:
        """
        使所有层级的指定键失效

        Args:
            key: 缓存键
        """
        with self._lock:
            for level_name, cache in self._levels:
                cache.invalidate(key)

    def clear(self) -> None:
        """清空所有缓存层级"""
        with self._lock:
            for level_name, cache in self._levels:
                cache.clear()

    def get_or_compute(self, key: K, compute_func: Callable[[], V]) -> V:
        """
        获取缓存值，如果所有层级都未命中则计算并缓存

        Args:
            key: 缓存键
            compute_func: 计算函数

        Returns:
            V: 缓存值或计算结果
        """
        value = self.get(key)
        if value is not None:
            return value

        value = compute_func()
        self.put(key, value)
        return value

    def get_hit_rate(self) -> float:
        """
        获取缓存命中率

        Returns:
            float: 命中率（0.0-1.0）
        """
        total = self._stats['hits'] + self._stats['misses']
        if total == 0:
            return 0.0
        return self._stats['hits'] / total

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 统计信息字典
        """
        with self._lock:
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': self.get_hit_rate(),
                'level_hits': dict(self._stats['level_hits']),
                'level_count': len(self._levels),
            }

    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._stats['hits'] = 0
            self._stats['misses'] = 0
            for level_name in self._stats['level_hits']:
                self._stats['level_hits'][level_name] = 0

    def level_count(self) -> int:
        """
        获取缓存层级数量

        Returns:
            int: 层级数量
        """
        return len(self._levels)

    def __len__(self) -> int:
        """返回第一级缓存的大小"""
        if not self._levels:
            return 0
        return len(self._levels[0][1])

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"MultiLevelCache(levels={len(self._levels)}, "
                f"hit_rate={self.get_hit_rate():.2%})")

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
