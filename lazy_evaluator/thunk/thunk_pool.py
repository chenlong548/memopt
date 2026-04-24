"""
ThunkPool对象池管理模块

实现Thunk对象池，支持对象复用，减少GC压力。
"""

from typing import Optional, Callable, TypeVar, Generic
import threading
from queue import Queue
from .memo_thunk import Memothunk

T = TypeVar('T')


class ThunkPool:
    """
    Thunk对象池，支持对象复用

    该类实现了对象池模式，用于管理和复用Memothunk对象，
    减少频繁创建和销毁对象带来的GC压力。

    学术支撑：
    - Thunk Recycling - 对象池复用，减少GC压力

    Attributes:
        _pool: 对象池队列
        _lock: 线程锁
        _max_size: 池最大容量
        _current_size: 当前池大小
        _created_count: 已创建的对象总数

    Example:
        >>> pool = ThunkPool(max_size=10)
        >>> thunk = pool.acquire()
        >>> thunk.set_computation(lambda: 42)
        >>> result = thunk.get()
        >>> pool.release(thunk)
    """

    def __init__(self, max_size: int = 100):
        """
        初始化ThunkPool

        Args:
            max_size: 池最大容量，默认100
        """
        self._pool: Queue = Queue()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._current_size = 0
        self._created_count = 0

    def acquire(self, computation: Optional[Callable[[], T]] = None) -> Memothunk[T]:
        """
        从池中获取一个Memothunk对象

        如果池中有可用对象，返回复用的对象。
        如果池为空且未达到最大容量，创建新对象。
        如果池为空且已达到最大容量，创建新对象（但不会添加到池中）。

        Args:
            computation: 可选的计算函数，如果提供会设置到thunk上

        Returns:
            Memothunk[T]: Memothunk实例
        """
        # 尝试从池中获取
        try:
            thunk = self._pool.get_nowait()
            if computation is not None:
                thunk.set_computation(computation)
            return thunk  # type: ignore
        except:
            pass

        # 池为空，创建新对象
        with self._lock:
            self._created_count += 1
            if computation is not None:
                return Memothunk(computation)
            else:
                # 创建一个默认的空计算函数
                return Memothunk(lambda: None)  # type: ignore

    def release(self, thunk: Memothunk) -> None:
        """
        将Memothunk对象归还到池中

        Args:
            thunk: 要归还的Memothunk实例
        """
        # 重置thunk状态
        thunk.reset()

        # 如果池未满，添加到池中
        with self._lock:
            if self._current_size < self._max_size:
                self._pool.put(thunk)
                self._current_size += 1

    def clear(self) -> None:
        """
        清空对象池
        """
        with self._lock:
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except:
                    break
            self._current_size = 0

    def size(self) -> int:
        """
        获取当前池中可用对象数量

        Returns:
            int: 可用对象数量
        """
        return self._current_size

    def max_size(self) -> int:
        """
        获取池最大容量

        Returns:
            int: 最大容量
        """
        return self._max_size

    def created_count(self) -> int:
        """
        获取已创建的对象总数

        Returns:
            int: 已创建的对象总数
        """
        return self._created_count

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ThunkPool(size={self._current_size}/{self._max_size}, "
                f"created={self._created_count})")

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()


class PooledThunk(Generic[T]):
    """
    池化的Thunk包装器

    提供自动归还机制的Thunk包装器，使用上下文管理器模式。

    Example:
        >>> pool = ThunkPool()
        >>> with PooledThunk(pool, lambda: 42) as thunk:
        ...     result = thunk.get()
    """

    def __init__(self, pool: ThunkPool, computation: Callable[[], T]):
        """
        初始化PooledThunk

        Args:
            pool: ThunkPool实例
            computation: 计算函数
        """
        self._pool = pool
        self._computation = computation
        self._thunk: Optional[Memothunk[T]] = None

    def __enter__(self) -> Memothunk[T]:
        """进入上下文，获取thunk"""
        self._thunk = self._pool.acquire(self._computation)
        return self._thunk

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """退出上下文，归还thunk"""
        if self._thunk is not None:
            self._pool.release(self._thunk)
            self._thunk = None
