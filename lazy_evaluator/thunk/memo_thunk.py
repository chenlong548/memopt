"""
Memothunk实现模块

实现带记忆化的thunk，支持Call-by-need语义。
"""

from typing import TypeVar, Callable, Optional, Generic
import threading
import uuid

T = TypeVar('T')


class Memothunk(Generic[T]):
    """
    带记忆化的thunk，只计算一次

    该类实现了记忆化的延迟计算，确保计算只执行一次，
    后续访问直接返回缓存的结果。

    学术支撑：
    - Memothunk - 带记忆化的thunk实现，支持Call-by-need语义
    - Thunk Recycling - 对象池复用，减少GC压力

    Attributes:
        _computation: 延迟计算的函数
        _value: 缓存的计算结果
        _evaluated: 是否已经求值
        _lock: 线程锁，确保线程安全
        _id: Thunk唯一标识符

    Example:
        >>> thunk = Memothunk(lambda: expensive_computation())
        >>> result1 = thunk.get()  # 执行计算
        >>> result2 = thunk.get()  # 直接返回缓存结果
    """

    def __init__(self, computation: Callable[[], T]):
        """
        初始化Memothunk

        Args:
            computation: 延迟计算的函数，无参数，返回类型T的值
        """
        self._computation = computation
        self._value: Optional[T] = None
        self._evaluated = False
        self._lock = threading.Lock()
        self._id = str(uuid.uuid4())

    def get(self) -> T:
        """
        获取值，首次调用时计算

        如果尚未求值，执行计算并缓存结果。
        如果已经求值，直接返回缓存结果。
        线程安全：使用锁确保多线程环境下只计算一次。

        Returns:
            T: 计算结果
        """
        if self._evaluated:
            return self._value

        with self._lock:
            # 双重检查锁定模式
            if self._evaluated:
                return self._value

            self._value = self._computation()
            self._evaluated = True
            return self._value

    def is_evaluated(self) -> bool:
        """
        检查是否已经求值

        Returns:
            bool: 如果已经求值返回True，否则返回False
        """
        return self._evaluated

    def reset(self) -> None:
        """
        重置状态，允许重新计算

        注意：此操作会清除缓存的值，下次访问时会重新计算
        """
        with self._lock:
            self._value = None
            self._evaluated = False

    def set_computation(self, computation: Callable[[], T]) -> None:
        """
        设置新的计算函数

        Args:
            computation: 新的计算函数

        注意：此操作会重置状态
        """
        with self._lock:
            self._computation = computation
            self._value = None
            self._evaluated = False

    def get_id(self) -> str:
        """
        获取Thunk唯一标识符

        Returns:
            str: UUID字符串
        """
        return self._id

    def __repr__(self) -> str:
        """字符串表示"""
        if self._evaluated:
            return f"Memothunk(id={self._id[:8]}, evaluated=True, value={self._value})"
        return f"Memothunk(id={self._id[:8]}, evaluated=False)"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
