"""
Lazy[T] 泛型类实现

实现支持Call-by-need语义的惰性值包装器。
"""

from enum import Enum
from typing import Generic, TypeVar, Callable, Optional
import threading
from .exceptions import LazyEvaluationError

T = TypeVar('T')
U = TypeVar('U')


class ThunkState(Enum):
    """Thunk状态枚举"""
    UNEVALUATED = "unevaluated"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"


class Lazy(Generic[T]):
    """
    惰性值包装器，支持Call-by-need语义（线程安全）

    该类实现了惰性求值，只有在真正需要值时才进行计算。
    计算结果会被缓存，后续访问直接返回缓存值。
    
    线程安全实现：
    - 使用RLock保护状态转换
    - 使用Condition让等待线程阻塞直到计算完成
    - 使用ThreadLocal检测同一线程内的循环依赖

    Attributes:
        _thunk: 延迟计算的函数
        _state: 当前求值状态
        _value: 缓存的计算结果
        _error: 计算过程中的错误（如果有）
        _lock: 线程锁，保护状态转换
        _condition: 条件变量，用于线程等待
        _evaluating_thread: 当前正在求值的线程ID

    Example:
        >>> lazy_val = Lazy(lambda: expensive_computation())
        >>> result = lazy_val.force()  # 此时才执行计算
    """

    def __init__(self, thunk: Callable[[], T]):
        """
        初始化惰性值

        Args:
            thunk: 延迟计算的函数，无参数，返回类型T的值
        """
        self._thunk = thunk
        self._state = ThunkState.UNEVALUATED
        self._value: Optional[T] = None
        self._error: Optional[Exception] = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._evaluating_thread: Optional[int] = None

    def force(self) -> T:
        """
        强制求值，返回计算结果（线程安全）

        如果尚未求值，执行计算并缓存结果。
        如果已经求值，直接返回缓存结果。
        如果正在求值（同一线程内的循环依赖），抛出异常。
        如果其他线程正在求值，等待其完成。

        线程安全：使用条件变量实现等待/通知机制。

        Returns:
            T: 计算结果

        Raises:
            LazyEvaluationError: 如果计算过程中出现错误或检测到循环依赖
        """
        with self._condition:
            # 已求值：直接返回结果或抛出错误
            if self._state == ThunkState.EVALUATED:
                if self._error is not None:
                    raise self._error
                if self._value is None:
                    raise LazyEvaluationError("Lazy value is None after evaluation")
                return self._value

            # 正在求值：检查是否是同一线程（循环依赖）
            if self._state == ThunkState.EVALUATING:
                current_thread = threading.current_thread().ident
                if self._evaluating_thread == current_thread:
                    # 同一线程内的循环依赖
                    raise LazyEvaluationError(
                        "Circular dependency detected during lazy evaluation"
                    )
                # 其他线程正在求值，等待完成
                while self._state == ThunkState.EVALUATING:
                    self._condition.wait()
                
                # 重新检查状态
                if self._error is not None:
                    raise self._error
                if self._value is None:
                    raise LazyEvaluationError("Lazy value is None after evaluation")
                return self._value

            # 开始求值
            self._state = ThunkState.EVALUATING
            self._evaluating_thread = threading.current_thread().ident

        # 在锁外执行计算，避免死锁
        try:
            value = self._thunk()
            with self._condition:
                self._value = value
                self._state = ThunkState.EVALUATED
                self._evaluating_thread = None
                # 通知所有等待的线程
                self._condition.notify_all()
            return value
        except Exception as e:
            # 包装为LazyEvaluationError
            lazy_error = LazyEvaluationError(f"Error during lazy evaluation: {e}") if not isinstance(e, LazyEvaluationError) else e
            with self._condition:
                self._error = lazy_error
                self._state = ThunkState.EVALUATED
                self._evaluating_thread = None
                # 通知所有等待的线程
                self._condition.notify_all()
            raise lazy_error from e

    def map(self, f: Callable[[T], U]) -> 'Lazy[U]':
        """
        映射操作，对惰性值应用函数

        Args:
            f: 映射函数，接受类型T的值，返回类型U的值

        Returns:
            Lazy[U]: 新的惰性值

        Example:
            >>> lazy_val = Lazy(lambda: 10)
            >>> mapped = lazy_val.map(lambda x: x * 2)
            >>> mapped.force()  # 返回 20
        """
        return Lazy(lambda: f(self.force()))

    def flat_map(self, f: Callable[[T], 'Lazy[U]']) -> 'Lazy[U]':
        """
        扁平映射操作，对惰性值应用返回惰性值的函数

        Args:
            f: 映射函数，接受类型T的值，返回Lazy[U]

        Returns:
            Lazy[U]: 新的惰性值

        Example:
            >>> lazy_val = Lazy(lambda: 10)
            >>> flat_mapped = lazy_val.flat_map(lambda x: Lazy(lambda: x * 2))
            >>> flat_mapped.force()  # 返回 20
        """
        return Lazy(lambda: f(self.force()).force())

    def is_evaluated(self) -> bool:
        """
        检查是否已经求值

        Returns:
            bool: 如果已经求值返回True，否则返回False
        """
        return self._state == ThunkState.EVALUATED

    def is_evaluating(self) -> bool:
        """
        检查是否正在求值

        Returns:
            bool: 如果正在求值返回True，否则返回False
        """
        return self._state == ThunkState.EVALUATING

    def get_state(self) -> ThunkState:
        """
        获取当前状态

        Returns:
            ThunkState: 当前求值状态
        """
        return self._state

    def reset(self) -> None:
        """
        重置状态，允许重新计算（线程安全）

        注意：此操作会清除缓存的值，下次访问时会重新计算
        """
        with self._condition:
            self._state = ThunkState.UNEVALUATED
            self._value = None
            self._error = None
            self._evaluating_thread = None

    def __repr__(self) -> str:
        """字符串表示"""
        state_str = self._state.value
        if self._state == ThunkState.EVALUATED and self._error is None:
            return f"Lazy(value={self._value}, state={state_str})"
        return f"Lazy(state={state_str})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
