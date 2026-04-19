"""
stream_processor 辅助函数

提供常用的辅助功能。
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
import time
import hashlib
import uuid
from functools import wraps


T = TypeVar('T')


def generate_id() -> str:
    """
    生成唯一ID

    Returns:
        str: 唯一ID
    """
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """
    生成短ID

    Returns:
        str: 短ID
    """
    return uuid.uuid4().hex[:8]


def hash_value(value: Any) -> str:
    """
    计算值的哈希

    Args:
        value: 值

    Returns:
        str: 哈希值
    """
    value_str = str(value)
    return hashlib.md5(value_str.encode()).hexdigest()


def timestamp_ms() -> int:
    """
    获取当前时间戳（毫秒）

    Returns:
        int: 毫秒时间戳
    """
    return int(time.time() * 1000)


def timestamp_s() -> int:
    """
    获取当前时间戳（秒）

    Returns:
        int: 秒时间戳
    """
    return int(time.time())


def format_duration(seconds: float) -> str:
    """
    格式化持续时间

    Args:
        seconds: 秒数

    Returns:
        str: 格式化字符串
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(size: int) -> str:
    """
    格式化字节数

    Args:
        size: 字节数

    Returns:
        str: 格式化字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PB"


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    分块列表

    Args:
        lst: 列表
        chunk_size: 块大小

    Returns:
        List[List[T]]: 分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(lst: List[List[T]]) -> List[T]:
    """
    展平列表

    Args:
        lst: 嵌套列表

    Returns:
        List[T]: 展平后的列表
    """
    result = []
    for item in lst:
        result.extend(item)
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并字典

    Args:
        *dicts: 字典列表

    Returns:
        Dict: 合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def retry(func: Optional[Callable] = None,
          max_attempts: int = 3,
          delay: float = 1.0,
          backoff: float = 2.0,
          max_delay: float = 60.0,
          exceptions: tuple = (Exception,),
          on_retry: Optional[Callable[[Exception, int, float], None]] = None) -> Callable:
    """
    重试装饰器

    Args:
        func: 函数（可选，支持无参数调用）
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 退避因子
        max_delay: 最大延迟（秒）
        exceptions: 可重试的异常类型元组
        on_retry: 重试回调函数 (exception, attempt, delay) -> None

    Returns:
        Callable: 装饰后的函数

    Raises:
        Exception: 所有尝试失败后抛出最后一次异常

    Examples:
        # 基本用法
        @retry(max_attempts=3)
        def my_function():
            pass

        # 指定可重试的异常类型
        @retry(exceptions=(ConnectionError, TimeoutError))
        def my_network_call():
            pass

        # 使用回调函数
        def on_retry_callback(exc, attempt, delay):
            print(f"Retry {attempt} after {delay}s due to {exc}")

        @retry(on_retry=on_retry_callback)
        def my_function():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            current_delay = min(delay, max_delay)
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # 如果不是最后一次尝试，则等待并重试
                    if attempt < max_attempts:
                        # 调用回调函数
                        if on_retry:
                            try:
                                on_retry(e, attempt, current_delay)
                            except Exception as callback_error:
                                # 回调函数出错不应影响重试逻辑
                                pass
                        
                        # 等待
                        time.sleep(current_delay)
                        
                        # 计算下次延迟
                        current_delay = min(current_delay * backoff, max_delay)
                    
                except Exception as e:
                    # 不可重试的异常，直接抛出
                    raise e

            # 所有尝试都失败了
            if last_exception is not None:
                # 添加重试信息到异常
                error_msg = (
                    f"Function {fn.__name__} failed after {max_attempts} attempts. "
                    f"Last error: {last_exception}"
                )
                # 尝试创建相同类型的异常并添加上下文信息
                try:
                    raise type(last_exception)(error_msg) from last_exception
                except:
                    # 如果无法创建新异常，抛出原始异常
                    raise last_exception
            
            # 理论上不应该到达这里
            raise RuntimeError(
                f"Function {fn.__name__} failed after {max_attempts} attempts "
                f"without capturing exception"
            )

        return wrapper
    
    # 支持 @retry 和 @retry(...) 两种用法
    if func is not None:
        return decorator(func)
    return decorator


def timeout(seconds: float) -> Callable:
    """
    超时装饰器

    Args:
        seconds: 超时时间

    Returns:
        Callable: 装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading

            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


class Singleton(type):
    """
    单例元类

    确保类只有一个实例。
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Lazy(Generic[T]):
    """
    惰性求值

    延迟计算值。
    """

    def __init__(self, func: Callable[[], T]):
        """
        初始化惰性对象

        Args:
            func: 计算函数
        """
        self._func = func
        self._value: Optional[T] = None
        self._computed = False

    def get(self) -> T:
        """
        获取值

        Returns:
            T: 值
        """
        if not self._computed:
            self._value = self._func()
            self._computed = True
        return self._value

    def is_computed(self) -> bool:
        """是否已计算"""
        return self._computed


class RateCalculator:
    """
    速率计算器

    计算事件速率。
    """

    def __init__(self, window_size: int = 60):
        """
        初始化速率计算器

        Args:
            window_size: 窗口大小（秒）
        """
        self._window_size = window_size
        self._events: List[float] = []

    def record(self, count: int = 1):
        """
        记录事件

        Args:
            count: 事件数量
        """
        current_time = time.time()
        for _ in range(count):
            self._events.append(current_time)

        cutoff_time = current_time - self._window_size
        self._events = [t for t in self._events if t > cutoff_time]

    def get_rate(self) -> float:
        """
        获取速率

        Returns:
            float: 速率（事件/秒）
        """
        if not self._events:
            return 0.0

        elapsed = time.time() - self._events[0]
        if elapsed == 0:
            return 0.0

        return len(self._events) / elapsed

    def reset(self):
        """重置"""
        self._events.clear()


def validate_config(config: Dict[str, Any],
                    required_keys: List[str]) -> bool:
    """
    验证配置

    Args:
        config: 配置字典
        required_keys: 必需的键列表

    Returns:
        bool: 是否有效
    """
    for key in required_keys:
        if key not in config:
            return False
    return True
