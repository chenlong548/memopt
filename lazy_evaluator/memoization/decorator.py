"""
memoize装饰器模块

提供函数记忆化装饰器，支持TTL过期。
"""

from typing import Callable, Optional, Any, Dict, Tuple
from functools import wraps
from collections import OrderedDict
import threading
import time
import hashlib
import json


def memoize(max_size: int = 1000, ttl: Optional[float] = None):
    """
    记忆化装饰器，支持TTL过期（优化版）

    该装饰器会缓存函数调用的结果，避免重复计算。
    支持设置最大缓存大小和过期时间。
    
    优化：
    - 使用OrderedDict实现O(1)时间复杂度的缓存淘汰
    - 使用安全的JSON序列化替代pickle
    - 减少不必要的time.time()调用

    Args:
        max_size: 最大缓存大小，默认1000
        ttl: 可选的过期时间（秒），None表示不过期

    Returns:
        装饰器函数

    Example:
        >>> @memoize(max_size=100, ttl=60)
        ... def expensive_function(x):
        ...     return x ** 2
        >>> result1 = expensive_function(10)  # 计算
        >>> result2 = expensive_function(10)  # 从缓存获取
    """
    def decorator(func: Callable) -> Callable:
        # 使用OrderedDict维护插入顺序，实现O(1)淘汰
        cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(func.__name__, args, kwargs)

            with lock:
                # 检查缓存
                if cache_key in cache:
                    value, timestamp = cache[cache_key]

                    # 检查是否过期
                    if ttl is None:
                        # 未设置TTL，直接返回
                        # 将访问的项移到末尾（最近使用）
                        cache.move_to_end(cache_key)
                        return value
                    else:
                        current_time = time.time()
                        if (current_time - timestamp) <= ttl:
                            # 未过期，将访问的项移到末尾
                            cache.move_to_end(cache_key)
                            return value
                        else:
                            # 过期，删除
                            del cache[cache_key]

            # 计算新值
            result = func(*args, **kwargs)

            with lock:
                # 检查缓存大小，淘汰最久未使用的项
                if len(cache) >= max_size:
                    # popitem(last=False) 删除最早的项，O(1)操作
                    cache.popitem(last=False)

                # 缓存结果，添加到末尾
                cache[cache_key] = (result, time.time())

            return result

        # 添加缓存管理方法
        def cache_clear():
            """清空缓存"""
            with lock:
                cache.clear()

        def cache_info():
            """获取缓存信息"""
            with lock:
                return {
                    'size': len(cache),
                    'max_size': max_size,
                    'ttl': ttl,
                }

        def cache_remove(*args, **kwargs):
            """移除特定参数的缓存"""
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            with lock:
                if cache_key in cache:
                    del cache[cache_key]

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        wrapper.cache_remove = cache_remove

        return wrapper

    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    生成缓存键（安全实现，不使用pickle）

    使用JSON序列化和MD5哈希生成缓存键，避免pickle的安全风险。
    pickle序列化不可信数据可能导致代码执行漏洞。

    Args:
        func_name: 函数名
        args: 位置参数
        kwargs: 关键字参数

    Returns:
        str: 缓存键（MD5哈希值）
    """
    try:
        # 使用JSON序列化，安全且跨平台兼容
        # 将参数转换为可JSON序列化的格式
        key_data = {
            'func': func_name,
            'args': _make_json_serializable(args),
            'kwargs': _make_json_serializable(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    except (TypeError, ValueError):
        # 如果JSON序列化失败，使用字符串表示
        key_str = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def _make_json_serializable(obj: Any) -> Any:
    """
    将对象转换为JSON可序列化的格式

    Args:
        obj: 要转换的对象

    Returns:
        Any: JSON可序列化的对象
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return sorted([_make_json_serializable(item) for item in obj])
    else:
        # 对于其他类型，尝试转换为字符串表示
        return str(obj)


def memoize_method(max_size: int = 1000, ttl: Optional[float] = None):
    """
    方法记忆化装饰器，支持TTL过期（优化版）

    该装饰器专门用于类方法，会为每个实例维护独立的缓存。
    
    优化：
    - 使用OrderedDict实现O(1)时间复杂度的缓存淘汰
    - 使用安全的JSON序列化替代pickle
    - 减少不必要的time.time()调用

    Args:
        max_size: 最大缓存大小，默认1000
        ttl: 可选的过期时间（秒），None表示不过期

    Returns:
        装饰器函数

    Example:
        >>> class MyClass:
        ...     @memoize_method(max_size=100)
        ...     def expensive_method(self, x):
        ...         return x ** 2
    """
    def decorator(func: Callable) -> Callable:
        # 使用实例字典存储缓存
        cache_attr_name = f'_memoize_cache_{func.__name__}'
        lock_attr_name = f'_memoize_lock_{func.__name__}'

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取或创建实例缓存
            if not hasattr(self, cache_attr_name):
                setattr(self, cache_attr_name, OrderedDict())
                setattr(self, lock_attr_name, threading.Lock())

            cache = getattr(self, cache_attr_name)
            lock = getattr(self, lock_attr_name)

            # 生成缓存键（不包含self）
            cache_key = _generate_cache_key(func.__name__, args, kwargs)

            with lock:
                # 检查缓存
                if cache_key in cache:
                    value, timestamp = cache[cache_key]

                    # 检查是否过期
                    if ttl is None:
                        # 未设置TTL，直接返回
                        cache.move_to_end(cache_key)
                        return value
                    else:
                        current_time = time.time()
                        if (current_time - timestamp) <= ttl:
                            cache.move_to_end(cache_key)
                            return value
                        else:
                            # 过期，删除
                            del cache[cache_key]

            # 计算新值
            result = func(self, *args, **kwargs)

            with lock:
                # 检查缓存大小
                if len(cache) >= max_size:
                    # popitem(last=False) 删除最早的项，O(1)操作
                    cache.popitem(last=False)

                # 缓存结果
                cache[cache_key] = (result, time.time())

            return result

        return wrapper

    return decorator


def memoize_property(ttl: Optional[float] = None):
    """
    属性记忆化装饰器，支持TTL过期

    该装饰器用于将方法转换为记忆化属性。

    Args:
        ttl: 可选的过期时间（秒），None表示不过期

    Returns:
        装饰器函数

    Example:
        >>> class MyClass:
        ...     @memoize_property(ttl=60)
        ...     def expensive_property(self):
        ...         return self._compute()
    """
    def decorator(func: Callable) -> property:
        cache_attr_name = f'_memoize_property_cache_{func.__name__}'
        timestamp_attr_name = f'_memoize_property_timestamp_{func.__name__}'

        @wraps(func)
        def getter(self):
            # 检查缓存
            if hasattr(self, cache_attr_name):
                # 检查是否过期
                if ttl is not None:
                    timestamp = getattr(self, timestamp_attr_name, 0)
                    if time.time() - timestamp > ttl:
                        # 过期，删除缓存
                        delattr(self, cache_attr_name)
                        delattr(self, timestamp_attr_name)
                    else:
                        return getattr(self, cache_attr_name)
                else:
                    return getattr(self, cache_attr_name)

            # 计算新值
            result = func(self)

            # 缓存结果
            setattr(self, cache_attr_name, result)
            if ttl is not None:
                setattr(self, timestamp_attr_name, time.time())

            return result

        def setter(self, value):
            """设置属性值"""
            setattr(self, cache_attr_name, value)
            if ttl is not None:
                setattr(self, timestamp_attr_name, time.time())

        def deleter(self):
            """删除属性"""
            if hasattr(self, cache_attr_name):
                delattr(self, cache_attr_name)
            if hasattr(self, timestamp_attr_name):
                delattr(self, timestamp_attr_name)

        return property(getter, setter, deleter)

    return decorator
