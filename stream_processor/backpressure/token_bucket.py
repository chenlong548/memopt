"""
stream_processor 令牌桶

实现令牌桶算法进行流量控制。
"""

from typing import Optional
import time
import threading


class TokenBucket:
    """
    令牌桶

    使用令牌桶算法进行流量控制。
    """

    def __init__(self,
                 capacity: int,
                 refill_rate: float,
                 initial_tokens: Optional[int] = None):
        """
        初始化令牌桶

        Args:
            capacity: 桶容量
            refill_rate: 令牌补充速率（令牌/秒）
            initial_tokens: 初始令牌数，默认为桶容量
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if refill_rate <= 0:
            raise ValueError(f"Refill rate must be positive, got {refill_rate}")

        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = initial_tokens if initial_tokens is not None else capacity
        self._last_refill_time = time.time()
        self._lock = threading.Lock()

    def try_consume(self, tokens: int = 1) -> bool:
        """
        尝试消费令牌

        Args:
            tokens: 要消费的令牌数

        Returns:
            bool: 是否成功
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    def consume(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        消费令牌（阻塞）

        Args:
            tokens: 要消费的令牌数
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            bool: 是否成功
        """
        start_time = time.time()

        while True:
            if self.try_consume(tokens):
                return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            time.sleep(0.001)

    def _refill(self):
        """补充令牌"""
        current_time = time.time()
        elapsed = current_time - self._last_refill_time

        tokens_to_add = elapsed * self._refill_rate

        self._tokens = min(self._capacity, self._tokens + tokens_to_add)
        self._last_refill_time = current_time

    def get_tokens(self) -> float:
        """
        获取当前令牌数

        Returns:
            float: 当前令牌数
        """
        with self._lock:
            self._refill()
            return self._tokens

    def get_capacity(self) -> int:
        """
        获取桶容量

        Returns:
            int: 桶容量
        """
        return self._capacity

    def get_refill_rate(self) -> float:
        """
        获取补充速率

        Returns:
            float: 补充速率
        """
        return self._refill_rate

    def set_refill_rate(self, rate: float):
        """
        设置补充速率

        Args:
            rate: 新的补充速率
        """
        with self._lock:
            self._refill_rate = rate

    def reset(self):
        """重置令牌桶"""
        with self._lock:
            self._tokens = self._capacity
            self._last_refill_time = time.time()

    def get_utilization(self) -> float:
        """
        获取令牌利用率

        Returns:
            float: 利用率（0-1）
        """
        with self._lock:
            self._refill()
            return 1.0 - (self._tokens / self._capacity)


class LeakyBucket:
    """
    漏桶

    使用漏桶算法进行流量整形。
    """

    def __init__(self, capacity: int, leak_rate: float):
        """
        初始化漏桶

        Args:
            capacity: 桶容量
            leak_rate: 漏出速率（请求/秒）
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if leak_rate <= 0:
            raise ValueError(f"Leak rate must be positive, got {leak_rate}")

        self._capacity = capacity
        self._leak_rate = leak_rate
        self._water = 0.0
        self._last_leak_time = time.time()
        self._lock = threading.Lock()

    def try_add(self, amount: int = 1) -> bool:
        """
        尝试添加水滴

        Args:
            amount: 水滴数量

        Returns:
            bool: 是否成功
        """
        with self._lock:
            self._leak()

            if self._water + amount <= self._capacity:
                self._water += amount
                return True

            return False

    def _leak(self):
        """漏水"""
        current_time = time.time()
        elapsed = current_time - self._last_leak_time

        leaked = elapsed * self._leak_rate

        self._water = max(0.0, self._water - leaked)
        self._last_leak_time = current_time

    def get_water_level(self) -> float:
        """
        获取当前水位

        Returns:
            float: 当前水位
        """
        with self._lock:
            self._leak()
            return self._water

    def get_capacity(self) -> int:
        """
        获取桶容量

        Returns:
            int: 桶容量
        """
        return self._capacity

    def get_leak_rate(self) -> float:
        """
        获取漏出速率

        Returns:
            float: 漏出速率
        """
        return self._leak_rate

    def reset(self):
        """重置漏桶"""
        with self._lock:
            self._water = 0.0
            self._last_leak_time = time.time()


class SlidingWindowCounter:
    """
    滑动窗口计数器

    使用滑动窗口算法进行限流。
    """

    def __init__(self, window_size: float, max_requests: int):
        """
        初始化滑动窗口计数器

        Args:
            window_size: 窗口大小（秒）
            max_requests: 最大请求数
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        if max_requests <= 0:
            raise ValueError(f"Max requests must be positive, got {max_requests}")

        self._window_size = window_size
        self._max_requests = max_requests
        self._requests: list = []
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """
        尝试获取许可

        Returns:
            bool: 是否成功
        """
        with self._lock:
            current_time = time.time()

            self._requests = [
                t for t in self._requests
                if current_time - t < self._window_size
            ]

            if len(self._requests) < self._max_requests:
                self._requests.append(current_time)
                return True

            return False

    def get_current_count(self) -> int:
        """
        获取当前计数

        Returns:
            int: 当前计数
        """
        with self._lock:
            current_time = time.time()

            self._requests = [
                t for t in self._requests
                if current_time - t < self._window_size
            ]

            return len(self._requests)

    def get_remaining(self) -> int:
        """
        获取剩余配额

        Returns:
            int: 剩余配额
        """
        return self._max_requests - self.get_current_count()

    def reset(self):
        """重置计数器"""
        with self._lock:
            self._requests.clear()
