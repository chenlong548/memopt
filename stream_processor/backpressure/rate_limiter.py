"""
stream_processor 限流器

实现流量限流功能。
"""

from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading

from .token_bucket import TokenBucket, LeakyBucket, SlidingWindowCounter


class RateLimitStrategy(Enum):
    """限流策略"""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """
    限流配置

    定义限流的配置参数。
    """

    rate: float

    capacity: int

    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    timeout: Optional[float] = None

    enable_metrics: bool = True


@dataclass
class RateLimitResult:
    """
    限流结果

    记录限流操作的结果。
    """

    allowed: bool

    remaining: int = 0

    retry_after: Optional[float] = None

    current_rate: float = 0.0


class RateLimiter:
    """
    限流器

    提供流量限流功能。
    """

    def __init__(self, config: RateLimitConfig):
        """
        初始化限流器

        Args:
            config: 限流配置
        """
        self._config = config
        self._limiter = self._create_limiter()
        self._total_requests = 0
        self._allowed_requests = 0
        self._rejected_requests = 0
        self._lock = threading.Lock()

    def _create_limiter(self):
        """创建限流器实例"""
        if self._config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucket(
                capacity=self._config.capacity,
                refill_rate=self._config.rate
            )
        elif self._config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return LeakyBucket(
                capacity=self._config.capacity,
                leak_rate=self._config.rate
            )
        else:
            return SlidingWindowCounter(
                window_size=1.0,
                max_requests=int(self._config.rate)
            )

    def try_acquire(self, tokens: int = 1) -> RateLimitResult:
        """
        尝试获取许可

        Args:
            tokens: 令牌数

        Returns:
            RateLimitResult: 限流结果
        """
        with self._lock:
            self._total_requests += 1

            if isinstance(self._limiter, TokenBucket):
                allowed = self._limiter.try_consume(tokens)
                remaining = int(self._limiter.get_tokens())
            elif isinstance(self._limiter, LeakyBucket):
                allowed = self._limiter.try_add(tokens)
                remaining = int(self._limiter.get_capacity() - self._limiter.get_water_level())
            else:
                allowed = self._limiter.try_acquire()
                remaining = self._limiter.get_remaining()

            if allowed:
                self._allowed_requests += 1
            else:
                self._rejected_requests += 1

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                retry_after=1.0 / self._config.rate if not allowed else None,
                current_rate=self._get_current_rate()
            )

    def acquire(self, tokens: int = 1) -> bool:
        """
        获取许可（阻塞）

        Args:
            tokens: 令牌数

        Returns:
            bool: 是否成功
        """
        if isinstance(self._limiter, TokenBucket):
            return self._limiter.consume(tokens, self._config.timeout)
        else:
            result = self.try_acquire(tokens)
            return result.allowed

    def _get_current_rate(self) -> float:
        """获取当前速率"""
        if self._total_requests == 0:
            return 0.0
        return self._allowed_requests / self._total_requests

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 统计信息
        """
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'allowed_requests': self._allowed_requests,
                'rejected_requests': self._rejected_requests,
                'allow_rate': self._get_current_rate()
            }

    def reset(self):
        """重置限流器"""
        with self._lock:
            if hasattr(self._limiter, 'reset'):
                self._limiter.reset()
            self._total_requests = 0
            self._allowed_requests = 0
            self._rejected_requests = 0


class AdaptiveRateLimiter(RateLimiter):
    """
    自适应限流器

    根据系统负载自动调整限流参数。
    """

    def __init__(self,
                 config: RateLimitConfig,
                 min_rate: float = 10.0,
                 max_rate: float = 1000.0):
        """
        初始化自适应限流器

        Args:
            config: 限流配置
            min_rate: 最小速率
            max_rate: 最大速率
        """
        super().__init__(config)
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._adjustment_interval = 5.0
        self._last_adjustment_time = time.time()
        self._latency_samples: list = []

    def record_latency(self, latency: float):
        """
        记录延迟

        Args:
            latency: 延迟（秒）
        """
        self._latency_samples.append(latency)

        if len(self._latency_samples) > 100:
            self._latency_samples.pop(0)

    def adjust_rate(self):
        """调整速率"""
        current_time = time.time()

        if current_time - self._last_adjustment_time < self._adjustment_interval:
            return

        if not self._latency_samples:
            return

        avg_latency = sum(self._latency_samples) / len(self._latency_samples)

        if avg_latency > 0.1:
            new_rate = max(self._min_rate, self._config.rate * 0.9)
        elif avg_latency < 0.01:
            new_rate = min(self._max_rate, self._config.rate * 1.1)
        else:
            new_rate = self._config.rate

        self._config.rate = new_rate

        if isinstance(self._limiter, TokenBucket):
            self._limiter.set_refill_rate(new_rate)

        self._last_adjustment_time = current_time
        self._latency_samples.clear()


class DistributedRateLimiter:
    """
    分布式限流器

    在分布式环境中协调限流。
    """

    def __init__(self,
                 config: RateLimitConfig,
                 instance_id: str,
                 total_instances: int):
        """
        初始化分布式限流器

        Args:
            config: 限流配置
            instance_id: 实例ID
            total_instances: 总实例数
        """
        self._config = config
        self._instance_id = instance_id
        self._total_instances = total_instances

        per_instance_rate = config.rate / total_instances
        per_instance_capacity = max(1, config.capacity // total_instances)

        instance_config = RateLimitConfig(
            rate=per_instance_rate,
            capacity=per_instance_capacity,
            strategy=config.strategy,
            timeout=config.timeout
        )

        self._local_limiter = RateLimiter(instance_config)

    def try_acquire(self, tokens: int = 1) -> RateLimitResult:
        """
        尝试获取许可

        Args:
            tokens: 令牌数

        Returns:
            RateLimitResult: 限流结果
        """
        return self._local_limiter.try_acquire(tokens)

    def get_instance_id(self) -> str:
        """
        获取实例ID

        Returns:
            str: 实例ID
        """
        return self._instance_id
