"""
指标收集模块

收集和统计缓冲区使用指标。
"""

import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class MetricSnapshot:
    """指标快照"""
    timestamp: float
    hit_rate: float
    avg_acquire_time: float
    pool_usage: float
    total_acquires: int
    total_releases: int
    total_hits: int
    total_misses: int


class BufferMetrics:
    """
    缓冲区指标收集器
    
    收集和统计缓冲区的各种使用指标。
    线程安全的实现。
    """
    
    def __init__(self, history_size: int = 1000):
        """
        初始化指标收集器
        
        Args:
            history_size: 历史记录大小
        """
        self._history_size = history_size
        
        # 计数器
        self._total_acquires = 0
        self._total_releases = 0
        self._total_hits = 0
        self._total_misses = 0
        
        # 延迟记录
        self._acquire_latencies: deque = deque(maxlen=history_size)
        self._release_latencies: deque = deque(maxlen=history_size)
        
        # 历史快照
        self._snapshots: deque = deque(maxlen=100)
        
        # 同步锁
        self._lock = threading.Lock()
        
        # 开始时间
        self._start_time = time.time()
    
    def record_acquire(self, latency: float) -> None:
        """
        记录获取操作
        
        Args:
            latency: 延迟时间（毫秒）
        """
        with self._lock:
            self._total_acquires += 1
            self._acquire_latencies.append(latency)
    
    def record_release(self, latency: float) -> None:
        """
        记录释放操作
        
        Args:
            latency: 延迟时间（毫秒）
        """
        with self._lock:
            self._total_releases += 1
            self._release_latencies.append(latency)
    
    def record_hit(self) -> None:
        """记录缓存命中"""
        with self._lock:
            self._total_hits += 1
    
    def record_miss(self) -> None:
        """记录缓存未命中"""
        with self._lock:
            self._total_misses += 1
    
    def take_snapshot(self, pool_usage: float = 0.0) -> MetricSnapshot:
        """
        创建指标快照
        
        Args:
            pool_usage: 池使用率
        
        Returns:
            MetricSnapshot对象
        """
        with self._lock:
            # 直接计算值，避免调用属性方法导致死锁
            total = self._total_hits + self._total_misses
            hit_rate = self._total_hits / total if total > 0 else 0.0
            
            avg_acquire = (
                sum(self._acquire_latencies) / len(self._acquire_latencies)
                if self._acquire_latencies else 0.0
            )
            
            snapshot = MetricSnapshot(
                timestamp=time.time(),
                hit_rate=hit_rate,
                avg_acquire_time=avg_acquire,
                pool_usage=pool_usage,
                total_acquires=self._total_acquires,
                total_releases=self._total_releases,
                total_hits=self._total_hits,
                total_misses=self._total_misses,
            )
            self._snapshots.append(snapshot)
            return snapshot
    
    @property
    def hit_rate(self) -> float:
        """获取命中率（线程安全）"""
        with self._lock:
            total = self._total_hits + self._total_misses
            if total == 0:
                return 0.0
            return self._total_hits / total
    
    @property
    def avg_acquire_time(self) -> float:
        """获取平均获取时间（毫秒）（线程安全）"""
        with self._lock:
            if not self._acquire_latencies:
                return 0.0
            return sum(self._acquire_latencies) / len(self._acquire_latencies)
    
    @property
    def avg_release_time(self) -> float:
        """获取平均释放时间（毫秒）（线程安全）"""
        with self._lock:
            if not self._release_latencies:
                return 0.0
            return sum(self._release_latencies) / len(self._release_latencies)
    
    @property
    def total_acquires(self) -> int:
        """获取总获取次数（线程安全）"""
        with self._lock:
            return self._total_acquires
    
    @property
    def total_releases(self) -> int:
        """获取总释放次数（线程安全）"""
        with self._lock:
            return self._total_releases
    
    @property
    def total_hits(self) -> int:
        """获取总命中次数（线程安全）"""
        with self._lock:
            return self._total_hits
    
    @property
    def total_misses(self) -> int:
        """获取总未命中次数（线程安全）"""
        with self._lock:
            return self._total_misses
    
    @property
    def uptime(self) -> float:
        """获取运行时间（秒）"""
        return time.time() - self._start_time
    
    @property
    def stats(self) -> Dict[str, float]:
        """获取统计信息"""
        with self._lock:
            return {
                "hit_rate": self.hit_rate,
                "avg_acquire_time_ms": self.avg_acquire_time,
                "avg_release_time_ms": self.avg_release_time,
                "total_acquires": self._total_acquires,
                "total_releases": self._total_releases,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "uptime_seconds": self.uptime,
            }
    
    @property
    def snapshots(self) -> List[MetricSnapshot]:
        """获取历史快照"""
        with self._lock:
            return list(self._snapshots)
    
    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._total_acquires = 0
            self._total_releases = 0
            self._total_hits = 0
            self._total_misses = 0
            self._acquire_latencies.clear()
            self._release_latencies.clear()
            self._snapshots.clear()
            self._start_time = time.time()
    
    def get_recent_latencies(self, count: int = 100) -> Dict[str, List[float]]:
        """
        获取最近的延迟记录
        
        Args:
            count: 返回的记录数量
        
        Returns:
            包含acquire和release延迟的字典
        """
        with self._lock:
            return {
                "acquire": list(self._acquire_latencies)[-count:],
                "release": list(self._release_latencies)[-count:],
            }
    
    def __repr__(self) -> str:
        return (
            f"BufferMetrics(acquires={self._total_acquires}, "
            f"releases={self._total_releases}, "
            f"hit_rate={self.hit_rate:.2%})"
        )
