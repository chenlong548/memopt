"""
自适应调整模块

根据运行时状态自适应调整缓冲区参数。
"""

import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class AdaptationLevel(Enum):
    """自适应调整级别"""
    CONSERVATIVE = "conservative"  # 保守调整
    MODERATE = "moderate"  # 适度调整
    AGGRESSIVE = "aggressive"  # 激进调整


@dataclass
class AdaptationStats:
    """自适应统计信息"""
    current_hit_rate: float = 0.0
    target_hit_rate: float = 0.9
    adjustments_made: int = 0
    last_adjustment_time: float = 0.0
    current_pool_size: int = 0
    current_buffer_size: int = 0


class AdaptiveStrategy:
    """
    自适应策略
    
    根据运行时指标自动调整缓冲区参数。
    """
    
    def __init__(
        self,
        buffer_pool,
        target_hit_rate: float = 0.9,
        min_pool_size: int = 4,
        max_pool_size: int = 64,
        adjustment_interval: float = 5.0,
    ):
        """
        初始化自适应策略
        
        Args:
            buffer_pool: 缓冲池实例
            target_hit_rate: 目标命中率
            min_pool_size: 最小池大小
            max_pool_size: 最大池大小
            adjustment_interval: 调整间隔（秒）
        """
        self._buffer_pool = buffer_pool
        self._target_hit_rate = target_hit_rate
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._adjustment_interval = adjustment_interval
        
        # 当前调整级别
        self._level = AdaptationLevel.MODERATE
        
        # 统计信息
        self._stats = AdaptationStats(
            target_hit_rate=target_hit_rate,
            current_pool_size=buffer_pool.capacity,
            current_buffer_size=buffer_pool.buffer_size,
        )
        
        # 上一次检查时间
        self._last_check_time = time.time()
        
        # 历史命中率
        self._hit_rate_history: list = []
        self._history_size = 10
        
        # 同步锁
        self._lock = threading.Lock()
        
        # 调整回调
        self._on_adjust: Optional[Callable] = None
    
    def update_metrics(self, hit_rate: float) -> None:
        """
        更新指标
        
        Args:
            hit_rate: 当前命中率
        """
        with self._lock:
            self._stats.current_hit_rate = hit_rate
            self._hit_rate_history.append(hit_rate)
            
            if len(self._hit_rate_history) > self._history_size:
                self._hit_rate_history.pop(0)
    
    def should_adjust(self) -> bool:
        """
        检查是否应该调整
        
        Returns:
            是否应该调整
        """
        current_time = time.time()
        
        if current_time - self._last_check_time < self._adjustment_interval:
            return False
        
        self._last_check_time = current_time
        return True
    
    def adjust(self) -> Optional[dict]:
        """
        执行自适应调整
        
        Returns:
            调整结果，如果未调整返回None
        """
        if not self.should_adjust():
            return None
        
        with self._lock:
            avg_hit_rate = self._get_average_hit_rate()
            
            if avg_hit_rate >= self._target_hit_rate:
                # 命中率达标，可能可以缩小池
                return self._try_shrink()
            else:
                # 命中率不达标，尝试扩大池
                return self._try_expand()
    
    def _get_average_hit_rate(self) -> float:
        """获取平均命中率"""
        if not self._hit_rate_history:
            return 0.0
        return sum(self._hit_rate_history) / len(self._hit_rate_history)
    
    def _try_expand(self) -> Optional[dict]:
        """尝试扩大池"""
        current_size = self._buffer_pool.capacity
        
        if current_size >= self._max_pool_size:
            return None
        
        # 根据调整级别确定扩展量
        if self._level == AdaptationLevel.CONSERVATIVE:
            new_size = current_size + 1
        elif self._level == AdaptationLevel.MODERATE:
            new_size = min(current_size * 2, self._max_pool_size)
        else:  # AGGRESSIVE
            new_size = min(current_size * 4, self._max_pool_size)
        
        self._buffer_pool.resize(new_size)
        
        self._stats.adjustments_made += 1
        self._stats.last_adjustment_time = time.time()
        self._stats.current_pool_size = new_size
        
        result = {
            "action": "expand",
            "old_size": current_size,
            "new_size": new_size,
            "reason": f"hit_rate {self._stats.current_hit_rate:.2%} < target {self._target_hit_rate:.2%}",
        }
        
        if self._on_adjust:
            self._on_adjust(result)
        
        return result
    
    def _try_shrink(self) -> Optional[dict]:
        """尝试缩小池"""
        current_size = self._buffer_pool.capacity
        
        if current_size <= self._min_pool_size:
            return None
        
        # 需要至少5个历史记录才能判断趋势
        min_history_for_shrink = 5
        if len(self._hit_rate_history) < min_history_for_shrink:
            return None
        
        # 只有命中率持续高于目标才缩小
        recent_history = self._hit_rate_history[-min_history_for_shrink:]
        if all(rate >= self._target_hit_rate for rate in recent_history):
            new_size = max(current_size - 1, self._min_pool_size)
            
            self._buffer_pool.resize(new_size)
            
            self._stats.adjustments_made += 1
            self._stats.last_adjustment_time = time.time()
            self._stats.current_pool_size = new_size
            
            result = {
                "action": "shrink",
                "old_size": current_size,
                "new_size": new_size,
                "reason": f"hit_rate {self._stats.current_hit_rate:.2%} >= target {self._target_hit_rate:.2%}",
            }
            
            if self._on_adjust:
                self._on_adjust(result)
            
            return result
        
        return None
    
    def set_level(self, level: AdaptationLevel) -> None:
        """
        设置调整级别
        
        Args:
            level: 调整级别
        """
        with self._lock:
            self._level = level
    
    def set_callback(self, callback: Callable[[dict], None]) -> None:
        """
        设置调整回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            self._on_adjust = callback
    
    def force_adjust(self, new_size: int) -> dict:
        """
        强制调整池大小
        
        Args:
            new_size: 新的池大小
        
        Returns:
            调整结果
        """
        with self._lock:
            old_size = self._buffer_pool.capacity
            actual_size = max(self._min_pool_size, min(new_size, self._max_pool_size))
            
            self._buffer_pool.resize(actual_size)
            
            self._stats.adjustments_made += 1
            self._stats.last_adjustment_time = time.time()
            self._stats.current_pool_size = actual_size
            
            result = {
                "action": "force_adjust",
                "old_size": old_size,
                "new_size": actual_size,
                "reason": "manual adjustment",
            }
            
            if self._on_adjust:
                self._on_adjust(result)
            
            return result
    
    @property
    def stats(self) -> AdaptationStats:
        """获取统计信息"""
        with self._lock:
            return AdaptationStats(
                current_hit_rate=self._stats.current_hit_rate,
                target_hit_rate=self._stats.target_hit_rate,
                adjustments_made=self._stats.adjustments_made,
                last_adjustment_time=self._stats.last_adjustment_time,
                current_pool_size=self._stats.current_pool_size,
                current_buffer_size=self._stats.current_buffer_size,
            )
    
    @property
    def level(self) -> AdaptationLevel:
        """获取当前调整级别"""
        return self._level
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveStrategy(target_hit_rate={self._target_hit_rate:.0%}, "
            f"level={self._level.value})"
        )
