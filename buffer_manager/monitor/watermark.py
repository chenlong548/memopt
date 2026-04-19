"""
水位线管理模块

管理缓冲区使用的水位线，支持回调通知。
"""

import threading
import time
from enum import Enum
from typing import Callable, Dict, Optional
from dataclasses import dataclass


class WatermarkLevel(Enum):
    """水位线级别"""
    LOW = "low"           # 低水位线
    NORMAL = "normal"     # 正常水位
    HIGH = "high"         # 高水位线
    CRITICAL = "critical" # 临界水位线


@dataclass
class WatermarkConfig:
    """水位线配置"""
    low: float = 0.25      # 低水位线（25%）
    high: float = 0.75     # 高水位线（75%）
    critical: float = 0.90 # 临界水位线（90%）


class WatermarkManager:
    """
    水位线管理器
    
    监控缓冲区使用率，在达到不同水位线时触发回调。
    线程安全的实现。
    """
    
    def __init__(
        self,
        low: float = 0.25,
        high: float = 0.75,
        critical: float = 0.90,
    ):
        """
        初始化水位线管理器
        
        Args:
            low: 低水位线（0-1）
            high: 高水位线（0-1）
            critical: 临界水位线（0-1）
        
        Raises:
            ValueError: 水位线配置无效
        """
        if not (0 < low < high < critical <= 1.0):
            raise ValueError(
                f"Invalid watermark levels: must satisfy 0 < low < high < critical <= 1.0, "
                f"got low={low}, high={high}, critical={critical}"
            )
        
        self._config = WatermarkConfig(low=low, high=high, critical=critical)
        
        # 当前水位
        self._current_level = WatermarkLevel.NORMAL
        self._current_usage = 0.0
        
        # 回调函数
        self._callbacks: Dict[WatermarkLevel, Callable[[float], None]] = {}
        
        # 同步锁
        self._lock = threading.Lock()
        
        # 统计信息
        self._level_changes = 0
        self._last_change_time = 0.0
    
    def check(self, current: float) -> WatermarkLevel:
        """
        检查当前水位
        
        Args:
            current: 当前使用率（0-1）
        
        Returns:
            当前水位线级别
        """
        if current < 0 or current > 1:
            raise ValueError(f"Current usage must be between 0 and 1, got {current}")
        
        with self._lock:
            self._current_usage = current
            
            # 确定水位线级别
            if current >= self._config.critical:
                new_level = WatermarkLevel.CRITICAL
            elif current >= self._config.high:
                new_level = WatermarkLevel.HIGH
            elif current <= self._config.low:
                new_level = WatermarkLevel.LOW
            else:
                new_level = WatermarkLevel.NORMAL
            
            # 检查是否发生变化
            if new_level != self._current_level:
                self._handle_level_change(new_level)
            
            return self._current_level
    
    def _handle_level_change(self, new_level: WatermarkLevel) -> None:
        """
        处理水位线变化
        
        Args:
            new_level: 新的水位线级别
        """
        old_level = self._current_level
        self._current_level = new_level
        self._level_changes += 1
        self._last_change_time = time.time()
        
        # 触发回调
        if new_level in self._callbacks:
            try:
                self._callbacks[new_level](self._current_usage)
            except Exception:
                pass  # 忽略回调异常
    
    def set_callback(
        self,
        level: WatermarkLevel,
        callback: Callable[[float], None]
    ) -> None:
        """
        设置水位线回调
        
        Args:
            level: 水位线级别
            callback: 回调函数，接收当前使用率作为参数
        """
        with self._lock:
            self._callbacks[level] = callback
    
    def remove_callback(self, level: WatermarkLevel) -> bool:
        """
        移除水位线回调
        
        Args:
            level: 水位线级别
        
        Returns:
            是否成功移除
        """
        with self._lock:
            if level in self._callbacks:
                del self._callbacks[level]
                return True
            return False
    
    def get_threshold(self, level: WatermarkLevel) -> float:
        """
        获取指定级别的阈值
        
        Args:
            level: 水位线级别
        
        Returns:
            阈值
        """
        if level == WatermarkLevel.LOW:
            return self._config.low
        elif level == WatermarkLevel.HIGH:
            return self._config.high
        elif level == WatermarkLevel.CRITICAL:
            return self._config.critical
        return 0.0
    
    def update_thresholds(
        self,
        low: Optional[float] = None,
        high: Optional[float] = None,
        critical: Optional[float] = None,
    ) -> None:
        """
        更新水位线阈值
        
        Args:
            low: 新的低水位线
            high: 新的高水位线
            critical: 新的临界水位线
        """
        with self._lock:
            if low is not None:
                self._config.low = low
            if high is not None:
                self._config.high = high
            if critical is not None:
                self._config.critical = critical
    
    @property
    def current_level(self) -> WatermarkLevel:
        """获取当前水位线级别"""
        return self._current_level
    
    @property
    def current_usage(self) -> float:
        """获取当前使用率"""
        return self._current_usage
    
    @property
    def config(self) -> WatermarkConfig:
        """获取水位线配置"""
        return self._config
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "current_level": self._current_level.value,
                "current_usage": self._current_usage,
                "low_threshold": self._config.low,
                "high_threshold": self._config.high,
                "critical_threshold": self._config.critical,
                "level_changes": self._level_changes,
                "last_change_time": self._last_change_time,
            }
    
    def reset(self) -> None:
        """重置水位线状态"""
        with self._lock:
            self._current_level = WatermarkLevel.NORMAL
            self._current_usage = 0.0
            self._level_changes = 0
            self._last_change_time = 0.0
    
    def __repr__(self) -> str:
        return (
            f"WatermarkManager(level={self._current_level.value}, "
            f"usage={self._current_usage:.2%})"
        )
