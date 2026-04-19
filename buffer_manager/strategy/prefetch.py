"""
预取策略模块

实现智能预取功能，预测并预加载可能需要的数据。
"""

import threading
from typing import Any, Optional, List, Callable
from collections import defaultdict, deque


class Prefetcher:
    """
    智能预取器
    
    基于访问模式预测下一次访问，并预取数据。
    支持顺序访问检测和热点数据识别。
    """
    
    def __init__(self, buffer_pool, window_size: int = 10):
        """
        初始化预取器
        
        Args:
            buffer_pool: 缓冲池实例
            window_size: 访问历史窗口大小
        """
        self._buffer_pool = buffer_pool
        self._window_size = window_size
        
        # 访问历史
        self._access_history: deque = deque(maxlen=window_size)
        
        # 访问频率统计
        self._access_count: defaultdict = defaultdict(int)
        
        # 预取回调
        self._prefetch_callback: Optional[Callable] = None
        
        # 同步锁
        self._lock = threading.Lock()
        
        # 统计信息
        self._prefetch_count = 0
        self._successful_prefetch = 0
    
    def record_access(self, key: Any) -> None:
        """
        记录访问
        
        Args:
            key: 访问的键
        """
        with self._lock:
            self._access_history.append(key)
            self._access_count[key] += 1
    
    def predict_next(self) -> Optional[Any]:
        """
        预测下一个访问
        
        Returns:
            预测的键，如果无法预测返回None
        """
        with self._lock:
            if len(self._access_history) < 2:
                return None
            
            history = list(self._access_history)
            
            # 检测顺序访问模式
            sequential_key = self._detect_sequential_pattern(history)
            if sequential_key is not None:
                return sequential_key
            
            # 检测重复访问模式
            repeat_key = self._detect_repeat_pattern(history)
            if repeat_key is not None:
                return repeat_key
            
            # 返回最常访问的键
            if self._access_count:
                return max(self._access_count.keys(), 
                          key=lambda k: self._access_count[k])
            
            return None
    
    def _detect_sequential_pattern(self, history: List[Any]) -> Optional[Any]:
        """
        检测顺序访问模式
        
        Args:
            history: 访问历史
        
        Returns:
            预测的下一个键
        """
        # 检查是否是数字序列
        try:
            numbers = [int(k) for k in history[-3:]]
            if len(numbers) >= 2:
                # 检查是否是等差序列
                diff = numbers[-1] - numbers[-2]
                if diff == numbers[-2] - numbers[-3] if len(numbers) >= 3 else True:
                    return str(numbers[-1] + diff)
        except (ValueError, TypeError, IndexError):
            pass
        
        return None
    
    def _detect_repeat_pattern(self, history: List[Any]) -> Optional[Any]:
        """
        检测重复访问模式
        
        Args:
            history: 访问历史
        
        Returns:
            预测的下一个键
        """
        if len(history) < 4:
            return None
        
        # 检查最后几个访问是否有重复模式
        last_four = history[-4:]
        if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
            return last_four[0]
        
        return None
    
    def prefetch(self) -> Optional[Any]:
        """
        执行预取
        
        Returns:
            预取的键，如果没有预取返回None
        """
        predicted = self.predict_next()
        
        if predicted is None:
            return None
        
        with self._lock:
            self._prefetch_count += 1
            
            if self._prefetch_callback is not None:
                try:
                    self._prefetch_callback(predicted)
                    self._successful_prefetch += 1
                except Exception:
                    pass
        
        return predicted
    
    def set_callback(self, callback: Callable[[Any], None]) -> None:
        """
        设置预取回调函数
        
        Args:
            callback: 回调函数，接收预测的键作为参数
        """
        with self._lock:
            self._prefetch_callback = callback
    
    def get_hot_keys(self, top_n: int = 10) -> List[Any]:
        """
        获取热点键
        
        Args:
            top_n: 返回的热点键数量
        
        Returns:
            热点键列表
        """
        with self._lock:
            sorted_keys = sorted(
                self._access_count.keys(),
                key=lambda k: self._access_count[k],
                reverse=True
            )
            return sorted_keys[:top_n]
    
    def clear(self) -> None:
        """清空访问历史"""
        with self._lock:
            self._access_history.clear()
            self._access_count.clear()
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "window_size": self._window_size,
                "history_length": len(self._access_history),
                "unique_keys": len(self._access_count),
                "prefetch_count": self._prefetch_count,
                "successful_prefetch": self._successful_prefetch,
                "success_rate": (
                    self._successful_prefetch / self._prefetch_count 
                    if self._prefetch_count > 0 else 0.0
                ),
            }
    
    def __repr__(self) -> str:
        return f"Prefetcher(window_size={self._window_size})"
