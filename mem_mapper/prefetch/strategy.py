"""
mem_mapper 预取策略模块

提供内存预取策略的定义和选择。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from ..utils.alignment import PAGE_SIZE_4KB


class PrefetchStrategyType(Enum):
    """预取策略类型"""
    SEQUENTIAL = "sequential"    # 顺序预取
    RANDOM = "random"            # 随机预取
    ADAPTIVE = "adaptive"        # 自适应预取
    NONE = "none"                # 不预取


@dataclass
class PrefetchRange:
    """
    预取范围
    
    表示一个需要预取的内存范围。
    """
    
    offset: int     # 偏移量
    size: int       # 大小
    
    def to_tuple(self) -> Tuple[int, int]:
        """
        转换为元组
        
        Returns:
            (offset, size) 元组
        """
        return (self.offset, self.size)


class PrefetchStrategy(ABC):
    """
    预取策略基类
    
    定义预取策略的接口。
    """
    
    def __init__(self, page_size: int = PAGE_SIZE_4KB):
        """
        初始化预取策略
        
        Args:
            page_size: 页面大小
        """
        self.page_size = page_size
    
    @abstractmethod
    def get_prefetch_ranges(self, 
                           total_size: int,
                           access_offset: Optional[int] = None) -> List[PrefetchRange]:
        """
        获取预取范围
        
        Args:
            total_size: 总大小
            access_offset: 当前访问偏移（可选）
            
        Returns:
            预取范围列表
        """
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> PrefetchStrategyType:
        """
        获取策略类型
        
        Returns:
            策略类型
        """
        pass
    
    def get_page_count(self, size: int) -> int:
        """
        计算页面数量
        
        Args:
            size: 大小
            
        Returns:
            页面数量
        """
        return (size + self.page_size - 1) // self.page_size


class SequentialPrefetchStrategy(PrefetchStrategy):
    """
    顺序预取策略
    
    适用于顺序访问模式。
    """
    
    def __init__(self, 
                 page_size: int = PAGE_SIZE_4KB,
                 window_size: int = 16,
                 stride: int = 1):
        """
        初始化顺序预取策略
        
        Args:
            page_size: 页面大小
            window_size: 预取窗口大小（页面数）
            stride: 步长（页面数）
        """
        super().__init__(page_size)
        self.window_size = window_size
        self.stride = stride
    
    def get_prefetch_ranges(self, 
                           total_size: int,
                           access_offset: Optional[int] = None) -> List[PrefetchRange]:
        """
        获取预取范围
        
        Args:
            total_size: 总大小
            access_offset: 当前访问偏移
            
        Returns:
            预取范围列表
        """
        ranges = []
        
        # 确定起始位置
        if access_offset is not None:
            start_page = access_offset // self.page_size
        else:
            start_page = 0
        
        # 计算预取范围
        prefetch_size = self.window_size * self.page_size
        prefetch_offset = start_page * self.page_size
        
        # 确保不超出总大小
        if prefetch_offset < total_size:
            actual_size = min(prefetch_size, total_size - prefetch_offset)
            ranges.append(PrefetchRange(offset=prefetch_offset, size=actual_size))
        
        return ranges
    
    def get_strategy_type(self) -> PrefetchStrategyType:
        """
        获取策略类型
        
        Returns:
            策略类型
        """
        return PrefetchStrategyType.SEQUENTIAL


class RandomPrefetchStrategy(PrefetchStrategy):
    """
    随机预取策略
    
    适用于随机访问模式，预取热点页面。
    """
    
    def __init__(self, 
                 page_size: int = PAGE_SIZE_4KB,
                 hot_pages: Optional[List[int]] = None,
                 prefetch_ratio: float = 0.3):
        """
        初始化随机预取策略
        
        Args:
            page_size: 页面大小
            hot_pages: 热点页面列表（页面偏移）
            prefetch_ratio: 预取比例
        """
        super().__init__(page_size)
        self.hot_pages = hot_pages or []
        self.prefetch_ratio = prefetch_ratio
    
    def get_prefetch_ranges(self, 
                           total_size: int,
                           access_offset: Optional[int] = None) -> List[PrefetchRange]:
        """
        获取预取范围
        
        Args:
            total_size: 总大小
            access_offset: 当前访问偏移
            
        Returns:
            预取范围列表
        """
        ranges = []
        
        # 如果有热点页面，预取热点页面
        if self.hot_pages:
            # 选择前N个热点页面
            num_pages = max(1, int(len(self.hot_pages) * self.prefetch_ratio))
            selected_pages = self.hot_pages[:num_pages]
            
            # 转换为预取范围
            for page_offset in selected_pages:
                if page_offset < total_size:
                    actual_size = min(self.page_size, total_size - page_offset)
                    ranges.append(PrefetchRange(offset=page_offset, size=actual_size))
        
        # 如果没有热点页面，预取当前访问位置
        elif access_offset is not None:
            if access_offset < total_size:
                actual_size = min(self.page_size, total_size - access_offset)
                ranges.append(PrefetchRange(offset=access_offset, size=actual_size))
        
        return ranges
    
    def get_strategy_type(self) -> PrefetchStrategyType:
        """
        获取策略类型
        
        Returns:
            策略类型
        """
        return PrefetchStrategyType.RANDOM


class AdaptivePrefetchStrategy(PrefetchStrategy):
    """
    自适应预取策略
    
    根据访问模式动态调整预取策略。
    """
    
    def __init__(self, 
                 page_size: int = PAGE_SIZE_4KB,
                 initial_window: int = 4,
                 max_window: int = 64,
                 learning_rate: float = 0.1):
        """
        初始化自适应预取策略
        
        Args:
            page_size: 页面大小
            initial_window: 初始窗口大小
            max_window: 最大窗口大小
            learning_rate: 学习率
        """
        super().__init__(page_size)
        self.initial_window = initial_window
        self.max_window = max_window
        self.learning_rate = learning_rate
        
        # 当前窗口大小
        self.current_window = initial_window
        
        # 访问历史
        self.access_history: List[int] = []
        self.max_history = 100
    
    def get_prefetch_ranges(self, 
                           total_size: int,
                           access_offset: Optional[int] = None) -> List[PrefetchRange]:
        """
        获取预取范围
        
        Args:
            total_size: 总大小
            access_offset: 当前访问偏移
            
        Returns:
            预取范围列表
        """
        # 更新访问历史
        if access_offset is not None:
            self.access_history.append(access_offset)
            if len(self.access_history) > self.max_history:
                self.access_history = self.access_history[-self.max_history:]
            
            # 调整窗口大小
            self._adjust_window()
        
        # 使用当前窗口大小进行预取
        prefetch_size = self.current_window * self.page_size
        
        ranges = []
        if access_offset is not None and access_offset < total_size:
            actual_size = min(prefetch_size, total_size - access_offset)
            ranges.append(PrefetchRange(offset=access_offset, size=actual_size))
        
        return ranges
    
    def _adjust_window(self):
        """调整窗口大小"""
        if len(self.access_history) < 2:
            return
        
        # 分析最近的访问模式
        recent = self.access_history[-10:]
        
        # 计算步长
        strides = []
        for i in range(1, len(recent)):
            stride = abs(recent[i] - recent[i-1])
            strides.append(stride)
        
        if not strides:
            return
        
        avg_stride = sum(strides) / len(strides)
        
        # 如果步长小且一致，增加窗口
        if avg_stride <= self.page_size * 2:
            self.current_window = min(
                self.max_window,
                int(self.current_window * (1 + self.learning_rate))
            )
        # 如果步长大，减少窗口
        elif avg_stride > self.page_size * 4:
            self.current_window = max(
                1,
                int(self.current_window * (1 - self.learning_rate))
            )
    
    def get_strategy_type(self) -> PrefetchStrategyType:
        """
        获取策略类型
        
        Returns:
            策略类型
        """
        return PrefetchStrategyType.ADAPTIVE


class NoPrefetchStrategy(PrefetchStrategy):
    """
    不预取策略
    
    禁用预取功能。
    """
    
    def get_prefetch_ranges(self, 
                           total_size: int,
                           access_offset: Optional[int] = None) -> List[PrefetchRange]:
        """
        获取预取范围
        
        Args:
            total_size: 总大小
            access_offset: 当前访问偏移
            
        Returns:
            空列表
        """
        return []
    
    def get_strategy_type(self) -> PrefetchStrategyType:
        """
        获取策略类型
        
        Returns:
            策略类型
        """
        return PrefetchStrategyType.NONE


def create_prefetch_strategy(strategy_type: PrefetchStrategyType,
                            page_size: int = PAGE_SIZE_4KB,
                            **kwargs) -> PrefetchStrategy:
    """
    创建预取策略
    
    Args:
        strategy_type: 策略类型
        page_size: 页面大小
        **kwargs: 策略参数
        
    Returns:
        预取策略实例
    """
    if strategy_type == PrefetchStrategyType.SEQUENTIAL:
        return SequentialPrefetchStrategy(page_size=page_size, **kwargs)
    elif strategy_type == PrefetchStrategyType.RANDOM:
        return RandomPrefetchStrategy(page_size=page_size, **kwargs)
    elif strategy_type == PrefetchStrategyType.ADAPTIVE:
        return AdaptivePrefetchStrategy(page_size=page_size, **kwargs)
    else:
        return NoPrefetchStrategy(page_size=page_size)
