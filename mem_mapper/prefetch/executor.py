"""
mem_mapper 预取执行器模块

提供内存预取的执行功能。
"""

import sys
from typing import List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from .strategy import PrefetchStrategy, PrefetchRange, PrefetchStrategyType
from ..core.region import MappedRegion
from ..core.exceptions import PrefetchError
from ..utils.stats import Timer


@dataclass
class PrefetchStats:
    """
    预取统计信息
    """
    
    total_prefetched: int = 0      # 总预取字节数
    total_ranges: int = 0          # 总预取范围数
    total_time: float = 0.0        # 总预取时间
    avg_time_per_range: float = 0.0  # 平均每个范围的时间
    errors: int = 0                # 错误数
    
    def to_dict(self) -> dict:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            'total_prefetched': self.total_prefetched,
            'total_ranges': self.total_ranges,
            'total_time_ms': self.total_time * 1000,
            'avg_time_per_range_ms': self.avg_time_per_range * 1000,
            'errors': self.errors,
        }


class PrefetchExecutor:
    """
    预取执行器
    
    执行内存预取操作。
    """
    
    def __init__(self, 
                 platform: Optional[Any] = None,
                 async_mode: bool = False,
                 max_workers: int = 4):
        """
        初始化预取执行器
        
        Args:
            platform: 平台实例
            async_mode: 是否异步模式
            max_workers: 最大工作线程数
        """
        self.platform = platform
        self.async_mode = async_mode
        self.max_workers = max_workers
        
        # 线程池（用于异步模式）
        self._executor: Optional[ThreadPoolExecutor] = None
        self._async_tasks: List[Future] = []
        
        # 统计信息
        self.stats = PrefetchStats()
        
        # 锁
        self._lock = threading.Lock()
    
    def prefetch(self, 
                region: MappedRegion,
                strategy: PrefetchStrategy,
                access_offset: Optional[int] = None) -> PrefetchStats:
        """
        执行预取
        
        Args:
            region: 映射区域
            strategy: 预取策略
            access_offset: 当前访问偏移
            
        Returns:
            预取统计信息
            
        Raises:
            PrefetchError: 预取失败时抛出
        """
        # 获取预取范围
        ranges = strategy.get_prefetch_ranges(region.size, access_offset)
        
        if not ranges:
            return PrefetchStats()
        
        # 执行预取
        if self.async_mode:
            return self._prefetch_async(region, ranges)
        else:
            return self._prefetch_sync(region, ranges)
    
    def _prefetch_sync(self, 
                      region: MappedRegion,
                      ranges: List[PrefetchRange]) -> PrefetchStats:
        """
        同步预取
        
        Args:
            region: 映射区域
            ranges: 预取范围列表
            
        Returns:
            预取统计信息
        """
        stats = PrefetchStats()
        timer = Timer("prefetch")
        
        for prefetch_range in ranges:
            try:
                timer.start()
                
                # 执行预取
                self._do_prefetch(region, prefetch_range)
                
                duration = timer.stop()
                
                # 更新统计
                stats.total_prefetched += prefetch_range.size
                stats.total_ranges += 1
                stats.total_time += duration
                
            except Exception as e:
                stats.errors += 1
                # 记录错误但不中断
                import warnings
                warnings.warn(f"Prefetch failed for range {prefetch_range}: {e}")
        
        # 计算平均时间
        if stats.total_ranges > 0:
            stats.avg_time_per_range = stats.total_time / stats.total_ranges
        
        # 更新全局统计
        with self._lock:
            self.stats.total_prefetched += stats.total_prefetched
            self.stats.total_ranges += stats.total_ranges
            self.stats.total_time += stats.total_time
            self.stats.errors += stats.errors
        
        return stats
    
    def _prefetch_async(self, 
                       region: MappedRegion,
                       ranges: List[PrefetchRange]) -> PrefetchStats:
        """
        异步预取
        
        Args:
            region: 映射区域
            ranges: 预取范围列表
            
        Returns:
            预取统计信息
        """
        # 创建线程池
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 提交任务
        for prefetch_range in ranges:
            future = self._executor.submit(
                self._prefetch_range_task,
                region,
                prefetch_range
            )
            self._async_tasks.append(future)
        
        # 返回初始统计（异步模式下无法立即获得完整统计）
        stats = PrefetchStats()
        stats.total_ranges = len(ranges)
        return stats
    
    def _prefetch_range_task(self, 
                            region: MappedRegion,
                            prefetch_range: PrefetchRange):
        """
        预取范围任务（用于异步模式）
        
        Args:
            region: 映射区域
            prefetch_range: 预取范围
        """
        try:
            self._do_prefetch(region, prefetch_range)
            
            # 更新统计
            with self._lock:
                self.stats.total_prefetched += prefetch_range.size
                self.stats.total_ranges += 1
                
        except Exception as e:
            with self._lock:
                self.stats.errors += 1
    
    def _do_prefetch(self, region: MappedRegion, prefetch_range: PrefetchRange):
        """
        执行实际的预取操作
        
        Args:
            region: 映射区域
            prefetch_range: 预取范围
        """
        # 计算实际地址
        addr = region.base_address + prefetch_range.offset
        size = prefetch_range.size
        
        # 根据平台执行预取
        if sys.platform.startswith('linux'):
            self._prefetch_linux(addr, size)
        elif sys.platform == 'win32':
            self._prefetch_windows(addr, size)
        else:
            # 其他平台：通过读取触发页面加载
            self._prefetch_by_read(region, prefetch_range)
    
    def _prefetch_linux(self, addr: int, size: int):
        """
        Linux平台预取
        
        使用madvise MADV_WILLNEED
        
        Args:
            addr: 内存地址
            size: 大小
        """
        try:
            import ctypes
            
            # 加载libc
            libc = ctypes.CDLL(None, use_errno=True)
            
            # 调用madvise
            madvise = libc.madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int
            
            # MADV_WILLNEED = 3
            result = madvise(ctypes.c_void_p(addr), size, 3)
            
            if result != 0:
                # madvise失败不抛出异常，只记录
                import warnings
                err = ctypes.get_errno()
                warnings.warn(f"madvise failed: errno={err}")
                
        except Exception as e:
            import warnings
            warnings.warn(f"Linux prefetch failed: {e}")
    
    def _prefetch_windows(self, addr: int, size: int):
        """
        Windows平台预取
        
        使用PrefetchVirtualMemory（Windows 8+）
        
        Args:
            addr: 内存地址
            size: 大小
        """
        try:
            # Windows 8+支持PrefetchVirtualMemory
            # 这里简化实现，使用读取方式
            pass
            
        except Exception as e:
            import warnings
            warnings.warn(f"Windows prefetch failed: {e}")
    
    def _prefetch_by_read(self, region: MappedRegion, prefetch_range: PrefetchRange):
        """
        通过读取触发预取
        
        Args:
            region: 映射区域
            prefetch_range: 预取范围
        """
        try:
            # 将内存区域转换为字节视图
            # 这里简化实现，实际需要访问内存
            pass
            
        except Exception as e:
            import warnings
            warnings.warn(f"Read-based prefetch failed: {e}")
    
    def wait_async_tasks(self, timeout: Optional[float] = None):
        """
        等待异步任务完成
        
        Args:
            timeout: 超时时间（秒）
        """
        if not self._async_tasks:
            return
        
        # 等待所有任务完成
        for future in self._async_tasks:
            try:
                future.result(timeout=timeout)
            except Exception:
                pass
        
        # 清空任务列表
        self._async_tasks.clear()
    
    def get_stats(self) -> PrefetchStats:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        with self._lock:
            # 计算平均时间
            if self.stats.total_ranges > 0:
                self.stats.avg_time_per_range = self.stats.total_time / self.stats.total_ranges
            return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self.stats = PrefetchStats()
    
    def shutdown(self):
        """关闭执行器"""
        # 等待所有异步任务完成
        self.wait_async_tasks()
        
        # 关闭线程池
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def prefault_region(region: MappedRegion,
                   strategy: Optional[PrefetchStrategy] = None,
                   platform: Optional[Any] = None) -> PrefetchStats:
    """
    预触发区域（全局函数）
    
    Args:
        region: 映射区域
        strategy: 预取策略，None则使用默认策略
        platform: 平台实例
        
    Returns:
        预取统计信息
    """
    if strategy is None:
        from .strategy import SequentialPrefetchStrategy
        strategy = SequentialPrefetchStrategy()
    
    executor = PrefetchExecutor(platform=platform)
    return executor.prefetch(region, strategy)
