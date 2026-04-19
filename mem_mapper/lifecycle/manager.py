"""
mem_mapper 生命周期管理模块

提供内存映射的生命周期管理功能。
"""

import time
import threading
from typing import Optional, Dict
from queue import PriorityQueue, Empty
from dataclasses import dataclass

from ..core.region import MappedRegion, MappingState
from ..core.registry import MappingRegistry
from ..core.exceptions import LifecycleError, RegionError


@dataclass
class CleanupTask:
    """
    清理任务
    
    表示一个待清理的映射区域。
    """
    
    region: MappedRegion        # 待清理的区域
    scheduled_time: float       # 计划执行时间
    priority: int = 0           # 优先级（数值越小优先级越高）
    
    def __lt__(self, other):
        """比较运算符，用于优先队列排序"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.scheduled_time < other.scheduled_time


class LifecycleManager:
    """
    生命周期管理器
    
    管理内存映射的完整生命周期，包括创建、引用计数、清理等。
    """
    
    def __init__(self, 
                 registry: MappingRegistry,
                 cleanup_delay: float = 60.0,
                 idle_threshold: float = 300.0):
        """
        初始化生命周期管理器
        
        Args:
            registry: 映射注册表
            cleanup_delay: 清理延迟时间（秒）
            idle_threshold: 空闲阈值（秒）
        """
        self.registry = registry
        self.cleanup_delay = cleanup_delay
        self.idle_threshold = idle_threshold
        
        # 清理队列
        self.cleanup_queue = PriorityQueue()
        
        # 清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 统计信息
        self._total_cleanups = 0
        self._total_delayed_cleanups = 0
        
        # 锁
        self._lock = threading.Lock()
    
    def acquire(self, region: MappedRegion):
        """
        获取映射引用
        
        Args:
            region: 映射区域
        """
        region.acquire()
        region.state = MappingState.ACTIVE
    
    def release(self, region: MappedRegion):
        """
        释放映射引用
        
        Args:
            region: 映射区域
        """
        ref_count = region.release()
        
        # 如果引用计数为0，调度清理
        if ref_count == 0:
            region.state = MappingState.INACTIVE
            self._schedule_cleanup(region)
    
    def _schedule_cleanup(self, region: MappedRegion, delay: Optional[float] = None):
        """
        调度清理任务
        
        Args:
            region: 映射区域
            delay: 延迟时间（秒），None则使用默认值
        """
        if delay is None:
            delay = self.cleanup_delay
        
        scheduled_time = time.time() + delay
        task = CleanupTask(region=region, scheduled_time=scheduled_time)
        
        self.cleanup_queue.put(task)
    
    def start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """停止清理线程"""
        if self._cleanup_thread is None:
            return
        
        self._stop_event.set()
        self._cleanup_thread.join(timeout=5.0)
        self._cleanup_thread = None
    
    def _cleanup_worker(self):
        """清理工作线程"""
        while not self._stop_event.is_set():
            try:
                # 获取清理任务
                task = self.cleanup_queue.get(timeout=1.0)
                
                # 检查是否到达执行时间
                if time.time() >= task.scheduled_time:
                    self._do_cleanup(task.region)
                else:
                    # 未到执行时间，放回队列
                    self.cleanup_queue.put(task)
                    
            except Empty:
                continue
            except Exception as e:
                # 记录错误但继续运行
                import warnings
                warnings.warn(f"Cleanup worker error: {e}")
    
    def _do_cleanup(self, region: MappedRegion):
        """
        执行清理
        
        Args:
            region: 映射区域
        """
        try:
            # 检查引用计数
            if region.get_ref_count() > 0:
                # 还有引用，不清理
                return
            
            # 检查空闲时间
            idle_time = region.get_idle_time()
            if idle_time < self.idle_threshold:
                # 未达到空闲阈值，重新调度
                self._schedule_cleanup(region)
                self._total_delayed_cleanups += 1
                return
            
            # 标记为僵尸状态
            region.state = MappingState.ZOMBIE
            
            # 执行清理
            self._cleanup_region(region)
            
            # 从注册表移除
            self.registry.remove(region.region_id)
            
            # 更新统计
            with self._lock:
                self._total_cleanups += 1
                
        except Exception as e:
            raise LifecycleError(f"Failed to cleanup region {region.region_id}: {e}")
    
    def _cleanup_region(self, region: MappedRegion):
        """
        清理映射区域
        
        Args:
            region: 映射区域
        """
        # 这里需要调用平台相关的清理函数
        # 实际实现需要在MemoryMapper中完成
        pass
    
    def force_cleanup(self, region: MappedRegion):
        """
        强制清理映射区域
        
        Args:
            region: 映射区域
        """
        self._do_cleanup(region)
    
    def cleanup_all(self):
        """清理所有映射"""
        regions = self.registry.get_all()
        for region in regions:
            try:
                self._do_cleanup(region)
            except Exception:
                pass
    
    def get_cleanup_queue_size(self) -> int:
        """
        获取清理队列大小
        
        Returns:
            队列大小
        """
        return self.cleanup_queue.qsize()
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'total_cleanups': self._total_cleanups,
                'total_delayed_cleanups': self._total_delayed_cleanups,
                'cleanup_queue_size': self.get_cleanup_queue_size(),
                'cleanup_delay': self.cleanup_delay,
                'idle_threshold': self.idle_threshold,
            }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_cleanup_thread()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_cleanup_thread()
        return False


class RegionGuard:
    """
    区域守卫
    
    自动管理映射区域的引用计数。
    """
    
    def __init__(self, region: MappedRegion, lifecycle: LifecycleManager):
        """
        初始化区域守卫
        
        Args:
            region: 映射区域
            lifecycle: 生命周期管理器
        """
        self.region = region
        self.lifecycle = lifecycle
    
    def __enter__(self):
        """上下文管理器入口"""
        self.lifecycle.acquire(self.region)
        return self.region
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.lifecycle.release(self.region)
        return False


def create_region_guard(region: MappedRegion, 
                       lifecycle: LifecycleManager) -> RegionGuard:
    """
    创建区域守卫（全局函数）
    
    Args:
        region: 映射区域
        lifecycle: 生命周期管理器
        
    Returns:
        区域守卫实例
    """
    return RegionGuard(region, lifecycle)
