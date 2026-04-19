"""
mem_mapper 大页池管理模块

提供大页池的分配和管理功能。
"""

import uuid
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from .detector import HugePageDetector, get_hugepage_detector
from ..core.exceptions import (
    HugePageNotAvailableError, 
    HugePagePoolExhaustedError
)
from ..utils.alignment import align_to_huge_page


@dataclass
class HugePageAllocation:
    """
    大页分配记录
    
    记录一次大页分配的详细信息。
    """
    
    allocation_id: uuid.UUID        # 分配ID
    page_size: int                  # 大页大小
    page_count: int                 # 页面数量
    total_size: int                 # 总大小
    region_id: Optional[str]        # 关联的映射区域ID
    allocation_time: float          # 分配时间
    numa_node: Optional[int] = None # NUMA节点
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            'allocation_id': str(self.allocation_id),
            'page_size': self.page_size,
            'page_count': self.page_count,
            'total_size': self.total_size,
            'region_id': self.region_id,
            'allocation_time': self.allocation_time,
            'numa_node': self.numa_node,
        }


@dataclass
class HugePagePoolStats:
    """
    大页池统计信息
    """
    
    total_allocations: int = 0      # 总分配次数
    total_deallocations: int = 0    # 总释放次数
    total_pages_allocated: int = 0  # 总分配页面数
    total_pages_freed: int = 0      # 总释放页面数
    active_allocations: int = 0     # 活跃分配数
    active_pages: int = 0           # 活跃页面数
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            'total_allocations': self.total_allocations,
            'total_deallocations': self.total_deallocations,
            'total_pages_allocated': self.total_pages_allocated,
            'total_pages_freed': self.total_pages_freed,
            'active_allocations': self.active_allocations,
            'active_pages': self.active_pages,
        }


class HugePagePool:
    """
    大页池管理器
    
    管理大页的分配、释放和统计。
    """
    
    def __init__(self, detector: Optional[HugePageDetector] = None):
        """
        初始化大页池
        
        Args:
            detector: 大页检测器，None则使用全局实例
        """
        self.detector = detector or get_hugepage_detector()
        
        # 分配记录
        self.allocations: Dict[uuid.UUID, HugePageAllocation] = {}
        
        # 按页面大小分组的分配
        self.allocations_by_size: Dict[int, List[uuid.UUID]] = {}
        
        # 按区域ID索引的分配
        self.allocations_by_region: Dict[str, uuid.UUID] = {}
        
        # 统计信息
        self.stats = HugePagePoolStats()
        
        # 线程锁
        self._lock = threading.RLock()
    
    def allocate(self,
                 size: int,
                 page_size: Optional[int] = None,
                 region_id: Optional[str] = None,
                 numa_node: Optional[int] = None) -> HugePageAllocation:
        """
        分配大页
        
        Args:
            size: 需要分配的大小（字节）
            page_size: 大页大小，None则自动选择
            region_id: 关联的映射区域ID
            numa_node: NUMA节点
            
        Returns:
            大页分配记录
            
        Raises:
            HugePageNotAvailableError: 大页不可用时抛出
            HugePagePoolExhaustedError: 大页池耗尽时抛出
        """
        with self._lock:
            # 获取配置
            config = self.detector.detect()
            
            # 选择大页大小
            if page_size is None:
                page_size = self.detector.recommend_page_size(size)
            
            # 检查是否支持
            if not config.is_size_supported(page_size):
                raise HugePageNotAvailableError(page_size)
            
            # 对齐大小
            aligned_size = align_to_huge_page(size, page_size)
            
            # 计算需要的页面数
            page_count = aligned_size // page_size
            
            # 检查是否有足够的大页
            info = self.detector.get_page_info(page_size)
            if not info or info.free_pages < page_count:
                raise HugePagePoolExhaustedError(
                    requested=page_count,
                    available=info.free_pages if info else 0,
                    page_size=page_size
                )
            
            # 创建分配记录
            allocation = HugePageAllocation(
                allocation_id=uuid.uuid4(),
                page_size=page_size,
                page_count=page_count,
                total_size=aligned_size,
                region_id=region_id,
                allocation_time=time.time(),
                numa_node=numa_node
            )
            
            # 记录分配
            self.allocations[allocation.allocation_id] = allocation
            
            # 更新索引
            if page_size not in self.allocations_by_size:
                self.allocations_by_size[page_size] = []
            self.allocations_by_size[page_size].append(allocation.allocation_id)
            
            if region_id is not None:
                self.allocations_by_region[region_id] = allocation.allocation_id
            
            # 更新统计
            self.stats.total_allocations += 1
            self.stats.total_pages_allocated += page_count
            self.stats.active_allocations += 1
            self.stats.active_pages += page_count
            
            return allocation
    
    def deallocate(self, allocation_id: uuid.UUID) -> bool:
        """
        释放大页分配
        
        Args:
            allocation_id: 分配ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            # 从索引中移除
            if allocation.page_size in self.allocations_by_size:
                try:
                    self.allocations_by_size[allocation.page_size].remove(allocation_id)
                except ValueError:
                    pass
            
            if allocation.region_id is not None:
                if allocation.region_id in self.allocations_by_region:
                    del self.allocations_by_region[allocation.region_id]
            
            # 从分配记录中移除
            del self.allocations[allocation_id]
            
            # 更新统计
            self.stats.total_deallocations += 1
            self.stats.total_pages_freed += allocation.page_count
            self.stats.active_allocations -= 1
            self.stats.active_pages -= allocation.page_count
            
            return True
    
    def deallocate_by_region(self, region_id: str) -> bool:
        """
        根据区域ID释放大页分配
        
        Args:
            region_id: 区域ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if region_id not in self.allocations_by_region:
                return False
            
            allocation_id = self.allocations_by_region[region_id]
            return self.deallocate(allocation_id)
    
    def get_allocation(self, allocation_id: uuid.UUID) -> Optional[HugePageAllocation]:
        """
        获取分配记录
        
        Args:
            allocation_id: 分配ID
            
        Returns:
            分配记录
        """
        with self._lock:
            return self.allocations.get(allocation_id)
    
    def get_allocation_by_region(self, region_id: str) -> Optional[HugePageAllocation]:
        """
        根据区域ID获取分配记录
        
        Args:
            region_id: 区域ID
            
        Returns:
            分配记录
        """
        with self._lock:
            allocation_id = self.allocations_by_region.get(region_id)
            if allocation_id:
                return self.allocations.get(allocation_id)
            return None
    
    def get_allocations_by_size(self, page_size: int) -> List[HugePageAllocation]:
        """
        获取指定大小的所有分配
        
        Args:
            page_size: 页面大小
            
        Returns:
            分配记录列表
        """
        with self._lock:
            allocation_ids = self.allocations_by_size.get(page_size, [])
            return [self.allocations[aid] for aid in allocation_ids if aid in self.allocations]
    
    def get_all_allocations(self) -> List[HugePageAllocation]:
        """
        获取所有分配
        
        Returns:
            所有分配记录列表
        """
        with self._lock:
            return list(self.allocations.values())
    
    def get_stats(self) -> HugePagePoolStats:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        with self._lock:
            return self.stats
    
    def get_available_pages(self, page_size: int) -> int:
        """
        获取指定大小的可用页面数
        
        Args:
            page_size: 页面大小
            
        Returns:
            可用页面数
        """
        info = self.detector.get_page_info(page_size)
        if not info:
            return 0
        
        # 计算已分配的页面数
        allocated = sum(
            alloc.page_count 
            for alloc in self.get_allocations_by_size(page_size)
        )
        
        # 可用 = 系统空闲 - 已分配
        return max(0, info.free_pages - allocated)
    
    def can_allocate(self, size: int, page_size: Optional[int] = None) -> bool:
        """
        检查是否可以分配
        
        Args:
            size: 需要分配的大小
            page_size: 页面大小
            
        Returns:
            是否可以分配
        """
        if page_size is None:
            page_size = self.detector.recommend_page_size(size)
        
        aligned_size = align_to_huge_page(size, page_size)
        page_count = aligned_size // page_size
        
        available = self.get_available_pages(page_size)
        return available >= page_count
    
    def clear(self):
        """清空所有分配记录"""
        with self._lock:
            self.allocations.clear()
            self.allocations_by_size.clear()
            self.allocations_by_region.clear()
            
            # 重置统计
            self.stats.active_allocations = 0
            self.stats.active_pages = 0
    
    def get_summary(self) -> Dict:
        """
        获取摘要信息
        
        Returns:
            摘要信息字典
        """
        with self._lock:
            # 按大小分组统计
            size_stats = {}
            for page_size, allocation_ids in self.allocations_by_size.items():
                allocations = [self.allocations[aid] for aid in allocation_ids if aid in self.allocations]
                size_stats[page_size] = {
                    'allocations': len(allocations),
                    'total_pages': sum(a.page_count for a in allocations),
                    'total_size': sum(a.total_size for a in allocations),
                    'available_pages': self.get_available_pages(page_size),
                }
            
            return {
                'stats': self.stats.to_dict(),
                'size_stats': size_stats,
                'total_allocations': len(self.allocations),
            }


# 全局大页池实例
_global_pool = None


def get_hugepage_pool() -> HugePagePool:
    """
    获取全局大页池实例
    
    Returns:
        大页池实例
    """
    global _global_pool
    
    if _global_pool is None:
        _global_pool = HugePagePool()
    
    return _global_pool
