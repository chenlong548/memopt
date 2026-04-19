"""
mem_mapper 映射注册表模块

提供全局映射注册表，管理所有内存映射区域。
"""

from typing import Dict, List, Optional
import uuid
import threading
from queue import PriorityQueue, Empty
import time

from .region import MappedRegion, AtomicCounter


class CleanupTask:
    """
    清理任务
    
    表示一个待清理的映射区域。
    """
    
    def __init__(self, region: MappedRegion, scheduled_time: float, priority: int = 0):
        """
        初始化清理任务
        
        Args:
            region: 待清理的映射区域
            scheduled_time: 计划执行时间
            priority: 优先级（数值越小优先级越高）
        """
        self.region = region
        self.scheduled_time = scheduled_time
        self.priority = priority
    
    def __lt__(self, other):
        """比较运算符，用于优先队列排序"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.scheduled_time < other.scheduled_time


class MappingRegistry:
    """
    全局映射注册表
    
    管理所有内存映射区域，提供查找、添加、删除等功能。
    支持按区域ID、文件路径、地址等多种方式查找。
    """
    
    def __init__(self, max_mappings: int = 10000):
        """
        初始化映射注册表
        
        Args:
            max_mappings: 最大映射数量
        """
        # 主映射表：region_id -> MappedRegion
        self.mappings: Dict[uuid.UUID, MappedRegion] = {}
        
        # 文件索引：file_path -> [region_id, ...]
        self.file_index: Dict[str, List[uuid.UUID]] = {}
        
        # 地址索引：base_address -> region_id
        self.addr_index: Dict[int, uuid.UUID] = {}
        
        # 统计信息
        self.total_mappings = AtomicCounter(0)
        self.total_mapped_bytes = AtomicCounter(0)
        
        # 清理队列
        self.cleanup_queue = PriorityQueue()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 最大映射数量
        self.max_mappings = max_mappings
    
    def add(self, region: MappedRegion) -> bool:
        """
        添加映射到注册表
        
        Args:
            region: 要添加的映射区域
            
        Returns:
            是否添加成功
            
        Raises:
            ValueError: 如果映射已存在或超过最大数量
        """
        with self._lock:
            # 检查是否已存在
            if region.region_id in self.mappings:
                raise ValueError(f"Region {region.region_id} already exists")
            
            # 检查是否超过最大数量
            if len(self.mappings) >= self.max_mappings:
                raise ValueError(f"Maximum mappings ({self.max_mappings}) reached")
            
            # 添加到主映射表
            self.mappings[region.region_id] = region
            
            # 更新文件索引
            if region.file_path not in self.file_index:
                self.file_index[region.file_path] = []
            self.file_index[region.file_path].append(region.region_id)
            
            # 更新地址索引
            self.addr_index[region.base_address] = region.region_id
            
            # 更新统计
            self.total_mappings.increment()
            self.total_mapped_bytes.increment(region.size)
            
            return True
    
    def remove(self, region_id: uuid.UUID) -> Optional[MappedRegion]:
        """
        从注册表移除映射
        
        Args:
            region_id: 映射区域ID
            
        Returns:
            被移除的映射区域，如果不存在则返回None
        """
        with self._lock:
            if region_id not in self.mappings:
                return None
            
            region = self.mappings[region_id]
            
            # 从文件索引移除
            if region.file_path in self.file_index:
                try:
                    self.file_index[region.file_path].remove(region_id)
                    if not self.file_index[region.file_path]:
                        del self.file_index[region.file_path]
                except ValueError:
                    pass
            
            # 从地址索引移除
            if region.base_address in self.addr_index:
                del self.addr_index[region.base_address]
            
            # 从主映射表移除
            del self.mappings[region_id]
            
            # 更新统计
            self.total_mappings.decrement()
            self.total_mapped_bytes.decrement(region.size)
            
            return region
    
    def get(self, region_id: uuid.UUID) -> Optional[MappedRegion]:
        """
        根据ID获取映射
        
        Args:
            region_id: 映射区域ID
            
        Returns:
            映射区域，如果不存在则返回None
        """
        with self._lock:
            return self.mappings.get(region_id)
    
    def find_by_file(self, file_path: str) -> List[MappedRegion]:
        """
        根据文件路径查找映射
        
        Args:
            file_path: 文件路径
            
        Returns:
            映射区域列表
        """
        with self._lock:
            if file_path not in self.file_index:
                return []
            return [
                self.mappings[rid] 
                for rid in self.file_index[file_path] 
                if rid in self.mappings
            ]
    
    def find_by_addr(self, addr: int) -> Optional[MappedRegion]:
        """
        根据地址查找映射
        
        Args:
            addr: 内存地址
            
        Returns:
            映射区域，如果不存在则返回None
        """
        with self._lock:
            region_id = self.addr_index.get(addr)
            if region_id:
                return self.mappings.get(region_id)
        return None
    
    def find_containing(self, addr: int) -> Optional[MappedRegion]:
        """
        查找包含指定地址的映射
        
        Args:
            addr: 内存地址
            
        Returns:
            包含该地址的映射区域，如果不存在则返回None
        """
        with self._lock:
            for region in self.mappings.values():
                if region.contains(addr):
                    return region
        return None
    
    def find_by_numa_node(self, numa_node: int) -> List[MappedRegion]:
        """
        查找指定NUMA节点上的映射
        
        Args:
            numa_node: NUMA节点ID
            
        Returns:
            映射区域列表
        """
        with self._lock:
            return [
                region for region in self.mappings.values()
                if region.numa_node == numa_node
            ]
    
    def find_huge_page_mappings(self) -> List[MappedRegion]:
        """
        查找使用大页的映射
        
        Returns:
            使用大页的映射区域列表
        """
        with self._lock:
            return [
                region for region in self.mappings.values()
                if region.uses_huge_pages
            ]
    
    def find_gpu_mappings(self) -> List[MappedRegion]:
        """
        查找有GPU映射的区域
        
        Returns:
            有GPU映射的区域列表
        """
        with self._lock:
            return [
                region for region in self.mappings.values()
                if region.gpu_mapping is not None
            ]
    
    def get_all(self) -> List[MappedRegion]:
        """
        获取所有映射
        
        Returns:
            所有映射区域列表
        """
        with self._lock:
            return list(self.mappings.values())
    
    def get_count(self) -> int:
        """
        获取映射数量
        
        Returns:
            映射数量
        """
        return self.total_mappings.get()
    
    def get_total_size(self) -> int:
        """
        获取总映射大小
        
        Returns:
            总映射大小（字节）
        """
        return self.total_mapped_bytes.get()
    
    def get_stats(self) -> Dict:
        """
        获取注册表统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            # 计算各种统计信息
            huge_page_count = sum(1 for r in self.mappings.values() if r.uses_huge_pages)
            gpu_mapping_count = sum(1 for r in self.mappings.values() if r.gpu_mapping is not None)
            locked_count = sum(1 for r in self.mappings.values() if r.is_locked)
            dirty_count = sum(1 for r in self.mappings.values() if r.is_dirty)
            
            # 计算NUMA节点分布
            numa_distribution: Dict[int, int] = {}
            for region in self.mappings.values():
                node = region.numa_node
                numa_distribution[node] = numa_distribution.get(node, 0) + 1
            
            # 计算文件分布
            file_count = len(self.file_index)
            
            return {
                'total_mappings': self.total_mappings.get(),
                'total_mapped_bytes': self.total_mapped_bytes.get(),
                'unique_files': file_count,
                'huge_page_mappings': huge_page_count,
                'gpu_mappings': gpu_mapping_count,
                'locked_mappings': locked_count,
                'dirty_mappings': dirty_count,
                'numa_distribution': numa_distribution,
                'max_mappings': self.max_mappings
            }
    
    def schedule_cleanup(self, region: MappedRegion, delay: float = 60.0, priority: int = 0):
        """
        调度清理任务
        
        Args:
            region: 待清理的映射区域
            delay: 延迟时间（秒）
            priority: 优先级
        """
        scheduled_time = time.time() + delay
        task = CleanupTask(region, scheduled_time, priority)
        self.cleanup_queue.put(task)
    
    def get_cleanup_task(self, timeout: float = 0.1) -> Optional[CleanupTask]:
        """
        获取待执行的清理任务
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            清理任务，如果没有则返回None
        """
        try:
            task = self.cleanup_queue.get(timeout=timeout)
            # 检查是否到达执行时间
            if time.time() >= task.scheduled_time:
                return task
            else:
                # 未到执行时间，放回队列
                self.cleanup_queue.put(task)
                return None
        except Empty:
            return None
    
    def clear(self):
        """清空注册表"""
        with self._lock:
            self.mappings.clear()
            self.file_index.clear()
            self.addr_index.clear()
            self.total_mappings.set(0)
            self.total_mapped_bytes.set(0)
    
    def __len__(self) -> int:
        """获取映射数量"""
        return len(self.mappings)
    
    def __contains__(self, region_id: uuid.UUID) -> bool:
        """检查映射是否存在"""
        return region_id in self.mappings
    
    def __iter__(self):
        """迭代所有映射"""
        with self._lock:
            return iter(list(self.mappings.values()))
    
    def __repr__(self) -> str:
        return (
            f"MappingRegistry(count={self.get_count()}, "
            f"size={self.get_total_size()}, "
            f"files={len(self.file_index)})"
        )


class MappingRegistryView:
    """
    映射注册表视图
    
    提供对注册表的只读访问视图。
    """
    
    def __init__(self, registry: MappingRegistry):
        """
        初始化视图
        
        Args:
            registry: 映射注册表
        """
        self._registry = registry
    
    def get(self, region_id: uuid.UUID) -> Optional[MappedRegion]:
        """根据ID获取映射"""
        return self._registry.get(region_id)
    
    def find_by_file(self, file_path: str) -> List[MappedRegion]:
        """根据文件路径查找映射"""
        return self._registry.find_by_file(file_path)
    
    def find_by_addr(self, addr: int) -> Optional[MappedRegion]:
        """根据地址查找映射"""
        return self._registry.find_by_addr(addr)
    
    def find_containing(self, addr: int) -> Optional[MappedRegion]:
        """查找包含指定地址的映射"""
        return self._registry.find_containing(addr)
    
    def get_all(self) -> List[MappedRegion]:
        """获取所有映射"""
        return self._registry.get_all()
    
    def get_count(self) -> int:
        """获取映射数量"""
        return self._registry.get_count()
    
    def get_total_size(self) -> int:
        """获取总映射大小"""
        return self._registry.get_total_size()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self._registry.get_stats()
