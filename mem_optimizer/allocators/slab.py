"""
mem_optimizer Slab分配器

实现Slab内存分配算法。
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.base import (
    AllocatorBase, AllocatorType, AllocationRequest, AllocationResult,
    MemoryBlock, MemoryRegionState
)
from ..core.config import SlabAllocatorConfig
from ..core.exceptions import AllocationError, OutOfMemoryError


@dataclass
class SlabObject:
    """Slab对象"""
    address: int
    size: int
    is_free: bool = True
    slab_id: int = 0


@dataclass
class Slab:
    """Slab结构"""
    slab_id: int
    address: int
    size: int
    object_size: int
    objects: List[SlabObject] = field(default_factory=list)
    free_count: int = 0
    in_use_count: int = 0

    @property
    def capacity(self) -> int:
        return len(self.objects)

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.in_use_count / self.capacity

    def get_free_object(self) -> Optional[SlabObject]:
        """获取空闲对象"""
        for obj in self.objects:
            if obj.is_free:
                return obj
        return None


@dataclass
class ObjectCache:
    """对象缓存"""
    object_size: int
    slabs: List[Slab] = field(default_factory=list)
    total_objects: int = 0
    free_objects: int = 0

    def get_utilization(self) -> float:
        if self.total_objects == 0:
            return 0.0
        return (self.total_objects - self.free_objects) / self.total_objects


class SlabAllocator(AllocatorBase):
    """
    Slab分配器

    使用Slab算法管理小对象内存分配，适合频繁分配释放的小对象。
    """

    def __init__(self,
                total_size: int,
                base_address: int = 0,
                config: Optional[SlabAllocatorConfig] = None):
        """
        初始化Slab分配器

        Args:
            total_size: 总内存大小
            base_address: 基地址
            config: 分配器配置
        """
        super().__init__(total_size, base_address)

        self.config = config or SlabAllocatorConfig()
        self.allocator_type = AllocatorType.SLAB

        self.slab_size = self.config.slab_size
        self.object_sizes = sorted(self.config.object_sizes)
        self.cache_limit = self.config.cache_limit
        self.enable_coloring = self.config.enable_coloring

        self._caches: Dict[int, ObjectCache] = {
            size: ObjectCache(object_size=size) for size in self.object_sizes
        }

        self._allocated: Dict[int, SlabObject] = {}
        self._slab_counter = 0
        self._next_address = base_address

        self._coloring_offset = 0

    def _get_object_size(self, size: int) -> int:
        """
        获取合适的对象大小

        Args:
            size: 请求大小

        Returns:
            int: 对象大小
        """
        for obj_size in self.object_sizes:
            if obj_size >= size:
                return obj_size

        return self.object_sizes[-1] if self.object_sizes else size

    def _create_slab(self, object_size: int) -> Optional[Slab]:
        """
        创建新的Slab

        Args:
            object_size: 对象大小

        Returns:
            Slab: 新创建的Slab
        """
        if self._next_address + self.slab_size > self.base_address + self.total_size:
            return None

        self._slab_counter += 1

        coloring = 0
        if self.enable_coloring:
            coloring = (self._coloring_offset * 64) % (self.slab_size // 4)
            self._coloring_offset = (self._coloring_offset + 1) % 8

        slab = Slab(
            slab_id=self._slab_counter,
            address=self._next_address + coloring,
            size=self.slab_size,
            object_size=object_size
        )

        capacity = (self.slab_size - coloring) // object_size

        for i in range(capacity):
            obj = SlabObject(
                address=slab.address + i * object_size,
                size=object_size,
                is_free=True,
                slab_id=slab.slab_id
            )
            slab.objects.append(obj)

        slab.free_count = capacity
        self._next_address += self.slab_size

        return slab

    def _find_or_create_slab(self, object_size: int) -> Optional[Slab]:
        """
        查找或创建Slab

        Args:
            object_size: 对象大小

        Returns:
            Slab: 可用的Slab
        """
        cache = self._caches.get(object_size)
        if cache is None:
            cache = ObjectCache(object_size=object_size)
            self._caches[object_size] = cache

        for slab in cache.slabs:
            if slab.free_count > 0:
                return slab

        if len(cache.slabs) >= self.cache_limit:
            for slab in cache.slabs:
                if slab.in_use_count == 0:
                    return slab

        new_slab = self._create_slab(object_size)
        if new_slab:
            cache.slabs.append(new_slab)
            cache.total_objects += new_slab.capacity
            cache.free_objects += new_slab.capacity

        return new_slab

    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """
        分配内存

        Args:
            request: 分配请求

        Returns:
            AllocationResult: 分配结果
        """
        start_time = time.time()

        try:
            size = request.size

            if size > max(self.object_sizes) * 2:
                return AllocationResult(
                    success=False,
                    error_message=f"Size too large for slab allocator: {size}"
                )

            object_size = self._get_object_size(size)

            slab = self._find_or_create_slab(object_size)
            if slab is None:
                self.stats.allocation_failures += 1
                return AllocationResult(
                    success=False,
                    error_message="Out of memory"
                )

            obj = slab.get_free_object()
            if obj is None:
                return AllocationResult(
                    success=False,
                    error_message="No free objects in slab"
                )

            obj.is_free = False
            slab.free_count -= 1
            slab.in_use_count += 1

            cache = self._caches[object_size]
            cache.free_objects -= 1

            self._allocated[obj.address] = obj

            self.stats.allocation_count += 1
            self.stats.used_size += object_size
            self.stats.free_size -= object_size

            if self.stats.used_size > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.used_size

            allocation_time = time.time() - start_time

            return AllocationResult(
                success=True,
                address=obj.address,
                size=request.size,
                actual_size=object_size,
                allocator_type=self.allocator_type,
                numa_node=request.numa_node,
                fragmentation=(object_size - request.size) / object_size if object_size > 0 else 0,
                allocation_time=allocation_time
            )

        except Exception as e:
            return AllocationResult(
                success=False,
                error_message=str(e)
            )

    def deallocate(self, address: int) -> bool:
        """
        释放内存

        Args:
            address: 内存地址

        Returns:
            bool: 是否成功
        """
        try:
            obj = self._allocated.get(address)
            if obj is None:
                return False

            del self._allocated[address]

            obj.is_free = True

            cache = self._caches.get(obj.size)
            if cache:
                for slab in cache.slabs:
                    if slab.slab_id == obj.slab_id:
                        slab.free_count += 1
                        slab.in_use_count -= 1
                        cache.free_objects += 1
                        break

            self.stats.deallocation_count += 1
            self.stats.used_size -= obj.size
            self.stats.free_size += obj.size

            return True

        except Exception:
            return False

    def reallocate(self, address: int, new_size: int) -> AllocationResult:
        """
        重新分配内存

        Args:
            address: 原内存地址
            new_size: 新大小

        Returns:
            AllocationResult: 分配结果
        """
        try:
            old_obj = self._allocated.get(address)
            if old_obj is None:
                return AllocationResult(
                    success=False,
                    error_message=f"Block not found at address: {address}"
                )

            new_object_size = self._get_object_size(new_size)

            if new_object_size <= old_obj.size:
                return AllocationResult(
                    success=True,
                    address=address,
                    size=new_size,
                    actual_size=old_obj.size,
                    allocator_type=self.allocator_type
                )

            self.deallocate(address)

            request = AllocationRequest(size=new_size)
            result = self.allocate(request)

            return result

        except Exception as e:
            return AllocationResult(
                success=False,
                error_message=str(e)
            )

    def get_free_blocks(self) -> List[MemoryBlock]:
        """
        获取空闲块列表

        Returns:
            List[MemoryBlock]: 空闲块列表
        """
        blocks = []
        for object_size, cache in self._caches.items():
            for slab in cache.slabs:
                for obj in slab.objects:
                    if obj.is_free:
                        block = MemoryBlock(
                            address=obj.address,
                            size=obj.size,
                            state=MemoryRegionState.FREE,
                            allocator_type=self.allocator_type
                        )
                        blocks.append(block)
        return blocks

    def get_allocated_blocks(self) -> List[MemoryBlock]:
        """
        获取已分配块列表

        Returns:
            List[MemoryBlock]: 已分配块列表
        """
        blocks = []
        for address, obj in self._allocated.items():
            block = MemoryBlock(
                address=obj.address,
                size=obj.size,
                state=MemoryRegionState.ALLOCATED,
                allocator_type=self.allocator_type
            )
            blocks.append(block)
        return blocks

    def get_fragmentation_score(self) -> float:
        """
        获取碎片评分

        Returns:
            float: 碎片评分 (0-1)
        """
        total_internal_frag = 0
        total_allocated = 0

        for address, obj in self._allocated.items():
            total_internal_frag += obj.size
            total_allocated += obj.size

        if total_allocated == 0:
            return 0.0

        external_frag = 0
        for cache in self._caches.values():
            for slab in cache.slabs:
                if slab.free_count > 0 and slab.in_use_count > 0:
                    external_frag += slab.free_count / slab.capacity

        return min(external_frag / max(len(self._caches), 1), 1.0)

    def defragment(self) -> int:
        """
        执行碎片整理

        Returns:
            int: 整理的块数量
        """
        cleaned = 0

        for object_size, cache in self._caches.items():
            empty_slabs = []
            for slab in cache.slabs:
                if slab.in_use_count == 0 and len(cache.slabs) > 1:
                    empty_slabs.append(slab)

            for slab in empty_slabs:
                cache.slabs.remove(slab)
                cache.total_objects -= slab.capacity
                cache.free_objects -= slab.free_count
                cleaned += 1

        self.stats.defragmentation_count += 1
        return cleaned

    def get_cache_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        获取缓存统计

        Returns:
            Dict: 对象大小到统计的映射
        """
        stats = {}
        for object_size, cache in self._caches.items():
            stats[object_size] = {
                'slab_count': len(cache.slabs),
                'total_objects': cache.total_objects,
                'free_objects': cache.free_objects,
                'utilization': cache.get_utilization()
            }
        return stats

    def shrink_caches(self, target_free_ratio: float = 0.25) -> int:
        """
        收缩缓存

        Args:
            target_free_ratio: 目标空闲比例

        Returns:
            int: 释放的Slab数量
        """
        released = 0

        for cache in self._caches.values():
            if cache.get_utilization() < (1 - target_free_ratio):
                empty_slabs = [s for s in cache.slabs if s.in_use_count == 0]

                for slab in empty_slabs[:len(empty_slabs) // 2]:
                    cache.slabs.remove(slab)
                    cache.total_objects -= slab.capacity
                    cache.free_objects -= slab.free_count
                    released += 1

        return released
