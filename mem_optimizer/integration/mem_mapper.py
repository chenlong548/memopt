"""
mem_optimizer mem_mapper集成模块

实现与mem_mapper模块的集成。
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..core.base import AllocatorType, AllocationRequest, AllocationResult
from ..core.exceptions import IntegrationError


@dataclass
class MappedMemoryInfo:
    """映射内存信息"""
    region_id: str
    address: int
    size: int
    file_path: Optional[str]
    numa_node: int
    uses_huge_pages: bool


class MemMapperIntegration:
    """
    mem_mapper集成器

    提供与mem_mapper模块的无缝集成。
    """

    def __init__(self, memory_pool: Optional[Any] = None):
        """
        初始化集成器

        Args:
            memory_pool: 内存池引用
        """
        self._memory_pool = memory_pool
        self._mapper: Optional[Any] = None
        self._mapped_regions: Dict[str, MappedMemoryInfo] = {}
        self._integration_enabled = False

        self._init_mapper()

    def _init_mapper(self):
        """初始化mem_mapper"""
        try:
            import sys
            import os

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from mem_mapper.core.mapper import MemoryMapper, MapperConfig

            mapper_config = MapperConfig(
                use_numa=True,
                use_huge_pages=True,
                use_prefetch=True
            )

            self._mapper = MemoryMapper(mapper_config)
            self._integration_enabled = True

        except ImportError:
            self._mapper = None
            self._integration_enabled = False
        except Exception:
            self._mapper = None
            self._integration_enabled = False

    def is_available(self) -> bool:
        """
        检查mem_mapper是否可用

        Returns:
            bool: 是否可用
        """
        return self._integration_enabled and self._mapper is not None

    def map_file_to_pool(self,
                        file_path: str,
                        mode: str = 'readonly',
                        numa_node: int = -1,
                        use_huge_pages: Optional[bool] = None) -> Optional[MappedMemoryInfo]:
        """
        将文件映射到内存池

        Args:
            file_path: 文件路径
            mode: 访问模式
            numa_node: NUMA节点
            use_huge_pages: 是否使用大页

        Returns:
            MappedMemoryInfo: 映射信息
        """
        if not self.is_available():
            return None

        try:
            region = self._mapper.map_file(
                path=file_path,
                mode=mode,
                numa_node=numa_node,
                use_huge_pages=use_huge_pages
            )

            info = MappedMemoryInfo(
                region_id=str(region.region_id),
                address=region.base_address,
                size=region.size,
                file_path=file_path,
                numa_node=region.numa_node,
                uses_huge_pages=region.uses_huge_pages
            )

            self._mapped_regions[info.region_id] = info

            if self._memory_pool:
                self._register_with_pool(info)

            return info

        except Exception as e:
            raise IntegrationError(f"Failed to map file: {e}", module="mem_mapper")

    def _register_with_pool(self, info: MappedMemoryInfo):
        """
        在内存池中注册映射区域

        Args:
            info: 映射信息
        """
        pass

    def unmap_region(self, region_id: str) -> bool:
        """
        解除映射

        Args:
            region_id: 区域ID

        Returns:
            bool: 是否成功
        """
        if not self.is_available():
            return False

        info = self._mapped_regions.get(region_id)
        if info is None:
            return False

        try:
            import uuid
            from mem_mapper.core.region import MappedRegion

            region = MappedRegion(
                region_id=uuid.UUID(region_id),
                file_path=info.file_path or "",
                base_address=info.address,
                size=info.size
            )

            self._mapper.unmap(region)

            del self._mapped_regions[region_id]

            return True

        except Exception:
            return False

    def get_mapped_regions(self) -> List[MappedMemoryInfo]:
        """
        获取所有映射区域

        Returns:
            List[MappedMemoryInfo]: 映射区域列表
        """
        return list(self._mapped_regions.values())

    def get_region_info(self, region_id: str) -> Optional[MappedMemoryInfo]:
        """
        获取区域信息

        Args:
            region_id: 区域ID

        Returns:
            MappedMemoryInfo: 区域信息
        """
        return self._mapped_regions.get(region_id)

    def advise_region(self, region_id: str, advice: str) -> bool:
        """
        对映射区域提供建议

        Args:
            region_id: 区域ID
            advice: 建议类型

        Returns:
            bool: 是否成功
        """
        if not self.is_available():
            return False

        info = self._mapped_regions.get(region_id)
        if info is None:
            return False

        try:
            import uuid
            from mem_mapper.core.region import MappedRegion

            region = MappedRegion(
                region_id=uuid.UUID(region_id),
                file_path=info.file_path or "",
                base_address=info.address,
                size=info.size
            )

            self._mapper.advise(region, advice)
            return True

        except Exception:
            return False

    def sync_region(self, region_id: str, async_mode: bool = False) -> bool:
        """
        同步映射区域

        Args:
            region_id: 区域ID
            async_mode: 是否异步

        Returns:
            bool: 是否成功
        """
        if not self.is_available():
            return False

        info = self._mapped_regions.get(region_id)
        if info is None:
            return False

        try:
            import uuid
            from mem_mapper.core.region import MappedRegion

            region = MappedRegion(
                region_id=uuid.UUID(region_id),
                file_path=info.file_path or "",
                base_address=info.address,
                size=info.size
            )

            self._mapper.sync(region, async_mode)
            return True

        except Exception:
            return False

    def get_mapper_stats(self) -> Dict[str, Any]:
        """
        获取mapper统计信息

        Returns:
            Dict: 统计信息
        """
        if not self.is_available():
            return {'available': False}

        try:
            stats = self._mapper.get_stats()
            return {
                'available': True,
                'mapper_stats': stats,
                'mapped_regions_count': len(self._mapped_regions)
            }
        except Exception:
            return {'available': False}

    def cleanup(self):
        """清理所有映射"""
        for region_id in list(self._mapped_regions.keys()):
            self.unmap_region(region_id)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
        return False
