"""
mem_mapper 核心映射器模块

提供MemoryMapper主类，整合所有功能模块。
"""

import os
import uuid
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import time

from .region import (
    MappedRegion, ProtectionFlags, MappingType, 
    NUMAPolicy, MappingState, AccessStatistics
)
from .registry import MappingRegistry
from .exceptions import (
    MemMapperError, MMapError, FileError, 
    RegionError, ConfigError
)
from ..platform.base import PlatformBase, PlatformFactory
from ..numa.topology import get_numa_topology, NUMATopology
from ..numa.policy import NUMAPolicyManager
from ..hugepage.detector import get_hugepage_detector, HugePageDetector
from ..hugepage.pool import get_hugepage_pool, HugePagePool
from ..prefetch.strategy import SequentialPrefetchStrategy, PrefetchStrategyType
from ..prefetch.executor import PrefetchExecutor
from ..lifecycle.manager import LifecycleManager
from ..utils.alignment import align_to_page, align_to_huge_page
from ..utils.stats import PerformanceTracker
from ..utils.security import (
    SecurityConfig, PathValidator, PermissionChecker, 
    ResourceLimiter, ErrorSanitizer, FileDescriptorTracker,
    get_security_config, set_security_config, get_fd_tracker
)

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class MapperConfig:
    """
    映射器配置
    
    MemoryMapper的配置选项。
    """
    
    # 基本配置
    use_numa: bool = True                    # 是否使用NUMA优化
    use_huge_pages: bool = True              # 是否使用大页
    use_prefetch: bool = True                # 是否使用预取
    
    # NUMA配置
    numa_policy: NUMAPolicy = NUMAPolicy.DEFAULT  # 默认NUMA策略
    numa_node: int = -1                      # 默认NUMA节点
    
    # 大页配置
    huge_page_size: int = 0                  # 大页大小（0表示自动）
    min_size_for_huge_pages: int = 2 * 1024 * 1024  # 使用大页的最小大小
    
    # 预取配置
    prefetch_strategy: PrefetchStrategyType = PrefetchStrategyType.SEQUENTIAL
    prefetch_window: int = 16                # 预取窗口大小（页面数）
    
    # 生命周期配置
    cleanup_delay: float = 60.0              # 清理延迟（秒）
    idle_threshold: float = 300.0            # 空闲阈值（秒）
    max_mappings: int = 10000                # 最大映射数量
    
    # 性能配置
    enable_stats: bool = True                # 是否启用统计
    async_prefetch: bool = False             # 是否异步预取
    
    # 安全配置
    security_config: Optional[SecurityConfig] = None  # 安全配置（None使用默认）


class MemoryMapper:
    """
    内存映射器主类
    
    提供高性能的内存映射功能，整合NUMA、大页、预取等优化。
    """
    
    def __init__(self, config: Optional[MapperConfig] = None):
        """
        初始化内存映射器
        
        Args:
            config: 映射器配置，None则使用默认配置
        """
        self.config = config or MapperConfig()
        
        # 初始化安全组件
        self.security_config = self.config.security_config or get_security_config()
        self.path_validator = PathValidator(self.security_config)
        self.permission_checker = PermissionChecker(self.security_config)
        self.resource_limiter = ResourceLimiter(self.security_config)
        self.error_sanitizer = ErrorSanitizer(self.security_config)
        self.fd_tracker = get_fd_tracker()
        
        # 设置全局安全配置
        set_security_config(self.security_config)
        
        # 初始化平台
        self.platform: PlatformBase = PlatformFactory.create()
        
        # 初始化注册表
        self.registry = MappingRegistry(max_mappings=self.config.max_mappings)
        
        # 初始化NUMA管理器
        self.numa_topology: Optional[NUMATopology] = None
        self.numa_policy_manager: Optional[NUMAPolicyManager] = None
        
        if self.config.use_numa:
            try:
                self.numa_topology = get_numa_topology()
                self.numa_policy_manager = NUMAPolicyManager(self.numa_topology)
            except Exception:
                pass
        
        # 初始化大页管理器
        self.hugepage_detector: Optional[HugePageDetector] = None
        self.hugepage_pool: Optional[HugePagePool] = None
        
        if self.config.use_huge_pages:
            try:
                self.hugepage_detector = get_hugepage_detector()
                self.hugepage_pool = get_hugepage_pool()
            except Exception:
                pass
        
        # 初始化预取执行器
        self.prefetch_executor: Optional[PrefetchExecutor] = None
        
        if self.config.use_prefetch:
            self.prefetch_executor = PrefetchExecutor(
                platform=self.platform,
                async_mode=self.config.async_prefetch
            )
        
        # 初始化生命周期管理器
        self.lifecycle = LifecycleManager(
            registry=self.registry,
            cleanup_delay=self.config.cleanup_delay,
            idle_threshold=self.config.idle_threshold
        )
        
        # 性能跟踪器
        self.tracker = PerformanceTracker() if self.config.enable_stats else None
        
        # 启动生命周期管理线程
        self.lifecycle.start_cleanup_thread()
    
    def map_file(self,
                 path: str,
                 mode: str = 'readonly',
                 offset: int = 0,
                 size: int = 0,
                 numa_node: int = -1,
                 use_huge_pages: Optional[bool] = None,
                 prefetch: bool = True) -> MappedRegion:
        """
        映射文件到内存
        
        Args:
            path: 文件路径
            mode: 访问模式（'readonly', 'readwrite', 'writecopy'）
            offset: 文件偏移
            size: 映射大小（0表示整个文件）
            numa_node: NUMA节点（-1表示自动选择）
            use_huge_pages: 是否使用大页（None表示自动决定）
            prefetch: 是否预取
            
        Returns:
            映射区域
            
        Raises:
            FileError: 文件操作失败时抛出
            MMapError: 映射失败时抛出
            ConfigError: 配置错误时抛出
        """
        fd = None
        
        try:
            # 1. 路径安全验证
            is_valid, validated_path, error_msg = self.path_validator.validate_path(path)
            if not is_valid:
                sanitized_msg = self.error_sanitizer.sanitize_error_message(error_msg)
                raise FileError(sanitized_msg, file_path=self.error_sanitizer.sanitize_path(path))
            
            # 使用验证后的路径
            safe_path = validated_path
            
            # 2. 检查文件是否存在
            if not os.path.exists(safe_path):
                raise FileError(
                    "File not found",
                    file_path=self.error_sanitizer.sanitize_path(safe_path)
                )
            
            # 3. 权限检查
            has_permission, perm_error = self.permission_checker.check_file_permission(safe_path, mode)
            if not has_permission:
                raise FileError(
                    perm_error or "Permission denied",
                    file_path=self.error_sanitizer.sanitize_path(safe_path)
                )
            
            # 4. 获取文件大小
            try:
                file_size = os.path.getsize(safe_path)
            except OSError as e:
                raise FileError(
                    f"Failed to get file size: {e}",
                    file_path=self.error_sanitizer.sanitize_path(safe_path),
                    errno=e.errno
                )
            
            # 5. 资源限制检查
            is_allowed, resource_error = self.resource_limiter.check_file_size(
                file_size, offset, size
            )
            if not is_allowed:
                raise FileError(
                    resource_error or "Resource limit exceeded",
                    file_path=self.error_sanitizer.sanitize_path(safe_path)
                )
            
            # 6. 检查映射限制
            current_count = self.registry.get_count()
            current_total = self.registry.get_total_size()
            actual_size = file_size - offset if size == 0 else size
            
            is_allowed, limit_error = self.resource_limiter.check_mapping_limit(
                actual_size, current_count, current_total
            )
            if not is_allowed:
                raise FileError(
                    limit_error or "Mapping limit exceeded",
                    file_path=self.error_sanitizer.sanitize_path(safe_path)
                )
            
            # 7. 确定映射大小
            if size == 0:
                size = file_size - offset
            
            if offset + size > file_size:
                raise FileError(
                    "Mapping range exceeds file size",
                    file_path=self.error_sanitizer.sanitize_path(safe_path)
                )
            
            # 8. 打开文件描述符
            try:
                if mode == 'readonly':
                    fd = os.open(safe_path, os.O_RDONLY)
                    prot = ProtectionFlags.READ
                    mapping_type = MappingType.PRIVATE
                elif mode == 'readwrite':
                    fd = os.open(safe_path, os.O_RDWR)
                    prot = ProtectionFlags.READ | ProtectionFlags.WRITE
                    mapping_type = MappingType.SHARED
                elif mode == 'writecopy':
                    fd = os.open(safe_path, os.O_RDONLY)
                    prot = ProtectionFlags.READ | ProtectionFlags.WRITE
                    mapping_type = MappingType.PRIVATE
                else:
                    raise ConfigError(f"Invalid mode: {mode}", config_key='mode')
                
                # 注册文件描述符
                self.fd_tracker.register(fd)
                logger.debug(f"Opened file descriptor {fd} for {self.error_sanitizer.sanitize_path(safe_path)}")
                
            except OSError as e:
                raise FileError(
                    f"Failed to open file: {e}",
                    file_path=self.error_sanitizer.sanitize_path(safe_path),
                    errno=e.errno
                )
            
            # 9. 创建映射
            region = self._create_mapping(
                fd=fd,
                offset=offset,
                size=size,
                prot=prot,
                mapping_type=mapping_type,
                numa_node=numa_node,
                use_huge_pages=use_huge_pages,
                file_path=safe_path
            )
            
            # 10. 预取
            if prefetch and self.prefetch_executor and self.config.use_prefetch:
                self._prefetch_region(region)
            
            return region
            
        except Exception as e:
            # 记录错误
            logger.error(f"map_file failed: {self.error_sanitizer.sanitize_error_message(str(e))}")
            raise
            
        finally:
            # 11. 关闭文件描述符（映射保持有效）
            if fd is not None:
                try:
                    if self.fd_tracker.close(fd):
                        logger.debug(f"Closed file descriptor {fd}")
                    else:
                        logger.warning(f"Failed to close file descriptor {fd}")
                except Exception as e:
                    logger.error(f"Error closing file descriptor {fd}: {e}")
    
    def _create_mapping(self,
                       fd: int,
                       offset: int,
                       size: int,
                       prot: ProtectionFlags,
                       mapping_type: MappingType,
                       numa_node: int,
                       use_huge_pages: Optional[bool],
                       file_path: str) -> MappedRegion:
        """
        创建内存映射
        
        Args:
            fd: 文件描述符
            offset: 文件偏移
            size: 映射大小
            prot: 保护标志
            mapping_type: 映射类型
            numa_node: NUMA节点
            use_huge_pages: 是否使用大页
            file_path: 文件路径
            
        Returns:
            映射区域
        """
        # 决定是否使用大页
        actual_use_huge_pages = False
        huge_page_size = 0
        
        if use_huge_pages is None:
            use_huge_pages = self.config.use_huge_pages
        
        if use_huge_pages and self.hugepage_detector and size >= self.config.min_size_for_huge_pages:
            # 选择大页大小
            if self.config.huge_page_size > 0:
                huge_page_size = self.config.huge_page_size
            else:
                huge_page_size = self.hugepage_detector.recommend_page_size(size)
            
            # 检查是否可用
            if self.hugepage_detector.can_allocate(size, huge_page_size):
                actual_use_huge_pages = True
        
        # 对齐大小
        if actual_use_huge_pages:
            aligned_size = align_to_huge_page(size, huge_page_size)
            aligned_offset = align_to_huge_page(offset, huge_page_size)
        else:
            page_size = self.platform.get_page_size()
            aligned_size = align_to_page(size, page_size)
            aligned_offset = align_to_page(offset, page_size)
        
        # 构建映射标志
        flags = self._build_mapping_flags(mapping_type, actual_use_huge_pages, huge_page_size)
        
        # 转换保护标志
        prot_flags = self._prot_to_int(prot)
        
        # 执行映射
        try:
            addr = self.platform.mmap(
                addr=None,
                length=aligned_size,
                prot=prot_flags,
                flags=flags,
                fd=fd,
                offset=aligned_offset
            )
        except Exception as e:
            raise MMapError(0, f"mmap failed: {e}")
        
        # 选择NUMA节点
        if numa_node is None or numa_node < 0:
            if self.numa_topology:
                numa_node = self.numa_topology.find_best_node(size)
        
        if numa_node is None or numa_node < 0:
            numa_node = 0
        
        # 绑定到NUMA节点
        if self.config.use_numa and self.numa_policy_manager and numa_node >= 0:
            try:
                self.numa_policy_manager.bind_memory(addr, aligned_size, numa_node)
            except Exception:
                # 绑定失败不影响映射
                pass
        
        # 创建映射区域对象
        region = MappedRegion(
            region_id=uuid.uuid4(),
            file_path=file_path,
            file_descriptor=fd,
            base_address=addr,
            size=size,
            aligned_size=aligned_size,
            protection=prot,
            mapping_type=mapping_type,
            numa_node=numa_node,
            numa_policy=self.config.numa_policy,
            uses_huge_pages=actual_use_huge_pages,
            huge_page_size=huge_page_size if actual_use_huge_pages else 0,
            creation_time=time.time(),
            last_access_time=time.time(),
            access_stats=AccessStatistics(),
            gpu_mapping=None,
            is_dirty=False,
            is_locked=False,
            state=MappingState.ACTIVE
        )
        
        # 添加到注册表
        self.registry.add(region)
        
        # 记录统计
        if self.tracker:
            self.tracker.record('map_file', 0.0, success=True)
        
        return region
    
    def _build_mapping_flags(self, 
                            mapping_type: MappingType,
                            use_huge_pages: bool,
                            huge_page_size: int) -> int:
        """
        构建映射标志
        
        Args:
            mapping_type: 映射类型
            use_huge_pages: 是否使用大页
            huge_page_size: 大页大小
            
        Returns:
            映射标志
        """
        flags = 0
        
        # 映射类型
        if mapping_type == MappingType.SHARED:
            flags |= 0x01  # MAP_SHARED
        elif mapping_type == MappingType.PRIVATE:
            flags |= 0x02  # MAP_PRIVATE
        
        # 大页标志
        if use_huge_pages:
            flags |= 0x40000  # MAP_HUGETLB
            
            # 设置大页大小
            if huge_page_size > 0:
                # 计算大页大小的shift
                shift = 0
                temp = huge_page_size
                while temp > 4096:  # 大于标准页面
                    temp >>= 1
                    shift += 1
                
                flags |= (shift << 26)  # MAP_HUGE_SHIFT
        
        return flags
    
    def _prot_to_int(self, prot: ProtectionFlags) -> int:
        """
        将保护标志转换为整数
        
        Args:
            prot: 保护标志
            
        Returns:
            整数标志
        """
        value = 0
        
        if prot.value & ProtectionFlags.READ.value:
            value |= 0x1  # PROT_READ
        
        if prot.value & ProtectionFlags.WRITE.value:
            value |= 0x2  # PROT_WRITE
        
        if prot.value & ProtectionFlags.EXEC.value:
            value |= 0x4  # PROT_EXEC
        
        return value
    
    def _prefetch_region(self, region: MappedRegion):
        """
        预取映射区域
        
        Args:
            region: 映射区域
        """
        if not self.prefetch_executor:
            return
        
        # 创建预取策略
        strategy = SequentialPrefetchStrategy(
            page_size=self.platform.get_page_size(),
            window_size=self.config.prefetch_window
        )
        
        # 执行预取
        try:
            self.prefetch_executor.prefetch(region, strategy)
        except Exception:
            # 预取失败不影响映射
            pass
    
    def unmap(self, region: MappedRegion):
        """
        解除映射
        
        Args:
            region: 映射区域
        """
        try:
            # 同步脏数据
            if region.is_dirty:
                self.platform.msync(region.base_address, region.aligned_size, 0x4)  # MS_SYNC
            
            # 解除映射
            self.platform.munmap(region.base_address, region.aligned_size)
            
            # 从注册表移除
            self.registry.remove(region.region_id)
            
            # 记录统计
            if self.tracker:
                self.tracker.record('unmap', 0.0, success=True)
                
        except Exception as e:
            if self.tracker:
                self.tracker.record('unmap', 0.0, success=False, error=str(e))
            raise
    
    def advise(self, region: MappedRegion, advice: str):
        """
        提供访问模式建议
        
        Args:
            region: 映射区域
            advice: 建议类型（'sequential', 'random', 'willneed', 'dontneed'）
        """
        advice_map = {
            'sequential': 2,   # MADV_SEQUENTIAL
            'random': 1,       # MADV_RANDOM
            'willneed': 3,     # MADV_WILLNEED
            'dontneed': 4,     # MADV_DONTNEED
        }
        
        advice_value = advice_map.get(advice.lower())
        if advice_value is None:
            raise ConfigError(f"Invalid advice: {advice}", config_key='advice')
        
        try:
            self.platform.madvise(region.base_address, region.aligned_size, advice_value)
        except Exception:
            # madvise失败不影响功能
            pass
    
    def sync(self, region: MappedRegion, async_mode: bool = False):
        """
        同步映射到文件
        
        Args:
            region: 映射区域
            async_mode: 是否异步同步
        """
        flags = 0x1 if async_mode else 0x4  # MS_ASYNC or MS_SYNC
        
        try:
            self.platform.msync(region.base_address, region.aligned_size, flags)
            region.is_dirty = False
        except Exception as e:
            raise MemMapperError(f"Failed to sync region: {e}")
    
    def lock(self, region: MappedRegion):
        """
        锁定映射（防止被交换）
        
        Args:
            region: 映射区域
        """
        try:
            self.platform.mlock(region.base_address, region.aligned_size)
            region.is_locked = True
        except Exception as e:
            raise MemMapperError(f"Failed to lock region: {e}")
    
    def unlock(self, region: MappedRegion):
        """
        解锁映射
        
        Args:
            region: 映射区域
        """
        try:
            self.platform.munlock(region.base_address, region.aligned_size)
            region.is_locked = False
        except Exception as e:
            raise MemMapperError(f"Failed to unlock region: {e}")
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'registry': self.registry.get_stats(),
            'lifecycle': self.lifecycle.get_stats(),
        }
        
        if self.tracker:
            stats['performance'] = self.tracker.get_summary()
        
        if self.hugepage_pool:
            stats['hugepage'] = self.hugepage_pool.get_summary()
        
        return stats
    
    def get_region(self, region_id: uuid.UUID) -> Optional[MappedRegion]:
        """
        获取映射区域
        
        Args:
            region_id: 区域ID
            
        Returns:
            映射区域
        """
        return self.registry.get(region_id)
    
    def get_all_regions(self) -> List[MappedRegion]:
        """
        获取所有映射区域
        
        Returns:
            映射区域列表
        """
        return self.registry.get_all()
    
    def cleanup(self):
        """清理所有映射"""
        self.lifecycle.cleanup_all()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
        self.lifecycle.stop_cleanup_thread()
        return False
    
    def __del__(self):
        """析构函数"""
        try:
            self.lifecycle.stop_cleanup_thread()
        except Exception:
            pass
