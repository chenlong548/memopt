"""
mem_mapper 映射区域模块

定义了内存映射区域的核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum, Flag
import uuid
import threading
import time


class ProtectionFlags(Flag):
    """内存保护标志"""
    READ = 1      # 可读
    WRITE = 2     # 可写
    EXEC = 4      # 可执行


class MappingType(Enum):
    """映射类型"""
    SHARED = 1      # 共享映射
    PRIVATE = 2     # 私有映射（写时复制）
    ANONYMOUS = 3   # 匿名映射


class NUMAPolicy(Enum):
    """NUMA策略"""
    DEFAULT = 0     # 默认策略
    BIND = 1        # 绑定到指定节点
    INTERLEAVE = 2  # 交错分配
    PREFERRED = 3   # 优先使用指定节点


class MappingState(Enum):
    """映射状态"""
    ACTIVE = 1      # 活跃状态
    INACTIVE = 2    # 非活跃状态
    ZOMBIE = 3      # 僵尸状态（等待清理）


class GPUMappingStrategy(Enum):
    """GPU映射策略"""
    FULL = 1        # 完全映射到GPU
    PARTIAL = 2     # 部分映射
    ON_DEMAND = 3   # 按需映射


class SyncState(Enum):
    """同步状态"""
    SYNCED = 1      # 已同步
    DIRTY = 2       # 有脏数据
    SYNCING = 3     # 同步中


class AtomicCounter:
    """
    原子计数器
    
    线程安全的计数器实现。
    """
    
    def __init__(self, initial_value: int = 0):
        """
        初始化原子计数器
        
        Args:
            initial_value: 初始值
        """
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, delta: int = 1) -> int:
        """
        增加计数
        
        Args:
            delta: 增量
            
        Returns:
            增加后的值
        """
        with self._lock:
            self._value += delta
            return self._value
    
    def decrement(self, delta: int = 1) -> int:
        """
        减少计数
        
        Args:
            delta: 减量
            
        Returns:
            减少后的值
        """
        with self._lock:
            self._value -= delta
            return self._value
    
    def get(self) -> int:
        """
        获取当前值
        
        Returns:
            当前计数值
        """
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """
        设置值
        
        Args:
            value: 新值
            
        Returns:
            设置后的值
        """
        with self._lock:
            self._value = value
            return self._value


@dataclass
class AccessStatistics:
    """
    访问统计信息
    
    记录映射区域的访问统计信息。
    """
    
    # 基本统计
    total_reads: int = 0           # 总读取次数
    total_writes: int = 0          # 总写入次数
    page_faults: int = 0           # 页面错误次数
    cache_hits: int = 0            # 缓存命中次数
    
    # 访问模式统计
    sequential_accesses: int = 0   # 顺序访问次数
    random_accesses: int = 0       # 随机访问次数
    
    # 热点页面列表 [(offset, count), ...]
    hot_pages: List[Tuple[int, int]] = field(default_factory=list)
    
    # 性能指标
    avg_access_latency: float = 0.0  # 平均访问延迟（微秒）
    throughput_mbps: float = 0.0     # 吞吐量（MB/s）
    
    def update_read(self, latency: float = 0.0):
        """
        更新读取统计
        
        Args:
            latency: 访问延迟（微秒）
        """
        self.total_reads += 1
        if latency > 0:
            # 更新平均延迟（移动平均）
            self.avg_access_latency = (
                (self.avg_access_latency * (self.total_reads - 1) + latency) 
                / self.total_reads
            )
    
    def update_write(self, latency: float = 0.0):
        """
        更新写入统计
        
        Args:
            latency: 访问延迟（微秒）
        """
        self.total_writes += 1
        if latency > 0:
            total_ops = self.total_reads + self.total_writes
            self.avg_access_latency = (
                (self.avg_access_latency * (total_ops - 1) + latency) 
                / total_ops
            )
    
    def update_page_fault(self):
        """更新页面错误统计"""
        self.page_faults += 1
    
    def update_cache_hit(self):
        """更新缓存命中统计"""
        self.cache_hits += 1
    
    def update_sequential_access(self):
        """更新顺序访问统计"""
        self.sequential_accesses += 1
    
    def update_random_access(self):
        """更新随机访问统计"""
        self.random_accesses += 1
    
    def add_hot_page(self, offset: int, count: int):
        """
        添加热点页面
        
        Args:
            offset: 页面偏移
            count: 访问次数
        """
        self.hot_pages.append((offset, count))
        # 保持热点页面列表有序（按访问次数降序）
        self.hot_pages.sort(key=lambda x: x[1], reverse=True)
        # 只保留前100个热点页面
        if len(self.hot_pages) > 100:
            self.hot_pages = self.hot_pages[:100]
    
    def get_summary(self) -> Dict:
        """
        获取统计摘要
        
        Returns:
            统计摘要字典
        """
        total_ops = self.total_reads + self.total_writes
        return {
            'total_operations': total_ops,
            'reads': self.total_reads,
            'writes': self.total_writes,
            'page_faults': self.page_faults,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / total_ops if total_ops > 0 else 0.0,
            'sequential_ratio': (
                self.sequential_accesses / total_ops 
                if total_ops > 0 else 0.0
            ),
            'random_ratio': (
                self.random_accesses / total_ops 
                if total_ops > 0 else 0.0
            ),
            'avg_latency_us': self.avg_access_latency,
            'throughput_mbps': self.throughput_mbps,
            'hot_pages_count': len(self.hot_pages)
        }


@dataclass
class GPUMappingInfo:
    """
    GPU映射信息
    
    记录CPU到GPU的映射信息。
    """
    
    device_id: int                           # GPU设备ID
    device_name: str                         # GPU设备名称
    gpu_address: int                         # GPU内存地址
    cpu_address: int                         # CPU内存地址
    mapping_strategy: GPUMappingStrategy     # 映射策略
    sync_state: SyncState                    # 同步状态
    dirty_regions: List[Tuple[int, int]] = field(default_factory=list)  # 脏区域列表 [(offset, size), ...]
    
    def add_dirty_region(self, offset: int, size: int):
        """
        添加脏区域
        
        Args:
            offset: 区域偏移
            size: 区域大小
        """
        self.dirty_regions.append((offset, size))
        self.sync_state = SyncState.DIRTY
    
    def clear_dirty_regions(self):
        """清除脏区域"""
        self.dirty_regions.clear()
        self.sync_state = SyncState.SYNCED
    
    def get_dirty_size(self) -> int:
        """
        获取脏数据总大小
        
        Returns:
            脏数据总大小
        """
        return sum(size for _, size in self.dirty_regions)


@dataclass
class MappedRegion:
    """
    内存映射区域
    
    核心数据结构，表示一个内存映射区域。
    """
    
    # 基本信息
    region_id: uuid.UUID                      # 区域唯一标识符
    file_path: str                            # 文件路径
    file_descriptor: int                      # 文件描述符
    
    # 地址和大小
    base_address: int                         # 基地址
    size: int                                 # 映射大小
    aligned_size: int                         # 对齐后的大小
    
    # 保护标志和映射类型
    protection: ProtectionFlags               # 保护标志
    mapping_type: MappingType                 # 映射类型
    
    # NUMA信息
    numa_node: int                            # NUMA节点ID
    numa_policy: NUMAPolicy                   # NUMA策略
    
    # 大页信息
    uses_huge_pages: bool                     # 是否使用大页
    huge_page_size: int                       # 大页大小（字节）
    
    # 引用计数和时间戳
    ref_count: Optional[AtomicCounter] = None  # 引用计数
    creation_time: Optional[float] = None      # 创建时间
    last_access_time: Optional[float] = None   # 最后访问时间
    
    # 统计和GPU映射
    access_stats: Optional[AccessStatistics] = None  # 访问统计
    gpu_mapping: Optional[GPUMappingInfo] = None     # GPU映射信息
    
    # 状态标志
    is_dirty: bool = False                            # 是否有脏数据
    is_locked: bool = False                           # 是否被锁定
    state: MappingState = MappingState.ACTIVE         # 映射状态
    
    def __post_init__(self):
        """初始化后处理"""
        if self.ref_count is None:
            self.ref_count = AtomicCounter(1)
        if self.access_stats is None:
            self.access_stats = AccessStatistics()
        if self.creation_time is None:
            self.creation_time = time.time()
        if self.last_access_time is None:
            self.last_access_time = time.time()
    
    def acquire(self):
        """获取引用"""
        self.ref_count.increment()
        self.last_access_time = time.time()
    
    def release(self) -> int:
        """
        释放引用
        
        Returns:
            释放后的引用计数
        """
        return self.ref_count.decrement()
    
    def get_ref_count(self) -> int:
        """
        获取引用计数
        
        Returns:
            当前引用计数
        """
        return self.ref_count.get()
    
    def update_access_time(self):
        """更新最后访问时间"""
        self.last_access_time = time.time()
    
    def get_age(self) -> float:
        """
        获取区域年龄
        
        Returns:
            从创建到现在的时间（秒）
        """
        creation = self.creation_time if self.creation_time is not None else time.time()
        return time.time() - creation
    
    def get_idle_time(self) -> float:
        """
        获取空闲时间
        
        Returns:
            从最后访问到现在的时间（秒）
        """
        last_access = self.last_access_time if self.last_access_time is not None else time.time()
        return time.time() - last_access
    
    def contains(self, address: int) -> bool:
        """
        检查地址是否在此区域内
        
        Args:
            address: 要检查的地址
            
        Returns:
            是否包含该地址
        """
        return self.base_address <= address < self.base_address + self.size
    
    def get_offset(self, address: int) -> Optional[int]:
        """
        获取地址在区域内的偏移
        
        Args:
            address: 地址
            
        Returns:
            偏移量，如果地址不在区域内则返回None
        """
        if self.contains(address):
            return address - self.base_address
        return None
    
    def get_address(self, offset: int) -> int:
        """
        根据偏移获取地址
        
        Args:
            offset: 偏移量
            
        Returns:
            地址
        """
        return self.base_address + offset
    
    def is_readable(self) -> bool:
        """
        检查是否可读
        
        Returns:
            是否可读
        """
        return ProtectionFlags.READ in self.protection or \
               self.protection.value & ProtectionFlags.READ.value
    
    def is_writable(self) -> bool:
        """
        检查是否可写
        
        Returns:
            是否可写
        """
        return ProtectionFlags.WRITE in self.protection or \
               self.protection.value & ProtectionFlags.WRITE.value
    
    def is_executable(self) -> bool:
        """
        检查是否可执行
        
        Returns:
            是否可执行
        """
        return ProtectionFlags.EXEC in self.protection or \
               self.protection.value & ProtectionFlags.EXEC.value
    
    def is_shared(self) -> bool:
        """
        检查是否共享映射
        
        Returns:
            是否共享映射
        """
        return self.mapping_type == MappingType.SHARED
    
    def is_private(self) -> bool:
        """
        检查是否私有映射
        
        Returns:
            是否私有映射
        """
        return self.mapping_type == MappingType.PRIVATE
    
    def is_anonymous(self) -> bool:
        """
        检查是否匿名映射
        
        Returns:
            是否匿名映射
        """
        return self.mapping_type == MappingType.ANONYMOUS
    
    def get_page_count(self, page_size: int = 4096) -> int:
        """
        获取页面数量
        
        Args:
            page_size: 页面大小
            
        Returns:
            页面数量
        """
        return (self.size + page_size - 1) // page_size
    
    def get_huge_page_count(self) -> int:
        """
        获取大页数量
        
        Returns:
            大页数量
        """
        if self.uses_huge_pages and self.huge_page_size > 0:
            return (self.size + self.huge_page_size - 1) // self.huge_page_size
        return 0
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        # 导入安全工具
        from ..utils.security import get_security_config, ErrorSanitizer
        
        security_config = get_security_config()
        sanitizer = ErrorSanitizer(security_config)
        
        return {
            'region_id': str(self.region_id),
            'file_path': sanitizer.sanitize_path(self.file_path),
            'file_descriptor': self.file_descriptor,
            'base_address': sanitizer.sanitize_address(self.base_address),
            'size': self.size,
            'aligned_size': self.aligned_size,
            'protection': self.protection.name,
            'mapping_type': self.mapping_type.name,
            'numa_node': self.numa_node,
            'numa_policy': self.numa_policy.name,
            'uses_huge_pages': self.uses_huge_pages,
            'huge_page_size': self.huge_page_size,
            'ref_count': self.get_ref_count(),
            'creation_time': self.creation_time,
            'last_access_time': self.last_access_time,
            'is_dirty': self.is_dirty,
            'is_locked': self.is_locked,
            'state': self.state.name,
            'access_stats': self.access_stats.get_summary(),
            'has_gpu_mapping': self.gpu_mapping is not None
        }
    
    def __repr__(self) -> str:
        return (
            f"MappedRegion(id={str(self.region_id)[:8]}, "
            f"path='{self.file_path}', "
            f"addr={hex(self.base_address)}, "
            f"size={self.size}, "
            f"refs={self.get_ref_count()})"
        )
