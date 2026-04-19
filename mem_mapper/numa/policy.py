"""
mem_mapper NUMA策略管理模块

提供NUMA策略配置和管理功能。
"""

import sys
from typing import List, Optional, Dict
from enum import Enum
from dataclasses import dataclass

from .topology import NUMATopology, get_numa_topology
from ..core.exceptions import NUMAError, NUMABindingError
from ..core.region import NUMAPolicy


class NUMAPolicyMode(Enum):
    """NUMA策略模式"""
    DEFAULT = 0      # 默认策略
    BIND = 1         # 绑定到指定节点
    INTERLEAVE = 2   # 交错分配
    PREFERRED = 3    # 优先使用指定节点
    LOCAL = 4        # 本地节点优先


@dataclass
class NUMAPolicyConfig:
    """
    NUMA策略配置
    
    定义NUMA内存分配策略的配置。
    """
    
    mode: NUMAPolicyMode = NUMAPolicyMode.DEFAULT  # 策略模式
    nodes: Optional[List[int]] = None              # 目标节点列表
    preferred_node: Optional[int] = None           # 优先节点
    strict: bool = False                           # 是否严格模式
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保nodes是列表
        if self.nodes is None:
            self.nodes = []
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            'mode': self.mode.name,
            'nodes': self.nodes,
            'preferred_node': self.preferred_node,
            'strict': self.strict,
        }


class NUMAPolicyManager:
    """
    NUMA策略管理器
    
    管理NUMA内存分配策略。
    """
    
    def __init__(self, topology: Optional[NUMATopology] = None):
        """
        初始化NUMA策略管理器
        
        Args:
            topology: NUMA拓扑，None则自动检测
        """
        self.topology = topology or get_numa_topology()
        self.current_policy = NUMAPolicyConfig()
    
    def set_policy(self, config: NUMAPolicyConfig) -> bool:
        """
        设置NUMA策略
        
        Args:
            config: NUMA策略配置
            
        Returns:
            是否设置成功
            
        Raises:
            NUMAError: 设置失败时抛出
        """
        # 验证配置
        self._validate_config(config)
        
        # 根据平台设置策略
        if sys.platform.startswith('linux'):
            return self._set_policy_linux(config)
        elif sys.platform == 'win32':
            return self._set_policy_windows(config)
        else:
            # 不支持的平台，只更新配置
            self.current_policy = config
            return True
    
    def _validate_config(self, config: NUMAPolicyConfig):
        """
        验证策略配置
        
        Args:
            config: 策略配置
            
        Raises:
            NUMAError: 配置无效时抛出
        """
        # 检查节点是否存在
        if config.nodes:
            for node_id in config.nodes:
                if node_id not in self.topology.node_memory:
                    raise NUMAError(f"NUMA node {node_id} does not exist")
        
        # 检查优先节点
        if config.preferred_node is not None:
            if config.preferred_node not in self.topology.node_memory:
                raise NUMAError(f"NUMA node {config.preferred_node} does not exist")
        
        # 检查策略模式
        if config.mode == NUMAPolicyMode.BIND and not config.nodes:
            raise NUMAError("BIND mode requires at least one node")
        
        if config.mode == NUMAPolicyMode.PREFERRED and config.preferred_node is None:
            raise NUMAError("PREFERRED mode requires a preferred node")
    
    def _set_policy_linux(self, config: NUMAPolicyConfig) -> bool:
        """
        在Linux上设置NUMA策略
        
        Args:
            config: 策略配置
            
        Returns:
            是否成功
        """
        try:
            import ctypes
            
            # 加载libc
            libc = ctypes.CDLL(None, use_errno=True)
            
            # 设置mbind
            # 这里简化实现，实际需要调用mbind系统调用
            # 由于mbind需要内存地址，这里只更新配置
            
            self.current_policy = config
            return True
            
        except Exception as e:
            raise NUMAError(f"Failed to set NUMA policy on Linux: {e}")
    
    def _set_policy_windows(self, config: NUMAPolicyConfig) -> bool:
        """
        在Windows上设置NUMA策略
        
        Args:
            config: 策略配置
            
        Returns:
            是否成功
        """
        try:
            # Windows的NUMA策略通常通过VirtualAllocExNuma设置
            # 这里简化实现，只更新配置
            
            self.current_policy = config
            return True
            
        except Exception as e:
            raise NUMAError(f"Failed to set NUMA policy on Windows: {e}")
    
    def bind_to_node(self, node_id: int, strict: bool = False) -> bool:
        """
        绑定到指定NUMA节点
        
        Args:
            node_id: NUMA节点ID
            strict: 是否严格模式
            
        Returns:
            是否成功
        """
        config = NUMAPolicyConfig(
            mode=NUMAPolicyMode.BIND,
            nodes=[node_id],
            strict=strict
        )
        return self.set_policy(config)
    
    def set_interleave(self, nodes: List[int]) -> bool:
        """
        设置交错分配策略
        
        Args:
            nodes: NUMA节点列表
            
        Returns:
            是否成功
        """
        config = NUMAPolicyConfig(
            mode=NUMAPolicyMode.INTERLEAVE,
            nodes=nodes
        )
        return self.set_policy(config)
    
    def set_preferred(self, node_id: int) -> bool:
        """
        设置优先节点
        
        Args:
            node_id: 优先节点ID
            
        Returns:
            是否成功
        """
        config = NUMAPolicyConfig(
            mode=NUMAPolicyMode.PREFERRED,
            preferred_node=node_id
        )
        return self.set_policy(config)
    
    def set_default(self) -> bool:
        """
        设置默认策略
        
        Returns:
            是否成功
        """
        config = NUMAPolicyConfig(mode=NUMAPolicyMode.DEFAULT)
        return self.set_policy(config)
    
    def get_current_policy(self) -> NUMAPolicyConfig:
        """
        获取当前策略
        
        Returns:
            当前NUMA策略配置
        """
        return self.current_policy
    
    def recommend_policy(self, 
                        size: int,
                        access_pattern: str = 'random') -> NUMAPolicyConfig:
        """
        推荐NUMA策略
        
        根据内存大小和访问模式推荐最优的NUMA策略。
        
        Args:
            size: 内存大小（字节）
            access_pattern: 访问模式（'sequential', 'random', 'mixed'）
            
        Returns:
            推荐的NUMA策略配置
        """
        node_count = self.topology.get_node_count()
        
        # 单节点系统，使用默认策略
        if node_count <= 1:
            return NUMAPolicyConfig(mode=NUMAPolicyMode.DEFAULT)
        
        # 根据访问模式选择策略
        if access_pattern == 'sequential':
            # 顺序访问：绑定到单个节点
            # 选择内存最充足的节点
            best_node = self._find_best_node_for_size(size)
            if best_node is not None:
                return NUMAPolicyConfig(
                    mode=NUMAPolicyMode.BIND,
                    nodes=[best_node]
                )
        
        elif access_pattern == 'random':
            # 随机访问：交错分配
            # 使用所有可用节点
            all_nodes = list(self.topology.node_memory.keys())
            return NUMAPolicyConfig(
                mode=NUMAPolicyMode.INTERLEAVE,
                nodes=all_nodes
            )
        
        else:  # mixed
            # 混合访问：优先使用本地节点
            return NUMAPolicyConfig(mode=NUMAPolicyMode.LOCAL)
        
        return NUMAPolicyConfig(mode=NUMAPolicyMode.DEFAULT)
    
    def _find_best_node_for_size(self, size: int) -> Optional[int]:
        """
        找到最适合分配指定大小的节点
        
        Args:
            size: 内存大小
            
        Returns:
            最佳节点ID
        """
        best_node = None
        best_available = 0
        
        for node_id, node in self.topology.node_memory.items():
            if node.memory_available >= size:
                if node.memory_available > best_available:
                    best_available = node.memory_available
                    best_node = node_id
        
        return best_node
    
    def bind_memory(self, addr: int, size: int, node_id: int) -> bool:
        """
        绑定内存区域到NUMA节点
        
        Args:
            addr: 内存地址
            size: 内存大小
            node_id: NUMA节点ID
            
        Returns:
            是否成功
            
        Raises:
            NUMABindingError: 绑定失败时抛出
        """
        if sys.platform.startswith('linux'):
            return self._bind_memory_linux(addr, size, node_id)
        elif sys.platform == 'win32':
            return self._bind_memory_windows(addr, size, node_id)
        else:
            return True
    
    def _bind_memory_linux(self, addr: int, size: int, node_id: int) -> bool:
        """
        在Linux上绑定内存到NUMA节点
        
        Args:
            addr: 内存地址
            size: 内存大小
            node_id: NUMA节点ID
            
        Returns:
            是否成功
        """
        try:
            import ctypes
            import os
            
            # 加载libc
            libc = ctypes.CDLL(None, use_errno=True)
            
            # 构建节点掩码
            max_node = self.topology.get_node_count()
            mask_size = (max_node + 63) // 64
            nodemask = (ctypes.c_ulong * mask_size)()
            
            # 设置节点位
            idx = node_id // 64
            bit = node_id % 64
            nodemask[idx] |= (1 << bit)
            
            # 调用mbind
            # syscall number for mbind on x86_64 is 237
            import sys
            if sys.maxsize > 2**32:
                syscall_num = 237
            else:
                syscall_num = 274
            
            syscall = libc.syscall
            syscall.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_ulong),
                              ctypes.c_ulong, ctypes.c_int]
            syscall.restype = ctypes.c_int
            
            result = syscall(
                syscall_num,
                ctypes.c_void_p(addr),
                size,
                1,  # MPOL_BIND
                nodemask,
                max_node,
                1   # MPOL_MF_STRICT
            )
            
            if result != 0:
                err = ctypes.get_errno()
                raise NUMABindingError(node_id, err)
            
            return True
            
        except Exception as e:
            if isinstance(e, NUMABindingError):
                raise
            raise NUMABindingError(node_id, str(e))
    
    def _bind_memory_windows(self, addr: int, size: int, node_id: int) -> bool:
        """
        在Windows上绑定内存到NUMA节点
        
        Args:
            addr: 内存地址
            size: 内存大小
            node_id: NUMA节点ID
            
        Returns:
            是否成功
        """
        # Windows没有直接绑定已分配内存的API
        # 需要在分配时指定节点
        # 这里返回True表示成功（实际需要在分配时处理）
        return True
    
    def get_node_for_address(self, addr: int) -> Optional[int]:
        """
        获取内存地址所在的NUMA节点
        
        Args:
            addr: 内存地址
            
        Returns:
            NUMA节点ID
        """
        if sys.platform.startswith('linux'):
            return self._get_node_for_address_linux(addr)
        else:
            return None
    
    def _get_node_for_address_linux(self, addr: int) -> Optional[int]:
        """
        在Linux上获取内存地址所在的NUMA节点
        
        Args:
            addr: 内存地址
            
        Returns:
            NUMA节点ID
        """
        try:
            # 读取/proc/self/numa_maps
            with open('/proc/self/numa_maps', 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        # 解析地址
                        addr_str = parts[0]
                        map_addr = int(addr_str, 16)
                        
                        # 查找N<node>=<count>字段
                        for part in parts[1:]:
                            if part.startswith('N') and '=' in part:
                                node_str = part[1:part.index('=')]
                                try:
                                    return int(node_str)
                                except ValueError:
                                    pass
            
        except (IOError, OSError):
            pass
        
        return None
    
    def get_policy_summary(self) -> Dict:
        """
        获取策略摘要
        
        Returns:
            策略摘要字典
        """
        return {
            'current_policy': self.current_policy.to_dict(),
            'topology': {
                'node_count': self.topology.get_node_count(),
                'total_memory': self.topology.get_total_memory(),
                'available_memory': self.topology.get_available_memory(),
            }
        }


# 全局NUMA策略管理器实例
_global_policy_manager = None


def get_numa_policy_manager() -> NUMAPolicyManager:
    """
    获取全局NUMA策略管理器
    
    Returns:
        NUMA策略管理器实例
    """
    global _global_policy_manager
    
    if _global_policy_manager is None:
        _global_policy_manager = NUMAPolicyManager()
    
    return _global_policy_manager
