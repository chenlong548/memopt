"""
mem_mapper NUMA拓扑检测模块

提供NUMA拓扑检测和管理功能。
"""

import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..core.exceptions import NUMAError, NUMANotSupportedError


@dataclass
class NUMANode:
    """
    NUMA节点信息
    
    表示单个NUMA节点的详细信息。
    """
    
    node_id: int                       # 节点ID
    cpus: List[int] = field(default_factory=list)  # CPU核心列表
    memory_total: int = 0              # 总内存（字节）
    memory_free: int = 0               # 空闲内存（字节）
    memory_available: int = 0          # 可用内存（字节）
    distance: List[int] = field(default_factory=list)  # 到其他节点的距离
    
    def get_memory_usage(self) -> float:
        """
        获取内存使用率
        
        Returns:
            内存使用率（0.0-1.0）
        """
        if self.memory_total == 0:
            return 0.0
        return (self.memory_total - self.memory_available) / self.memory_total
    
    def get_cpu_count(self) -> int:
        """
        获取CPU核心数量
        
        Returns:
            CPU核心数量
        """
        return len(self.cpus)


@dataclass
class NUMATopology:
    """
    NUMA拓扑结构
    
    表示整个系统的NUMA拓扑信息。
    """
    
    nodes: List[NUMANode] = field(default_factory=list)  # NUMA节点列表
    distance_matrix: List[List[int]] = field(default_factory=list)  # 距离矩阵
    cpu_to_node: Dict[int, int] = field(default_factory=dict)  # CPU到节点的映射
    node_memory: Dict[int, 'NUMANode'] = field(default_factory=dict)  # 节点内存信息
    
    def get_node(self, node_id: int) -> Optional[NUMANode]:
        """
        获取指定节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            NUMA节点，如果不存在则返回None
        """
        return self.node_memory.get(node_id)
    
    def get_node_count(self) -> int:
        """
        获取节点数量
        
        Returns:
            节点数量
        """
        return len(self.nodes)
    
    def get_cpu_node(self, cpu_id: int) -> Optional[int]:
        """
        获取CPU所在的NUMA节点
        
        Args:
            cpu_id: CPU ID
            
        Returns:
            NUMA节点ID，如果不存在则返回None
        """
        return self.cpu_to_node.get(cpu_id)
    
    def get_distance(self, node1: int, node2: int) -> int:
        """
        获取两个节点之间的距离
        
        Args:
            node1: 节点1 ID
            node2: 节点2 ID
            
        Returns:
            距离值
        """
        if node1 < len(self.distance_matrix) and node2 < len(self.distance_matrix[node1]):
            return self.distance_matrix[node1][node2]
        return 0
    
    def get_total_memory(self) -> int:
        """
        获取总内存
        
        Returns:
            总内存（字节）
        """
        return sum(node.memory_total for node in self.nodes)
    
    def get_available_memory(self) -> int:
        """
        获取可用内存
        
        Returns:
            可用内存（字节）
        """
        return sum(node.memory_available for node in self.nodes)
    
    def find_best_node(self, 
                       size: int,
                       prefer_local: bool = False,
                       current_node: Optional[int] = None) -> Optional[int]:
        """
        找到最适合分配指定大小内存的节点
        
        Args:
            size: 需要分配的内存大小
            prefer_local: 是否优先本地节点
            current_node: 当前节点
            
        Returns:
            最佳节点ID
        """
        if prefer_local and current_node is not None:
            node = self.get_node(current_node)
            if node and node.memory_available >= size:
                return current_node
        
        # 找到有足够内存且距离最近的节点
        best_node = None
        best_score = float('inf')
        
        for node in self.nodes:
            if node.memory_available < size:
                continue
            
            # 计算分数（考虑距离和内存使用率）
            distance = 0
            if current_node is not None:
                distance = self.get_distance(current_node, node.node_id)
            
            usage = node.get_memory_usage()
            score = distance * 0.5 + usage * 100  # 权重平衡
            
            if score < best_score:
                best_score = score
                best_node = node.node_id
        
        return best_node


class NUMATopologyDetector:
    """
    NUMA拓扑检测器
    
    检测系统的NUMA拓扑结构。
    """
    
    def __init__(self):
        """初始化NUMA拓扑检测器"""
        self._topology = None
        self._cached = False
    
    def detect(self) -> NUMATopology:
        """
        检测NUMA拓扑
        
        Returns:
            NUMA拓扑结构
            
        Raises:
            NUMANotSupportedError: 系统不支持NUMA时抛出
        """
        if self._cached and self._topology is not None:
            return self._topology
        
        # 根据平台选择检测方法
        if sys.platform.startswith('linux'):
            self._topology = self._detect_linux()
        elif sys.platform == 'win32':
            self._topology = self._detect_windows()
        else:
            # 不支持的平台，返回单节点拓扑
            self._topology = self._create_single_node_topology()
        
        self._cached = True
        return self._topology
    
    def _detect_linux(self) -> NUMATopology:
        """
        检测Linux系统的NUMA拓扑
        
        Returns:
            NUMA拓扑结构
        """
        topology = NUMATopology()
        
        # 检查NUMA是否可用
        numa_dir = '/sys/devices/system/node'
        if not os.path.exists(numa_dir):
            return self._create_single_node_topology()
        
        # 读取节点信息
        node_dirs = [d for d in os.listdir(numa_dir) if d.startswith('node')]
        
        if not node_dirs:
            return self._create_single_node_topology()
        
        # 解析每个节点
        for node_dir in sorted(node_dirs):
            node_id = int(node_dir.replace('node', ''))
            node_path = os.path.join(numa_dir, node_dir)
            
            node = NUMANode(node_id=node_id)
            
            # 读取CPU信息
            cpulist_path = os.path.join(node_path, 'cpulist')
            if os.path.exists(cpulist_path):
                try:
                    with open(cpulist_path, 'r') as f:
                        cpulist = f.read().strip()
                        node.cpus = self._parse_cpulist(cpulist)
                except IOError:
                    pass
            
            # 读取内存信息
            meminfo_path = os.path.join(node_path, 'meminfo')
            if os.path.exists(meminfo_path):
                try:
                    with open(meminfo_path, 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_total = int(parts[2]) * 1024  # kB to bytes
                            elif 'MemFree' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_free = int(parts[2]) * 1024
                            elif 'MemAvailable' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_available = int(parts[2]) * 1024
                except IOError:
                    pass
            
            # 读取距离信息
            distance_path = os.path.join(node_path, 'distance')
            if os.path.exists(distance_path):
                try:
                    with open(distance_path, 'r') as f:
                        distances = f.read().strip().split()
                        node.distance = [int(d) for d in distances]
                except IOError:
                    pass
            
            topology.nodes.append(node)
            topology.node_memory[node_id] = node
            
            # 更新CPU到节点的映射
            for cpu in node.cpus:
                topology.cpu_to_node[cpu] = node_id
        
        # 构建距离矩阵
        node_count = len(topology.nodes)
        topology.distance_matrix = [[0] * node_count for _ in range(node_count)]
        
        for node in topology.nodes:
            if node.distance:
                for i, dist in enumerate(node.distance):
                    if i < node_count:
                        topology.distance_matrix[node.node_id][i] = dist
        
        return topology
    
    def _detect_windows(self) -> NUMATopology:
        """
        检测Windows系统的NUMA拓扑
        
        Returns:
            NUMA拓扑结构
        """
        topology = NUMATopology()
        
        try:
            import ctypes
            
            # 获取NUMA节点数量
            kernel32 = ctypes.windll.kernel32
            
            GetNumaHighestNodeNumber = kernel32.GetNumaHighestNodeNumber
            GetNumaHighestNodeNumber.argtypes = [
                ctypes.POINTER(ctypes.c_ulong)
            ]
            GetNumaHighestNodeNumber.restype = ctypes.c_int
            
            highest_node = ctypes.c_ulong()
            if not GetNumaHighestNodeNumber(ctypes.byref(highest_node)):
                return self._create_single_node_topology()
            
            node_count = highest_node.value + 1
            
            # 获取每个节点的信息
            for node_id in range(node_count):
                node = NUMANode(node_id=node_id)
                
                # 获取节点的CPU掩码
                GetNumaNodeProcessorMask = kernel32.GetNumaNodeProcessorMask
                GetNumaNodeProcessorMask.argtypes = [
                    ctypes.c_ubyte,
                    ctypes.POINTER(ctypes.c_ulonglong)
                ]
                GetNumaNodeProcessorMask.restype = ctypes.c_int
                
                processor_mask = ctypes.c_ulonglong()
                if GetNumaNodeProcessorMask(node_id, ctypes.byref(processor_mask)):
                    # 解析CPU掩码
                    mask = processor_mask.value
                    cpu_id = 0
                    while mask:
                        if mask & 1:
                            node.cpus.append(cpu_id)
                        mask >>= 1
                        cpu_id += 1
                
                # Windows没有直接获取节点内存的API
                # 使用系统总内存除以节点数作为估计
                import ctypes.wintypes
                
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.wintypes.DWORD),
                        ('dwMemoryLoad', ctypes.wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]
                
                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                
                if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                    total_mem = mem_status.ullTotalPhys
                    avail_mem = mem_status.ullAvailPhys
                    
                    # 平均分配到各节点
                    node.memory_total = total_mem // node_count
                    node.memory_available = avail_mem // node_count
                
                topology.nodes.append(node)
                topology.node_memory[node_id] = node
                
                # 更新CPU到节点的映射
                for cpu in node.cpus:
                    topology.cpu_to_node[cpu] = node_id
            
            # Windows没有节点距离的概念，使用默认值
            topology.distance_matrix = [[10 if i == j else 20 
                                         for j in range(node_count)]
                                        for i in range(node_count)]
            
        except Exception:
            return self._create_single_node_topology()
        
        return topology
    
    def _create_single_node_topology(self) -> NUMATopology:
        """
        创建单节点拓扑
        
        Returns:
            单节点NUMA拓扑
        """
        topology = NUMATopology()
        
        # 创建单个节点
        node = NUMANode(node_id=0)
        
        # 获取CPU数量
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            node.cpus = list(range(cpu_count))
        except Exception:
            node.cpus = [0]
        
        # 获取内存信息
        try:
            import psutil
            mem = psutil.virtual_memory()
            node.memory_total = mem.total
            node.memory_available = mem.available
            node.memory_free = mem.free
        except ImportError:
            # 如果没有psutil，使用platform模块
            import platform
            if sys.platform.startswith('linux'):
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                        for line in meminfo.split('\n'):
                            if line.startswith('MemTotal:'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_total = int(parts[1]) * 1024
                            elif line.startswith('MemAvailable:'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_available = int(parts[1]) * 1024
                            elif line.startswith('MemFree:'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    node.memory_free = int(parts[1]) * 1024
                except IOError:
                    pass
        
        topology.nodes.append(node)
        topology.node_memory[0] = node
        
        # 更新CPU到节点的映射
        for cpu in node.cpus:
            topology.cpu_to_node[cpu] = 0
        
        # 单节点的距离矩阵
        topology.distance_matrix = [[10]]
        
        return topology
    
    def _parse_cpulist(self, cpulist: str) -> List[int]:
        """
        解析CPU列表字符串
        
        Args:
            cpulist: CPU列表字符串（如 "0-3,5,7-9"）
            
        Returns:
            CPU ID列表
        """
        cpus = []
        
        if not cpulist:
            return cpus
        
        for part in cpulist.split(','):
            part = part.strip()
            if '-' in part:
                # 范围，如 "0-3"
                start, end = part.split('-')
                try:
                    start_id = int(start)
                    end_id = int(end)
                    cpus.extend(range(start_id, end_id + 1))
                except ValueError:
                    pass
            else:
                # 单个CPU
                try:
                    cpus.append(int(part))
                except ValueError:
                    pass
        
        return sorted(cpus)
    
    def refresh(self) -> NUMATopology:
        """
        刷新拓扑信息
        
        Returns:
            最新的NUMA拓扑
        """
        self._cached = False
        return self.detect()
    
    def is_numa_available(self) -> bool:
        """
        检查NUMA是否可用
        
        Returns:
            NUMA是否可用
        """
        try:
            topology = self.detect()
            return topology.get_node_count() > 1
        except Exception:
            return False


# 全局拓扑检测器实例
_global_detector = None


def get_numa_topology() -> NUMATopology:
    """
    获取NUMA拓扑（全局函数）
    
    Returns:
        NUMA拓扑结构
    """
    global _global_detector
    
    if _global_detector is None:
        _global_detector = NUMATopologyDetector()
    
    return _global_detector.detect()


def is_numa_available() -> bool:
    """
    检查NUMA是否可用（全局函数）
    
    Returns:
        NUMA是否可用
    """
    global _global_detector
    
    if _global_detector is None:
        _global_detector = NUMATopologyDetector()
    
    return _global_detector.is_numa_available()
