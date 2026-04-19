"""
mem_monitor NUMA感知模块

实现NUMA拓扑感知和内存平衡。

基于NeoMem论文的设备端分析和硬件加速思路。
"""

import time
import sys
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# 配置模块日志记录器
logger = logging.getLogger(__name__)


class NUMAPolicy(Enum):
    """NUMA策略"""
    DEFAULT = "default"       # 默认
    BIND = "bind"             # 绑定到指定节点
    INTERLEAVE = "interleave" # 交错分配
    PREFERRED = "preferred"   # 优先使用指定节点
    LOCAL = "local"           # 本地节点优先


@dataclass
class NUMANodeInfo:
    """
    NUMA节点信息

    记录单个NUMA节点的详细信息。
    """

    node_id: int                             # 节点ID

    # CPU信息
    cpus: List[int] = field(default_factory=list)  # CPU核心列表
    cpu_count: int = 0                       # CPU数量

    # 内存信息
    memory_total: int = 0                    # 总内存
    memory_free: int = 0                     # 空闲内存
    memory_used: int = 0                     # 已用内存
    memory_available: int = 0                # 可用内存

    # 距离信息
    distances: List[int] = field(default_factory=list)  # 到其他节点的距离

    # 统计
    allocation_count: int = 0                # 分配次数
    migration_in_count: int = 0              # 迁入次数
    migration_out_count: int = 0             # 迁出次数

    def get_usage_ratio(self) -> float:
        """获取内存使用率"""
        if self.memory_total == 0:
            return 0.0
        return self.memory_used / self.memory_total

    def get_available_ratio(self) -> float:
        """获取可用内存比例"""
        if self.memory_total == 0:
            return 0.0
        return self.memory_available / self.memory_total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'cpu_count': self.cpu_count,
            'memory_total_mb': self.memory_total / (1024 * 1024),
            'memory_used_mb': self.memory_used / (1024 * 1024),
            'memory_available_mb': self.memory_available / (1024 * 1024),
            'usage_ratio': self.get_usage_ratio(),
            'allocation_count': self.allocation_count,
            'migration_in_count': self.migration_in_count,
            'migration_out_count': self.migration_out_count,
        }


@dataclass
class NUMATopologyInfo:
    """
    NUMA拓扑信息

    记录整个系统的NUMA拓扑结构。
    """

    # 节点列表
    nodes: List[NUMANodeInfo] = field(default_factory=list)

    # 节点映射
    node_map: Dict[int, NUMANodeInfo] = field(default_factory=dict)

    # CPU到节点映射
    cpu_to_node: Dict[int, int] = field(default_factory=dict)

    # 距离矩阵
    distance_matrix: List[List[int]] = field(default_factory=list)

    # 是否支持NUMA
    numa_available: bool = False

    def get_node(self, node_id: int) -> Optional[NUMANodeInfo]:
        """获取节点信息"""
        return self.node_map.get(node_id)

    def get_node_count(self) -> int:
        """获取节点数量"""
        return len(self.nodes)

    def get_cpu_node(self, cpu_id: int) -> Optional[int]:
        """获取CPU所在的NUMA节点"""
        return self.cpu_to_node.get(cpu_id)

    def get_distance(self, node1: int, node2: int) -> int:
        """获取两个节点之间的距离"""
        if node1 < len(self.distance_matrix) and node2 < len(self.distance_matrix[node1]):
            return self.distance_matrix[node1][node2]
        return 0

    def get_total_memory(self) -> int:
        """获取总内存"""
        return sum(node.memory_total for node in self.nodes)

    def get_total_available(self) -> int:
        """获取总可用内存"""
        return sum(node.memory_available for node in self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'numa_available': self.numa_available,
            'node_count': self.get_node_count(),
            'total_memory_mb': self.get_total_memory() / (1024 * 1024),
            'total_available_mb': self.get_total_available() / (1024 * 1024),
            'nodes': [n.to_dict() for n in self.nodes],
        }


@dataclass
class MigrationPlan:
    """
    迁移计划

    描述一次内存迁移的详细计划。
    """

    # 迁移信息
    source_node: int                         # 源节点
    target_node: int                         # 目标节点
    size: int                                # 迁移大小
    page_count: int                          # 页面数量

    # 原因
    reason: str = ""                         # 迁移原因

    # 优先级
    priority: int = 0                        # 优先级 (越高越优先)

    # 预估收益
    estimated_benefit: float = 0.0           # 预估收益

    # 页面列表
    pages: List[int] = field(default_factory=list)  # 要迁移的页面列表


class NUMABalancer:
    """
    NUMA平衡器

    平衡各NUMA节点的内存使用。

    算法思路：
    1. 监控各节点内存使用
    2. 检测不平衡状态
    3. 生成迁移计划
    4. 执行迁移

    平衡策略：
    - 基于内存使用率
    - 考虑节点距离
    - 考虑访问模式
    """

    def __init__(self, threshold: float = 0.2):
        """
        初始化NUMA平衡器

        Args:
            threshold: 平衡阈值（使用率差异超过此值触发平衡）
        """
        self._threshold = threshold
        self._balance_history: List[Dict[str, Any]] = []

    def check_balance(self, topology: NUMATopologyInfo) -> Tuple[bool, float]:
        """
        检查是否需要平衡

        Args:
            topology: NUMA拓扑

        Returns:
            Tuple[bool, float]: (是否需要平衡, 不平衡程度)
        """
        if len(topology.nodes) < 2:
            return False, 0.0

        # 计算各节点使用率
        usage_rates = [node.get_usage_ratio() for node in topology.nodes]

        # 计算最大差异
        max_usage = max(usage_rates)
        min_usage = min(usage_rates)
        imbalance = max_usage - min_usage

        return imbalance > self._threshold, imbalance

    def generate_balance_plan(self,
                             topology: NUMATopologyInfo,
                             hot_pages: Dict[int, int]) -> List[MigrationPlan]:
        """
        生成平衡计划

        Args:
            topology: NUMA拓扑
            hot_pages: 热页面到节点的映射

        Returns:
            List[MigrationPlan]: 迁移计划列表
        """
        plans = []

        # 检查平衡状态
        needs_balance, imbalance = self.check_balance(topology)

        if not needs_balance:
            return plans

        # 找出高使用率和低使用率节点
        node_usage = [
            (node.node_id, node.get_usage_ratio(), node.memory_available)
            for node in topology.nodes
        ]
        node_usage.sort(key=lambda x: x[1], reverse=True)

        high_node = node_usage[0]
        low_node = node_usage[-1]

        # 计算需要迁移的大小
        target_usage = (high_node[1] + low_node[1]) / 2
        migrate_size = int((high_node[1] - target_usage) * topology.get_node(high_node[0]).memory_total)

        # 创建迁移计划
        if migrate_size > 0:
            plan = MigrationPlan(
                source_node=high_node[0],
                target_node=low_node[0],
                size=migrate_size,
                page_count=migrate_size // 4096,
                reason=f"Balance memory usage (imbalance: {imbalance:.2%})",
                priority=int(imbalance * 100),
                estimated_benefit=imbalance * migrate_size,
            )
            plans.append(plan)

        return plans

    def get_balance_stats(self) -> Dict[str, Any]:
        """获取平衡统计"""
        return {
            'balance_count': len(self._balance_history),
            'recent_balances': self._balance_history[-10:],
        }


class MigrationPlanner:
    """
    迁移规划器

    规划内存迁移操作。

    基于NOMAD论文的事务性迁移：
    - 批量迁移
    - 原子性保证
    - 回滚机制
    """

    def __init__(self, batch_size: int = 100):
        """
        初始化迁移规划器

        Args:
            batch_size: 批量迁移大小
        """
        self._batch_size = batch_size
        self._pending_migrations: List[MigrationPlan] = []
        self._completed_migrations: List[Dict[str, Any]] = []

    def add_migration(self, plan: MigrationPlan):
        """添加迁移计划"""
        self._pending_migrations.append(plan)

    def get_next_batch(self) -> List[MigrationPlan]:
        """获取下一批迁移计划"""
        batch = self._pending_migrations[:self._batch_size]
        self._pending_migrations = self._pending_migrations[self._batch_size:]
        return batch

    def mark_completed(self, plan: MigrationPlan, success: bool):
        """标记迁移完成"""
        self._completed_migrations.append({
            'timestamp': time.time(),
            'source_node': plan.source_node,
            'target_node': plan.target_node,
            'size': plan.size,
            'success': success,
        })

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        successful = sum(1 for m in self._completed_migrations if m['success'])
        failed = len(self._completed_migrations) - successful

        return {
            'pending_count': len(self._pending_migrations),
            'completed_count': len(self._completed_migrations),
            'successful_count': successful,
            'failed_count': failed,
        }


class NUMAAwareManager:
    """
    NUMA感知管理器

    提供NUMA拓扑感知和内存管理功能。

    功能：
    1. 检测NUMA拓扑
    2. 监控节点内存使用
    3. 生成平衡建议
    4. 规划内存迁移
    """

    def __init__(self, config):
        """
        初始化NUMA感知管理器

        Args:
            config: 分层配置
        """
        self._config = config

        # 拓扑信息
        self._topology = NUMATopologyInfo()

        # 平衡器
        self._balancer = NUMABalancer(config.numa_balance_threshold)

        # 迁移规划器
        self._planner = MigrationPlanner(config.migration_batch_size)

        # 热页面追踪
        self._hot_pages: Dict[int, int] = {}  # page -> node

        # 初始化拓扑
        self._detect_topology()

    def _detect_topology(self):
        """检测NUMA拓扑"""
        self._topology.numa_available = False

        if sys.platform.startswith('linux'):
            self._detect_linux_topology()
        elif sys.platform == 'win32':
            self._detect_windows_topology()
        else:
            self._create_single_node_topology()

    def _detect_linux_topology(self):
        """检测Linux NUMA拓扑"""
        numa_dir = '/sys/devices/system/node'

        if not os.path.exists(numa_dir):
            self._create_single_node_topology()
            return

        # 读取节点信息
        try:
            node_dirs = [d for d in os.listdir(numa_dir) if d.startswith('node')]
        except OSError as e:
            logger.warning(f"Failed to list NUMA nodes: {e}")
            self._create_single_node_topology()
            return

        if not node_dirs:
            self._create_single_node_topology()
            return

        self._topology.numa_available = True

        for node_dir in sorted(node_dirs):
            node_id = int(node_dir.replace('node', ''))
            node_path = os.path.join(numa_dir, node_dir)

            node_info = NUMANodeInfo(node_id=node_id)

            # 读取CPU信息
            cpulist_path = os.path.join(node_path, 'cpulist')
            if os.path.exists(cpulist_path):
                try:
                    with open(cpulist_path, 'r') as f:
                        cpulist = f.read().strip()
                        node_info.cpus = self._parse_cpulist(cpulist)
                        node_info.cpu_count = len(node_info.cpus)
                except (OSError, IOError) as e:
                    logger.debug(f"Failed to read cpulist for node {node_id}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error reading cpulist for node {node_id}: {e}")

            # 读取内存信息
            meminfo_path = os.path.join(node_path, 'meminfo')
            if os.path.exists(meminfo_path):
                try:
                    with open(meminfo_path, 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node_info.memory_total = int(parts[2]) * 1024
                            elif 'MemFree' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node_info.memory_free = int(parts[2]) * 1024
                            elif 'MemAvailable' in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    node_info.memory_available = int(parts[2]) * 1024
                except (OSError, IOError) as e:
                    logger.debug(f"Failed to read meminfo for node {node_id}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error reading meminfo for node {node_id}: {e}")

            # 读取距离信息
            distance_path = os.path.join(node_path, 'distance')
            if os.path.exists(distance_path):
                try:
                    with open(distance_path, 'r') as f:
                        distances = f.read().strip().split()
                        node_info.distances = [int(d) for d in distances]
                except (OSError, IOError) as e:
                    logger.debug(f"Failed to read distance for node {node_id}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error reading distance for node {node_id}: {e}")

            # 计算已用内存
            node_info.memory_used = node_info.memory_total - node_info.memory_available

            self._topology.nodes.append(node_info)
            self._topology.node_map[node_id] = node_info

            # 更新CPU映射
            for cpu in node_info.cpus:
                self._topology.cpu_to_node[cpu] = node_id

        # 构建距离矩阵
        node_count = len(self._topology.nodes)
        self._topology.distance_matrix = [[0] * node_count for _ in range(node_count)]

        for node in self._topology.nodes:
            if node.distances:
                for i, dist in enumerate(node.distances):
                    if i < node_count:
                        self._topology.distance_matrix[node.node_id][i] = dist

    def _detect_windows_topology(self):
        """检测Windows NUMA拓扑"""
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32

            # 获取最高节点号
            GetNumaHighestNodeNumber = kernel32.GetNumaHighestNodeNumber
            GetNumaHighestNodeNumber.argtypes = [ctypes.POINTER(ctypes.c_ulong)]
            GetNumaHighestNodeNumber.restype = ctypes.c_int

            highest_node = ctypes.c_ulong()
            if not GetNumaHighestNodeNumber(ctypes.byref(highest_node)):
                logger.debug("GetNumaHighestNodeNumber failed, using single node topology")
                self._create_single_node_topology()
                return

            node_count = highest_node.value + 1
            self._topology.numa_available = node_count > 1

            # 获取系统内存信息
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

            total_mem = 0
            avail_mem = 0

            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                total_mem = mem_status.ullTotalPhys
                avail_mem = mem_status.ullAvailPhys
            else:
                logger.debug("GlobalMemoryStatusEx failed, using default memory values")

            # 创建节点信息
            for node_id in range(node_count):
                node_info = NUMANodeInfo(node_id=node_id)

                # 平均分配内存
                node_info.memory_total = total_mem // node_count
                node_info.memory_available = avail_mem // node_count
                node_info.memory_used = node_info.memory_total - node_info.memory_available

                self._topology.nodes.append(node_info)
                self._topology.node_map[node_id] = node_info

            # Windows默认距离矩阵
            self._topology.distance_matrix = [
                [10 if i == j else 20 for j in range(node_count)]
                for i in range(node_count)
            ]

        except Exception as e:
            logger.warning(f"Failed to detect Windows NUMA topology: {e}", exc_info=True)
            self._create_single_node_topology()

    def _create_single_node_topology(self):
        """创建单节点拓扑"""
        node_info = NUMANodeInfo(node_id=0)

        # 获取CPU数量
        try:
            import multiprocessing
            node_info.cpu_count = multiprocessing.cpu_count()
            node_info.cpus = list(range(node_info.cpu_count))
        except Exception as e:
            logger.debug(f"Failed to get CPU count: {e}")
            node_info.cpu_count = 1
            node_info.cpus = [0]

        # 获取内存信息
        try:
            import psutil
            mem = psutil.virtual_memory()
            node_info.memory_total = mem.total
            node_info.memory_available = mem.available
            node_info.memory_free = mem.free
            node_info.memory_used = mem.used
        except ImportError:
            logger.debug("psutil not available, using default memory values")
            # 使用默认值
            node_info.memory_total = 8 * 1024 * 1024 * 1024  # 8GB
            node_info.memory_available = 4 * 1024 * 1024 * 1024  # 4GB
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            # 使用默认值
            node_info.memory_total = 8 * 1024 * 1024 * 1024  # 8GB
            node_info.memory_available = 4 * 1024 * 1024 * 1024  # 4GB

        self._topology.nodes.append(node_info)
        self._topology.node_map[0] = node_info
        self._topology.distance_matrix = [[10]]
        self._topology.numa_available = False

    def _parse_cpulist(self, cpulist: str) -> List[int]:
        """解析CPU列表"""
        cpus = []

        for part in cpulist.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                try:
                    cpus.extend(range(int(start), int(end) + 1))
                except ValueError:
                    pass
            else:
                try:
                    cpus.append(int(part))
                except ValueError:
                    pass

        return sorted(cpus)

    def get_topology(self) -> NUMATopologyInfo:
        """获取NUMA拓扑"""
        return self._topology

    def get_current_node(self) -> int:
        """
        获取当前CPU所在的NUMA节点
        
        Returns:
            int: NUMA节点ID
        """
        # Linux: 使用os.sched_getcpu()
        if sys.platform.startswith('linux'):
            try:
                cpu_id = os.sched_getcpu()
                return self._topology.get_cpu_node(cpu_id) or 0
            except (AttributeError, OSError) as e:
                logger.debug(f"sched_getcpu not available: {e}")
                return 0
        
        # Windows: 使用ctypes获取当前处理器编号
        elif sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                GetCurrentProcessorNumber = kernel32.GetCurrentProcessorNumber
                GetCurrentProcessorNumber.restype = ctypes.c_ulong
                cpu_id = GetCurrentProcessorNumber()
                return self._topology.get_cpu_node(cpu_id) or 0
            except Exception as e:
                logger.debug(f"GetCurrentProcessorNumber not available: {e}")
                return 0
        
        # 其他平台: 返回默认节点
        return 0

    def get_best_node_for_allocation(self, size: int, prefer_local: bool = True) -> int:
        """
        获取最适合分配的节点

        Args:
            size: 分配大小
            prefer_local: 是否优先本地节点

        Returns:
            int: 推荐节点ID
        """
        if prefer_local:
            local_node = self.get_current_node()
            local_info = self._topology.get_node(local_node)
            if local_info and local_info.memory_available >= size:
                return local_node

        # 找到有足够内存且最空闲的节点
        best_node = 0
        best_available = 0

        for node in self._topology.nodes:
            if node.memory_available >= size and node.memory_available > best_available:
                best_node = node.node_id
                best_available = node.memory_available

        return best_node

    def update_hot_page(self, page: int, node: int):
        """更新热页面信息"""
        self._hot_pages[page] = node

    def get_balance_recommendations(self) -> List[Dict[str, Any]]:
        """获取平衡建议"""
        recommendations = []

        # 生成平衡计划
        plans = self._balancer.generate_balance_plan(self._topology, self._hot_pages)

        for plan in plans:
            recommendations.append({
                'type': 'numa_balance',
                'source_node': plan.source_node,
                'target_node': plan.target_node,
                'size': plan.size,
                'reason': plan.reason,
                'priority': plan.priority,
            })

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'topology': self._topology.to_dict(),
            'balancer': self._balancer.get_balance_stats(),
            'planner': self._planner.get_stats(),
            'hot_page_count': len(self._hot_pages),
        }

    def refresh_topology(self):
        """刷新拓扑信息"""
        self._detect_topology()
