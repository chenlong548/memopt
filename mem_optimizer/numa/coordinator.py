"""
mem_optimizer NUMA协调器

实现NUMA感知的内存分配协调。
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from ..core.base import AllocationRequest, NUMACoordinatorBase
from ..core.config import NUMAConfig, NUMAPolicy
from ..core.exceptions import NUMAError, NUMANotAvailableError, NUMAMigrationError


class NodeState(Enum):
    """NUMA节点状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class NUMANode:
    """NUMA节点信息"""
    node_id: int
    state: NodeState = NodeState.ONLINE
    total_memory: int = 0
    free_memory: int = 0
    used_memory: int = 0
    cpus: List[int] = field(default_factory=list)
    distance: Dict[int, int] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        if self.total_memory == 0:
            return 0.0
        return self.used_memory / self.total_memory

    @property
    def available_ratio(self) -> float:
        if self.total_memory == 0:
            return 0.0
        return self.free_memory / self.total_memory


@dataclass
class MemoryDistribution:
    """内存分布"""
    allocations: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_allocated: int = 0

    def record(self, node: int, size: int):
        self.allocations[node] += size
        self.total_allocated += size

    def get_balance_score(self) -> float:
        if not self.allocations:
            return 1.0

        values = list(self.allocations.values())
        if len(values) <= 1:
            return 1.0

        avg = sum(values) / len(values)
        if avg == 0:
            return 1.0

        variance = sum((v - avg) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5

        cv = std_dev / avg
        return max(0.0, 1.0 - cv)


class NUMACoordinator(NUMACoordinatorBase):
    """
    NUMA协调器

    提供NUMA感知的内存分配和管理功能。
    """

    def __init__(self, config: Optional[NUMAConfig] = None):
        """
        初始化NUMA协调器

        Args:
            config: NUMA配置
        """
        self.config = config or NUMAConfig()

        self._nodes: Dict[int, NUMANode] = {}
        self._distribution = MemoryDistribution()
        self._is_available = False
        self._lock = threading.Lock()

        self._migration_count = 0
        self._balance_count = 0

        self._initialize()

    def _initialize(self):
        """初始化NUMA信息"""
        self._is_available = self._detect_numa()

        if not self._is_available:
            self._create_single_node()

    def _detect_numa(self) -> bool:
        """
        检测NUMA支持

        Returns:
            bool: 是否支持NUMA
        """
        if os.name == 'nt':
            return False

        numa_path = "/sys/devices/system/node"
        if not os.path.exists(numa_path):
            return False

        try:
            node_dirs = [d for d in os.listdir(numa_path) if d.startswith('node')]

            for node_dir in node_dirs:
                node_id = int(node_dir[4:])
                node = self._read_node_info(numa_path, node_id)
                if node:
                    self._nodes[node_id] = node

            return len(self._nodes) > 0

        except Exception:
            return False

    def _read_node_info(self, numa_path: str, node_id: int) -> Optional[NUMANode]:
        """
        读取节点信息

        Args:
            numa_path: NUMA路径
            node_id: 节点ID

        Returns:
            NUMANode: 节点信息
        """
        node_path = os.path.join(numa_path, f"node{node_id}")

        if not os.path.exists(node_path):
            return None

        node = NUMANode(node_id=node_id)

        meminfo_path = os.path.join(node_path, "meminfo")
        if os.path.exists(meminfo_path):
            try:
                with open(meminfo_path, 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    node.total_memory = int(parts[2]) * 1024
                                except (ValueError, IndexError):
                                    pass
                        elif 'MemFree' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    node.free_memory = int(parts[2]) * 1024
                                except (ValueError, IndexError):
                                    pass
            except Exception:
                pass

        cpulist_path = os.path.join(node_path, "cpulist")
        if os.path.exists(cpulist_path):
            try:
                with open(cpulist_path, 'r') as f:
                    cpulist = f.read().strip()
                    node.cpus = self._parse_cpulist(cpulist)
            except Exception:
                pass

        distance_path = os.path.join(node_path, "distance")
        if os.path.exists(distance_path):
            try:
                with open(distance_path, 'r') as f:
                    distances = f.read().strip().split()
                    for i, d in enumerate(distances):
                        try:
                            node.distance[i] = int(d)
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass

        node.used_memory = node.total_memory - node.free_memory

        return node

    def _parse_cpulist(self, cpulist: str) -> List[int]:
        """
        解析CPU列表

        Args:
            cpulist: CPU列表字符串

        Returns:
            List[int]: CPU ID列表
        """
        cpus = []
        for part in cpulist.split(','):
            if '-' in part:
                start, end = part.split('-')
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return cpus

    def _create_single_node(self):
        """创建单节点模拟"""
        node = NUMANode(
            node_id=0,
            state=NodeState.ONLINE,
            total_memory=8 * 1024 * 1024 * 1024,
            free_memory=4 * 1024 * 1024 * 1024,
            used_memory=4 * 1024 * 1024 * 1024,
            cpus=list(range(os.cpu_count() or 4))
        )
        self._nodes[0] = node

    def get_numa_nodes(self) -> List[int]:
        """
        获取NUMA节点列表

        Returns:
            List[int]: NUMA节点ID列表
        """
        return list(self._nodes.keys())

    def get_node_memory_info(self, node: int) -> Dict[str, Any]:
        """
        获取节点内存信息

        Args:
            node: NUMA节点ID

        Returns:
            Dict: 内存信息
        """
        numa_node = self._nodes.get(node)
        if numa_node is None:
            return {}

        return {
            'node_id': numa_node.node_id,
            'state': numa_node.state.value,
            'total_memory': numa_node.total_memory,
            'free_memory': numa_node.free_memory,
            'used_memory': numa_node.used_memory,
            'utilization': numa_node.utilization,
            'available_ratio': numa_node.available_ratio,
            'cpu_count': len(numa_node.cpus)
        }

    def select_node(self, request: AllocationRequest) -> int:
        """
        选择最优NUMA节点

        Args:
            request: 分配请求

        Returns:
            int: 选择的节点ID
        """
        if request.numa_node >= 0 and request.numa_node in self._nodes:
            return request.numa_node

        policy = self.config.policy

        if policy == NUMAPolicy.LOCAL:
            return self._select_local_node()
        elif policy == NUMAPolicy.INTERLEAVE:
            return self._select_interleave_node()
        elif policy == NUMAPolicy.PREFERRED:
            return self._select_preferred_node()
        elif policy == NUMAPolicy.BIND:
            return self._select_bind_node()
        else:
            return self._select_best_node(request)

    def _select_local_node(self) -> int:
        """选择本地节点"""
        return 0

    def _select_interleave_node(self) -> int:
        """选择交错节点"""
        nodes = list(self._nodes.keys())
        if not nodes:
            return 0

        idx = int(time.time() * 1000) % len(nodes)
        return nodes[idx]

    def _select_preferred_node(self) -> int:
        """选择首选节点"""
        if self.config.preferred_node >= 0 and self.config.preferred_node in self._nodes:
            return self.config.preferred_node
        return self._select_best_node(AllocationRequest(size=0))

    def _select_bind_node(self) -> int:
        """选择绑定节点"""
        return self._select_best_node(AllocationRequest(size=0))

    def _select_best_node(self, request: AllocationRequest) -> int:
        """
        选择最佳节点

        Args:
            request: 分配请求

        Returns:
            int: 节点ID
        """
        best_node = None
        best_score = float('-inf')

        for node_id, node in self._nodes.items():
            if node.state != NodeState.ONLINE:
                continue

            if node.free_memory < request.size:
                continue

            score = self._calculate_node_score(node, request)

            if score > best_score:
                best_score = score
                best_node = node_id

        return best_node if best_node is not None else 0

    def _calculate_node_score(self, node: NUMANode, request: AllocationRequest) -> float:
        """
        计算节点评分

        Args:
            node: 节点
            request: 请求

        Returns:
            float: 评分
        """
        score = 0.0

        available_ratio = node.available_ratio
        score += available_ratio * 100

        utilization = node.utilization
        score -= utilization * 50

        allocated = self._distribution.allocations.get(node.node_id, 0)
        if self._distribution.total_allocated > 0:
            balance = 1.0 - (allocated / self._distribution.total_allocated)
            score += balance * 30

        return score

    def migrate(self, address: int, size: int, target_node: int) -> bool:
        """
        迁移内存到目标节点

        Args:
            address: 内存地址
            size: 内存大小
            target_node: 目标节点

        Returns:
            bool: 是否成功
        """
        if not self._is_available:
            return False

        if target_node not in self._nodes:
            return False

        target = self._nodes[target_node]
        if target.free_memory < size:
            return False

        self._migration_count += 1

        return True

    def get_interleave_policy(self) -> Dict[str, Any]:
        """
        获取交错策略

        Returns:
            Dict: 交错策略配置
        """
        interleave_nodes = self.config.interleave_nodes

        if not interleave_nodes:
            interleave_nodes = list(self._nodes.keys())

        return {
            'policy': 'interleave',
            'nodes': interleave_nodes,
            'current_index': int(time.time() * 1000) % max(len(interleave_nodes), 1)
        }

    def record_allocation(self, node: int, size: int):
        """
        记录分配

        Args:
            node: 节点ID
            size: 分配大小
        """
        with self._lock:
            self._distribution.record(node, size)

            if node in self._nodes:
                self._nodes[node].used_memory += size
                self._nodes[node].free_memory -= size

    def record_deallocation(self, node: int, size: int):
        """
        记录释放

        Args:
            node: 节点ID
            size: 释放大小
        """
        with self._lock:
            if node in self._distribution.allocations:
                self._distribution.allocations[node] = max(
                    0, self._distribution.allocations[node] - size
                )
                self._distribution.total_allocated = max(
                    0, self._distribution.total_allocated - size
                )

            if node in self._nodes:
                self._nodes[node].used_memory = max(
                    0, self._nodes[node].used_memory - size
                )
                self._nodes[node].free_memory = min(
                    self._nodes[node].total_memory,
                    self._nodes[node].free_memory + size
                )

    def balance_memory(self) -> Dict[str, Any]:
        """
        平衡内存分布

        Returns:
            Dict: 平衡结果
        """
        with self._lock:
            balance_score = self._distribution.get_balance_score()

            if balance_score > 0.8:
                return {
                    'balanced': True,
                    'score': balance_score,
                    'migrations': 0
                }

            migrations = []

            node_allocations = [
                (node_id, self._distribution.allocations.get(node_id, 0))
                for node_id in self._nodes.keys()
            ]

            node_allocations.sort(key=lambda x: x[1])

            if len(node_allocations) >= 2:
                low_node, low_alloc = node_allocations[0]
                high_node, high_alloc = node_allocations[-1]

                if high_alloc > low_alloc * 2:
                    migrate_size = (high_alloc - low_alloc) // 2
                    migrations.append({
                        'from_node': high_node,
                        'to_node': low_node,
                        'size': migrate_size
                    })

            self._balance_count += 1

            return {
                'balanced': len(migrations) == 0,
                'score': balance_score,
                'migrations': migrations
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        with self._lock:
            return {
                'available': self._is_available,
                'node_count': len(self._nodes),
                'nodes': {
                    node_id: self.get_node_memory_info(node_id)
                    for node_id in self._nodes.keys()
                },
                'distribution': {
                    'allocations': dict(self._distribution.allocations),
                    'total_allocated': self._distribution.total_allocated,
                    'balance_score': self._distribution.get_balance_score()
                },
                'migration_count': self._migration_count,
                'balance_count': self._balance_count,
                'policy': self.config.policy.value
            }

    def is_available(self) -> bool:
        """
        检查NUMA是否可用

        Returns:
            bool: 是否可用
        """
        return self._is_available

    def get_node_for_address(self, address: int) -> int:
        """
        获取地址所在的NUMA节点

        Args:
            address: 内存地址

        Returns:
            int: 节点ID
        """
        return 0
