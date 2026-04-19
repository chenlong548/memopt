"""
依赖图实现模块

实现依赖图数据结构，支持增量计算。
"""

from typing import Dict, Set, List, Optional, Callable, Any, Deque
from collections import deque
import threading
from ..core.exceptions import DependencyGraphError, CircularDependencyError


class DependencyGraph:
    """
    依赖图，支持增量计算

    该类实现了有向无环图（DAG）数据结构，用于管理节点之间的依赖关系。
    支持拓扑排序、循环依赖检测、依赖追踪等功能。

    学术支撑：
    - Self-adjusting computation - 基于依赖图的增量计算

    Attributes:
        _nodes: 节点字典，存储节点ID到计算函数的映射
        _edges: 边字典，存储节点ID到依赖集合的映射
        _reverse_edges: 反向边字典，存储节点ID到被依赖集合的映射
        _lock: 线程锁

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_node("A", lambda: 1)
        >>> graph.add_node("B", lambda: 2)
        >>> graph.add_edge("A", "B")  # A依赖B
        >>> order = graph.topological_sort()
    """

    def __init__(self):
        """初始化依赖图"""
        self._nodes: Dict[str, Callable] = {}
        self._edges: Dict[str, Set[str]] = {}  # node -> set of dependencies
        self._reverse_edges: Dict[str, Set[str]] = {}  # node -> set of dependents
        self._lock = threading.RLock()

    def add_node(self, node_id: str, computation: Callable[[], Any]) -> None:
        """
        添加节点

        Args:
            node_id: 节点ID
            computation: 计算函数

        Raises:
            DependencyGraphError: 如果节点已存在
        """
        with self._lock:
            if node_id in self._nodes:
                raise DependencyGraphError(
                    node_id=node_id,
                    reason="Node already exists"
                )

            self._nodes[node_id] = computation
            self._edges[node_id] = set()
            self._reverse_edges[node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """
        移除节点

        Args:
            node_id: 节点ID

        Raises:
            DependencyGraphError: 如果节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise DependencyGraphError(
                    node_id=node_id,
                    reason="Node not found"
                )

            # 移除所有相关的边
            for dep in self._edges[node_id]:
                self._reverse_edges[dep].discard(node_id)

            for dependent in self._reverse_edges[node_id]:
                self._edges[dependent].discard(node_id)

            # 移除节点
            del self._nodes[node_id]
            del self._edges[node_id]
            del self._reverse_edges[node_id]

    def add_edge(self, from_id: str, to_id: str) -> None:
        """
        添加依赖边

        Args:
            from_id: 依赖方节点ID
            to_id: 被依赖方节点ID

        Raises:
            DependencyGraphError: 如果节点不存在
        """
        with self._lock:
            if from_id not in self._nodes:
                raise DependencyGraphError(
                    node_id=from_id,
                    reason="Node not found"
                )
            if to_id not in self._nodes:
                raise DependencyGraphError(
                    node_id=to_id,
                    reason="Node not found"
                )

            self._edges[from_id].add(to_id)
            self._reverse_edges[to_id].add(from_id)

    def remove_edge(self, from_id: str, to_id: str) -> None:
        """
        移除依赖边

        Args:
            from_id: 依赖方节点ID
            to_id: 被依赖方节点ID
        """
        with self._lock:
            if from_id in self._edges:
                self._edges[from_id].discard(to_id)
            if to_id in self._reverse_edges:
                self._reverse_edges[to_id].discard(from_id)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """
        获取节点的依赖

        Args:
            node_id: 节点ID

        Returns:
            Set[str]: 依赖的节点ID集合

        Raises:
            DependencyGraphError: 如果节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise DependencyGraphError(
                    node_id=node_id,
                    reason="Node not found"
                )
            return self._edges[node_id].copy()

    def get_dependents(self, node_id: str) -> Set[str]:
        """
        获取依赖该节点的节点（反向依赖）

        Args:
            node_id: 节点ID

        Returns:
            Set[str]: 依赖该节点的节点ID集合

        Raises:
            DependencyGraphError: 如果节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise DependencyGraphError(
                    node_id=node_id,
                    reason="Node not found"
                )
            return self._reverse_edges[node_id].copy()

    def get_computation(self, node_id: str) -> Optional[Callable]:
        """
        获取节点的计算函数

        Args:
            node_id: 节点ID

        Returns:
            Optional[Callable]: 计算函数，如果节点不存在返回None
        """
        with self._lock:
            return self._nodes.get(node_id)

    def topological_sort(self) -> List[str]:
        """
        拓扑排序（优化版，使用deque）

        使用Kahn算法进行拓扑排序，时间复杂度O(V+E)。
        使用deque替代list，将pop(0)操作从O(n)优化为O(1)。

        Returns:
            List[str]: 拓扑排序结果

        Raises:
            CircularDependencyError: 如果存在循环依赖
        """
        with self._lock:
            # 先检测循环依赖
            cycle = self.detect_cycle()
            if cycle:
                raise CircularDependencyError(cycle)

            # Kahn算法
            in_degree = {node: 0 for node in self._nodes}
            for node in self._nodes:
                for dep in self._edges[node]:
                    in_degree[node] += 1

            # 使用deque替代list，popleft()是O(1)操作
            queue: Deque[str] = deque(node for node, degree in in_degree.items() if degree == 0)
            result = []

            while queue:
                # popleft()是O(1)操作，而list.pop(0)是O(n)操作
                node = queue.popleft()
                result.append(node)

                for dependent in self._reverse_edges[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            return result

    def detect_cycle(self) -> Optional[List[str]]:
        """
        检测循环依赖

        Returns:
            Optional[List[str]]: 如果存在循环依赖，返回循环路径；否则返回None
        """
        with self._lock:
            visited = set()
            rec_stack = set()
            path = []

            def dfs(node: str) -> Optional[List[str]]:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in self._edges[node]:
                    if neighbor not in visited:
                        result = dfs(neighbor)
                        if result:
                            return result
                    elif neighbor in rec_stack:
                        # 找到循环
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]

                path.pop()
                rec_stack.remove(node)
                return None

            for node in self._nodes:
                if node not in visited:
                    result = dfs(node)
                    if result:
                        return result

            return None

    def get_all_nodes(self) -> Set[str]:
        """
        获取所有节点ID

        Returns:
            Set[str]: 节点ID集合
        """
        with self._lock:
            return set(self._nodes.keys())

    def get_all_edges(self) -> Dict[str, Set[str]]:
        """
        获取所有边

        Returns:
            Dict[str, Set[str]]: 边字典
        """
        with self._lock:
            return {k: v.copy() for k, v in self._edges.items()}

    def clear(self) -> None:
        """清空图"""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._reverse_edges.clear()

    def __len__(self) -> int:
        """返回节点数量"""
        with self._lock:
            return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """检查是否包含指定节点"""
        with self._lock:
            return node_id in self._nodes

    def __repr__(self) -> str:
        """字符串表示"""
        with self._lock:
            edge_count = sum(len(deps) for deps in self._edges.values())
            return f"DependencyGraph(nodes={len(self._nodes)}, edges={edge_count})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
