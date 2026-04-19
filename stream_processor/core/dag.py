"""
stream_processor DAG任务图

定义流处理的任务拓扑结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Iterator
from collections import defaultdict
import threading

from .exceptions import CyclicDependencyError, InvalidOperatorError


@dataclass
class DAGNode:
    """
    DAG节点

    表示任务图中的一个节点。
    """

    node_id: str

    name: str

    operator_type: str

    parallelism: int = 1

    config: Dict[str, Any] = field(default_factory=dict)

    upstream_nodes: Set[str] = field(default_factory=set)

    downstream_nodes: Set[str] = field(default_factory=set)

    def add_upstream(self, node_id: str):
        """添加上游节点"""
        self.upstream_nodes.add(node_id)

    def add_downstream(self, node_id: str):
        """添加下游节点"""
        self.downstream_nodes.add(node_id)

    def remove_upstream(self, node_id: str):
        """移除上游节点"""
        self.upstream_nodes.discard(node_id)

    def remove_downstream(self, node_id: str):
        """移除下游节点"""
        self.downstream_nodes.discard(node_id)

    def get_upstream_nodes(self) -> Set[str]:
        """获取上游节点"""
        return self.upstream_nodes.copy()

    def get_downstream_nodes(self) -> Set[str]:
        """获取下游节点"""
        return self.downstream_nodes.copy()

    def is_source(self) -> bool:
        """是否为源节点"""
        return len(self.upstream_nodes) == 0

    def is_sink(self) -> bool:
        """是否为汇节点"""
        return len(self.downstream_nodes) == 0


@dataclass
class DAGEdge:
    """
    DAG边

    表示节点之间的连接。
    """

    source_id: str

    target_id: str

    partition_strategy: str = "forward"

    key_selector: Optional[Callable] = None


class DAG:
    """
    DAG任务图

    表示流处理任务的拓扑结构。
    """

    def __init__(self, name: str = "default"):
        """
        初始化DAG

        Args:
            name: DAG名称
        """
        self.name = name
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: Dict[str, List[DAGEdge]] = defaultdict(list)
        self._node_counter = 0
        self._lock = threading.Lock()

    def add_node(self,
                 name: str,
                 operator_type: str,
                 parallelism: int = 1,
                 config: Optional[Dict[str, Any]] = None) -> str:
        """
        添加节点

        Args:
            name: 节点名称
            operator_type: 操作符类型
            parallelism: 并行度
            config: 配置

        Returns:
            str: 节点ID
        """
        with self._lock:
            self._node_counter += 1
            node_id = f"{name}_{self._node_counter}"

            node = DAGNode(
                node_id=node_id,
                name=name,
                operator_type=operator_type,
                parallelism=parallelism,
                config=config or {}
            )

            self._nodes[node_id] = node
            return node_id

    def remove_node(self, node_id: str):
        """
        移除节点

        Args:
            node_id: 节点ID
        """
        with self._lock:
            if node_id not in self._nodes:
                return

            node = self._nodes[node_id]

            for upstream_id in node.upstream_nodes:
                if upstream_id in self._nodes:
                    self._nodes[upstream_id].remove_downstream(node_id)

            for downstream_id in node.downstream_nodes:
                if downstream_id in self._nodes:
                    self._nodes[downstream_id].remove_upstream(node_id)

            del self._nodes[node_id]

            if node_id in self._edges:
                del self._edges[node_id]

    def add_edge(self,
                 source_id: str,
                 target_id: str,
                 partition_strategy: str = "forward",
                 key_selector: Optional[Callable] = None):
        """
        添加边

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            partition_strategy: 分区策略
            key_selector: 键选择器

        Raises:
            InvalidOperatorError: 节点不存在时抛出
            CyclicDependencyError: 存在循环依赖时抛出
        """
        with self._lock:
            if source_id not in self._nodes:
                raise InvalidOperatorError(f"Source node {source_id} not found")
            if target_id not in self._nodes:
                raise InvalidOperatorError(f"Target node {target_id} not found")

            if self._would_create_cycle(source_id, target_id):
                raise CyclicDependencyError(
                    f"Adding edge {source_id} -> {target_id} would create a cycle"
                )

            edge = DAGEdge(
                source_id=source_id,
                target_id=target_id,
                partition_strategy=partition_strategy,
                key_selector=key_selector
            )

            self._edges[source_id].append(edge)
            self._nodes[source_id].add_downstream(target_id)
            self._nodes[target_id].add_upstream(source_id)

    def remove_edge(self, source_id: str, target_id: str):
        """
        移除边

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
        """
        with self._lock:
            if source_id in self._edges:
                self._edges[source_id] = [
                    edge for edge in self._edges[source_id]
                    if edge.target_id != target_id
                ]

            if source_id in self._nodes:
                self._nodes[source_id].remove_downstream(target_id)

            if target_id in self._nodes:
                self._nodes[target_id].remove_upstream(source_id)

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        """
        获取节点

        Args:
            node_id: 节点ID

        Returns:
            Optional[DAGNode]: 节点
        """
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[DAGNode]:
        """
        获取所有节点

        Returns:
            List[DAGNode]: 节点列表
        """
        return list(self._nodes.values())

    def get_edges(self) -> List[DAGEdge]:
        """
        获取所有边

        Returns:
            List[DAGEdge]: 边列表
        """
        edges = []
        for edge_list in self._edges.values():
            edges.extend(edge_list)
        return edges

    def get_source_nodes(self) -> List[DAGNode]:
        """
        获取源节点

        Returns:
            List[DAGNode]: 源节点列表
        """
        return [node for node in self._nodes.values() if node.is_source()]

    def get_sink_nodes(self) -> List[DAGNode]:
        """
        获取汇节点

        Returns:
            List[DAGNode]: 汇节点列表
        """
        return [node for node in self._nodes.values() if node.is_sink()]

    def get_upstream_nodes(self, node_id: str) -> List[DAGNode]:
        """
        获取上游节点

        Args:
            node_id: 节点ID

        Returns:
            List[DAGNode]: 上游节点列表
        """
        node = self.get_node(node_id)
        if not node:
            return []

        return [
            self._nodes[upstream_id]
            for upstream_id in node.upstream_nodes
            if upstream_id in self._nodes
        ]

    def get_downstream_nodes(self, node_id: str) -> List[DAGNode]:
        """
        获取下游节点

        Args:
            node_id: 节点ID

        Returns:
            List[DAGNode]: 下游节点列表
        """
        node = self.get_node(node_id)
        if not node:
            return []

        return [
            self._nodes[downstream_id]
            for downstream_id in node.downstream_nodes
            if downstream_id in self._nodes
        ]

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """
        检查添加边是否会创建循环

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID

        Returns:
            bool: 是否会创建循环
        """
        visited = set()
        stack = [target_id]

        while stack:
            current = stack.pop()

            if current == source_id:
                return True

            if current in visited:
                continue

            visited.add(current)

            if current in self._nodes:
                for downstream_id in self._nodes[current].downstream_nodes:
                    stack.append(downstream_id)

        return False

    def topological_sort(self) -> List[str]:
        """
        拓扑排序

        Returns:
            List[str]: 拓扑排序后的节点ID列表
        """
        in_degree = {node_id: 0 for node_id in self._nodes}

        for node in self._nodes.values():
            for downstream_id in node.downstream_nodes:
                if downstream_id in in_degree:
                    in_degree[downstream_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            if current in self._nodes:
                for downstream_id in self._nodes[current].downstream_nodes:
                    if downstream_id in in_degree:
                        in_degree[downstream_id] -= 1
                        if in_degree[downstream_id] == 0:
                            queue.append(downstream_id)

        if len(result) != len(self._nodes):
            raise CyclicDependencyError("DAG contains a cycle")

        return result

    def get_execution_plan(self) -> List[List[str]]:
        """
        获取执行计划（分层）

        Returns:
            List[List[str]]: 分层的节点ID列表
        """
        levels: List[List[str]] = []
        assigned = set()

        nodes = self.topological_sort()

        while len(assigned) < len(nodes):
            current_level = []

            for node_id in nodes:
                if node_id in assigned:
                    continue

                node = self._nodes[node_id]
                upstream_assigned = all(
                    upstream_id in assigned
                    for upstream_id in node.upstream_nodes
                )

                if upstream_assigned or node.is_source():
                    current_level.append(node_id)
                    assigned.add(node_id)

            if current_level:
                levels.append(current_level)

        return levels

    def validate(self) -> bool:
        """
        验证DAG

        Returns:
            bool: 是否有效

        Raises:
            InvalidOperatorError: DAG无效时抛出
        """
        if not self._nodes:
            raise InvalidOperatorError("DAG has no nodes")

        source_nodes = self.get_source_nodes()
        if not source_nodes:
            raise InvalidOperatorError("DAG has no source nodes")

        sink_nodes = self.get_sink_nodes()
        if not sink_nodes:
            raise InvalidOperatorError("DAG has no sink nodes")

        try:
            self.topological_sort()
        except CyclicDependencyError as e:
            raise InvalidOperatorError(f"DAG validation failed: {e}")

        return True

    def copy(self) -> 'DAG':
        """
        复制DAG

        Returns:
            DAG: DAG副本
        """
        new_dag = DAG(name=f"{self.name}_copy")

        for node in self._nodes.values():
            new_dag._nodes[node.node_id] = DAGNode(
                node_id=node.node_id,
                name=node.name,
                operator_type=node.operator_type,
                parallelism=node.parallelism,
                config=node.config.copy(),
                upstream_nodes=node.upstream_nodes.copy(),
                downstream_nodes=node.downstream_nodes.copy()
            )

        for source_id, edges in self._edges.items():
            new_dag._edges[source_id] = [
                DAGEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    partition_strategy=edge.partition_strategy,
                    key_selector=edge.key_selector
                )
                for edge in edges
            ]

        new_dag._node_counter = self._node_counter

        return new_dag

    def __len__(self) -> int:
        """获取节点数量"""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """检查节点是否存在"""
        return node_id in self._nodes

    def __iter__(self) -> Iterator[DAGNode]:
        """迭代节点"""
        return iter(self._nodes.values())
