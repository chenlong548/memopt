"""
增量计算器实现模块

实现基于依赖图的增量计算。
"""

from typing import Dict, Set, Any, Optional, Deque
from collections import deque
import threading
from .graph import DependencyGraph
from ..core.exceptions import DependencyGraphError


class IncrementalEvaluator:
    """
    增量计算器，基于依赖图

    该类实现了增量计算机制，当某个节点的值发生变化时，
    只重新计算受影响的节点，而不是重新计算所有节点。

    学术支撑：
    - Self-adjusting computation - 基于依赖图的增量计算

    Attributes:
        _graph: 依赖图
        _values: 节点值缓存
        _dirty: 需要重新计算的节点集合
        _lock: 线程锁

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_node("A", lambda: 1)
        >>> graph.add_node("B", lambda: 2)
        >>> graph.add_edge("A", "B")
        >>> evaluator = IncrementalEvaluator(graph)
        >>> result = evaluator.evaluate("A")
    """

    def __init__(self, graph: DependencyGraph):
        """
        初始化增量计算器

        Args:
            graph: 依赖图实例
        """
        self._graph = graph
        self._values: Dict[str, Any] = {}
        self._dirty: Set[str] = set()
        self._lock = threading.RLock()

    def evaluate(self, node_id: str) -> Any:
        """
        求值指定节点

        如果节点是脏的（需要重新计算），则重新计算。
        否则返回缓存的值。

        Args:
            node_id: 节点ID

        Returns:
            Any: 计算结果

        Raises:
            DependencyGraphError: 如果节点不存在
        """
        with self._lock:
            if node_id not in self._graph:
                raise DependencyGraphError(
                    node_id=node_id,
                    reason="Node not found"
                )

            # 如果节点是脏的，重新计算
            if node_id in self._dirty or node_id not in self._values:
                # 先求值所有依赖
                dependencies = self._graph.get_dependencies(node_id)
                for dep_id in dependencies:
                    if dep_id in self._dirty or dep_id not in self._values:
                        self.evaluate(dep_id)

                # 计算当前节点
                computation = self._graph.get_computation(node_id)
                if computation is not None:
                    self._values[node_id] = computation()

                # 从脏集合中移除
                self._dirty.discard(node_id)

            return self._values[node_id]

    def invalidate(self, node_id: str) -> None:
        """
        使节点及其依赖失效（迭代实现，避免栈溢出）

        当节点的值发生变化时，需要使该节点及其所有依赖节点失效，
        以便下次访问时重新计算。
        
        使用广度优先搜索（BFS）迭代遍历依赖图，避免深度依赖图导致的栈溢出。

        Args:
            node_id: 节点ID
        """
        with self._lock:
            # 使用队列进行BFS遍历，避免递归栈溢出
            queue: Deque[str] = deque()
            visited: Set[str] = set()
            
            queue.append(node_id)
            visited.add(node_id)
            
            while queue:
                current_id = queue.popleft()
                
                # 使当前节点失效
                self._dirty.add(current_id)
                
                # 获取依赖当前节点的所有节点
                dependents = self._graph.get_dependents(current_id)
                for dependent_id in dependents:
                    if dependent_id not in visited:
                        visited.add(dependent_id)
                        queue.append(dependent_id)

    def invalidate_all(self) -> None:
        """使所有节点失效"""
        with self._lock:
            self._dirty = self._graph.get_all_nodes()

    def get_dirty_nodes(self) -> Set[str]:
        """
        获取需要重新计算的节点

        Returns:
            Set[str]: 脏节点集合
        """
        with self._lock:
            return self._dirty.copy()

    def get_value(self, node_id: str) -> Optional[Any]:
        """
        获取节点的缓存值

        Args:
            node_id: 节点ID

        Returns:
            Optional[Any]: 缓存值，如果不存在返回None
        """
        with self._lock:
            return self._values.get(node_id)

    def set_value(self, node_id: str, value: Any) -> None:
        """
        设置节点的值

        Args:
            node_id: 节点ID
            value: 节点值
        """
        with self._lock:
            self._values[node_id] = value
            self._dirty.discard(node_id)

    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self._values.clear()
            self._dirty.clear()

    def is_dirty(self, node_id: str) -> bool:
        """
        检查节点是否是脏的

        Args:
            node_id: 节点ID

        Returns:
            bool: 如果节点需要重新计算返回True，否则返回False
        """
        with self._lock:
            return node_id in self._dirty

    def is_evaluated(self, node_id: str) -> bool:
        """
        检查节点是否已经求值

        Args:
            node_id: 节点ID

        Returns:
            bool: 如果节点已经求值返回True，否则返回False
        """
        with self._lock:
            return node_id in self._values

    def evaluate_all(self) -> Dict[str, Any]:
        """
        求值所有节点

        Returns:
            Dict[str, Any]: 节点ID到值的映射
        """
        with self._lock:
            # 获取拓扑排序
            order = self._graph.topological_sort()

            # 按顺序求值
            for node_id in order:
                if node_id in self._dirty or node_id not in self._values:
                    self.evaluate(node_id)

            return dict(self._values)

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        获取求值统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            return {
                'total_nodes': len(self._graph),
                'evaluated_nodes': len(self._values),
                'dirty_nodes': len(self._dirty),
                'cached_nodes': len(self._values) - len(self._dirty),
            }

    def __repr__(self) -> str:
        """字符串表示"""
        with self._lock:
            return (f"IncrementalEvaluator(nodes={len(self._graph)}, "
                    f"evaluated={len(self._values)}, dirty={len(self._dirty)})")

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
