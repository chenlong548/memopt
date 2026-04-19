"""
评估上下文模块 (Evaluation Context Module)

提供惰性计算的上下文管理，支持批量求值和依赖追踪。
"""

from typing import Dict, Set, Optional, Any, List, Deque
from collections import deque
import threading
from .lazy import Lazy, ThunkState
from .exceptions import LazyEvaluationError, CircularDependencyError


class EvaluationContext:
    """
    评估上下文，管理惰性值的批量求值（线程安全）

    该类提供了惰性计算的上下文环境，支持：
    - 注册和管理多个惰性值
    - 批量求值
    - 依赖追踪
    - 循环依赖检测
    
    线程安全：使用RLock保护所有状态操作。

    Attributes:
        _lazy_values: 注册的惰性值字典
        _dependencies: 依赖关系图
        _evaluation_stack: 当前求值栈（用于检测循环依赖）
        _lock: 线程锁

    Example:
        >>> ctx = EvaluationContext()
        >>> lazy_val = Lazy(lambda: 42)
        >>> ctx.register("value1", lazy_val)
        >>> result = ctx.evaluate("value1")
    """

    def __init__(self):
        """初始化评估上下文"""
        self._lazy_values: Dict[str, Lazy] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._evaluation_stack: List[str] = []
        self._lock = threading.RLock()

    def register(self, name: str, lazy_value: Lazy) -> None:
        """
        注册惰性值（线程安全）

        Args:
            name: 惰性值名称
            lazy_value: Lazy实例

        Raises:
            LazyEvaluationError: 如果名称已存在
        """
        with self._lock:
            if name in self._lazy_values:
                raise LazyEvaluationError(f"Lazy value '{name}' already registered")
            self._lazy_values[name] = lazy_value
            self._dependencies[name] = set()

    def unregister(self, name: str) -> None:
        """
        注销惰性值（线程安全）

        Args:
            name: 惰性值名称
        """
        with self._lock:
            if name in self._lazy_values:
                del self._lazy_values[name]
                del self._dependencies[name]
                # 清除其他节点对该节点的依赖
                for deps in self._dependencies.values():
                    deps.discard(name)

    def get(self, name: str) -> Optional[Lazy]:
        """
        获取惰性值（线程安全）

        Args:
            name: 惰性值名称

        Returns:
            Optional[Lazy]: Lazy实例，如果不存在返回None
        """
        with self._lock:
            return self._lazy_values.get(name)

    def add_dependency(self, from_name: str, to_name: str) -> None:
        """
        添加依赖关系（线程安全）

        Args:
            from_name: 依赖方名称
            to_name: 被依赖方名称

        Raises:
            LazyEvaluationError: 如果节点不存在
        """
        with self._lock:
            if from_name not in self._lazy_values:
                raise LazyEvaluationError(f"Lazy value '{from_name}' not found")
            if to_name not in self._lazy_values:
                raise LazyEvaluationError(f"Lazy value '{to_name}' not found")

            self._dependencies[from_name].add(to_name)

    def evaluate(self, name: str) -> Any:
        """
        求值指定的惰性值（线程安全）

        Args:
            name: 惰性值名称

        Returns:
            Any: 计算结果

        Raises:
            LazyEvaluationError: 如果节点不存在
            CircularDependencyError: 如果检测到循环依赖
        """
        with self._lock:
            if name not in self._lazy_values:
                raise LazyEvaluationError(f"Lazy value '{name}' not found")

            # 检测循环依赖
            if name in self._evaluation_stack:
                cycle = self._evaluation_stack + [name]
                raise CircularDependencyError(cycle)

            lazy_value = self._lazy_values[name]

            # 如果已经求值，直接返回
            if lazy_value.is_evaluated():
                return lazy_value.force()

            # 添加到求值栈
            self._evaluation_stack.append(name)

        # 在锁外执行计算，避免死锁
        try:
            # 先求值依赖
            with self._lock:
                deps = self._dependencies[name].copy()
            
            for dep_name in deps:
                with self._lock:
                    if not self._lazy_values[dep_name].is_evaluated():
                        pass  # 将在递归调用中处理
                self.evaluate(dep_name)

            # 求值当前节点
            result = lazy_value.force()
            return result
        finally:
            # 从求值栈移除
            with self._lock:
                self._evaluation_stack.pop()

    def evaluate_all(self) -> Dict[str, Any]:
        """
        求值所有惰性值（线程安全）

        Returns:
            Dict[str, Any]: 名称到结果的映射

        Raises:
            CircularDependencyError: 如果检测到循环依赖
        """
        with self._lock:
            names = list(self._lazy_values.keys())
        
        results = {}
        for name in names:
            results[name] = self.evaluate(name)
        return results

    def reset(self, name: Optional[str] = None) -> None:
        """
        重置惰性值状态（线程安全）

        Args:
            name: 惰性值名称，如果为None则重置所有
        """
        with self._lock:
            if name is not None:
                if name in self._lazy_values:
                    self._lazy_values[name].reset()
            else:
                for lazy_value in self._lazy_values.values():
                    lazy_value.reset()

    def get_dependencies(self, name: str) -> Set[str]:
        """
        获取依赖关系（线程安全）

        Args:
            name: 惰性值名称

        Returns:
            Set[str]: 依赖的惰性值名称集合
        """
        with self._lock:
            return self._dependencies.get(name, set()).copy()

    def get_dependents(self, name: str) -> Set[str]:
        """
        获取被依赖关系（反向依赖）（线程安全）

        Args:
            name: 惰性值名称

        Returns:
            Set[str]: 依赖该惰性值的名称集合
        """
        with self._lock:
            dependents = set()
            for node_name, deps in self._dependencies.items():
                if name in deps:
                    dependents.add(node_name)
            return dependents

    def detect_cycle(self) -> Optional[List[str]]:
        """
        检测循环依赖（线程安全）

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

                for neighbor in self._dependencies.get(node, set()):
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

            for node in self._lazy_values:
                if node not in visited:
                    result = dfs(node)
                    if result:
                        return result

            return None

    def get_evaluation_order(self) -> List[str]:
        """
        获取求值顺序（拓扑排序）（线程安全，优化版）

        使用Kahn算法进行拓扑排序，时间复杂度O(V+E)。
        使用deque替代list，将pop(0)操作从O(n)优化为O(1)。

        Returns:
            List[str]: 求值顺序列表

        Raises:
            CircularDependencyError: 如果存在循环依赖
        """
        with self._lock:
            # 先检测循环依赖
            cycle = self.detect_cycle()
            if cycle:
                raise CircularDependencyError(cycle)

            # 拓扑排序
            in_degree = {node: 0 for node in self._lazy_values}
            for node in self._dependencies:
                for dep in self._dependencies[node]:
                    in_degree[node] += 1

            # 使用Kahn算法，使用deque优化
            queue: Deque[str] = deque(node for node, degree in in_degree.items() if degree == 0)
            result = []

            while queue:
                # popleft()是O(1)操作
                node = queue.popleft()
                result.append(node)

                for dependent in self.get_dependents(node):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            return result

    def __len__(self) -> int:
        """返回注册的惰性值数量"""
        return len(self._lazy_values)

    def __contains__(self, name: str) -> bool:
        """检查是否包含指定名称的惰性值"""
        return name in self._lazy_values

    def __repr__(self) -> str:
        """字符串表示"""
        return f"EvaluationContext(size={len(self._lazy_values)})"
