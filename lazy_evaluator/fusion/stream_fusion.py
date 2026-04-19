"""
流融合优化模块

实现流操作的融合优化，减少中间数据结构。
"""

from typing import Callable, List, Any, Optional, Tuple
import inspect
from ..core.exceptions import FusionError


class StreamFusion:
    """
    流操作融合优化

    该类实现了流操作的融合优化，将多个连续的操作合并为单个操作，
    减少中间数据结构的创建，提高性能。

    学术支撑：
    - Stream Fusion - 流操作融合优化

    Attributes:
        _optimization_rules: 优化规则字典

    Example:
        >>> fusion = StreamFusion()
        >>> operations = [lambda x: x * 2, lambda x: x + 1]
        >>> fused = fusion.fuse(operations)
        >>> result = fused(5)  # 返回 11
    """

    def __init__(self):
        """初始化StreamFusion"""
        self._optimization_rules = {
            'map_map': self._fuse_map_map,
            'filter_filter': self._fuse_filter_filter,
            'map_filter': self._fuse_map_filter,
            'filter_map': self._fuse_filter_map,
        }

    def fuse(self, operations: List[Callable]) -> Callable:
        """
        融合多个操作为单个操作

        Args:
            operations: 操作列表

        Returns:
            Callable: 融合后的操作

        Raises:
            FusionError: 如果操作列表为空
        """
        if not operations:
            raise FusionError(
                operation="fuse",
                reason="Operations list is empty"
            )

        if len(operations) == 1:
            return operations[0]

        # 尝试应用优化规则
        fused_ops = list(operations)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(fused_ops) - 1:
                op1 = fused_ops[i]
                op2 = fused_ops[i + 1]

                # 尝试融合相邻的两个操作
                fused = self._try_fuse_pair(op1, op2)
                if fused is not None:
                    fused_ops[i] = fused
                    fused_ops.pop(i + 1)
                    changed = True
                else:
                    i += 1

        # 如果还有多个操作，创建组合函数
        if len(fused_ops) == 1:
            return fused_ops[0]

        return self._create_composed_function(fused_ops)

    def optimize(self, pipeline: List[str]) -> List[str]:
        """
        优化管道操作序列

        Args:
            pipeline: 操作名称列表

        Returns:
            List[str]: 优化后的操作名称列表
        """
        if not pipeline:
            return pipeline

        optimized = list(pipeline)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(optimized) - 1:
                op1 = optimized[i]
                op2 = optimized[i + 1]

                # 检查是否可以融合
                if self._can_fuse_operations(op1, op2):
                    # 创建融合后的操作名称
                    fused_name = f"{op1}_{op2}"
                    optimized[i] = fused_name
                    optimized.pop(i + 1)
                    changed = True
                else:
                    i += 1

        return optimized

    def _try_fuse_pair(self, op1: Callable, op2: Callable) -> Optional[Callable]:
        """
        尝试融合两个操作

        Args:
            op1: 第一个操作
            op2: 第二个操作

        Returns:
            Optional[Callable]: 融合后的操作，如果无法融合返回None
        """
        # 获取操作类型
        op1_type = self._get_operation_type(op1)
        op2_type = self._get_operation_type(op2)

        # 查找对应的融合规则
        rule_key = f"{op1_type}_{op2_type}"
        if rule_key in self._optimization_rules:
            return self._optimization_rules[rule_key](op1, op2)

        return None

    def _get_operation_type(self, op: Callable) -> str:
        """
        获取操作类型

        Args:
            op: 操作函数

        Returns:
            str: 操作类型（'map', 'filter', 'unknown'）
        """
        # 简单的类型推断
        # 在实际应用中，可以使用更复杂的分析方法
        if hasattr(op, '__name__'):
            name = op.__name__.lower()
            if 'map' in name:
                return 'map'
            elif 'filter' in name:
                return 'filter'

        # 默认返回map
        return 'map'

    def _can_fuse_operations(self, op1_name: str, op2_name: str) -> bool:
        """
        检查两个操作是否可以融合

        Args:
            op1_name: 第一个操作名称
            op2_name: 第二个操作名称

        Returns:
            bool: 如果可以融合返回True，否则返回False
        """
        # 简单的融合规则
        fusable_pairs = [
            ('map', 'map'),
            ('filter', 'filter'),
            ('map', 'filter'),
            ('filter', 'map'),
        ]

        return (op1_name, op2_name) in fusable_pairs

    def _fuse_map_map(self, op1: Callable, op2: Callable) -> Callable:
        """
        融合两个map操作

        Args:
            op1: 第一个map操作
            op2: 第二个map操作

        Returns:
            Callable: 融合后的操作
        """
        def fused(x):
            return op2(op1(x))

        fused.__name__ = 'fused_map_map'
        return fused

    def _fuse_filter_filter(self, op1: Callable, op2: Callable) -> Callable:
        """
        融合两个filter操作

        Args:
            op1: 第一个filter操作
            op2: 第二个filter操作

        Returns:
            Callable: 融合后的操作
        """
        def fused(x):
            return op1(x) and op2(x)

        fused.__name__ = 'fused_filter_filter'
        return fused

    def _fuse_map_filter(self, op1: Callable, op2: Callable) -> Callable:
        """
        融合map和filter操作

        Args:
            op1: map操作
            op2: filter操作

        Returns:
            Callable: 融合后的操作
        """
        def fused(x):
            mapped = op1(x)
            if op2(mapped):
                return mapped
            return None  # 使用None表示过滤

        fused.__name__ = 'fused_map_filter'
        return fused

    def _fuse_filter_map(self, op1: Callable, op2: Callable) -> Callable:
        """
        融合filter和map操作

        Args:
            op1: filter操作
            op2: map操作

        Returns:
            Callable: 融合后的操作
        """
        def fused(x):
            if op1(x):
                return op2(x)
            return None  # 使用None表示过滤

        fused.__name__ = 'fused_filter_map'
        return fused

    def _create_composed_function(self, operations: List[Callable]) -> Callable:
        """
        创建组合函数

        Args:
            operations: 操作列表

        Returns:
            Callable: 组合函数
        """
        def composed(x):
            result = x
            for op in operations:
                result = op(result)
                if result is None:
                    break
            return result

        composed.__name__ = 'composed'
        return composed

    def __repr__(self) -> str:
        """字符串表示"""
        return f"StreamFusion(rules={len(self._optimization_rules)})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
