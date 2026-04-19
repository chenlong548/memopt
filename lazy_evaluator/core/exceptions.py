"""
异常定义模块 (Exceptions Module)

定义惰性计算相关的异常类型。
"""


class LazyEvaluationError(Exception):
    """惰性计算基础异常"""

    def __init__(self, message: str = "Lazy evaluation error"):
        self.message = message
        super().__init__(self.message)


class CircularDependencyError(LazyEvaluationError):
    """循环依赖异常"""

    def __init__(self, cycle_path: list = None):
        self.cycle_path = cycle_path or []
        message = f"Circular dependency detected: {' -> '.join(map(str, self.cycle_path))}"
        super().__init__(message)


class ThunkEvaluationError(LazyEvaluationError):
    """Thunk求值异常"""

    def __init__(self, thunk_id: str = None, reason: str = None):
        self.thunk_id = thunk_id
        self.reason = reason
        message = f"Thunk evaluation error"
        if thunk_id:
            message += f" for thunk '{thunk_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class CacheError(LazyEvaluationError):
    """缓存相关异常"""

    def __init__(self, operation: str = None, key: str = None, reason: str = None):
        self.operation = operation
        self.key = key
        self.reason = reason
        message = "Cache error"
        if operation:
            message += f" during {operation}"
        if key:
            message += f" for key '{key}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class DependencyGraphError(LazyEvaluationError):
    """依赖图相关异常"""

    def __init__(self, node_id: str = None, reason: str = None):
        self.node_id = node_id
        self.reason = reason
        message = "Dependency graph error"
        if node_id:
            message += f" for node '{node_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class FusionError(LazyEvaluationError):
    """融合优化异常"""

    def __init__(self, operation: str = None, reason: str = None):
        self.operation = operation
        self.reason = reason
        message = "Fusion error"
        if operation:
            message += f" during {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
