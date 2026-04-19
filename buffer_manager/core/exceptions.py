"""
异常定义模块

定义缓冲区管理器中使用的所有异常类型。
"""


class BufferManagerError(Exception):
    """
    缓冲区管理器基础异常类
    
    注意：此类原名为 BufferError，为避免与 Python 内置异常同名而重命名。
    """
    pass


class BufferFullError(BufferManagerError):
    """缓冲区已满异常"""
    def __init__(self, message: str = "Buffer is full"):
        super().__init__(message)


class BufferEmptyError(BufferManagerError):
    """缓冲区为空异常"""
    def __init__(self, message: str = "Buffer is empty"):
        super().__init__(message)


class BufferTimeoutError(BufferManagerError):
    """缓冲区操作超时异常"""
    def __init__(self, message: str = "Buffer operation timeout"):
        super().__init__(message)


class PoolExhaustedError(BufferManagerError):
    """缓冲池耗尽异常"""
    def __init__(self, message: str = "Buffer pool exhausted"):
        super().__init__(message)


class InvalidAlignmentError(BufferManagerError):
    """无效对齐参数异常"""
    def __init__(self, alignment: int):
        super().__init__(f"Invalid alignment: {alignment}. Must be power of 2.")


class InvalidCapacityError(BufferManagerError):
    """无效容量参数异常"""
    def __init__(self, capacity: int):
        super().__init__(f"Invalid capacity: {capacity}. Must be positive.")


class RingBufferCapacityError(BufferManagerError):
    """环形缓冲区容量错误"""
    def __init__(self, capacity: int):
        super().__init__(f"Ring buffer capacity must be power of 2, got: {capacity}")


# 向后兼容别名
BufferError = BufferManagerError
