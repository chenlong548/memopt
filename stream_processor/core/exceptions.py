"""
stream_processor 核心异常定义

定义流处理器的异常类型。
"""


class StreamProcessorError(Exception):
    """流处理器基础异常"""

    pass


class DAGError(StreamProcessorError):
    """DAG任务图异常"""

    pass


class CyclicDependencyError(DAGError):
    """循环依赖异常"""

    pass


class InvalidOperatorError(DAGError):
    """无效操作符异常"""

    pass


class OperatorError(StreamProcessorError):
    """操作符异常"""

    pass


class SourceError(OperatorError):
    """数据源异常"""

    pass


class TransformError(OperatorError):
    """转换异常"""

    pass


class SinkError(OperatorError):
    """输出异常"""

    pass


class WindowError(OperatorError):
    """窗口异常"""

    pass


class WatermarkError(StreamProcessorError):
    """Watermark异常"""

    pass


class LateDataError(WatermarkError):
    """迟到数据异常"""

    pass


class CheckpointError(StreamProcessorError):
    """检查点异常"""

    pass


class SnapshotError(CheckpointError):
    """快照异常"""

    pass


class RecoveryError(CheckpointError):
    """恢复异常"""

    pass


class StateBackendError(CheckpointError):
    """状态后端异常"""

    pass


class BackpressureError(StreamProcessorError):
    """背压异常"""

    pass


class RateLimitExceededError(BackpressureError):
    """限流超限异常"""

    pass


class BufferOverflowError(BackpressureError):
    """缓冲区溢出异常"""

    pass


class SerializationError(StreamProcessorError):
    """序列化异常"""

    pass


class DeserializationError(SerializationError):
    """反序列化异常"""

    pass


class ExecutionError(StreamProcessorError):
    """执行异常"""

    pass


class TimeoutError(ExecutionError):
    """超时异常"""

    pass


class ConfigurationError(StreamProcessorError):
    """配置异常"""

    pass
