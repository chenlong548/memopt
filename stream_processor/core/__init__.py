"""
stream_processor 核心层

提供流处理的核心抽象和数据结构。
"""

from .exceptions import (
    StreamProcessorError,
    DAGError,
    CyclicDependencyError,
    InvalidOperatorError,
    OperatorError,
    SourceError,
    TransformError,
    SinkError,
    WindowError,
    WatermarkError,
    LateDataError,
    CheckpointError,
    SnapshotError,
    RecoveryError,
    StateBackendError,
    BackpressureError,
    RateLimitExceededError,
    BufferOverflowError,
    SerializationError,
    DeserializationError,
    ExecutionError,
    TimeoutError,
    ConfigurationError,
)

from .record import Record
from .watermark import Watermark, WatermarkGenerator, WatermarkStrategy, WatermarkTracker
from .stream import Stream, StreamRecord, StreamType, KeyedStream
from .execution_context import (
    ExecutionContext,
    ExecutionConfig,
    ExecutionState,
    TaskMetrics,
)
from .dag import DAG, DAGNode, DAGEdge

__all__ = [
    'StreamProcessorError',
    'DAGError',
    'CyclicDependencyError',
    'InvalidOperatorError',
    'OperatorError',
    'SourceError',
    'TransformError',
    'SinkError',
    'WindowError',
    'WatermarkError',
    'LateDataError',
    'CheckpointError',
    'SnapshotError',
    'RecoveryError',
    'StateBackendError',
    'BackpressureError',
    'RateLimitExceededError',
    'BufferOverflowError',
    'SerializationError',
    'DeserializationError',
    'ExecutionError',
    'TimeoutError',
    'ConfigurationError',
    'Record',
    'Watermark',
    'WatermarkGenerator',
    'WatermarkStrategy',
    'WatermarkTracker',
    'Stream',
    'StreamRecord',
    'StreamType',
    'KeyedStream',
    'ExecutionContext',
    'ExecutionConfig',
    'ExecutionState',
    'TaskMetrics',
    'DAG',
    'DAGNode',
    'DAGEdge',
]
