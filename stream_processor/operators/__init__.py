"""
stream_processor 操作符层

提供流处理的各种操作符。
"""

from .base import (
    Operator,
    OneInputOperator,
    TwoInputOperator,
    OperatorConfig,
    OperatorMetrics,
    OperatorType,
    OperatorState,
)

from .source import (
    SourceOperator,
    CollectionSource,
    IteratorSource,
    FunctionSource,
    FileSource,
    SocketSource,
    KafkaSource,
)

from .transform import (
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    KeyByOperator,
    ReduceOperator,
    AggregateOperator,
    UnionOperator,
    ProcessFunctionOperator,
    ProcessContext,
)

from .sink import (
    SinkOperator,
    PrintSink,
    FileSink,
    FunctionSink,
    CollectionSink,
    KafkaSink,
    DatabaseSink,
)

from .window import (
    WindowOperator,
    WindowContext,
    TumblingWindowOperator,
    SlidingWindowOperator,
    SessionWindowOperator,
    CountWindowOperator,
)

from .compression import (
    CompressionOperator,
    DecompressionOperator,
    StreamCompressionOperator,
)

__all__ = [
    'Operator',
    'OneInputOperator',
    'TwoInputOperator',
    'OperatorConfig',
    'OperatorMetrics',
    'OperatorType',
    'OperatorState',
    'SourceOperator',
    'CollectionSource',
    'IteratorSource',
    'FunctionSource',
    'FileSource',
    'SocketSource',
    'KafkaSource',
    'MapOperator',
    'FilterOperator',
    'FlatMapOperator',
    'KeyByOperator',
    'ReduceOperator',
    'AggregateOperator',
    'UnionOperator',
    'ProcessFunctionOperator',
    'ProcessContext',
    'SinkOperator',
    'PrintSink',
    'FileSink',
    'FunctionSink',
    'CollectionSink',
    'KafkaSink',
    'DatabaseSink',
    'WindowOperator',
    'WindowContext',
    'TumblingWindowOperator',
    'SlidingWindowOperator',
    'SessionWindowOperator',
    'CountWindowOperator',
    'CompressionOperator',
    'DecompressionOperator',
    'StreamCompressionOperator',
]
