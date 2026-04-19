"""
stream_processor 工具层

提供通用工具函数。
"""

from .serialization import (
    Serializer,
    JsonSerializer,
    PickleSerializer,
    BinarySerializer,
    CompositeSerializer,
    SerializationContext,
    serialize,
    deserialize,
)

from .metrics import (
    Metric,
    Counter,
    Gauge,
    Histogram,
    Meter,
    MetricType,
    MetricValue,
    MetricsRegistry,
    MetricsCollector,
)

from .helpers import (
    generate_id,
    generate_short_id,
    hash_value,
    timestamp_ms,
    timestamp_s,
    format_duration,
    format_bytes,
    chunk_list,
    flatten_list,
    merge_dicts,
    retry,
    timeout,
    Singleton,
    Lazy,
    RateCalculator,
    validate_config,
)

__all__ = [
    'Serializer',
    'JsonSerializer',
    'PickleSerializer',
    'BinarySerializer',
    'CompositeSerializer',
    'SerializationContext',
    'serialize',
    'deserialize',
    'Metric',
    'Counter',
    'Gauge',
    'Histogram',
    'Meter',
    'MetricType',
    'MetricValue',
    'MetricsRegistry',
    'MetricsCollector',
    'generate_id',
    'generate_short_id',
    'hash_value',
    'timestamp_ms',
    'timestamp_s',
    'format_duration',
    'format_bytes',
    'chunk_list',
    'flatten_list',
    'merge_dicts',
    'retry',
    'timeout',
    'Singleton',
    'Lazy',
    'RateCalculator',
    'validate_config',
]
