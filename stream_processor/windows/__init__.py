"""
stream_processor 窗口层

提供流处理的窗口机制。
"""

from .base import (
    Window,
    WindowAssigner,
    Trigger,
    WindowSerializer,
    WindowFunction,
    ReduceWindowFunction,
    AggregateWindowFunction,
    TimeWindowSerializer,
)

from .tumbling import (
    TumblingWindowAssigner,
    TumblingEventTimeWindows,
    TumblingProcessingTimeWindows,
)

from .sliding import (
    SlidingWindowAssigner,
    SlidingEventTimeWindows,
    SlidingProcessingTimeWindows,
)

from .session import (
    SessionWindow,
    SessionWindowAssigner,
    DynamicSessionWindowAssigner,
)

from .count import (
    CountWindow,
    CountWindowAssigner,
    SlidingCountWindowAssigner,
    CountWindowSerializer,
    GlobalWindowAssigner,
)

from .trigger import (
    TriggerResult,
    EventTimeTrigger,
    ProcessingTimeTrigger,
    CountTrigger,
    ContinuousEventTimeTrigger,
    NeverTrigger,
    PurgingTrigger,
    EarlyFiringTrigger,
)

__all__ = [
    'Window',
    'WindowAssigner',
    'Trigger',
    'WindowSerializer',
    'WindowFunction',
    'ReduceWindowFunction',
    'AggregateWindowFunction',
    'TimeWindowSerializer',
    'TumblingWindowAssigner',
    'TumblingEventTimeWindows',
    'TumblingProcessingTimeWindows',
    'SlidingWindowAssigner',
    'SlidingEventTimeWindows',
    'SlidingProcessingTimeWindows',
    'SessionWindow',
    'SessionWindowAssigner',
    'DynamicSessionWindowAssigner',
    'CountWindow',
    'CountWindowAssigner',
    'SlidingCountWindowAssigner',
    'CountWindowSerializer',
    'GlobalWindowAssigner',
    'TriggerResult',
    'EventTimeTrigger',
    'ProcessingTimeTrigger',
    'CountTrigger',
    'ContinuousEventTimeTrigger',
    'NeverTrigger',
    'PurgingTrigger',
    'EarlyFiringTrigger',
]
