"""
stream_processor 窗口操作符

定义窗口操作符。
"""

from typing import Any, Callable, List, Optional, Dict
from dataclasses import dataclass, field
import time

from .base import OneInputOperator, OperatorConfig, OperatorType, OperatorState
from ..core.record import Record
from ..core.execution_context import ExecutionContext
from ..core.watermark import Watermark
from ..core.exceptions import WindowError


@dataclass
class WindowContext:
    """
    窗口上下文

    提供窗口处理的上下文信息。
    """

    window_start: float

    window_end: float

    key: Optional[str] = None

    state: Dict[str, Any] = field(default_factory=dict)

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any):
        """设置状态"""
        self.state[key] = value


class WindowOperator(OneInputOperator):
    """
    窗口操作符基类

    对数据进行窗口化处理。
    """

    def __init__(self,
                 name: str,
                 window_assigner: Any,
                 trigger: Any,
                 window_func: Callable[[List[Record], WindowContext], List[Record]],
                 parallelism: int = 1,
                 max_windows: int = 10000,
                 max_window_age: float = 3600.0,
                 max_records_per_window: int = 10000):
        """
        初始化窗口操作符

        Args:
            name: 操作符名称
            window_assigner: 窗口分配器
            trigger: 触发器
            window_func: 窗口函数
            parallelism: 并行度
            max_windows: 最大窗口数量
            max_window_age: 窗口最大存活时间（秒）
            max_records_per_window: 每个窗口最大记录数
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.WINDOW,
            parallelism=parallelism
        )
        super().__init__(config)
        self._window_assigner = window_assigner
        self._trigger = trigger
        self._window_func = window_func
        self._windows: Dict[str, Dict[str, List[Record]]] = {}
        self._watermark = Watermark(timestamp=float('-inf'))
        self._max_windows = max_windows
        self._max_window_age = max_window_age
        self._max_records_per_window = max_records_per_window
        self._window_count = 0
        # 窗口创建时间跟踪
        self._window_timestamps: Dict[str, float] = {}
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 60.0  # 每60秒清理一次

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            key = record.key or record.get_key()

            windows = self._window_assigner.assign_windows(record)

            # 定期清理过期窗口
            current_time = time.time()
            if current_time - self._last_cleanup_time > self._cleanup_interval:
                self._cleanup_expired_windows()
                self._last_cleanup_time = current_time

            for window in windows:
                window_id = self._get_window_id(key, window)

                if key not in self._windows:
                    self._windows[key] = {}

                if window_id not in self._windows[key]:
                    # 检查窗口数量限制
                    if self._window_count >= self._max_windows:
                        self._cleanup_old_windows()
                        # 如果清理后仍然达到限制，抛出异常
                        if self._window_count >= self._max_windows:
                            raise WindowError(
                                f"Maximum window count ({self._max_windows}) reached. "
                                f"Consider increasing max_windows or checking for data skew."
                            )
                    
                    self._windows[key][window_id] = []
                    self._window_timestamps[window_id] = time.time()
                    self._window_count += 1

                # 检查单个窗口的记录数限制
                if len(self._windows[key][window_id]) >= self._max_records_per_window:
                    # 触发窗口以释放内存
                    result = self._fire_window(key, window_id, window)
                    # 重新创建窗口
                    self._windows[key][window_id] = []
                    self._window_timestamps[window_id] = time.time()
                    return result
                
                self._windows[key][window_id].append(record)

                if self._should_trigger(key, window_id, record):
                    return self._fire_window(key, window_id, window)

            return []

        except WindowError:
            raise
        except Exception as e:
            raise WindowError(f"Window operation failed: {e}") from e

    def _cleanup_expired_windows(self):
        """清理过期窗口"""
        current_time = time.time()
        expired_window_ids = []
        
        for window_id, create_time in list(self._window_timestamps.items()):
            if current_time - create_time > self._max_window_age:
                expired_window_ids.append(window_id)
        
        # 删除过期窗口
        for window_id in expired_window_ids:
            for key in list(self._windows.keys()):
                if window_id in self._windows[key]:
                    del self._windows[key][window_id]
                    self._window_count -= 1
                    if window_id in self._window_timestamps:
                        del self._window_timestamps[window_id]
                    break
            
            # 清理空的key
            keys_to_remove = [k for k, v in self._windows.items() if not v]
            for k in keys_to_remove:
                del self._windows[k]

    def _cleanup_old_windows(self):
        """清理旧窗口（LRU策略）"""
        # 首先清理过期窗口
        self._cleanup_expired_windows()
        
        # 如果仍然超过限制，使用LRU策略清理
        if self._window_count >= self._max_windows:
            # 按创建时间排序，删除最老的窗口
            sorted_windows = sorted(
                self._window_timestamps.items(),
                key=lambda x: x[1]
            )
            
            # 删除最老的10%窗口
            num_to_remove = max(1, len(sorted_windows) // 10)
            for window_id, _ in sorted_windows[:num_to_remove]:
                for key in list(self._windows.keys()):
                    if window_id in self._windows[key]:
                        del self._windows[key][window_id]
                        self._window_count -= 1
                        break
                
                if window_id in self._window_timestamps:
                    del self._window_timestamps[window_id]
            
            # 清理空的key
            keys_to_remove = [k for k, v in self._windows.items() if not v]
            for k in keys_to_remove:
                del self._windows[k]

    def _get_window_id(self, key: str, window: Any) -> str:
        """获取窗口ID"""
        return f"{key}_{window.start}_{window.end}"

    def _should_trigger(self, key: str, window_id: str, record: Record) -> bool:
        """判断是否应该触发"""
        return self._trigger.should_fire(
            self._windows[key][window_id],
            record,
            self._watermark
        )

    def _fire_window(self, key: str, window_id: str, window: Any) -> List[Record]:
        """触发窗口"""
        records = self._windows[key].pop(window_id, [])
        self._window_count -= 1
        
        # 清理窗口时间戳
        if window_id in self._window_timestamps:
            del self._window_timestamps[window_id]

        if not records:
            return []

        context = WindowContext(
            window_start=window.start,
            window_end=window.end,
            key=key
        )

        result = self._window_func(records, context)

        return result

    def process_watermark(self, watermark: Watermark) -> List[Record]:
        """
        处理watermark

        Args:
            watermark: watermark标记

        Returns:
            List[Record]: 输出记录列表
        """
        self._watermark = watermark

        results = []

        for key, windows in list(self._windows.items()):
            for window_id, records in list(windows.items()):
                if self._should_fire_on_watermark(key, window_id):
                    window = self._get_window_from_id(window_id)
                    result = self._fire_window(key, window_id, window)
                    results.extend(result)

        return results

    def _should_fire_on_watermark(self, key: str, window_id: str) -> bool:
        """判断是否应该在watermark时触发"""
        return False

    def _get_window_from_id(self, window_id: str) -> Any:
        """从ID获取窗口"""
        parts = window_id.split('_')
        if len(parts) >= 3:
            from ..windows.base import Window
            return Window(start=float(parts[-2]), end=float(parts[-1]))
        return None

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        self._windows.clear()
        self._window_timestamps.clear()
        self._window_count = 0
        self._last_cleanup_time = time.time()

    def close(self):
        """关闭操作符"""
        self._windows.clear()
        self._window_timestamps.clear()
        self._window_count = 0
        self.set_state(OperatorState.COMPLETED)


class TumblingWindowOperator(WindowOperator):
    """
    滚动窗口操作符

    固定大小、不重叠的窗口。
    """

    def __init__(self,
                 name: str,
                 window_size: float,
                 window_func: Callable[[List[Record], WindowContext], List[Record]],
                 parallelism: int = 1):
        """
        初始化滚动窗口操作符

        Args:
            name: 操作符名称
            window_size: 窗口大小（秒）
            window_func: 窗口函数
            parallelism: 并行度
        """
        from ..windows.tumbling import TumblingWindowAssigner
        from ..windows.trigger import EventTimeTrigger

        assigner = TumblingWindowAssigner(size=window_size)
        trigger = EventTimeTrigger()

        super().__init__(
            name=name,
            window_assigner=assigner,
            trigger=trigger,
            window_func=window_func,
            parallelism=parallelism
        )
        self._window_size = window_size

    def _should_fire_on_watermark(self, key: str, window_id: str) -> bool:
        """判断是否应该在watermark时触发"""
        window = self._get_window_from_id(window_id)
        if window:
            return self._watermark.timestamp >= window.end
        return False


class SlidingWindowOperator(WindowOperator):
    """
    滑动窗口操作符

    固定大小、可重叠的窗口。
    """

    def __init__(self,
                 name: str,
                 window_size: float,
                 slide_size: float,
                 window_func: Callable[[List[Record], WindowContext], List[Record]],
                 parallelism: int = 1):
        """
        初始化滑动窗口操作符

        Args:
            name: 操作符名称
            window_size: 窗口大小（秒）
            slide_size: 滑动大小（秒）
            window_func: 窗口函数
            parallelism: 并行度
        """
        from ..windows.sliding import SlidingWindowAssigner
        from ..windows.trigger import EventTimeTrigger

        assigner = SlidingWindowAssigner(size=window_size, slide=slide_size)
        trigger = EventTimeTrigger()

        super().__init__(
            name=name,
            window_assigner=assigner,
            trigger=trigger,
            window_func=window_func,
            parallelism=parallelism
        )
        self._window_size = window_size
        self._slide_size = slide_size

    def _should_fire_on_watermark(self, key: str, window_id: str) -> bool:
        """判断是否应该在watermark时触发"""
        window = self._get_window_from_id(window_id)
        if window:
            return self._watermark.timestamp >= window.end
        return False


class SessionWindowOperator(WindowOperator):
    """
    会话窗口操作符

    基于活动间隔的窗口。
    """

    def __init__(self,
                 name: str,
                 gap: float,
                 window_func: Callable[[List[Record], WindowContext], List[Record]],
                 parallelism: int = 1):
        """
        初始化会话窗口操作符

        Args:
            name: 操作符名称
            gap: 会话间隔（秒）
            window_func: 窗口函数
            parallelism: 并行度
        """
        from ..windows.session import SessionWindowAssigner
        from ..windows.trigger import EventTimeTrigger

        assigner = SessionWindowAssigner(gap=gap)
        trigger = EventTimeTrigger()

        super().__init__(
            name=name,
            window_assigner=assigner,
            trigger=trigger,
            window_func=window_func,
            parallelism=parallelism
        )
        self._gap = gap


class CountWindowOperator(WindowOperator):
    """
    计数窗口操作符

    基于元素数量的窗口。
    """

    def __init__(self,
                 name: str,
                 count: int,
                 window_func: Callable[[List[Record], WindowContext], List[Record]],
                 parallelism: int = 1):
        """
        初始化计数窗口操作符

        Args:
            name: 操作符名称
            count: 元素数量
            window_func: 窗口函数
            parallelism: 并行度
        """
        from ..windows.count import CountWindowAssigner
        from ..windows.trigger import CountTrigger

        assigner = CountWindowAssigner(count=count)
        trigger = CountTrigger(count=count)

        super().__init__(
            name=name,
            window_assigner=assigner,
            trigger=trigger,
            window_func=window_func,
            parallelism=parallelism
        )
        self._count = count

    def _should_trigger(self, key: str, window_id: str, record: Record) -> bool:
        """判断是否应该触发"""
        records = self._windows.get(key, {}).get(window_id, [])
        return len(records) >= self._count
