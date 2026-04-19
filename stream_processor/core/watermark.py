"""
stream_processor Watermark机制

处理流处理中的乱序数据。
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import time
import threading


@dataclass
class Watermark:
    """
    Watermark标记

    表示流处理中的时间进度标记。
    """

    timestamp: float

    partition: Optional[int] = None

    created_time: float = field(default_factory=time.time)

    def __lt__(self, other: 'Watermark') -> bool:
        return self.timestamp < other.timestamp

    def __le__(self, other: 'Watermark') -> bool:
        return self.timestamp <= other.timestamp

    def __gt__(self, other: 'Watermark') -> bool:
        return self.timestamp > other.timestamp

    def __ge__(self, other: 'Watermark') -> bool:
        return self.timestamp >= other.timestamp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Watermark):
            return False
        return self.timestamp == other.timestamp

    def __hash__(self) -> int:
        return hash(self.timestamp)


class WatermarkGenerator:
    """
    Watermark生成器

    生成watermark以处理乱序数据。
    """

    def __init__(self,
                 out_of_orderness: float = 5.0,
                 idle_timeout: float = 60.0,
                 watermark_callback: Optional[Callable[[Watermark], None]] = None):
        """
        初始化Watermark生成器

        Args:
            out_of_orderness: 允许的乱序时间（秒）
            idle_timeout: 空闲超时时间（秒）
            watermark_callback: watermark回调函数
        """
        self.out_of_orderness = out_of_orderness
        self.idle_timeout = idle_timeout
        self.watermark_callback = watermark_callback

        self._max_timestamp = float('-inf')
        self._last_watermark = Watermark(timestamp=float('-inf'))
        self._last_record_time = time.time()
        self._lock = threading.Lock()

    def update(self, record_timestamp: float, partition: Optional[int] = None) -> Optional[Watermark]:
        """
        根据记录时间戳更新watermark

        Args:
            record_timestamp: 记录时间戳
            partition: 分区号

        Returns:
            Optional[Watermark]: 新的watermark，如果没有更新则返回None
        """
        with self._lock:
            self._last_record_time = time.time()

            if record_timestamp > self._max_timestamp:
                self._max_timestamp = record_timestamp

                new_watermark_ts = self._max_timestamp - self.out_of_orderness

                if new_watermark_ts > self._last_watermark.timestamp:
                    new_watermark = Watermark(
                        timestamp=new_watermark_ts,
                        partition=partition
                    )
                    self._last_watermark = new_watermark

                    if self.watermark_callback:
                        self.watermark_callback(new_watermark)

                    return new_watermark

            return None

    def get_current_watermark(self) -> Watermark:
        """
        获取当前watermark

        Returns:
            Watermark: 当前watermark
        """
        with self._lock:
            current_time = time.time()

            if current_time - self._last_record_time > self.idle_timeout:
                idle_watermark = Watermark(
                    timestamp=current_time - self.out_of_orderness
                )
                if idle_watermark.timestamp > self._last_watermark.timestamp:
                    self._last_watermark = idle_watermark

            return self._last_watermark

    def reset(self):
        """重置watermark生成器"""
        with self._lock:
            self._max_timestamp = float('-inf')
            self._last_watermark = Watermark(timestamp=float('-inf'))
            self._last_record_time = time.time()


class WatermarkStrategy:
    """
    Watermark策略

    定义watermark生成策略。
    """

    @staticmethod
    def bounded_out_of_orderness(max_out_of_orderness: float) -> WatermarkGenerator:
        """
        有界乱序策略

        Args:
            max_out_of_orderness: 最大乱序时间（秒）

        Returns:
            WatermarkGenerator: watermark生成器
        """
        return WatermarkGenerator(out_of_orderness=max_out_of_orderness)

    @staticmethod
    def ascending_timestamps() -> WatermarkGenerator:
        """
        升序时间戳策略

        Returns:
            WatermarkGenerator: watermark生成器
        """
        return WatermarkGenerator(out_of_orderness=0.0)

    @staticmethod
    def for_monotonous_timestamps() -> WatermarkGenerator:
        """
        单调时间戳策略

        Returns:
            WatermarkGenerator: watermark生成器
        """
        return WatermarkGenerator(out_of_orderness=0.0)

    @staticmethod
    def for_bounded_out_of_orderness(max_delay: float) -> WatermarkGenerator:
        """
        有界延迟策略

        Args:
            max_delay: 最大延迟时间（秒）

        Returns:
            WatermarkGenerator: watermark生成器
        """
        return WatermarkGenerator(out_of_orderness=max_delay)


class WatermarkTracker:
    """
    Watermark跟踪器

    跟踪多个分区的watermark状态。
    """

    def __init__(self, num_partitions: int = 1):
        """
        初始化Watermark跟踪器

        Args:
            num_partitions: 分区数量
        """
        self.num_partitions = num_partitions
        self._partition_watermarks: dict[int, Watermark] = {
            i: Watermark(timestamp=float('-inf'))
            for i in range(num_partitions)
        }
        self._lock = threading.Lock()

    def update_partition_watermark(self, partition: int, watermark: Watermark):
        """
        更新分区watermark

        Args:
            partition: 分区号
            watermark: 新watermark
        """
        with self._lock:
            if partition in self._partition_watermarks:
                if watermark.timestamp > self._partition_watermarks[partition].timestamp:
                    self._partition_watermarks[partition] = watermark

    def get_global_watermark(self) -> Watermark:
        """
        获取全局watermark（所有分区中的最小值）

        Returns:
            Watermark: 全局watermark
        """
        with self._lock:
            min_timestamp = min(
                wm.timestamp for wm in self._partition_watermarks.values()
            )
            return Watermark(timestamp=min_timestamp)

    def get_partition_watermark(self, partition: int) -> Optional[Watermark]:
        """
        获取分区watermark

        Args:
            partition: 分区号

        Returns:
            Optional[Watermark]: 分区watermark
        """
        with self._lock:
            return self._partition_watermarks.get(partition)

    def reset(self):
        """重置所有watermark"""
        with self._lock:
            for partition in self._partition_watermarks:
                self._partition_watermarks[partition] = Watermark(timestamp=float('-inf'))
