"""
Stream Processor适配器模块

提供与stream_processor模块的集成适配器。
"""

from typing import Any, Callable, Optional, Dict, Iterable, TypeVar, Generic, List
import sys
import os

T = TypeVar('T')
U = TypeVar('U')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ..core.lazy import Lazy
from ..fusion.pipeline import LazyPipeline


class StreamAdapter:
    """
    Stream Processor适配器

    该类提供了lazy_evaluator与stream_processor模块的集成适配器，
    允许在流处理中使用惰性计算。

    Example:
        >>> adapter = StreamAdapter()
        >>> lazy_stream = adapter.create_lazy_stream([1, 2, 3, 4, 5])
        >>> result = lazy_stream.map(lambda x: x * 2).collect()
    """

    def __init__(self):
        """初始化StreamAdapter"""
        self._stream_processor = None
        try:
            # 尝试导入stream_processor模块
            from stream_processor import Stream
            self._stream_processor = Stream
        except ImportError:
            pass

    def is_available(self) -> bool:
        """
        检查stream_processor模块是否可用

        Returns:
            bool: 如果可用返回True，否则返回False
        """
        return self._stream_processor is not None

    def create_lazy_stream(self, source: Iterable[T]) -> LazyPipeline[T]:
        """
        创建惰性流

        Args:
            source: 数据源

        Returns:
            LazyPipeline: 惰性管道
        """
        return LazyPipeline(source)

    def from_stream_processor(self, stream: Iterable[T]) -> LazyPipeline[T]:
        """
        从stream_processor的Stream创建LazyPipeline

        Args:
            stream: stream_processor的Stream实例

        Returns:
            LazyPipeline: 惰性管道
        """
        # 将stream_processor的流转换为可迭代对象
        if hasattr(stream, '__iter__'):
            return LazyPipeline(stream)
        else:
            raise ValueError("Stream must be iterable")

    def to_stream_processor(self, pipeline: LazyPipeline) -> Any:
        """
        将LazyPipeline转换为stream_processor的Stream

        Args:
            pipeline: LazyPipeline实例

        Returns:
            Stream: stream_processor的Stream实例
        """
        if not self.is_available():
            raise RuntimeError("stream_processor module is not available")

        # 直接使用stream_processor包装pipeline，保持惰性计算
        if self._stream_processor is not None:
            return self._stream_processor(pipeline)  # type: ignore

    def lazy_transform(self, source: Any, transform_func: Callable) -> Lazy:
        """
        创建惰性转换

        Args:
            source: 数据源
            transform_func: 转换函数

        Returns:
            Lazy: 惰性值
        """
        return Lazy(lambda: transform_func(source))

    def batch_process(self, source: Iterable[T], batch_size: int, process_func: Callable[[List[T]], U]) -> LazyPipeline[U]:
        """
        批量处理

        Args:
            source: 数据源
            batch_size: 批次大小
            process_func: 处理函数

        Returns:
            LazyPipeline: 惰性管道
        """
        def batch_generator():
            batch = []
            for item in source:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield process_func(batch)
                    batch = []
            if batch:
                yield process_func(batch)

        return LazyPipeline(batch_generator())

    def create_windowed_stream(self, source: Iterable[T], window_size: int, step: int = 1) -> LazyPipeline[List[T]]:
        """
        创建滑动窗口流

        Args:
            source: 数据源
            window_size: 窗口大小
            step: 步长

        Returns:
            LazyPipeline: 惰性管道
        """
        def window_generator():
            window = []
            for item in source:
                window.append(item)
                if len(window) >= window_size:
                    yield list(window)
                    # 移动窗口
                    for _ in range(step):
                        if window:
                            window.pop(0)

        return LazyPipeline(window_generator())

    def __repr__(self) -> str:
        """字符串表示"""
        available = "available" if self.is_available() else "not available"
        return f"StreamAdapter(status={available})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
