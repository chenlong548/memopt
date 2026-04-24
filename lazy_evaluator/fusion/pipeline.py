"""
惰性管道实现模块

实现惰性管道，支持链式操作。
"""

from typing import Iterable, Callable, Any, List, Optional, TypeVar, Generic, Set, Tuple
import sys
from ..core.lazy import Lazy
from ..core.exceptions import LazyEvaluationError, FusionError

T = TypeVar('T')
U = TypeVar('U')

# 默认内存限制（元素数量）
DEFAULT_MEMORY_LIMIT = 1000000  # 100万元素


class MemoryLimitExceededError(FusionError):
    """内存限制超出异常"""
    
    def __init__(self, operation: Optional[str] = None, limit: Optional[int] = None):
        self.limit = limit
        message = "Memory limit exceeded"
        if operation:
            message += f" during {operation}"
        if limit:
            message += f" (limit: {limit} elements)"
        super().__init__(operation=operation or "", reason=message)


class LazyPipeline(Generic[T]):
    """
    惰性管道，支持链式操作（支持内存限制）

    该类实现了惰性管道模式，支持链式操作而不立即执行。
    只有在调用终端操作（如collect、reduce）时才真正执行计算。
    
    内存安全：
    - 支持设置内存限制，防止大数据集导致内存溢出
    - distinct操作会检查内存限制

    Attributes:
        _source: 数据源
        _operations: 操作列表
        _lazy_source: 惰性数据源
        _memory_limit: 内存限制（元素数量）

    Example:
        >>> pipeline = LazyPipeline([1, 2, 3, 4, 5])
        >>> result = pipeline.map(lambda x: x * 2).filter(lambda x: x > 4).collect()
        >>> # result = [6, 8, 10]
    """

    def __init__(self, source: Iterable[T], memory_limit: int = DEFAULT_MEMORY_LIMIT):
        """
        初始化惰性管道

        Args:
            source: 数据源
            memory_limit: 内存限制（元素数量），默认100万
        """
        self._source = source
        self._operations: List[Tuple[str, Any]] = []
        self._lazy_source: Optional[Lazy] = None
        self._memory_limit = memory_limit

    def map(self, f: Callable[[T], U]) -> 'LazyPipeline[U]':
        """
        映射操作

        Args:
            f: 映射函数

        Returns:
            LazyPipeline[U]: 新的管道
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline[U](source_generator(), self._memory_limit)  # type: ignore
        new_pipeline._operations = [('map', f)]
        return new_pipeline

    def filter(self, pred: Callable[[T], bool]) -> 'LazyPipeline[T]':
        """
        过滤操作

        Args:
            pred: 谓词函数

        Returns:
            LazyPipeline[T]: 新的管道
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline(source_generator(), self._memory_limit)
        new_pipeline._operations = [('filter', pred)]
        return new_pipeline

    def flat_map(self, f: Callable[[T], Iterable[U]]) -> 'LazyPipeline[U]':
        """
        扁平映射操作

        Args:
            f: 映射函数，返回可迭代对象

        Returns:
            LazyPipeline[U]: 新的管道
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline[U](source_generator(), self._memory_limit)  # type: ignore
        new_pipeline._operations = [('flat_map', f)]
        return new_pipeline

    def take(self, n: int) -> 'LazyPipeline[T]':
        """
        取前n个元素

        Args:
            n: 元素数量

        Returns:
            LazyPipeline[T]: 新的管道
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline(source_generator(), self._memory_limit)
        new_pipeline._operations = [('take', n)]
        return new_pipeline

    def drop(self, n: int) -> 'LazyPipeline[T]':
        """
        跳过前n个元素

        Args:
            n: 元素数量

        Returns:
            LazyPipeline[T]: 新的管道
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline(source_generator(), self._memory_limit)
        new_pipeline._operations = [('drop', n)]
        return new_pipeline

    def distinct(self) -> 'LazyPipeline[T]':
        """
        去重操作（带内存限制）

        Returns:
            LazyPipeline[T]: 新的管道

        Raises:
            MemoryLimitExceededError: 如果内存限制超出
        """
        # 创建一个生成器函数，它会执行当前管道的所有操作
        def source_generator():
            for item in self._execute():
                yield item

        new_pipeline = LazyPipeline(source_generator(), self._memory_limit)
        new_pipeline._operations = [('distinct', None)]
        return new_pipeline

    def collect(self) -> List[T]:
        """
        收集结果

        Returns:
            List[T]: 结果列表
        """
        result = []
        for item in self._execute():
            result.append(item)
        return result

    def reduce(self, f: Callable[[Any, T], Any], initial: Any = None) -> Any:
        """
        归约操作

        Args:
            f: 归约函数
            initial: 初始值

        Returns:
            Any: 归约结果
        """
        accumulator = initial
        first = True

        for item in self._execute():
            if first and initial is None:
                accumulator = item
                first = False
            else:
                accumulator = f(accumulator, item)
                first = False

        return accumulator

    def for_each(self, f: Callable[[T], None]) -> None:
        """
        对每个元素执行操作

        Args:
            f: 操作函数
        """
        for item in self._execute():
            f(item)

    def count(self) -> int:
        """
        计数

        Returns:
            int: 元素数量
        """
        count = 0
        for _ in self._execute():
            count += 1
        return count

    def first(self) -> Optional[T]:
        """
        获取第一个元素

        Returns:
            Optional[T]: 第一个元素，如果为空返回None
        """
        for item in self._execute():
            return item
        return None

    def last(self) -> Optional[T]:
        """
        获取最后一个元素

        Returns:
            Optional[T]: 最后一个元素，如果为空返回None
        """
        last_item = None
        for item in self._execute():
            last_item = item
        return last_item

    def any_match(self, pred: Callable[[T], bool]) -> bool:
        """
        检查是否有任意元素匹配

        Args:
            pred: 谓词函数

        Returns:
            bool: 如果有匹配返回True，否则返回False
        """
        for item in self._execute():
            if pred(item):
                return True
        return False

    def all_match(self, pred: Callable[[T], bool]) -> bool:
        """
        检查是否所有元素都匹配

        Args:
            pred: 谓词函数

        Returns:
            bool: 如果所有都匹配返回True，否则返回False
        """
        for item in self._execute():
            if not pred(item):
                return False
        return True

    def none_match(self, pred: Callable[[T], bool]) -> bool:
        """
        检查是否没有元素匹配

        Args:
            pred: 谓词函数

        Returns:
            bool: 如果没有匹配返回True，否则返回False
        """
        for item in self._execute():
            if pred(item):
                return False
        return True

    def _execute(self) -> Iterable[T]:
        """
        执行管道操作

        Yields:
            T: 处理后的元素

        Raises:
            MemoryLimitExceededError: 如果distinct操作超出内存限制
        """
        # 创建生成器链
        iterator = iter(self._source)

        # 应用所有操作
        for op_type, op_func in self._operations:
            if op_type == 'map':
                iterator = (op_func(item) for item in iterator)
            elif op_type == 'filter':
                iterator = (item for item in iterator if op_func(item))
            elif op_type == 'flat_map':
                def flat_map_gen(it, f):
                    for item in it:
                        for sub_item in f(item):
                            yield sub_item
                iterator = flat_map_gen(iterator, op_func)
            elif op_type == 'take':
                def take_gen(it, n):
                    count = 0
                    for item in it:
                        if count >= n:
                            break
                        yield item
                        count += 1
                iterator = take_gen(iterator, op_func)
            elif op_type == 'drop':
                def drop_gen(it, n):
                    count = 0
                    for item in it:
                        if count >= n:
                            yield item
                        else:
                            count += 1
                iterator = drop_gen(iterator, op_func)
            elif op_type == 'distinct':
                # 使用闭包捕获memory_limit
                memory_limit = self._memory_limit
                
                def distinct_gen(it, limit: int):
                    seen: Set[T] = set()
                    for item in it:
                        # 检查内存限制
                        if len(seen) >= limit:
                            raise MemoryLimitExceededError(
                                operation='distinct',
                                limit=limit
                            )
                        if item not in seen:
                            seen.add(item)
                            yield item
                iterator = distinct_gen(iterator, memory_limit)

        # 返回最终的迭代器
        return iterator  # type: ignore

    def __iter__(self) -> Iterable[T]:
        """返回迭代器"""
        return self._execute()  # type: ignore

    def __repr__(self) -> str:
        """字符串表示"""
        return f"LazyPipeline(operations={len(self._operations)})"

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
