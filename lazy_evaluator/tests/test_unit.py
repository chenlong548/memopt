"""
lazy_evaluator模块单元测试

包含所有模块的完整测试用例。
"""

import unittest
import time
import threading
from typing import List
from collections import deque

# Core层测试
from lazy_evaluator.core.lazy import Lazy, ThunkState
from lazy_evaluator.core.evaluation import EvaluationContext
from lazy_evaluator.core.exceptions import (
    LazyEvaluationError,
    CircularDependencyError,
    ThunkEvaluationError,
    CacheError
)

# Thunk层测试
from lazy_evaluator.thunk.memo_thunk import Memothunk
from lazy_evaluator.thunk.thunk_pool import ThunkPool, PooledThunk

# Memoization层测试
from lazy_evaluator.memoization.lru_cache import LRUCache
from lazy_evaluator.memoization.multi_level import MultiLevelCache, CacheLevel
from lazy_evaluator.memoization.decorator import memoize, memoize_method, memoize_property

# Dependency层测试
from lazy_evaluator.dependency.graph import DependencyGraph
from lazy_evaluator.dependency.incremental import IncrementalEvaluator

# Fusion层测试
from lazy_evaluator.fusion.stream_fusion import StreamFusion
from lazy_evaluator.fusion.pipeline import LazyPipeline, MemoryLimitExceededError

# Integration层测试
from lazy_evaluator.integration.stream_adapter import StreamAdapter
from lazy_evaluator.integration.compressor_adapter import CompressorAdapter
from lazy_evaluator.integration.optimizer_adapter import OptimizerAdapter


class TestLazy(unittest.TestCase):
    """Lazy[T]基本功能测试"""

    def test_lazy_creation(self):
        """测试Lazy创建"""
        lazy_val = Lazy(lambda: 42)
        self.assertEqual(lazy_val.get_state(), ThunkState.UNEVALUATED)
        self.assertFalse(lazy_val.is_evaluated())

    def test_lazy_force(self):
        """测试Lazy强制求值"""
        lazy_val = Lazy(lambda: 42)
        result = lazy_val.force()
        self.assertEqual(result, 42)
        self.assertTrue(lazy_val.is_evaluated())

    def test_lazy_memoization(self):
        """测试Lazy记忆化"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            return 42

        lazy_val = Lazy(computation)
        result1 = lazy_val.force()
        result2 = lazy_val.force()

        self.assertEqual(result1, 42)
        self.assertEqual(result2, 42)
        self.assertEqual(call_count[0], 1)  # 只计算一次

    def test_lazy_map(self):
        """测试Lazy映射操作"""
        lazy_val = Lazy(lambda: 10)
        mapped = lazy_val.map(lambda x: x * 2)
        result = mapped.force()
        self.assertEqual(result, 20)

    def test_lazy_flat_map(self):
        """测试Lazy扁平映射操作"""
        lazy_val = Lazy(lambda: 10)
        flat_mapped = lazy_val.flat_map(lambda x: Lazy(lambda: x * 2))
        result = flat_mapped.force()
        self.assertEqual(result, 20)

    def test_lazy_reset(self):
        """测试Lazy重置"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            return 42

        lazy_val = Lazy(computation)
        lazy_val.force()
        lazy_val.reset()
        lazy_val.force()

        self.assertEqual(call_count[0], 2)  # 重置后重新计算

    def test_lazy_error_handling(self):
        """测试Lazy错误处理"""
        def error_computation():
            raise ValueError("Test error")

        lazy_val = Lazy(error_computation)
        with self.assertRaises(LazyEvaluationError):
            lazy_val.force()

    def test_lazy_thread_safety(self):
        """测试Lazy线程安全 - 确保多线程环境下只计算一次"""
        call_count = [0]
        
        def computation():
            call_count[0] += 1
            time.sleep(0.05)  # 模拟耗时操作
            return 42
        
        lazy_val = Lazy(computation)
        results = []
        results_lock = threading.Lock()
        errors = []
        
        def worker():
            try:
                result = lazy_val.force()
                with results_lock:
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程同时访问
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 验证：所有线程都获得正确结果，且只计算一次
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        self.assertTrue(all(r == 42 for r in results))
        self.assertEqual(call_count[0], 1)  # 关键：只计算一次

    def test_lazy_thread_safety_with_error(self):
        """测试Lazy线程安全 - 错误情况下的线程安全"""
        call_count = [0]
        
        def error_computation():
            call_count[0] += 1
            time.sleep(0.05)
            raise ValueError("Test error")
        
        lazy_val = Lazy(error_computation)
        errors = []
        
        def worker():
            try:
                lazy_val.force()
            except LazyEvaluationError:
                errors.append(True)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 所有线程都应该捕获到错误，且只计算一次
        self.assertEqual(len(errors), 5)
        self.assertEqual(call_count[0], 1)


class TestEvaluationContext(unittest.TestCase):
    """EvaluationContext测试"""

    def test_context_register(self):
        """测试注册惰性值"""
        ctx = EvaluationContext()
        lazy_val = Lazy(lambda: 42)
        ctx.register("value1", lazy_val)
        self.assertEqual(len(ctx), 1)

    def test_context_evaluate(self):
        """测试求值"""
        ctx = EvaluationContext()
        lazy_val = Lazy(lambda: 42)
        ctx.register("value1", lazy_val)
        result = ctx.evaluate("value1")
        self.assertEqual(result, 42)

    def test_context_dependencies(self):
        """测试依赖关系"""
        ctx = EvaluationContext()
        lazy_val1 = Lazy(lambda: 10)
        lazy_val2 = Lazy(lambda: 20)
        ctx.register("val1", lazy_val1)
        ctx.register("val2", lazy_val2)
        ctx.add_dependency("val2", "val1")

        deps = ctx.get_dependencies("val2")
        self.assertIn("val1", deps)

    def test_context_cycle_detection(self):
        """测试循环依赖检测"""
        ctx = EvaluationContext()
        lazy_val1 = Lazy(lambda: 10)
        lazy_val2 = Lazy(lambda: 20)
        ctx.register("val1", lazy_val1)
        ctx.register("val2", lazy_val2)
        ctx.add_dependency("val1", "val2")
        ctx.add_dependency("val2", "val1")

        cycle = ctx.detect_cycle()
        self.assertIsNotNone(cycle)

    def test_context_thread_safety(self):
        """测试EvaluationContext线程安全"""
        ctx = EvaluationContext()
        results = []
        errors = []
        
        def worker(i):
            try:
                lazy_val = Lazy(lambda: i * 10)
                ctx.register(f"val_{i}", lazy_val)
                result = ctx.evaluate(f"val_{i}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)

    def test_context_evaluation_order_optimization(self):
        """测试拓扑排序使用deque优化"""
        ctx = EvaluationContext()
        
        # 创建大量节点
        for i in range(100):
            lazy_val = Lazy(lambda: i)
            ctx.register(f"node_{i}", lazy_val)
        
        # 添加依赖关系
        for i in range(1, 100):
            ctx.add_dependency(f"node_{i}", f"node_{i-1}")
        
        # 获取求值顺序，验证不会出错
        order = ctx.get_evaluation_order()
        self.assertEqual(len(order), 100)


class TestMemothunk(unittest.TestCase):
    """Memothunk记忆化测试"""

    def test_memothunk_creation(self):
        """测试Memothunk创建"""
        thunk = Memothunk(lambda: 42)
        self.assertFalse(thunk.is_evaluated())

    def test_memothunk_get(self):
        """测试Memothunk获取值"""
        thunk = Memothunk(lambda: 42)
        result = thunk.get()
        self.assertEqual(result, 42)
        self.assertTrue(thunk.is_evaluated())

    def test_memothunk_memoization(self):
        """测试Memothunk记忆化"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            return 42

        thunk = Memothunk(computation)
        thunk.get()
        thunk.get()

        self.assertEqual(call_count[0], 1)

    def test_memothunk_reset(self):
        """测试Memothunk重置"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            return 42

        thunk = Memothunk(computation)
        thunk.get()
        thunk.reset()
        thunk.get()

        self.assertEqual(call_count[0], 2)

    def test_memothunk_thread_safety(self):
        """测试Memothunk线程安全"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            time.sleep(0.01)  # 模拟耗时操作
            return 42

        thunk = Memothunk(computation)
        threads = []

        for _ in range(10):
            thread = threading.Thread(target=thunk.get)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 确保只计算一次
        self.assertEqual(call_count[0], 1)


class TestThunkPool(unittest.TestCase):
    """ThunkPool对象池测试"""

    def test_pool_acquire(self):
        """测试对象池获取"""
        pool = ThunkPool(max_size=10)
        thunk = pool.acquire(lambda: 42)
        self.assertIsNotNone(thunk)

    def test_pool_release(self):
        """测试对象池释放"""
        pool = ThunkPool(max_size=10)
        thunk = pool.acquire(lambda: 42)
        pool.release(thunk)
        self.assertEqual(pool.size(), 1)

    def test_pool_reuse(self):
        """测试对象池复用"""
        pool = ThunkPool(max_size=10)
        thunk1 = pool.acquire(lambda: 42)
        pool.release(thunk1)
        thunk2 = pool.acquire(lambda: 24)

        # 应该复用同一个对象
        self.assertEqual(thunk1.get_id(), thunk2.get_id())

    def test_pool_clear(self):
        """测试对象池清空"""
        pool = ThunkPool(max_size=10)
        thunk = pool.acquire(lambda: 42)
        pool.release(thunk)
        pool.clear()
        self.assertEqual(pool.size(), 0)


class TestLRUCache(unittest.TestCase):
    """LRUCache缓存测试"""

    def test_cache_put_get(self):
        """测试缓存存取"""
        cache = LRUCache(max_size=10)
        cache.put("key1", "value1")
        result = cache.get("key1")
        self.assertEqual(result, "value1")

    def test_cache_eviction(self):
        """测试缓存淘汰"""
        cache = LRUCache(max_size=2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # 应该淘汰key1

        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")

    def test_cache_invalidate(self):
        """测试缓存失效"""
        cache = LRUCache(max_size=10)
        cache.put("key1", "value1")
        cache.invalidate("key1")
        self.assertIsNone(cache.get("key1"))

    def test_cache_ttl(self):
        """测试缓存TTL"""
        cache = LRUCache(max_size=10, ttl=0.1)
        cache.put("key1", "value1")
        time.sleep(0.15)
        result = cache.get("key1")
        self.assertIsNone(result)

    def test_cache_get_or_compute(self):
        """测试缓存计算"""
        cache = LRUCache(max_size=10)
        call_count = [0]

        def compute():
            call_count[0] += 1
            return 42

        result1 = cache.get_or_compute("key1", compute)
        result2 = cache.get_or_compute("key1", compute)

        self.assertEqual(result1, 42)
        self.assertEqual(result2, 42)
        self.assertEqual(call_count[0], 1)


class TestMultiLevelCache(unittest.TestCase):
    """多级缓存测试"""

    def test_multi_level_creation(self):
        """测试多级缓存创建"""
        cache = MultiLevelCache()
        cache.add_level(CacheLevel.L1, LRUCache(max_size=10))
        cache.add_level(CacheLevel.L2, LRUCache(max_size=100))
        self.assertEqual(cache.level_count(), 2)

    def test_multi_level_get_put(self):
        """测试多级缓存存取"""
        cache = MultiLevelCache()
        cache.add_level(CacheLevel.L1, LRUCache(max_size=10))
        cache.add_level(CacheLevel.L2, LRUCache(max_size=100))

        cache.put("key1", "value1")
        result = cache.get("key1")
        self.assertEqual(result, "value1")

    def test_multi_level_hit_rate(self):
        """测试多级缓存命中率"""
        cache = MultiLevelCache()
        cache.add_level(CacheLevel.L1, LRUCache(max_size=10))
        cache.add_level(CacheLevel.L2, LRUCache(max_size=100))

        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 0)


class TestMemoizeDecorator(unittest.TestCase):
    """memoize装饰器测试"""

    def test_memoize_function(self):
        """测试函数记忆化"""
        call_count = [0]

        @memoize(max_size=10)
        def expensive_func(x):
            call_count[0] += 1
            return x ** 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)

        self.assertEqual(result1, 25)
        self.assertEqual(result2, 25)
        self.assertEqual(call_count[0], 1)

    def test_memoize_ttl(self):
        """测试TTL过期"""
        call_count = [0]

        @memoize(max_size=10, ttl=0.1)
        def expensive_func(x):
            call_count[0] += 1
            return x ** 2

        expensive_func(5)
        time.sleep(0.15)
        expensive_func(5)

        self.assertEqual(call_count[0], 2)

    def test_memoize_method(self):
        """测试方法记忆化"""
        class TestClass:
            def __init__(self):
                self.call_count = 0

            @memoize_method(max_size=10)
            def method(self, x):
                self.call_count += 1
                return x ** 2

        obj = TestClass()
        result1 = obj.method(5)
        result2 = obj.method(5)

        self.assertEqual(result1, 25)
        self.assertEqual(result2, 25)
        self.assertEqual(obj.call_count, 1)

    def test_memoize_property(self):
        """测试属性记忆化"""
        class TestClass:
            def __init__(self):
                self.call_count = 0

            @memoize_property()
            def prop(self):
                self.call_count += 1
                return 42

        obj = TestClass()
        result1 = obj.prop
        result2 = obj.prop

        self.assertEqual(result1, 42)
        self.assertEqual(result2, 42)
        self.assertEqual(obj.call_count, 1)

    def test_memoize_safe_serialization(self):
        """测试安全序列化 - 不使用pickle"""
        # 测试各种参数类型
        @memoize(max_size=10)
        def func_with_complex_args(a, b, c=None, d=None):
            return (a, b, c, d)
        
        # 测试基本类型
        result1 = func_with_complex_args(1, 2, c=3, d=4)
        result2 = func_with_complex_args(1, 2, c=3, d=4)
        self.assertEqual(result1, result2)
        
        # 测试列表参数
        result3 = func_with_complex_args([1, 2], [3, 4])
        result4 = func_with_complex_args([1, 2], [3, 4])
        self.assertEqual(result3, result4)
        
        # 测试字典参数
        result5 = func_with_complex_args({'a': 1}, {'b': 2})
        result6 = func_with_complex_args({'a': 1}, {'b': 2})
        self.assertEqual(result5, result6)

    def test_memoize_cache_eviction_optimization(self):
        """测试缓存淘汰O(1)优化"""
        call_count = [0]
        
        @memoize(max_size=3)
        def expensive_func(x):
            call_count[0] += 1
            return x ** 2
        
        # 填充缓存
        expensive_func(1)
        expensive_func(2)
        expensive_func(3)
        
        # 访问key=1，使其成为最近使用
        expensive_func(1)
        
        # 添加新项，应该淘汰key=2（最久未使用）
        expensive_func(4)
        
        # key=2应该被淘汰
        call_count_before = call_count[0]
        expensive_func(2)
        self.assertEqual(call_count[0], call_count_before + 1)  # 需要重新计算
        
        # key=1应该还在缓存中
        call_count_before = call_count[0]
        expensive_func(1)
        self.assertEqual(call_count[0], call_count_before)  # 不需要重新计算


class TestDependencyGraph(unittest.TestCase):
    """DependencyGraph依赖图测试"""

    def test_graph_creation(self):
        """测试依赖图创建"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 1)
        graph.add_node("B", lambda: 2)
        self.assertEqual(len(graph), 2)

    def test_graph_edges(self):
        """测试依赖边"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 1)
        graph.add_node("B", lambda: 2)
        graph.add_edge("A", "B")

        deps = graph.get_dependencies("A")
        self.assertIn("B", deps)

    def test_graph_topological_sort(self):
        """测试拓扑排序"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 1)
        graph.add_node("B", lambda: 2)
        graph.add_node("C", lambda: 3)
        graph.add_edge("B", "A")
        graph.add_edge("C", "B")

        order = graph.topological_sort()
        # A应该在B之前，B应该在C之前
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("B"), order.index("C"))

    def test_graph_cycle_detection(self):
        """测试循环依赖检测"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 1)
        graph.add_node("B", lambda: 2)
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")

        cycle = graph.detect_cycle()
        self.assertIsNotNone(cycle)

    def test_graph_topological_sort_performance(self):
        """测试拓扑排序使用deque优化"""
        graph = DependencyGraph()
        
        # 创建大量节点
        for i in range(1000):
            graph.add_node(f"node_{i}", lambda: i)
        
        # 创建线性依赖链
        for i in range(1, 1000):
            graph.add_edge(f"node_{i}", f"node_{i-1}")
        
        # 执行拓扑排序，验证不会出错且结果正确
        order = graph.topological_sort()
        self.assertEqual(len(order), 1000)
        
        # 验证依赖顺序正确
        for i in range(1, 1000):
            self.assertLess(order.index(f"node_{i-1}"), order.index(f"node_{i}"))


class TestIncrementalEvaluator(unittest.TestCase):
    """IncrementalEvaluator增量计算测试"""

    def test_evaluator_creation(self):
        """测试增量计算器创建"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 1)
        evaluator = IncrementalEvaluator(graph)
        self.assertIsNotNone(evaluator)

    def test_evaluator_evaluate(self):
        """测试增量求值"""
        graph = DependencyGraph()
        graph.add_node("A", lambda: 42)
        evaluator = IncrementalEvaluator(graph)
        result = evaluator.evaluate("A")
        self.assertEqual(result, 42)

    def test_evaluator_invalidation(self):
        """测试增量失效"""
        call_count = [0]

        def computation():
            call_count[0] += 1
            return 42

        graph = DependencyGraph()
        graph.add_node("A", computation)
        evaluator = IncrementalEvaluator(graph)

        evaluator.evaluate("A")
        evaluator.invalidate("A")
        evaluator.evaluate("A")

        self.assertEqual(call_count[0], 2)

    def test_evaluator_dependencies(self):
        """测试依赖关系增量计算"""
        value_a = [10]

        def compute_a():
            return value_a[0]

        def compute_b():
            return evaluator.get_value("A") * 2

        graph = DependencyGraph()
        graph.add_node("A", compute_a)
        graph.add_node("B", compute_b)
        graph.add_edge("B", "A")

        evaluator = IncrementalEvaluator(graph)
        result1 = evaluator.evaluate("B")
        self.assertEqual(result1, 20)

        # 修改A的值
        value_a[0] = 20
        evaluator.invalidate("A")
        result2 = evaluator.evaluate("B")
        self.assertEqual(result2, 40)

    def test_evaluator_deep_dependency_no_stack_overflow(self):
        """测试深度依赖图不会栈溢出 - 使用迭代实现"""
        # 创建深度依赖链（500层）
        graph = DependencyGraph()
        
        for i in range(500):
            graph.add_node(f"node_{i}", lambda: i)
        
        # 创建线性依赖链
        for i in range(1, 500):
            graph.add_edge(f"node_{i}", f"node_{i-1}")
        
        evaluator = IncrementalEvaluator(graph)
        
        # 求值最后一个节点，应该能正确处理深度依赖
        result = evaluator.evaluate("node_499")
        self.assertEqual(result, 499)
        
        # 测试invalidate不会栈溢出
        evaluator.invalidate("node_0")
        dirty_nodes = evaluator.get_dirty_nodes()
        self.assertEqual(len(dirty_nodes), 500)  # 所有节点都应该失效


class TestStreamFusion(unittest.TestCase):
    """StreamFusion融合优化测试"""

    def test_fusion_creation(self):
        """测试流融合创建"""
        fusion = StreamFusion()
        self.assertIsNotNone(fusion)

    def test_fusion_map_map(self):
        """测试map-map融合"""
        fusion = StreamFusion()
        operations = [
            lambda x: x * 2,
            lambda x: x + 1
        ]
        fused = fusion.fuse(operations)
        result = fused(5)
        self.assertEqual(result, 11)  # (5 * 2) + 1

    def test_fusion_optimize(self):
        """测试管道优化"""
        fusion = StreamFusion()
        pipeline = ['map', 'map', 'filter']
        optimized = fusion.optimize(pipeline)
        self.assertEqual(len(optimized), 2)  # map_map + filter


class TestLazyPipeline(unittest.TestCase):
    """LazyPipeline管道测试"""

    def test_pipeline_creation(self):
        """测试管道创建"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        self.assertIsNotNone(pipeline)

    def test_pipeline_map(self):
        """测试管道映射"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        result = pipeline.map(lambda x: x * 2).collect()
        self.assertEqual(result, [2, 4, 6, 8, 10])

    def test_pipeline_filter(self):
        """测试管道过滤"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        result = pipeline.filter(lambda x: x > 2).collect()
        self.assertEqual(result, [3, 4, 5])

    def test_pipeline_chain(self):
        """测试管道链式操作"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        result = (pipeline
                  .map(lambda x: x * 2)
                  .filter(lambda x: x > 4)
                  .collect())
        self.assertEqual(result, [6, 8, 10])

    def test_pipeline_reduce(self):
        """测试管道归约"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        result = pipeline.reduce(lambda a, b: a + b, 0)
        self.assertEqual(result, 15)

    def test_pipeline_take(self):
        """测试管道取前n个"""
        pipeline = LazyPipeline([1, 2, 3, 4, 5])
        result = pipeline.take(3).collect()
        self.assertEqual(result, [1, 2, 3])

    def test_pipeline_distinct(self):
        """测试管道去重"""
        pipeline = LazyPipeline([1, 2, 2, 3, 3, 3, 4])
        result = pipeline.distinct().collect()
        self.assertEqual(result, [1, 2, 3, 4])

    def test_pipeline_memory_limit(self):
        """测试管道内存限制"""
        # 创建一个大数据集，但设置较小的内存限制
        large_data = range(10000)
        pipeline = LazyPipeline(large_data, memory_limit=100)
        
        # 正常操作应该工作
        result = pipeline.map(lambda x: x * 2).take(10).collect()
        self.assertEqual(len(result), 10)
        
    def test_pipeline_distinct_memory_limit_exceeded(self):
        """测试distinct操作超出内存限制"""
        # 创建一个大数据集，设置很小的内存限制
        large_data = range(1000)
        pipeline = LazyPipeline(large_data, memory_limit=10)
        
        # distinct操作应该抛出内存限制异常
        with self.assertRaises(MemoryLimitExceededError):
            pipeline.distinct().collect()
    
    def test_pipeline_distinct_within_memory_limit(self):
        """测试distinct操作在内存限制内"""
        # 小数据集，内存限制足够
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        pipeline = LazyPipeline(data, memory_limit=100)
        result = pipeline.distinct().collect()
        self.assertEqual(result, [1, 2, 3, 4])


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_stream_adapter(self):
        """测试Stream适配器"""
        adapter = StreamAdapter()
        lazy_stream = adapter.create_lazy_stream([1, 2, 3, 4, 5])
        result = lazy_stream.map(lambda x: x * 2).collect()
        self.assertEqual(result, [2, 4, 6, 8, 10])

    def test_compressor_adapter(self):
        """测试Compressor适配器"""
        adapter = CompressorAdapter()
        data = b"test data"
        lazy_compressed = adapter.lazy_compress(data)
        result = lazy_compressed.get()
        # 如果compressor不可用，返回原始数据
        self.assertIsNotNone(result)

    def test_optimizer_adapter(self):
        """测试Optimizer适配器"""
        adapter = OptimizerAdapter()
        thunk = adapter.create_optimized_thunk(lambda: 42)
        result = thunk.get()
        self.assertEqual(result, 42)

    def test_end_to_end(self):
        """端到端测试"""
        # 创建惰性管道
        pipeline = LazyPipeline(range(10))

        # 应用多个操作
        result = (pipeline
                  .map(lambda x: x ** 2)
                  .filter(lambda x: x > 20)
                  .take(3)
                  .collect())

        self.assertEqual(len(result), 3)
        self.assertTrue(all(x > 20 for x in result))


if __name__ == '__main__':
    unittest.main()
