"""
memopt 基准测试

测试各模块的性能指标。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import Callable, Any, Dict, List


class BenchmarkResult:
    """基准测试结果"""
    
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.memory: List[int] = []
    
    def add(self, time_ms: float, memory_bytes: int = 0):
        self.times.append(time_ms)
        self.memory.append(memory_bytes)
    
    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0
    
    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0
    
    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  平均时间: {self.avg_time:.3f} ms\n"
            f"  最小时间: {self.min_time:.3f} ms\n"
            f"  最大时间: {self.max_time:.3f} ms"
        )


def benchmark(func: Callable, iterations: int = 100, **kwargs) -> BenchmarkResult:
    """
    运行基准测试
    
    Args:
        func: 要测试的函数
        iterations: 迭代次数
        **kwargs: 传递给函数的参数
    
    Returns:
        BenchmarkResult: 测试结果
    """
    result = BenchmarkResult(func.__name__)
    
    for _ in range(iterations):
        start = time.perf_counter()
        func(**kwargs)
        end = time.perf_counter()
        result.add((end - start) * 1000)
    
    return result


def run_all_benchmarks() -> Dict[str, BenchmarkResult]:
    """运行所有基准测试"""
    results = {}
    
    results.update(benchmark_mem_mapper())
    results.update(benchmark_data_compressor())
    results.update(benchmark_stream_processor())
    results.update(benchmark_mem_optimizer())
    results.update(benchmark_lazy_evaluator())
    results.update(benchmark_sparse_array())
    results.update(benchmark_mem_monitor())
    results.update(benchmark_buffer_manager())
    
    return results


def benchmark_mem_mapper() -> Dict[str, BenchmarkResult]:
    """内存映射基准测试"""
    results = {}
    
    try:
        from mem_mapper import MemoryMapper, MapperConfig
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test" * 1024 * 1024)
            temp_path = f.name
        
        config = MapperConfig(use_numa=False, use_huge_pages=False)
        mapper = MemoryMapper(config)
        
        def map_unmap():
            region = mapper.map_file(temp_path, mode="readonly", size=4*1024*1024)
            mapper.unmap(region)
        
        results["mem_mapper"] = benchmark(map_unmap, iterations=50)
        
        os.unlink(temp_path)
        
    except ImportError as e:
        print(f"  [跳过] mem_mapper: {e}")
    
    return results


def benchmark_data_compressor() -> Dict[str, BenchmarkResult]:
    """数据压缩基准测试"""
    results = {}
    
    try:
        from data_compressor import DataCompressor, CompressionConfig, CompressionAlgorithm, CompressionLevel
        
        data = b"Hello, World!" * 10000
        
        def compress_zstd():
            config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD, level=CompressionLevel.FAST)
            compressor = DataCompressor(config)
            return compressor.compress(data)
        
        def compress_lz4():
            config = CompressionConfig(algorithm=CompressionAlgorithm.LZ4, level=CompressionLevel.FAST)
            compressor = DataCompressor(config)
            return compressor.compress(data)
        
        results["compress_zstd"] = benchmark(compress_zstd)
        results["compress_lz4"] = benchmark(compress_lz4)
        
    except ImportError as e:
        print(f"  [跳过] data_compressor: {e}")
    
    return results


def benchmark_stream_processor() -> Dict[str, BenchmarkResult]:
    """流式处理基准测试"""
    results = {}
    
    try:
        from stream_processor import CollectionSource
        
        data = list(range(10000))
        
        def stream_process():
            source = CollectionSource("source", data)
            return list(source.read())
        
        results["stream_process"] = benchmark(stream_process)
        
    except ImportError as e:
        print(f"  [跳过] stream_processor: {e}")
    
    return results


def benchmark_mem_optimizer() -> Dict[str, BenchmarkResult]:
    """内存优化基准测试"""
    results = {}
    
    try:
        from mem_optimizer import MemoryPool, OptimizerConfig
        
        config = OptimizerConfig(
            total_memory=10*1024*1024,
            max_allocation_size=1*1024*1024
        )
        pool = MemoryPool(config=config)
        
        def allocate_deallocate():
            result = pool.allocate(1024)
            if result.success:
                pool.deallocate(result.address)
        
        results["mem_optimizer"] = benchmark(allocate_deallocate)
        
    except ImportError as e:
        print(f"  [跳过] mem_optimizer: {e}")
    
    return results


def benchmark_lazy_evaluator() -> Dict[str, BenchmarkResult]:
    """惰性计算基准测试"""
    results = {}
    
    try:
        from lazy_evaluator import Lazy, memoize
        
        @memoize(max_size=1000)
        def fib(n):
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)
        
        def lazy_fib():
            lazy = Lazy(lambda: fib(30))
            return lazy.force()
        
        results["lazy_evaluator"] = benchmark(lazy_fib)
        
    except ImportError as e:
        print(f"  [跳过] lazy_evaluator: {e}")
    
    return results


def benchmark_sparse_array() -> Dict[str, BenchmarkResult]:
    """稀疏数组基准测试"""
    results = {}
    
    try:
        from sparse_array import SparseArray
        
        dense = np.random.rand(100, 100)
        dense[dense < 0.9] = 0
        sparse = SparseArray.from_dense(dense)
        
        def spmv():
            x = np.random.rand(100)
            return sparse @ x
        
        results["sparse_spmv"] = benchmark(spmv)
        
    except ImportError as e:
        print(f"  [跳过] sparse_array: {e}")
    
    return results


def benchmark_mem_monitor() -> Dict[str, BenchmarkResult]:
    """内存监控基准测试"""
    results = {}
    
    try:
        from mem_monitor import MemoryMonitor, MonitorConfig
        
        config = MonitorConfig()
        
        def monitor_snapshot():
            monitor = MemoryMonitor(config)
            return monitor.get_snapshot()
        
        results["mem_monitor"] = benchmark(monitor_snapshot, iterations=50)
        
    except ImportError as e:
        print(f"  [跳过] mem_monitor: {e}")
    
    return results


def benchmark_buffer_manager() -> Dict[str, BenchmarkResult]:
    """缓冲区管理基准测试"""
    results = {}
    
    try:
        from buffer_manager import BufferPool, SPSCQueue
        
        pool = BufferPool(buffer_size=4096, num_buffers=16)
        queue = SPSCQueue(capacity=1000)
        
        def pool_acquire_release():
            buffer = pool.acquire()
            pool.release(buffer)
        
        def queue_enqueue_dequeue():
            queue.enqueue("item")
            return queue.dequeue()
        
        results["buffer_pool"] = benchmark(pool_acquire_release)
        results["spsc_queue"] = benchmark(queue_enqueue_dequeue)
        
    except ImportError as e:
        print(f"  [跳过] buffer_manager: {e}")
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("memopt 基准测试")
    print("=" * 50)
    print()
    
    results = run_all_benchmarks()
    
    if not results:
        print("警告: 没有成功运行任何基准测试")
    else:
        print(f"成功运行 {len(results)} 项基准测试")
        print()
        for name, result in results.items():
            print(result)
            print()
