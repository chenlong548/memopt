"""
data_compressor 性能测试

测试压缩速度、压缩比、内存使用和并发性能。
"""

import unittest
import os
import sys
import time
import io
import threading
import json
from typing import Dict, List, Any

# 添加项目路径 - 确保data_compressor模块可以被导入
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np

from data_compressor import (
    DataCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressionStats,
    CompressedData,
    DataType,
)

from data_compressor.algorithms.zstd_wrapper import ZstdCompressor, ZSTD_AVAILABLE
from data_compressor.algorithms.lz4_wrapper import LZ4Compressor, LZ4_AVAILABLE
from data_compressor.algorithms.brotli_wrapper import BrotliCompressor, BROTLI_AVAILABLE
from data_compressor.stream.stream_compressor import StreamCompressor


class PerformanceTestResult:
    """性能测试结果"""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def add_result(self, test_name: str, metrics: Dict[str, Any]):
        """添加测试结果"""
        self.results[test_name] = metrics

    def get_summary(self) -> str:
        """获取摘要"""
        lines = ["=" * 60]
        lines.append("性能测试结果摘要")
        lines.append("=" * 60)

        for test_name, metrics in self.results.items():
            lines.append(f"\n{test_name}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.results


class TestCompressionSpeed(unittest.TestCase):
    """压缩速度测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.compressor = DataCompressor()
        cls.test_sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024]  # 1KB, 10KB, 100KB, 1MB
        cls.results = PerformanceTestResult()

    def test_compression_speed_by_size(self):
        """测试不同数据大小的压缩速度"""
        for size in self.test_sizes:
            data = os.urandom(size)

            start_time = time.time()
            compressed = self.compressor.compress(data)
            compress_time = time.time() - start_time

            start_time = time.time()
            decompressed = self.compressor.decompress(compressed)
            decompress_time = time.time() - start_time

            throughput = (size / (1024 * 1024)) / compress_time if compress_time > 0 else 0

            self.results.add_result(
                f"speed_size_{size}",
                {
                    'data_size_bytes': size,
                    'data_size_kb': size / 1024,
                    'compress_time_s': compress_time,
                    'decompress_time_s': decompress_time,
                    'throughput_mbps': throughput,
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': compressed.stats.compression_ratio if compressed.stats else 0,
                }
            )

            # 验证正确性
            self.assertEqual(decompressed, data)

    def test_algorithm_speed_comparison(self):
        """测试不同算法的速度比较"""
        data = os.urandom(100 * 1024)  # 100KB
        algorithms = []

        if ZSTD_AVAILABLE:
            algorithms.append(CompressionAlgorithm.ZSTD)
        if LZ4_AVAILABLE:
            algorithms.append(CompressionAlgorithm.LZ4)
        if BROTLI_AVAILABLE:
            algorithms.append(CompressionAlgorithm.BROTLI)

        for alg in algorithms:
            config = CompressionConfig(algorithm=alg)

            # 多次测试取平均
            times = []
            for _ in range(5):
                start_time = time.time()
                compressed = self.compressor.compress(data, config)
                compress_time = time.time() - start_time
                times.append(compress_time)

            avg_time = sum(times) / len(times)
            throughput = (len(data) / (1024 * 1024)) / avg_time if avg_time > 0 else 0

            self.results.add_result(
                f"algorithm_speed_{alg.value}",
                {
                    'algorithm': alg.value,
                    'avg_compress_time_s': avg_time,
                    'throughput_mbps': throughput,
                }
            )

    def test_compression_level_speed(self):
        """测试不同压缩级别的速度"""
        if not ZSTD_AVAILABLE:
            self.skipTest("zstandard library not available")

        data = os.urandom(100 * 1024)  # 100KB
        levels = [
            CompressionLevel.FASTEST,
            CompressionLevel.FAST,
            CompressionLevel.BALANCED,
            CompressionLevel.HIGH,
            CompressionLevel.MAXIMUM,
        ]

        for level in levels:
            config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD, level=level)

            start_time = time.time()
            compressed = self.compressor.compress(data, config)
            compress_time = time.time() - start_time

            self.results.add_result(
                f"level_speed_{level.name}",
                {
                    'level': level.name,
                    'compress_time_s': compress_time,
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': len(data) / compressed.compressed_size,
                }
            )


class TestCompressionRatio(unittest.TestCase):
    """压缩比测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.compressor = DataCompressor()
        cls.results = PerformanceTestResult()

    def test_compression_ratio_by_data_type(self):
        """测试不同数据类型的压缩比"""
        test_cases = {
            'text': b'This is a text string for compression testing. ' * 100,
            'json': json.dumps({'key': 'value', 'numbers': list(range(100))}).encode() * 10,
            'repeated': b'AAAA' * 1000,
            'sparse': self._create_sparse_data(10000),
            'random': os.urandom(10000),
            'binary_pattern': bytes(range(256)) * 40,
        }

        for name, data in test_cases.items():
            compressed = self.compressor.compress(data)
            ratio = len(data) / compressed.compressed_size

            self.results.add_result(
                f"ratio_{name}",
                {
                    'data_type': name,
                    'original_size': len(data),
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': ratio,
                    'space_saving': (1 - compressed.compressed_size / len(data)) * 100,
                }
            )

    def test_algorithm_ratio_comparison(self):
        """测试不同算法的压缩比比较"""
        # 使用文本数据（通常有较好的压缩比）
        data = b'This is a sample text for compression ratio testing. ' * 200

        algorithms = []
        if ZSTD_AVAILABLE:
            algorithms.append(CompressionAlgorithm.ZSTD)
        if LZ4_AVAILABLE:
            algorithms.append(CompressionAlgorithm.LZ4)
        if BROTLI_AVAILABLE:
            algorithms.append(CompressionAlgorithm.BROTLI)

        for alg in algorithms:
            config = CompressionConfig(algorithm=alg, level=CompressionLevel.MAXIMUM)
            compressed = self.compressor.compress(data, config)
            ratio = len(data) / compressed.compressed_size

            self.results.add_result(
                f"algorithm_ratio_{alg.value}",
                {
                    'algorithm': alg.value,
                    'original_size': len(data),
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': ratio,
                }
            )

    def test_model_compression_ratio(self):
        """测试模型数据压缩比"""
        # 模拟FP32模型权重
        fp32_weights = np.random.randn(1000).astype(np.float32)
        fp32_data = fp32_weights.tobytes()

        # 模拟BF16模型权重
        bf16_weights = np.random.randn(1000).astype(np.float16)
        bf16_data = bf16_weights.tobytes()

        # 压缩测试
        for name, data in [('fp32', fp32_data), ('bf16', bf16_data)]:
            compressed = self.compressor.compress(data)
            ratio = len(data) / compressed.compressed_size

            self.results.add_result(
                f"model_ratio_{name}",
                {
                    'model_type': name,
                    'original_size': len(data),
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': ratio,
                }
            )

    def _create_sparse_data(self, size: int) -> bytes:
        """创建稀疏数据"""
        data = bytearray(size)
        # 只设置少量非零值
        for i in range(0, size, 100):
            data[i] = i % 256
        return bytes(data)


class TestMemoryUsage(unittest.TestCase):
    """内存使用测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.compressor = DataCompressor()
        cls.results = PerformanceTestResult()

    def test_memory_efficiency(self):
        """测试内存效率"""
        try:
            import psutil
            import os as os_module
            process = psutil.Process(os_module.getpid())
        except ImportError:
            self.skipTest("psutil not available for memory testing")

        # 记录初始内存
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 压缩多个文件
        data_sizes = [100 * 1024, 500 * 1024, 1024 * 1024]
        peak_memory = initial_memory

        for size in data_sizes:
            data = os.urandom(size)
            compressed = self.compressor.compress(data)
            decompressed = self.compressor.decompress(compressed)

            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)

            del data, compressed, decompressed

        memory_increase = peak_memory - initial_memory

        self.results.add_result(
            'memory_efficiency',
            {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
            }
        )

    def test_stream_memory_usage(self):
        """测试流式压缩内存使用"""
        try:
            import psutil
            import os as os_module
            process = psutil.Process(os_module.getpid())
        except ImportError:
            self.skipTest("psutil not available for memory testing")

        initial_memory = process.memory_info().rss / (1024 * 1024)

        # 流式压缩大数据
        data_size = 10 * 1024 * 1024  # 10MB
        data = os.urandom(data_size)

        stream_compressor = StreamCompressor(CompressionConfig(chunk_size=1024 * 1024))
        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        stream_compressor.compress_stream(input_stream, output_stream)

        peak_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = peak_memory - initial_memory

        self.results.add_result(
            'stream_memory',
            {
                'data_size_mb': data_size / (1024 * 1024),
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'memory_efficiency_ratio': data_size / (1024 * 1024) / memory_increase if memory_increase > 0 else 0,
            }
        )


class TestConcurrencyPerformance(unittest.TestCase):
    """并发性能测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.results = PerformanceTestResult()

    def test_parallel_compression(self):
        """测试并行压缩"""
        from data_compressor.stream.stream_compressor import StreamCompressor

        data_size = 5 * 1024 * 1024  # 5MB
        data = os.urandom(data_size)

        # 串行压缩
        compressor = DataCompressor()
        start_time = time.time()
        compressed = compressor.compress(data)
        serial_time = time.time() - start_time

        # 并行压缩
        stream_compressor = StreamCompressor(CompressionConfig(chunk_size=512 * 1024))
        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        start_time = time.time()
        stream_compressor.compress_parallel(input_stream, output_stream, num_workers=4)
        parallel_time = time.time() - start_time

        speedup = serial_time / parallel_time if parallel_time > 0 else 0

        self.results.add_result(
            'parallel_compression',
            {
                'data_size_mb': data_size / (1024 * 1024),
                'serial_time_s': serial_time,
                'parallel_time_s': parallel_time,
                'speedup': speedup,
                'num_workers': 4,
            }
        )

    def test_concurrent_requests(self):
        """测试并发请求"""
        compressor = DataCompressor()
        data = os.urandom(10 * 1024)  # 10KB
        num_requests = 100
        num_threads = 10

        results = {'success': 0, 'failure': 0, 'times': []}
        lock = threading.Lock()

        def compress_worker():
            start_time = time.time()
            try:
                compressed = compressor.compress(data)
                decompressed = compressor.decompress(compressed)
                if decompressed == data:
                    with lock:
                        results['success'] += 1
                        results['times'].append(time.time() - start_time)
                else:
                    with lock:
                        results['failure'] += 1
            except Exception:
                with lock:
                    results['failure'] += 1

        # 创建线程
        threads = []
        start_time = time.time()

        for _ in range(num_requests):
            t = threading.Thread(target=compress_worker)
            threads.append(t)
            t.start()

            # 控制并发数
            while len([t for t in threads if t.is_alive()]) >= num_threads:
                time.sleep(0.001)

        # 等待完成
        for t in threads:
            t.join()

        total_time = time.time() - start_time
        avg_time = sum(results['times']) / len(results['times']) if results['times'] else 0
        throughput = results['success'] / total_time if total_time > 0 else 0

        self.results.add_result(
            'concurrent_requests',
            {
                'total_requests': num_requests,
                'successful': results['success'],
                'failed': results['failure'],
                'total_time_s': total_time,
                'avg_request_time_s': avg_time,
                'throughput_req_per_s': throughput,
            }
        )


class TestStreamPerformance(unittest.TestCase):
    """流式处理性能测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.results = PerformanceTestResult()

    def test_stream_throughput(self):
        """测试流式处理吞吐量"""
        data_sizes = [1024 * 1024, 5 * 1024 * 1024, 10 * 1024 * 1024]  # 1MB, 5MB, 10MB

        for size in data_sizes:
            data = os.urandom(size)
            stream_compressor = StreamCompressor(CompressionConfig(chunk_size=512 * 1024))

            # 压缩
            input_stream = io.BytesIO(data)
            output_stream = io.BytesIO()

            start_time = time.time()
            stats = stream_compressor.compress_stream(input_stream, output_stream)
            compress_time = time.time() - start_time

            # 解压
            output_stream.seek(0)
            result_stream = io.BytesIO()

            start_time = time.time()
            stream_compressor.decompress_stream(output_stream, result_stream)
            decompress_time = time.time() - start_time

            throughput = (size / (1024 * 1024)) / compress_time if compress_time > 0 else 0

            self.results.add_result(
                f"stream_throughput_{size // 1024 // 1024}MB",
                {
                    'data_size_mb': size / (1024 * 1024),
                    'compress_time_s': compress_time,
                    'decompress_time_s': decompress_time,
                    'throughput_mbps': throughput,
                    'compression_ratio': stats.compression_ratio,
                }
            )


class TestBenchmark(unittest.TestCase):
    """综合基准测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.compressor = DataCompressor()
        cls.results = PerformanceTestResult()

    def test_full_benchmark(self):
        """完整基准测试"""
        # 测试数据集
        test_datasets = {
            'text_1kb': b'Text data for compression benchmark testing. ' * 20,
            'text_100kb': b'Text data for compression benchmark testing. ' * 2000,
            'binary_1kb': os.urandom(1024),
            'binary_100kb': os.urandom(100 * 1024),
            'sparse_10kb': self._create_sparse_data(10 * 1024),
            'repeated_10kb': b'PATTERN' * 1400,
        }

        algorithms = []
        if ZSTD_AVAILABLE:
            algorithms.append(CompressionAlgorithm.ZSTD)
        if LZ4_AVAILABLE:
            algorithms.append(CompressionAlgorithm.LZ4)
        if BROTLI_AVAILABLE:
            algorithms.append(CompressionAlgorithm.BROTLI)

        benchmark_results = {}

        for dataset_name, data in test_datasets.items():
            dataset_results = {}

            for alg in algorithms:
                config = CompressionConfig(algorithm=alg)

                # 压缩
                start_time = time.time()
                compressed = self.compressor.compress(data, config)
                compress_time = time.time() - start_time

                # 解压
                start_time = time.time()
                decompressed = self.compressor.decompress(compressed)
                decompress_time = time.time() - start_time

                # 验证
                correct = decompressed == data

                dataset_results[alg.value] = {
                    'compress_time_ms': compress_time * 1000,
                    'decompress_time_ms': decompress_time * 1000,
                    'compressed_size': compressed.compressed_size,
                    'compression_ratio': len(data) / compressed.compressed_size,
                    'correct': correct,
                }

            benchmark_results[dataset_name] = dataset_results

        self.results.add_result('full_benchmark', benchmark_results)

    def test_adaptive_selection_performance(self):
        """测试自适应选择性能"""
        test_cases = [
            ('text', b'Text data ' * 1000),
            ('binary', os.urandom(10 * 1024)),
            ('sparse', self._create_sparse_data(10 * 1024)),
        ]

        results = {}

        for name, data in test_cases:
            config = CompressionConfig(algorithm=CompressionAlgorithm.AUTO)

            start_time = time.time()
            compressed = self.compressor.compress(data, config)
            compress_time = time.time() - start_time

            results[name] = {
                'selected_algorithm': compressed.algorithm.value,
                'compress_time_ms': compress_time * 1000,
                'compression_ratio': len(data) / compressed.compressed_size,
            }

        self.results.add_result('adaptive_selection', results)

    def _create_sparse_data(self, size: int) -> bytes:
        """创建稀疏数据"""
        data = bytearray(size)
        for i in range(0, size, 50):
            data[i] = i % 256
        return bytes(data)


class PerformanceReport:
    """测试报告生成器"""

    def __init__(self):
        self.all_results: Dict[str, PerformanceTestResult] = {}

    def add_test_results(self, test_class: str, results: PerformanceTestResult):
        """添加测试结果"""
        self.all_results[test_class] = results

    def generate_report(self) -> str:
        """生成报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("data_compressor 性能测试报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        for test_class, results in self.all_results.items():
            lines.append(f"\n{'-' * 50}")
            lines.append(f"{test_class}")
            lines.append(f"{'-' * 50}")

            for test_name, metrics in results.results.items():
                lines.append(f"\n  {test_name}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"    {key}: {value:.4f}")
                    elif isinstance(value, dict):
                        lines.append(f"    {key}:")
                        for k, v in value.items():
                            if isinstance(v, float):
                                lines.append(f"      {k}: {v:.4f}")
                            else:
                                lines.append(f"      {k}: {v}")
                    else:
                        lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def save_report(self, filepath: str):
        """保存报告"""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)


def run_performance_tests():
    """运行所有性能测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCompressionSpeed))
    suite.addTests(loader.loadTestsFromTestCase(TestCompressionRatio))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryUsage))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrencyPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmark))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_performance_tests()
