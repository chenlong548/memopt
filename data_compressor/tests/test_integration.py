"""
data_compressor 集成测试

测试完整流程和模块间集成。
"""

import unittest
import os
import sys
import tempfile
import io
import json
import time
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)

from data_compressor import (
    DataCompressor,
    StreamCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressedData,
    DataType,
    CompressionError,
    DecompressionError,
)

from data_compressor.integration.mem_mapper_integration import (
    MemMapperIntegration,
    CompressedMemoryMapper,
)


class TestEndToEndCompression(unittest.TestCase):
    """端到端压缩测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()

    def test_simple_text_compression(self):
        """测试简单文本压缩"""
        text = b"Hello, World! This is a simple text compression test."
        compressed = self.compressor.compress(text)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, text)

    def test_json_data_compression(self):
        """测试JSON数据压缩"""
        data = {
            'name': 'test',
            'values': list(range(100)),
            'nested': {'key': 'value'}
        }
        json_bytes = json.dumps(data).encode('utf-8')

        compressed = self.compressor.compress(json_bytes)
        decompressed = self.compressor.decompress(compressed)

        self.assertEqual(decompressed, json_bytes)
        self.assertEqual(json.loads(decompressed), data)

    def test_binary_data_compression(self):
        """测试二进制数据压缩"""
        binary_data = os.urandom(10 * 1024)  # 10KB

        compressed = self.compressor.compress(binary_data)
        decompressed = self.compressor.decompress(compressed)

        self.assertEqual(decompressed, binary_data)

    def test_numpy_array_compression(self):
        """测试NumPy数组压缩"""
        arr = np.random.randn(100, 100)
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        numpy_data = buffer.read()

        compressed = self.compressor.compress(numpy_data)
        decompressed = self.compressor.decompress(compressed)

        # 验证数据
        decompressed_buffer = io.BytesIO(decompressed)
        loaded_arr = np.load(decompressed_buffer)

        np.testing.assert_array_equal(arr, loaded_arr)

    def test_large_file_compression(self):
        """测试大文件压缩"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            data = os.urandom(1024 * 1024)  # 1MB
            f.write(data)

        try:
            # 读取并压缩
            with open(temp_path, 'rb') as f:
                file_data = f.read()

            compressed = self.compressor.compress(file_data)
            decompressed = self.compressor.decompress(compressed)

            self.assertEqual(decompressed, file_data)
        finally:
            os.unlink(temp_path)

    def test_multiple_compression_rounds(self):
        """测试多轮压缩"""
        data = b"Test data for multiple compression rounds."

        # 第一轮
        compressed1 = self.compressor.compress(data)
        decompressed1 = self.compressor.decompress(compressed1)
        self.assertEqual(decompressed1, data)

        # 第二轮（压缩已压缩的数据）
        compressed2 = self.compressor.compress(compressed1.data)
        decompressed2 = self.compressor.decompress(compressed2)
        self.assertEqual(decompressed2, compressed1.data)


class TestAlgorithmIntegration(unittest.TestCase):
    """算法集成测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()
        self.test_data = {
            'text': b'Text data for algorithm testing.' * 100,
            'binary': os.urandom(10 * 1024),
            'mixed': b'ABC' * 1000 + b'\x00' * 1000 + b'XYZ' * 1000,
        }

    def test_all_algorithms_produce_valid_output(self):
        """测试所有算法产生有效输出"""
        algorithms = [
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.BROTLI,
        ]

        for data_name, data in self.test_data.items():
            for alg in algorithms:
                with self.subTest(data=data_name, algorithm=alg.value):
                    try:
                        config = CompressionConfig(algorithm=alg)
                        compressed = self.compressor.compress(data, config)
                        decompressed = self.compressor.decompress(compressed)
                        self.assertEqual(decompressed, data)
                    except ImportError:
                        self.skipTest(f"{alg.value} library not available")

    def test_algorithm_switching(self):
        """测试算法切换"""
        data = b"Data for algorithm switching test."

        # 使用ZSTD
        config1 = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD)
        try:
            compressed1 = self.compressor.compress(data, config1)
            self.assertEqual(compressed1.algorithm, CompressionAlgorithm.ZSTD)
        except ImportError:
            self.skipTest("zstandard library not available")

        # 切换到LZ4
        config2 = CompressionConfig(algorithm=CompressionAlgorithm.LZ4)
        try:
            compressed2 = self.compressor.compress(data, config2)
            self.assertEqual(compressed2.algorithm, CompressionAlgorithm.LZ4)
        except ImportError:
            pass

    def test_auto_algorithm_selection(self):
        """测试自动算法选择"""
        test_cases = [
            (b'Text data ' * 1000, 'text'),
            (os.urandom(1024), 'random'),
            (b'AAAA' * 1000, 'repeated'),
        ]

        for data, description in test_cases:
            with self.subTest(data_type=description):
                config = CompressionConfig(algorithm=CompressionAlgorithm.AUTO)
                compressed = self.compressor.compress(data, config)

                # 应该选择一个有效算法
                self.assertIn(compressed.algorithm, [
                    CompressionAlgorithm.ZSTD,
                    CompressionAlgorithm.LZ4,
                    CompressionAlgorithm.BROTLI,
                ])

                # 解压应该成功
                decompressed = self.compressor.decompress(compressed)
                self.assertEqual(decompressed, data)


class TestStreamIntegration(unittest.TestCase):
    """流式处理集成测试"""

    def test_stream_roundtrip(self):
        """测试流式往返"""
        data = os.urandom(100 * 1024)  # 100KB

        # 压缩
        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        compressor = StreamCompressor()
        stats = compressor.compress_stream(input_stream, output_stream)

        self.assertEqual(stats.original_size, len(data))
        self.assertGreater(stats.compressed_size, 0)

        # 解压
        output_stream.seek(0)
        result_stream = io.BytesIO()

        compressor.decompress_stream(output_stream, result_stream)

        result_stream.seek(0)
        decompressed = result_stream.read()

        self.assertEqual(len(decompressed), len(data))

    def test_stream_with_progress_callback(self):
        """测试带进度回调的流式处理"""
        data = os.urandom(50 * 1024)
        progress_updates = []

        def progress_callback(current, total):
            progress_updates.append((current, total))

        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        compressor = StreamCompressor()
        compressor.compress_stream(input_stream, output_stream, progress_callback)

        self.assertGreater(len(progress_updates), 0)
        # 最后一次更新应该是完成状态
        self.assertEqual(progress_updates[-1][0], len(data))

    def test_parallel_stream_compression(self):
        """测试并行流式压缩"""
        data = os.urandom(500 * 1024)  # 500KB

        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        compressor = StreamCompressor()
        stats = compressor.compress_parallel(input_stream, output_stream, num_workers=2)

        self.assertEqual(stats.original_size, len(data))


class TestMemMapperIntegration(unittest.TestCase):
    """mem_mapper集成测试"""

    def setUp(self):
        """测试前准备"""
        self.integration = MemMapperIntegration()

    def test_compress_file(self):
        """测试文件压缩"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            temp_path = f.name
            f.write(b'Test data for mem_mapper integration.')

        try:
            # 获取压缩统计
            stats = self.integration.get_compression_stats(temp_path)

            self.assertIn('file_path', stats)
            self.assertIn('file_size', stats)
            self.assertIn('recommended_algorithm', stats)

        finally:
            os.unlink(temp_path)

    def test_compress_decompress_file(self):
        """测试文件压缩解压"""
        original_data = b'Test data for compression and decompression.'

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            original_path = f.name
            f.write(original_data)

        try:
            # 压缩
            compressed_path = original_path + '.dcmp'
            self.integration._compress_file(original_path, CompressionConfig())

            # 读取压缩文件
            compressed = self.integration._read_compressed_file(compressed_path)

            # 解压
            compressor = DataCompressor()
            decompressed = compressor.decompress(compressed)

            self.assertEqual(decompressed, original_data)

        finally:
            if os.path.exists(original_path):
                os.unlink(original_path)
            if os.path.exists(original_path + '.dcmp'):
                os.unlink(original_path + '.dcmp')


class TestStatisticsIntegration(unittest.TestCase):
    """统计集成测试"""

    def test_statistics_collection(self):
        """测试统计收集"""
        compressor = DataCompressor()

        # 执行多次压缩
        for i in range(10):
            data = f"Test data {i}".encode()
            compressed = compressor.compress(data)
            compressor.decompress(compressed)

        # 获取统计摘要
        summary = compressor.get_stats_summary()

        self.assertIn('total_operations', summary)
        self.assertGreater(summary['total_operations'], 0)

    def test_algorithm_stats_tracking(self):
        """测试算法统计跟踪"""
        compressor = DataCompressor()

        # 使用不同算法
        algorithms = [CompressionAlgorithm.ZSTD, CompressionAlgorithm.LZ4]

        for alg in algorithms:
            try:
                config = CompressionConfig(algorithm=alg)
                data = os.urandom(1024)
                compressed = compressor.compress(data, config)
            except ImportError:
                continue

        summary = compressor.get_stats_summary()
        self.assertIn('algorithms', summary)


class TestDataTypeDetectionIntegration(unittest.TestCase):
    """数据类型检测集成测试"""

    def test_text_detection(self):
        """测试文本检测"""
        compressor = DataCompressor()
        text_data = b'This is plain text data for detection testing.'

        analysis = compressor.analyze(text_data)

        self.assertEqual(analysis['data_type'], 'text')

    def test_json_detection(self):
        """测试JSON检测"""
        compressor = DataCompressor()
        json_data = json.dumps({'key': 'value'}).encode()

        analysis = compressor.analyze(json_data)

        self.assertEqual(analysis['data_type'], 'json')

    def test_binary_detection(self):
        """测试二进制检测"""
        compressor = DataCompressor()
        binary_data = os.urandom(1024)

        analysis = compressor.analyze(binary_data)

        # 二进制数据可能被检测为多种类型
        self.assertIn(analysis['data_type'], ['generic', 'binary', 'fp32_tensor'])


class TestFeatureExtractionIntegration(unittest.TestCase):
    """特征提取集成测试"""

    def test_feature_extraction_for_compression(self):
        """测试压缩特征提取"""
        compressor = DataCompressor()

        # 高冗余数据
        high_redundancy = b'AAAA' * 1000
        analysis_high = compressor.analyze(high_redundancy)

        # 低冗余数据
        low_redundancy = os.urandom(4000)
        analysis_low = compressor.analyze(low_redundancy)

        # 高冗余数据应该有更高的估算压缩比
        self.assertGreater(
            analysis_high.get('estimated_compression_ratio', 1.0),
            analysis_low.get('estimated_compression_ratio', 1.0)
        )


class TestBenchmarkIntegration(unittest.TestCase):
    """基准测试集成"""

    def test_benchmark_all_algorithms(self):
        """测试所有算法基准测试"""
        compressor = DataCompressor()
        data = b'Benchmark test data.' * 100

        results = compressor.benchmark(data)

        self.assertIsInstance(results, dict)
        # 至少有一个算法成功
        successful = [r for r in results.values() if r is not None]
        self.assertGreater(len(successful), 0)


class TestErrorRecoveryIntegration(unittest.TestCase):
    """错误恢复集成测试"""

    def test_recovery_from_invalid_data(self):
        """测试从无效数据恢复"""
        compressor = DataCompressor()

        # 有效数据
        valid_data = b'Valid data for compression.'

        # 压缩有效数据
        compressed = compressor.compress(valid_data)

        # 尝试解压损坏的数据
        corrupted = CompressedData(
            data=b'corrupted data',
            algorithm=compressed.algorithm,
            level=compressed.level,
            original_size=compressed.original_size,
            compressed_size=len(b'corrupted data')
        )

        try:
            compressor.decompress(corrupted)
        except (DecompressionError, Exception):
            pass  # 预期的异常

        # 应该仍能处理有效数据
        new_compressed = compressor.compress(valid_data)
        decompressed = compressor.decompress(new_compressed)
        self.assertEqual(decompressed, valid_data)


class TestWorkflowIntegration(unittest.TestCase):
    """工作流集成测试"""

    def test_complete_compression_workflow(self):
        """测试完整压缩工作流"""
        # 1. 创建测试数据
        data = b'Complete workflow test data.' * 100

        # 2. 分析数据
        compressor = DataCompressor()
        analysis = compressor.analyze(data)

        # 3. 根据分析选择算法
        recommended = analysis['recommended_algorithm']
        config = CompressionConfig(algorithm=CompressionAlgorithm(recommended))

        # 4. 压缩
        compressed = compressor.compress(data, config)

        # 5. 验证压缩结果
        self.assertLess(compressed.compressed_size, compressed.original_size)

        # 6. 解压
        decompressed = compressor.decompress(compressed)

        # 7. 验证解压结果
        self.assertEqual(decompressed, data)

        # 8. 获取统计
        summary = compressor.get_stats_summary()
        self.assertGreater(summary['total_operations'], 0)

    def test_batch_compression_workflow(self):
        """测试批量压缩工作流"""
        compressor = DataCompressor()

        # 创建多个测试文件
        test_files = []
        for i in range(5):
            data = f'Batch test file {i} data.'.encode() * 100
            test_files.append(data)

        # 批量压缩
        compressed_files = []
        for data in test_files:
            compressed = compressor.compress(data)
            compressed_files.append(compressed)

        # 批量解压
        decompressed_files = []
        for compressed in compressed_files:
            decompressed = compressor.decompress(compressed)
            decompressed_files.append(decompressed)

        # 验证
        for original, decompressed in zip(test_files, decompressed_files):
            self.assertEqual(original, decompressed)


class TestConfigurationIntegration(unittest.TestCase):
    """配置集成测试"""

    def test_config_propagation(self):
        """测试配置传播"""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.MAXIMUM,
            chunk_size=2048,
            enable_parallel=True,
        )

        compressor = DataCompressor(config)
        data = b'Config propagation test.'

        compressed = compressor.compress(data)
        self.assertEqual(compressed.algorithm, CompressionAlgorithm.ZSTD)
        self.assertEqual(compressed.level, CompressionLevel.MAXIMUM)

    def test_runtime_config_override(self):
        """测试运行时配置覆盖"""
        default_config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.BALANCED,
        )

        compressor = DataCompressor(default_config)
        data = b'Runtime config test.'

        # 使用不同配置覆盖
        override_config = CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            level=CompressionLevel.FASTEST,
        )

        try:
            compressed = compressor.compress(data, override_config)
            self.assertEqual(compressed.algorithm, CompressionAlgorithm.LZ4)
        except ImportError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
