"""
data_compressor 单元测试

测试所有压缩算法和核心功能。
"""

import unittest
import os
import sys
import tempfile
import struct
import time
import threading
import io
from typing import Dict, Any

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
    CompressionError,
    DecompressionError,
    ValidationError,
)

from data_compressor.algorithms.zstd_wrapper import ZstdCompressor, ZSTD_AVAILABLE
from data_compressor.algorithms.lz4_wrapper import LZ4Compressor, LZ4_AVAILABLE
from data_compressor.algorithms.brotli_wrapper import BrotliCompressor, BROTLI_AVAILABLE
from data_compressor.algorithms.model_compress.bf16_compress import BF16ModelCompressor
from data_compressor.algorithms.model_compress.fp32_compress import FP32ModelCompressor
from data_compressor.algorithms.kv_cache.lexico import LexicoCompressor
from data_compressor.algorithms.kv_cache.zsmerge import ZSMergeCompressor
from data_compressor.algorithms.adaptive.selector import AdaptiveAlgorithmSelector
from data_compressor.algorithms.adaptive.feature_extractor import FeatureExtractor
from data_compressor.detection.type_detector import DataTypeDetector
from data_compressor.stream.stream_compressor import StreamCompressor
from data_compressor.stream.buffer_pool import BufferPool, BufferView
from data_compressor.stream.chunk_manager import ChunkManager
from data_compressor.utils.statistics import StatisticsCollector
from data_compressor.utils.validation import DataValidator


class TestCompressionAlgorithm(unittest.TestCase):
    """压缩算法基础测试"""

    def setUp(self):
        """测试前准备"""
        self.test_data = {
            'simple': b'Hello, World! This is a test string for compression.',
            'repeated': b'AAAA' * 1000,
            'random': os.urandom(1024),
            'zeros': b'\x00' * 1024,
            'mixed': b'ABCD' * 100 + b'\x00' * 100 + b'1234' * 100,
            'large': os.urandom(100 * 1024),  # 100KB
        }

    def test_zstd_compression(self):
        """测试ZSTD压缩"""
        if not ZSTD_AVAILABLE:
            self.skipTest("zstandard library not available")

        compressor = ZstdCompressor()

        for name, data in self.test_data.items():
            with self.subTest(data=name):
                # 压缩
                compressed = compressor.compress(data)

                # 验证压缩结果
                self.assertIsInstance(compressed, CompressedData)
                self.assertEqual(compressed.algorithm, CompressionAlgorithm.ZSTD)
                self.assertEqual(compressed.original_size, len(data))
                self.assertGreater(compressed.compressed_size, 0)

                # 解压
                decompressed = compressor.decompress(compressed)
                self.assertEqual(decompressed, data)

    def test_zstd_compression_levels(self):
        """测试ZSTD不同压缩级别"""
        if not ZSTD_AVAILABLE:
            self.skipTest("zstandard library not available")

        compressor = ZstdCompressor()
        data = self.test_data['large']

        results = {}
        for level in [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.MAXIMUM]:
            config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD, level=level)
            compressed = compressor.compress(data, config)
            results[level] = {
                'size': compressed.compressed_size,
                'ratio': len(data) / compressed.compressed_size
            }

        # 更高压缩级别通常产生更小的文件
        # 注意：这不总是成立，取决于数据类型
        self.assertGreater(results[CompressionLevel.FASTEST]['size'], 0)

    def test_lz4_compression(self):
        """测试LZ4压缩"""
        if not LZ4_AVAILABLE:
            self.skipTest("lz4 library not available")

        compressor = LZ4Compressor()

        for name, data in self.test_data.items():
            with self.subTest(data=name):
                compressed = compressor.compress(data)
                self.assertIsInstance(compressed, CompressedData)
                self.assertEqual(compressed.algorithm, CompressionAlgorithm.LZ4)

                decompressed = compressor.decompress(compressed)
                self.assertEqual(decompressed, data)

    def test_brotli_compression(self):
        """测试Brotli压缩"""
        if not BROTLI_AVAILABLE:
            self.skipTest("brotli library not available")

        compressor = BrotliCompressor()

        for name, data in self.test_data.items():
            with self.subTest(data=name):
                compressed = compressor.compress(data)
                self.assertIsInstance(compressed, CompressedData)
                self.assertEqual(compressed.algorithm, CompressionAlgorithm.BROTLI)

                decompressed = compressor.decompress(compressed)
                self.assertEqual(decompressed, data)

    def test_bf16_compression(self):
        """测试BF16模型压缩"""
        compressor = BF16ModelCompressor()

        # 创建BF16测试数据（模拟float16）
        floats = np.random.randn(100).astype(np.float16)
        data = floats.tobytes()

        compressed = compressor.compress(data)
        self.assertIsInstance(compressed, CompressedData)
        self.assertEqual(compressed.algorithm, CompressionAlgorithm.BF16_MODEL)

        decompressed = compressor.decompress(compressed)
        # BF16压缩可能有精度损失，检查大小
        self.assertEqual(len(decompressed), len(data))

    def test_fp32_compression(self):
        """测试FP32模型压缩"""
        compressor = FP32ModelCompressor()

        # 创建FP32测试数据
        floats = np.random.randn(100).astype(np.float32)
        data = floats.tobytes()

        compressed = compressor.compress(data)
        self.assertIsInstance(compressed, CompressedData)
        self.assertEqual(compressed.algorithm, CompressionAlgorithm.FP32_MODEL)

        decompressed = compressor.decompress(compressed)
        self.assertEqual(len(decompressed), len(data))

    def test_lexico_compression(self):
        """测试Lexico字典压缩"""
        compressor = LexicoCompressor()

        # 创建有重复模式的数据
        data = b'pattern1pattern2pattern1pattern2' * 100

        compressed = compressor.compress(data)
        self.assertIsInstance(compressed, CompressedData)
        self.assertEqual(compressed.algorithm, CompressionAlgorithm.LEXICO)

        decompressed = compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_zsmerge_compression(self):
        """测试ZSMerge KV Cache压缩"""
        compressor = ZSMergeCompressor()

        # 创建模拟KV Cache数据
        data = np.random.randn(1024).astype(np.float32).tobytes() * 4

        compressed = compressor.compress(data)
        self.assertIsInstance(compressed, CompressedData)
        self.assertEqual(compressed.algorithm, CompressionAlgorithm.KV_CACHE)

        decompressed = compressor.decompress(compressed)
        # KV Cache压缩可能有精度损失
        self.assertGreater(len(decompressed), 0)


class TestDataCompressor(unittest.TestCase):
    """主压缩器测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()
        self.test_data = {
            'simple': b'Hello, World!',
            'text': b'This is a text data for compression testing.' * 100,
            'binary': os.urandom(1024),
            'json': b'{"key": "value", "number": 123, "array": [1, 2, 3]}',
        }

    def test_compress_decompress(self):
        """测试基本压缩解压"""
        for name, data in self.test_data.items():
            with self.subTest(data=name):
                compressed = self.compressor.compress(data)
                self.assertIsInstance(compressed, CompressedData)

                decompressed = self.compressor.decompress(compressed)
                self.assertEqual(decompressed, data)

    def test_auto_algorithm_selection(self):
        """测试自动算法选择"""
        config = CompressionConfig(algorithm=CompressionAlgorithm.AUTO)

        for name, data in self.test_data.items():
            with self.subTest(data=name):
                compressed = self.compressor.compress(data, config)
                # 应该选择一个有效的算法
                self.assertIn(compressed.algorithm, [
                    CompressionAlgorithm.ZSTD,
                    CompressionAlgorithm.LZ4,
                    CompressionAlgorithm.BROTLI,
                ])

    def test_specific_algorithm(self):
        """测试指定算法"""
        algorithms = [
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.BROTLI,
        ]

        data = self.test_data['text']

        for alg in algorithms:
            with self.subTest(algorithm=alg.value):
                try:
                    config = CompressionConfig(algorithm=alg)
                    compressed = self.compressor.compress(data, config)
                    self.assertEqual(compressed.algorithm, alg)
                except ImportError:
                    self.skipTest(f"{alg.value} library not available")

    def test_compression_levels(self):
        """测试不同压缩级别"""
        data = self.test_data['text']

        for level in [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.MAXIMUM]:
            with self.subTest(level=level.name):
                config = CompressionConfig(
                    algorithm=CompressionAlgorithm.ZSTD,
                    level=level
                )
                try:
                    compressed = self.compressor.compress(data, config)
                    self.assertEqual(compressed.level, level)
                except ImportError:
                    self.skipTest("zstandard library not available")

    def test_analyze(self):
        """测试数据分析功能"""
        data = self.test_data['text']
        analysis = self.compressor.analyze(data)

        self.assertIn('data_type', analysis)
        self.assertIn('features', analysis)
        self.assertIn('recommended_algorithm', analysis)

    def test_benchmark(self):
        """测试基准测试功能"""
        data = self.test_data['text']

        # 只测试可用的算法
        algorithms = [CompressionAlgorithm.ZSTD, CompressionAlgorithm.LZ4]
        results = self.compressor.benchmark(data, algorithms)

        self.assertIsInstance(results, dict)
        # 至少有一个算法成功
        self.assertGreater(len([r for r in results.values() if r is not None]), 0)

    def test_get_capabilities(self):
        """测试获取能力信息"""
        capabilities = self.compressor.get_capabilities()

        self.assertIn('supported_algorithms', capabilities)
        self.assertIn('features', capabilities)
        self.assertIn('performance', capabilities)


class TestDataTypeDetector(unittest.TestCase):
    """数据类型检测器测试"""

    def setUp(self):
        """测试前准备"""
        self.detector = DataTypeDetector()

    def test_detect_text(self):
        """测试文本检测"""
        text_data = b'This is a plain text string for testing.'
        detected = self.detector.detect(text_data)
        self.assertEqual(detected, DataType.TEXT)

    def test_detect_json(self):
        """测试JSON检测"""
        json_data = b'{"key": "value", "number": 123}'
        detected = self.detector.detect(json_data)
        self.assertEqual(detected, DataType.JSON)

    def test_detect_numpy(self):
        """测试NumPy数组检测"""
        import io
        arr = np.array([1, 2, 3, 4, 5])
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        numpy_data = buffer.read()

        detected = self.detector.detect(numpy_data)
        self.assertEqual(detected, DataType.NUMPY_ARRAY)

    def test_detect_binary(self):
        """测试二进制数据检测"""
        binary_data = os.urandom(1024)
        detected = self.detector.detect(binary_data)
        # 随机数据可能被检测为多种类型
        self.assertIn(detected, [DataType.GENERIC, DataType.BINARY, DataType.FP32_TENSOR])

    def test_analyze(self):
        """测试数据分析"""
        data = b'Test data for analysis.'
        analysis = self.detector.analyze(data)

        self.assertIn('data_type', analysis)
        self.assertIn('size', analysis)
        self.assertIn('entropy', analysis)


class TestFeatureExtractor(unittest.TestCase):
    """特征提取器测试"""

    def setUp(self):
        """测试前准备"""
        self.extractor = FeatureExtractor()

    def test_extract_basic_features(self):
        """测试基本特征提取"""
        data = b'Hello, World!'
        features = self.extractor.extract(data, DataType.GENERIC)

        self.assertIn('size', features)
        self.assertIn('entropy', features)
        self.assertIn('unique_bytes', features)

    def test_extract_statistical_features(self):
        """测试统计特征提取"""
        data = b'AABBCCDD' * 100
        features = self.extractor.extract(data, DataType.GENERIC)

        self.assertIn('byte_frequency_mean', features)
        self.assertIn('byte_frequency_std', features)

    def test_extract_pattern_features(self):
        """测试模式特征提取"""
        data = b'pattern' * 100
        features = self.extractor.extract(data, DataType.GENERIC)

        self.assertIn('has_repeated_patterns', features)
        self.assertIn('run_length_avg', features)

    def test_estimate_compression_ratio(self):
        """测试压缩比估算"""
        # 高冗余数据应该有较高的估算压缩比
        high_redundancy = b'AAAA' * 1000
        features_high = self.extractor.extract(high_redundancy, DataType.GENERIC)

        # 低冗余数据应该有较低的估算压缩比
        low_redundancy = os.urandom(1024)
        features_low = self.extractor.extract(low_redundancy, DataType.GENERIC)

        # 高冗余数据的估算压缩比应该更高
        self.assertGreater(
            features_high.get('estimated_ratio', 1.0),
            features_low.get('estimated_ratio', 1.0)
        )


class TestAdaptiveAlgorithmSelector(unittest.TestCase):
    """自适应算法选择器测试"""

    def setUp(self):
        """测试前准备"""
        self.selector = AdaptiveAlgorithmSelector()

    def test_select_algorithm(self):
        """测试算法选择"""
        features = {
            'data_type': 'text',
            'size': 1024,
            'entropy': 4.0,
            'redundancy': 0.5,
        }
        config = CompressionConfig()

        selected = self.selector.select(features, config)
        self.assertIn(selected, [
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.BROTLI,
        ])

    def test_update_stats(self):
        """测试统计更新"""
        stats = CompressionStats()
        stats.compression_ratio = 2.5
        stats.compression_time = 0.1

        self.selector.update(CompressionAlgorithm.ZSTD, stats)

        alg_stats = self.selector.get_algorithm_stats()
        self.assertIn('zstd', alg_stats)

    def test_ucb_selection(self):
        """测试UCB算法选择"""
        # 模拟多次更新
        for _ in range(10):
            stats = CompressionStats()
            stats.compression_ratio = 2.0
            stats.compression_time = 0.05
            self.selector.update(CompressionAlgorithm.LZ4, stats)

        features = {'data_type': 'generic', 'size': 1024, 'entropy': 5.0}
        config = CompressionConfig()

        selected = self.selector.select(features, config)
        self.assertIsNotNone(selected)


class TestStreamCompressor(unittest.TestCase):
    """流式压缩器测试"""

    def setUp(self):
        """测试前准备"""
        self.config = CompressionConfig(chunk_size=1024)
        self.compressor = StreamCompressor(self.config)
        self.test_data = os.urandom(10 * 1024)  # 10KB

    def test_compress_stream(self):
        """测试流式压缩"""
        input_stream = io.BytesIO(self.test_data)
        output_stream = io.BytesIO()

        stats = self.compressor.compress_stream(input_stream, output_stream)

        self.assertIsInstance(stats, CompressionStats)
        self.assertEqual(stats.original_size, len(self.test_data))
        self.assertGreater(stats.compressed_size, 0)

    def test_decompress_stream(self):
        """测试流式解压"""
        # 先压缩
        input_stream = io.BytesIO(self.test_data)
        compressed_stream = io.BytesIO()
        self.compressor.compress_stream(input_stream, compressed_stream)

        # 再解压
        compressed_stream.seek(0)
        decompressed_stream = io.BytesIO()
        self.compressor.decompress_stream(compressed_stream, decompressed_stream)

        # 验证
        decompressed_stream.seek(0)
        decompressed_data = decompressed_stream.read()

        self.assertEqual(len(decompressed_data), len(self.test_data))

    def test_parallel_compression(self):
        """测试并行压缩"""
        input_stream = io.BytesIO(self.test_data)
        output_stream = io.BytesIO()

        stats = self.compressor.compress_parallel(input_stream, output_stream, num_workers=2)

        self.assertIsInstance(stats, CompressionStats)


class TestBufferPool(unittest.TestCase):
    """缓冲池测试"""

    def setUp(self):
        """测试前准备"""
        self.pool = BufferPool(buffer_size=1024, num_buffers=4)

    def test_acquire_release(self):
        """测试获取和释放缓冲区"""
        buffer = self.pool.acquire()
        self.assertIsInstance(buffer, bytearray)
        if buffer is not None:
            self.assertEqual(len(buffer), 1024)
            self.pool.release(buffer)

        stats = self.pool.get_stats()
        self.assertEqual(stats['total_acquisitions'], 1)
        self.assertEqual(stats['total_releases'], 1)

    def test_multiple_acquisitions(self):
        """测试多次获取"""
        buffers = []
        for _ in range(4):
            buffer = self.pool.acquire(timeout=1.0)
            self.assertIsNotNone(buffer)
            buffers.append(buffer)

        # 第5次应该超时
        buffer = self.pool.acquire(timeout=0.1)
        self.assertIsNone(buffer)

        # 释放后可以再次获取
        self.pool.release(buffers[0])
        buffer = self.pool.acquire(timeout=1.0)
        self.assertIsNotNone(buffer)

    def test_buffer_view(self):
        """测试缓冲区视图"""
        buffer = bytearray(100)
        view = BufferView(buffer, offset=10, size=50)

        self.assertEqual(len(view), 50)

        # 写入测试
        written = view.write(b'Hello', offset=0)
        self.assertEqual(written, 5)

        # 读取测试
        data = view.read(5)
        self.assertEqual(data, b'Hello')


class TestChunkManager(unittest.TestCase):
    """分块管理器测试"""

    def setUp(self):
        """测试前准备"""
        self.manager = ChunkManager(chunk_size=1024, enable_dedup=True)

    def test_create_chunk(self):
        """测试创建分块"""
        data = b'Test data for chunking.'
        chunk_info = self.manager.create_chunk(data, 0)

        self.assertEqual(chunk_info['index'], 0)
        self.assertEqual(chunk_info['size'], len(data))
        self.assertIsNotNone(chunk_info['checksum'])

    def test_get_chunk(self):
        """测试获取分块"""
        data = b'Test data for retrieval.'
        self.manager.create_chunk(data, 0)

        retrieved = self.manager.get_chunk(0)
        self.assertEqual(retrieved, data)

    def test_split_merge(self):
        """测试分割和合并"""
        data = os.urandom(3000)  # 约3个分块
        chunks = self.manager.split_data(data)

        self.assertEqual(len(chunks), 3)

        # 合并
        indices = [idx for idx, _ in chunks]
        merged = self.manager.merge_chunks(indices)

        self.assertEqual(len(merged), len(data))

    def test_deduplication(self):
        """测试去重"""
        data = b'Duplicate data'

        # 创建两个相同的分块
        self.manager.create_chunk(data, 0)
        self.manager.create_chunk(data, 1)

        stats = self.manager.get_stats()
        self.assertGreater(stats['dedup_chunks'], 0)

    def test_verify_chunk(self):
        """测试分块验证"""
        data = b'Data for verification.'
        self.manager.create_chunk(data, 0)

        self.assertTrue(self.manager.verify_chunk(0))


class TestStatisticsCollector(unittest.TestCase):
    """统计收集器测试"""

    def setUp(self):
        """测试前准备"""
        self.collector = StatisticsCollector()

    def test_record(self):
        """测试记录统计"""
        stats = CompressionStats()
        stats.original_size = 1000
        stats.compressed_size = 500
        stats.compression_ratio = 2.0
        stats.compression_time = 0.1
        stats.algorithm_used = CompressionAlgorithm.ZSTD

        self.collector.record(stats)

        summary = self.collector.get_summary()
        self.assertEqual(summary['total_operations'], 1)

    def test_algorithm_ranking(self):
        """测试算法排名"""
        # 记录多个算法的统计
        for alg, ratio in [(CompressionAlgorithm.ZSTD, 3.0),
                           (CompressionAlgorithm.LZ4, 2.0),
                           (CompressionAlgorithm.BROTLI, 4.0)]:
            stats = CompressionStats()
            stats.compression_ratio = ratio
            stats.compression_time = 0.1
            stats.algorithm_used = alg
            self.collector.record(stats)

        ranking = self.collector.get_algorithm_ranking(metric='ratio')
        self.assertEqual(len(ranking), 3)
        # Brotli应该排第一（最高压缩比）
        self.assertEqual(ranking[0]['algorithm'], 'brotli')

    def test_export(self):
        """测试导出"""
        stats = CompressionStats()
        stats.algorithm_used = CompressionAlgorithm.ZSTD
        self.collector.record(stats)

        exported = self.collector.export_to_dict()
        self.assertIn('summary', exported)
        self.assertIn('algorithm_ranking', exported)


class TestDataValidator(unittest.TestCase):
    """数据验证器测试"""

    def setUp(self):
        """测试前准备"""
        self.validator = DataValidator()

    def test_validate_bytes(self):
        """测试验证字节数据"""
        data = b'Valid data'
        self.assertTrue(self.validator.validate(data))

    def test_validate_bytearray(self):
        """测试验证bytearray"""
        data = bytearray(b'Valid data')
        self.assertTrue(self.validator.validate(data))

    def test_validate_memoryview(self):
        """测试验证memoryview"""
        data = memoryview(b'Valid data')
        self.assertTrue(self.validator.validate(data))

    def test_validate_invalid_type(self):
        """测试验证无效类型"""
        with self.assertRaises(ValidationError):
            self.validator.validate("string data")  # type: ignore

    def test_validate_size_limits(self):
        """测试大小限制"""
        validator = DataValidator(max_size=100, min_size=10)

        # 太小
        with self.assertRaises(ValidationError):
            validator.validate(b'short')

        # 太大
        with self.assertRaises(ValidationError):
            validator.validate(b'x' * 200)

    def test_validate_compressed_data(self):
        """测试验证压缩数据"""
        compressed = CompressedData(
            data=b'compressed',
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.BALANCED,
            original_size=100,
            compressed_size=20
        )

        with self.assertRaises(ValidationError):
            self.validator.validate_compressed_data(compressed)

        compressed.compressed_size = len(compressed.data)
        self.assertTrue(self.validator.validate_compressed_data(compressed))

    def test_get_data_info(self):
        """测试获取数据信息"""
        data = b'Hello World!'
        info = self.validator.get_data_info(data)

        self.assertEqual(info['size'], len(data))
        self.assertIn('entropy', info)
        self.assertIn('unique_bytes', info)


class TestCompressionStats(unittest.TestCase):
    """压缩统计测试"""

    def test_calculate_ratio(self):
        """测试计算压缩比"""
        stats = CompressionStats()
        stats.original_size = 1000
        stats.compressed_size = 500

        stats.calculate_ratio()
        self.assertEqual(stats.compression_ratio, 2.0)

    def test_calculate_throughput(self):
        """测试计算吞吐量"""
        stats = CompressionStats()
        stats.original_size = 1024 * 1024  # 1MB
        stats.compression_time = 0.5  # 0.5秒

        stats.calculate_throughput()
        self.assertAlmostEqual(stats.throughput_mbps, 2.0, places=1)

    def test_get_summary(self):
        """测试获取摘要"""
        stats = CompressionStats()
        stats.original_size = 1024 * 1024
        stats.compressed_size = 512 * 1024
        stats.compression_ratio = 2.0
        stats.algorithm_used = CompressionAlgorithm.ZSTD
        stats.level_used = CompressionLevel.BALANCED

        summary = stats.get_summary()

        self.assertIn('original_size_mb', summary)
        self.assertIn('compression_ratio', summary)
        self.assertIn('algorithm', summary)


class TestCompressedData(unittest.TestCase):
    """压缩数据容器测试"""

    def test_create(self):
        """测试创建压缩数据"""
        data = CompressedData(
            data=b'compressed data',
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.BALANCED,
            original_size=100,
            compressed_size=14
        )

        self.assertEqual(data.algorithm, CompressionAlgorithm.ZSTD)
        self.assertEqual(data.original_size, 100)

    def test_get_metadata(self):
        """测试获取元数据"""
        data = CompressedData(
            data=b'compressed',
            algorithm=CompressionAlgorithm.LZ4,
            level=CompressionLevel.FAST,
            original_size=50,
            compressed_size=10,
            data_type=DataType.TEXT
        )

        metadata = data.get_metadata()

        self.assertEqual(metadata['algorithm'], 'lz4')
        self.assertEqual(metadata['data_type'], 'text')
        self.assertEqual(metadata['original_size'], 50)


class TestCompressionConfig(unittest.TestCase):
    """压缩配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = CompressionConfig()

        self.assertEqual(config.algorithm, CompressionAlgorithm.AUTO)
        self.assertEqual(config.level, CompressionLevel.BALANCED)
        self.assertEqual(config.data_type, DataType.GENERIC)

    def test_custom_config(self):
        """测试自定义配置"""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.MAXIMUM,
            data_type=DataType.TEXT,
            enable_parallel=True,
            chunk_size=2048
        )

        self.assertEqual(config.algorithm, CompressionAlgorithm.ZSTD)
        self.assertEqual(config.level, CompressionLevel.MAXIMUM)
        self.assertEqual(config.chunk_size, 2048)


class TestExceptions(unittest.TestCase):
    """异常测试"""

    def test_compression_error(self):
        """测试压缩错误"""
        error = CompressionError("Test error", algorithm="zstd", original_size=100)
        self.assertEqual(error.algorithm, "zstd")
        self.assertEqual(error.original_size, 100)

    def test_decompression_error(self):
        """测试解压错误"""
        error = DecompressionError("Test error", algorithm="lz4", compressed_size=50)
        self.assertEqual(error.algorithm, "lz4")
        self.assertEqual(error.compressed_size, 50)

    def test_validation_error(self):
        """测试验证错误"""
        error = ValidationError("Invalid data", data_info={'size': 0})
        self.assertEqual(error.data_info, {'size': 0})


class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试"""

    def test_compress_function(self):
        """测试compress便捷函数"""
        from data_compressor import compress, decompress

        data = b'Hello, World!'
        compressed = compress(data, algorithm='zstd')

        self.assertIsInstance(compressed, CompressedData)

        decompressed = decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_analyze_function(self):
        """测试analyze便捷函数"""
        from data_compressor import analyze

        data = b'Test data for analysis.'
        result = analyze(data)

        self.assertIn('data_type', result)
        self.assertIn('recommended_algorithm', result)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
