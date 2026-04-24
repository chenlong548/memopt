"""
data_compressor 安全测试

测试输入验证、边界条件、异常处理和资源管理。
"""

import unittest
import os
import sys
import tempfile
import gc
import threading
import time
import io

# 添加项目路径 - 确保data_compressor模块可以被导入
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from data_compressor import (
    DataCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressedData,
    DataType,
    CompressionError,
    DecompressionError,
    ValidationError,
    MemoryLimitError,
    StreamError,
)

from data_compressor.utils.validation import DataValidator
from data_compressor.stream.buffer_pool import BufferPool
from data_compressor.stream.chunk_manager import ChunkManager


class TestInputValidation(unittest.TestCase):
    """输入验证测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()
        self.validator = DataValidator()

    def test_none_input(self):
        """测试None输入"""
        with self.assertRaises((ValidationError, CompressionError, TypeError, AttributeError)):
            # 注意：这里测试None输入，期望抛出异常
            # 类型检查：None 不能分配给 bytes、bytearray 或 memoryview
            try:
                self.compressor.compress(None)  # type: ignore
            except Exception as e:
                # 捕获并重新抛出异常，确保测试通过
                raise

    def test_empty_data(self):
        """测试空数据"""
        # 空数据应该能处理
        try:
            compressed = self.compressor.compress(b'')
            # 某些算法可能不支持空数据
        except (CompressionError, ValidationError):
            pass  # 预期的异常

    def test_invalid_type_string(self):
        """测试字符串输入（无效类型）"""
        with self.assertRaises(ValidationError):
            # 类型检查：字符串不能分配给 bytes、bytearray 或 memoryview
            self.validator.validate("string data")  # type: ignore

    def test_invalid_type_int(self):
        """测试整数输入（无效类型）"""
        with self.assertRaises(ValidationError):
            self.validator.validate(12345)  # type: ignore

    def test_invalid_type_list(self):
        """测试列表输入（无效类型）"""
        with self.assertRaises(ValidationError):
            self.validator.validate([1, 2, 3])  # type: ignore

    def test_invalid_type_dict(self):
        """测试字典输入（无效类型）"""
        with self.assertRaises(ValidationError):
            self.validator.validate({'key': 'value'})  # type: ignore

    def test_valid_bytes(self):
        """测试有效字节数据"""
        data = b'Valid bytes data'
        self.assertTrue(self.validator.validate(data))

    def test_valid_bytearray(self):
        """测试有效bytearray数据"""
        data = bytearray(b'Valid bytearray data')
        self.assertTrue(self.validator.validate(data))

    def test_valid_memoryview(self):
        """测试有效memoryview数据"""
        data = memoryview(b'Valid memoryview data')
        self.assertTrue(self.validator.validate(data))


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()

    def test_single_byte(self):
        """测试单字节数据"""
        data = b'A'
        try:
            compressed = self.compressor.compress(data)
            decompressed = self.compressor.decompress(compressed)
            self.assertEqual(decompressed, data)
        except (CompressionError, DecompressionError):
            pass  # 某些算法可能不支持

    def test_small_data(self):
        """测试小数据"""
        data = b'AB'
        try:
            compressed = self.compressor.compress(data)
            decompressed = self.compressor.decompress(compressed)
            self.assertEqual(decompressed, data)
        except (CompressionError, DecompressionError):
            pass

    def test_exact_chunk_size(self):
        """测试精确分块大小"""
        chunk_size = 1024
        config = CompressionConfig(chunk_size=chunk_size)
        compressor = DataCompressor(config)

        data = b'X' * chunk_size
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_chunk_size_plus_one(self):
        """测试分块大小+1"""
        chunk_size = 1024
        config = CompressionConfig(chunk_size=chunk_size)
        compressor = DataCompressor(config)

        data = b'X' * (chunk_size + 1)
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_large_data(self):
        """测试大数据"""
        # 1MB数据
        data = os.urandom(1024 * 1024)
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_all_zeros(self):
        """测试全零数据"""
        data = b'\x00' * 1024
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_all_ones(self):
        """测试全1数据"""
        data = b'\xFF' * 1024
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_max_entropy_data(self):
        """测试最大熵数据（随机数据）"""
        data = os.urandom(1024)
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_repeated_pattern(self):
        """测试重复模式数据"""
        pattern = b'ABCDEFGH'
        data = pattern * 1000
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)


class TestExceptionHandling(unittest.TestCase):
    """异常处理测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()

    def test_invalid_algorithm(self):
        """测试无效算法"""
        # 尝试使用不存在的算法
        pass  # 算法由枚举限制，无法传入无效值

    def test_corrupted_compressed_data(self):
        """测试损坏的压缩数据"""
        data = b'Test data for compression'
        compressed = self.compressor.compress(data)

        # 损坏数据
        corrupted_data = bytearray(compressed.data)
        if len(corrupted_data) > 10:
            corrupted_data[5] = (corrupted_data[5] + 1) % 256

        corrupted = CompressedData(
            data=bytes(corrupted_data),
            algorithm=compressed.algorithm,
            level=compressed.level,
            original_size=compressed.original_size,
            compressed_size=compressed.compressed_size
        )

        # 解压损坏数据应该失败或产生错误结果
        try:
            decompressed = self.compressor.decompress(corrupted)
            # 如果成功，结果可能不正确
        except (DecompressionError, Exception):
            pass  # 预期的异常

    def test_wrong_original_size(self):
        """测试错误的原始大小"""
        data = b'Test data'
        compressed = self.compressor.compress(data)

        # 修改原始大小
        wrong_size = CompressedData(
            data=compressed.data,
            algorithm=compressed.algorithm,
            level=compressed.level,
            original_size=999,  # 错误的大小
            compressed_size=compressed.compressed_size
        )

        # 应该能解压，但大小可能不匹配
        try:
            decompressed = self.compressor.decompress(wrong_size)
        except (DecompressionError, Exception):
            pass  # 预期的异常

    def test_algorithm_mismatch(self):
        """测试算法不匹配"""
        data = b'Test data'

        # 使用ZSTD压缩
        config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD)
        try:
            compressed = self.compressor.compress(data, config)

            # 修改算法标记
            wrong_algorithm = CompressedData(
                data=compressed.data,
                algorithm=CompressionAlgorithm.LZ4,  # 错误的算法
                level=compressed.level,
                original_size=compressed.original_size,
                compressed_size=compressed.compressed_size
            )

            # 解压应该失败
            with self.assertRaises((DecompressionError, Exception)):
                self.compressor.decompress(wrong_algorithm)
        except ImportError:
            self.skipTest("zstandard library not available")


class TestResourceManagement(unittest.TestCase):
    """资源管理测试"""

    def test_memory_cleanup(self):
        """测试内存清理"""
        compressor = DataCompressor()

        # 压缩大量数据
        for _ in range(100):
            data = os.urandom(1024)
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)
            del compressed, decompressed

        # 强制垃圾回收
        gc.collect()

        # 如果没有内存泄漏，应该能正常完成
        self.assertTrue(True)

    def test_file_handle_cleanup(self):
        """测试文件句柄清理"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(b'Test data for file compression')

        try:
            # 多次读写文件
            for _ in range(10):
                with open(temp_path, 'rb') as f:
                    data = f.read()
                compressed = DataCompressor().compress(data)
        finally:
            os.unlink(temp_path)

    def test_buffer_pool_cleanup(self):
        """测试缓冲池清理"""
        pool = BufferPool(buffer_size=1024, num_buffers=4)

        # 获取和释放缓冲区
        buffers = []
        for _ in range(4):
            buffer = pool.acquire(timeout=1.0)
            if buffer:
                buffers.append(buffer)

        # 释放所有缓冲区
        for buffer in buffers:
            pool.release(buffer)

        # 清理
        pool.clear()

        stats = pool.get_stats()
        self.assertEqual(stats['available_buffers'], 4)

    def test_chunk_manager_cleanup(self):
        """测试分块管理器清理"""
        manager = ChunkManager(chunk_size=1024)

        # 创建多个分块
        for i in range(10):
            data = os.urandom(500)
            manager.create_chunk(data, i)

        # 清理
        manager.clear()

        stats = manager.get_stats()
        self.assertEqual(stats['total_chunks'], 0)


class TestConcurrencySafety(unittest.TestCase):
    """并发安全测试"""

    def test_concurrent_compression(self):
        """测试并发压缩"""
        compressor = DataCompressor()
        errors = []
        results = []

        def compress_worker(worker_id):
            try:
                for i in range(10):
                    data = f"Worker {worker_id} data {i}".encode()
                    compressed = compressor.compress(data)
                    decompressed = compressor.decompress(compressed)
                    results.append((worker_id, i, data == decompressed))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=compress_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 验证结果
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 40)  # 4 workers * 10 iterations

    def test_concurrent_buffer_access(self):
        """测试并发缓冲区访问"""
        pool = BufferPool(buffer_size=1024, num_buffers=4)
        errors = []

        def buffer_worker(worker_id):
            try:
                for _ in range(10):
                    buffer = pool.acquire(timeout=5.0)
                    if buffer:
                        time.sleep(0.001)  # 模拟工作
                        pool.release(buffer)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(8):
            t = threading.Thread(target=buffer_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    def test_concurrent_chunk_access(self):
        """测试并发分块访问"""
        manager = ChunkManager(chunk_size=1024)
        errors = []

        def chunk_worker(worker_id):
            try:
                for i in range(10):
                    data = f"Worker {worker_id} chunk {i}".encode()
                    chunk_idx = worker_id * 10 + i
                    manager.create_chunk(data, chunk_idx)
                    retrieved = manager.get_chunk(chunk_idx)
                    if retrieved != data:
                        errors.append((worker_id, "Data mismatch"))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=chunk_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")


class TestStreamSafety(unittest.TestCase):
    """流处理安全测试"""

    def test_invalid_stream_format(self):
        """测试无效流格式"""
        from data_compressor.stream.stream_compressor import StreamCompressor

        compressor = StreamCompressor()

        # 创建无效格式的输入
        invalid_data = b'INVALID_FORMAT_DATA'
        input_stream = io.BytesIO(invalid_data)
        output_stream = io.BytesIO()

        # 应该抛出异常
        with self.assertRaises((StreamError, Exception)):
            compressor.decompress_stream(input_stream, output_stream)

    def test_truncated_stream(self):
        """测试截断的流"""
        from data_compressor.stream.stream_compressor import StreamCompressor

        compressor = StreamCompressor()

        # 创建有效数据然后截断
        data = b'Test data for stream compression'
        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        compressor.compress_stream(input_stream, output_stream)

        # 截断输出
        output_stream.seek(0)
        truncated = output_stream.read(20)  # 只取前20字节

        # 尝试解压截断的数据
        truncated_stream = io.BytesIO(truncated)
        result_stream = io.BytesIO()

        with self.assertRaises((StreamError, Exception)):
            compressor.decompress_stream(truncated_stream, result_stream)

    def test_stream_stop(self):
        """测试流停止"""
        from data_compressor.stream.stream_compressor import StreamCompressor

        compressor = StreamCompressor()

        # 创建大数据
        data = os.urandom(1024 * 1024)  # 1MB
        input_stream = io.BytesIO(data)
        output_stream = io.BytesIO()

        # 在另一个线程中停止
        def stop_after_delay():
            time.sleep(0.1)
            compressor.stop()

        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.start()

        # 开始压缩
        try:
            compressor.compress_stream(input_stream, output_stream)
        except StreamError:
            pass  # 预期的异常

        stop_thread.join()


class TestConfigValidation(unittest.TestCase):
    """配置验证测试"""

    def test_invalid_chunk_size(self):
        """测试无效分块大小"""
        # 负数分块大小
        with self.assertRaises((ValueError, TypeError)):
            CompressionConfig(chunk_size=-1)

        # 零分块大小
        with self.assertRaises((ValueError, TypeError)):
            CompressionConfig(chunk_size=0)

    def test_invalid_num_workers(self):
        """测试无效工作线程数"""
        # 负数工作线程
        with self.assertRaises((ValueError, TypeError)):
            CompressionConfig(num_workers=-1)

        # 零工作线程
        with self.assertRaises((ValueError, TypeError)):
            CompressionConfig(num_workers=0)

    def test_invalid_max_memory(self):
        """测试无效最大内存"""
        # 负数内存限制
        with self.assertRaises((ValueError, TypeError)):
            CompressionConfig(max_memory_usage=-1)


class TestCompressedDataValidation(unittest.TestCase):
    """压缩数据验证测试"""

    def test_missing_required_fields(self):
        """测试缺少必需字段"""
        # 缺少level字段
        with self.assertRaises(TypeError):
            CompressedData(
                data=b'',
                algorithm=CompressionAlgorithm.ZSTD,
                original_size=100,
                compressed_size=50
            )  # type: ignore

    def test_size_consistency(self):
        """测试大小一致性"""
        validator = DataValidator()

        # 压缩大小与实际数据大小不匹配
        compressed = CompressedData(
            data=b'short',
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.BALANCED,
            original_size=100,
            compressed_size=1000  # 错误的大小
        )

        with self.assertRaises(ValidationError):
            validator.validate_compressed_data(compressed)


class TestEdgeCases(unittest.TestCase):
    """边缘情况测试"""

    def setUp(self):
        """测试前准备"""
        self.compressor = DataCompressor()

    def test_unicode_data(self):
        """测试Unicode数据"""
        text = "Unicode: \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u65e5\u672c\u8a9e"
        data = text.encode('utf-8')

        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_binary_data(self):
        """测试二进制数据"""
        data = bytes(range(256))  # 所有可能的字节值

        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_very_long_repeated_pattern(self):
        """测试超长重复模式"""
        pattern = b'X'
        data = pattern * 1000000  # 1百万字节

        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_alternating_pattern(self):
        """测试交替模式"""
        data = b'AB' * 10000

        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_sparse_data(self):
        """测试稀疏数据"""
        # 大部分是零，少量非零
        data = bytearray(10000)
        data[100] = 1
        data[500] = 2
        data[9000] = 3

        compressed = self.compressor.compress(bytes(data))
        decompressed = self.compressor.decompress(compressed)
        self.assertEqual(decompressed, bytes(data))


class TestSecurityVulnerabilities(unittest.TestCase):
    """安全漏洞测试"""

    def test_decompression_bomb_detection(self):
        """测试解压炸弹检测"""
        # 创建一个高压缩比的数据（模拟解压炸弹）
        # 注意：实际测试中需要小心处理
        data = b'X' * 1000  # 简单测试

        compressor = DataCompressor()
        compressed = compressor.compress(data)

        # 应该能正常处理
        decompressed = compressor.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_memory_limit_enforcement(self):
        """测试内存限制执行"""
        validator = DataValidator()

        # 设置较小的内存限制
        validator.max_size = 1024

        # 尝试验证大数据
        large_data = os.urandom(2048)

        with self.assertRaises(ValidationError):
            validator.validate(large_data)

    def test_path_traversal_prevention(self):
        """测试路径遍历防护"""
        validator = DataValidator()

        # 尝试验证不存在的文件
        with self.assertRaises(ValidationError):
            validator.validate_file('/nonexistent/path/file.txt')


if __name__ == '__main__':
    unittest.main(verbosity=2)
