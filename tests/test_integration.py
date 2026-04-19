"""
memopt 集成测试

测试各模块之间的集成功能。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tempfile


class TestIntegration(unittest.TestCase):
    """集成测试基类"""
    
    def setUp(self):
        """测试前准备"""
        pass
    
    def tearDown(self):
        """测试后清理"""
        pass


class TestMemMapperDataCompressorIntegration(TestIntegration):
    """mem_mapper 与 data_compressor 集成测试"""
    
    def test_map_and_compress(self):
        """测试映射文件后压缩"""
        from mem_mapper import MemoryMapper, MapperConfig
        from data_compressor import DataCompressor, CompressionConfig, CompressionAlgorithm, CompressionLevel
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data" * 10000)
            temp_path = f.name
        
        config = MapperConfig(use_numa=False, use_huge_pages=False)
        mapper = MemoryMapper(config)
        
        region = mapper.map_file(temp_path, mode="readonly", size=90000)
        
        self.assertIsNotNone(region)
        self.assertEqual(region.size, 90000)
        
        data = b"test data" * 10000
        comp_config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.FAST
        )
        compressor = DataCompressor(comp_config)
        compressed = compressor.compress(data[:90000])
        
        self.assertLess(len(compressed.data), len(data[:90000]))
        
        decompressed = compressor.decompress(compressed)
        self.assertEqual(data[:90000], decompressed)
        
        mapper.unmap(region)
        os.unlink(temp_path)


class TestStreamProcessorBufferManagerIntegration(TestIntegration):
    """stream_processor 与 buffer_manager 集成测试"""
    
    def test_stream_with_buffer_pool(self):
        """测试流处理使用缓冲池"""
        from stream_processor import CollectionSource
        from buffer_manager import BufferPool
        
        pool = BufferPool(buffer_size=4096, num_buffers=8)
        
        data = [b"data" * 100 for _ in range(10)]
        
        source = CollectionSource("source", data)
        records = list(source.read())
        
        self.assertEqual(len(records), 10)
        
        buffer = pool.acquire()
        self.assertIsNotNone(buffer)
        pool.release(buffer)


class TestMemOptimizerMemMonitorIntegration(TestIntegration):
    """mem_optimizer 与 mem_monitor 集成测试"""
    
    def test_memory_pool_with_monitoring(self):
        """测试内存池监控"""
        from mem_optimizer import MemoryPool, OptimizerConfig
        from mem_monitor import MemoryMonitor, MonitorConfig
        
        config = MonitorConfig()
        monitor = MemoryMonitor(config)
        
        opt_config = OptimizerConfig(
            total_memory=10 * 1024 * 1024,
            max_allocation_size=1 * 1024 * 1024
        )
        pool = MemoryPool(config=opt_config)
        
        blocks = []
        for _ in range(10):
            result = pool.allocate(1024)
            if result.success:
                blocks.append(result.address)
        
        snapshot = monitor.get_snapshot()
        self.assertIsNotNone(snapshot)
        
        for address in blocks:
            pool.deallocate(address)
        
        report = monitor.stop()
        self.assertIsNotNone(report)


class TestSparseArrayLazyEvaluatorIntegration(TestIntegration):
    """sparse_array 与 lazy_evaluator 集成测试"""
    
    def test_lazy_sparse_matrix(self):
        """测试惰性稀疏矩阵计算"""
        from sparse_array import SparseArray
        from lazy_evaluator import Lazy
        
        dense = np.random.rand(100, 100)
        dense[dense < 0.9] = 0
        
        lazy_sparse = Lazy(lambda: SparseArray.from_dense(dense))
        sparse = lazy_sparse.force()
        
        x = np.random.rand(100)
        lazy_result = Lazy(lambda: sparse @ x)
        result = lazy_result.force()
        
        self.assertEqual(len(result), 100)


class TestFullPipeline(TestIntegration):
    """完整流水线测试"""
    
    def test_data_pipeline(self):
        """测试完整数据处理流水线"""
        from mem_mapper import MemoryMapper, MapperConfig
        from data_compressor import DataCompressor, CompressionConfig, CompressionAlgorithm, CompressionLevel
        from stream_processor import CollectionSource
        from buffer_manager import BufferPool
        from mem_monitor import MemoryMonitor, MonitorConfig
        
        config = MonitorConfig()
        monitor = MemoryMonitor(config)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"pipeline test data" * 10000)
            temp_path = f.name
        
        mapper_config = MapperConfig(use_numa=False, use_huge_pages=False)
        mapper = MemoryMapper(mapper_config)
        region = mapper.map_file(temp_path, mode="readonly", size=180000)
        
        data = b"pipeline test data" * 10000
        comp_config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.FAST
        )
        compressor = DataCompressor(comp_config)
        compressed = compressor.compress(data[:180000])
        
        chunks = [compressed.data[i:i+4096] for i in range(0, len(compressed.data), 4096)]
        source = CollectionSource("source", chunks)
        records = list(source.read())
        
        pool = BufferPool(buffer_size=4096, num_buffers=4)
        buffer = pool.acquire()
        self.assertIsNotNone(buffer)
        pool.release(buffer)
        
        mapper.unmap(region)
        os.unlink(temp_path)
        
        report = monitor.stop()
        
        self.assertGreater(len(records), 0)
        self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
