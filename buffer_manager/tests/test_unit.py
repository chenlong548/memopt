"""
单元测试模块

测试buffer_manager的所有核心功能。
"""

import unittest
import threading
import time
import io
from concurrent.futures import ThreadPoolExecutor

# 导入所有模块
from buffer_manager.core.buffer import Buffer
from buffer_manager.core.config import BufferConfig
from buffer_manager.core.exceptions import (
    BufferFullError,
    BufferEmptyError,
    BufferTimeoutError,
    PoolExhaustedError,
    InvalidAlignmentError,
    InvalidCapacityError,
    RingBufferCapacityError,
)
from buffer_manager.pools.buffer_pool import BufferPool, PoolStats
from buffer_manager.pools.ring_buffer import RingBuffer
from buffer_manager.pools.double_buffer import DoubleBuffer
from buffer_manager.queues.spsc import SPSCQueue
from buffer_manager.queues.mpsc import MPSCQueue
from buffer_manager.queues.mpmc import MPMCQueue
from buffer_manager.strategy.replacement import LRU, ARC
from buffer_manager.strategy.prefetch import Prefetcher
from buffer_manager.strategy.adaptive import AdaptiveStrategy, AdaptationLevel
from buffer_manager.monitor.metrics import BufferMetrics
from buffer_manager.monitor.watermark import WatermarkLevel, WatermarkManager
from buffer_manager.integration.stream_adapter import StreamAdapter
from buffer_manager.integration.compressor_adapter import CompressorAdapter, CompressionType


class TestBuffer(unittest.TestCase):
    """Buffer基本功能测试"""
    
    def test_create_buffer(self):
        """测试创建缓冲区"""
        buffer = Buffer(1024)
        self.assertEqual(buffer.capacity, 1024)
        self.assertEqual(buffer.position, 0)
        self.assertTrue(buffer.is_empty)
    
    def test_create_buffer_with_alignment(self):
        """测试创建带对齐的缓冲区"""
        buffer = Buffer(1024, alignment=64)
        self.assertEqual(buffer.alignment, 64)
    
    def test_invalid_capacity(self):
        """测试无效容量"""
        with self.assertRaises(InvalidCapacityError):
            Buffer(0)
        with self.assertRaises(InvalidCapacityError):
            Buffer(-1)
    
    def test_invalid_alignment(self):
        """测试无效对齐"""
        with self.assertRaises(InvalidAlignmentError):
            Buffer(1024, alignment=0)
        with self.assertRaises(InvalidAlignmentError):
            Buffer(1024, alignment=3)  # 不是2的幂
    
    def test_write_read(self):
        """测试读写操作"""
        buffer = Buffer(1024)
        data = b"Hello, World!"
        
        written = buffer.write(data)
        self.assertEqual(written, len(data))
        self.assertEqual(buffer.position, len(data))
        
        buffer.seek(0)
        read_data = buffer.read(len(data))
        self.assertEqual(read_data, data)
    
    def test_write_full(self):
        """测试缓冲区满"""
        buffer = Buffer(10)
        buffer.write(b"1234567890")
        
        with self.assertRaises(BufferFullError):
            buffer.write(b"extra")
    
    def test_read_empty(self):
        """测试读取空缓冲区"""
        buffer = Buffer(1024)
        
        # 缓冲区为空时，position=0，limit=capacity
        # 需要先flip使limit=position=0
        buffer.flip()
        
        with self.assertRaises(BufferEmptyError):
            buffer.read(10)
    
    def test_seek(self):
        """测试定位操作"""
        buffer = Buffer(1024)
        buffer.write(b"Hello")
        
        buffer.seek(2)
        self.assertEqual(buffer.position, 2)
        
        data = buffer.read(3)
        self.assertEqual(data, b"llo")
    
    def test_peek(self):
        """测试查看数据"""
        buffer = Buffer(1024)
        buffer.write(b"Hello")
        
        buffer.seek(0)
        data = buffer.peek(3)
        self.assertEqual(data, b"Hel")
        self.assertEqual(buffer.position, 0)  # 位置不变
    
    def test_clear(self):
        """测试清空缓冲区"""
        buffer = Buffer(1024)
        buffer.write(b"Hello")
        
        buffer.clear()
        self.assertEqual(buffer.position, 0)
        self.assertTrue(buffer.is_empty)
    
    def test_flip(self):
        """测试翻转操作"""
        buffer = Buffer(1024)
        buffer.write(b"Hello")
        
        buffer.flip()
        self.assertEqual(buffer.position, 0)
        self.assertEqual(buffer.limit, 5)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with Buffer(1024) as buffer:
            buffer.write(b"Hello")
            self.assertEqual(buffer.position, 5)
        
        # 退出后应该被清空
        self.assertEqual(buffer.position, 0)


class TestBufferPool(unittest.TestCase):
    """BufferPool测试"""
    
    def test_create_pool(self):
        """测试创建缓冲池"""
        pool = BufferPool(1024, 4)
        self.assertEqual(pool.capacity, 4)
        self.assertEqual(pool.available_count, 4)
    
    def test_acquire_release(self):
        """测试获取和释放缓冲区"""
        pool = BufferPool(1024, 4)
        
        buffer = pool.acquire()
        self.assertIsNotNone(buffer)
        self.assertEqual(pool.available_count, 3)
        self.assertEqual(pool.in_use_count, 1)
        
        pool.release(buffer)
        self.assertEqual(pool.available_count, 4)
        self.assertEqual(pool.in_use_count, 0)
    
    def test_pool_exhausted(self):
        """测试池耗尽"""
        pool = BufferPool(1024, 2)
        
        buf1 = pool.acquire()
        buf2 = pool.acquire()
        
        with self.assertRaises(BufferTimeoutError):
            pool.acquire(timeout=0.1)
        
        pool.release(buf1)
        buf3 = pool.acquire(timeout=0.1)
        self.assertIsNotNone(buf3)
    
    def test_pool_stats(self):
        """测试池统计信息"""
        pool = BufferPool(1024, 4)
        
        buf = pool.acquire()
        stats = pool.stats
        
        self.assertEqual(stats.total_buffers, 4)
        self.assertEqual(stats.available_buffers, 3)
        self.assertEqual(stats.total_acquires, 1)
    
    def test_pool_resize(self):
        """测试池大小调整"""
        pool = BufferPool(1024, 4)
        
        pool.resize(8)
        self.assertEqual(pool.capacity, 8)
        self.assertEqual(pool.available_count, 8)
        
        pool.resize(2)
        self.assertEqual(pool.capacity, 2)


class TestRingBuffer(unittest.TestCase):
    """RingBuffer测试"""
    
    def test_create_ring_buffer(self):
        """测试创建环形缓冲区"""
        rb = RingBuffer(1024)
        self.assertEqual(rb.capacity, 1024)
        self.assertTrue(rb.is_empty)
    
    def test_invalid_capacity(self):
        """测试无效容量"""
        with self.assertRaises(RingBufferCapacityError):
            RingBuffer(100)  # 不是2的幂
    
    def test_write_read(self):
        """测试读写操作"""
        rb = RingBuffer(1024)
        
        data = b"Hello, World!"
        written = rb.write(data)
        self.assertEqual(written, len(data))
        
        read_data = rb.read(len(data))
        self.assertEqual(read_data, data)
    
    def test_wrap_around(self):
        """测试环形边界"""
        rb = RingBuffer(16)  # 小缓冲区测试边界
        
        # 写入并读取，使写指针接近末尾
        rb.write(b"12345678")
        rb.read(8)
        
        # 再次写入，测试环形边界
        rb.write(b"abcdefgh")
        data = rb.read(8)
        self.assertEqual(data, b"abcdefgh")
    
    def test_peek(self):
        """测试查看数据"""
        rb = RingBuffer(1024)
        rb.write(b"Hello")
        
        data = rb.peek(3)
        self.assertEqual(data, b"Hel")
        self.assertEqual(rb.readable, 5)  # 数据未被读取
    
    def test_skip(self):
        """测试跳过数据"""
        rb = RingBuffer(1024)
        rb.write(b"Hello")
        
        rb.skip(2)
        data = rb.read(3)
        self.assertEqual(data, b"llo")
    
    def test_full_empty(self):
        """测试满和空状态"""
        rb = RingBuffer(16)
        
        # 写入直到满（16字节）
        rb.write(b"12345678")  # 8字节
        rb.write(b"12345678")  # 8字节
        
        self.assertTrue(rb.is_full)
        
        rb.read(16)
        self.assertTrue(rb.is_empty)


class TestDoubleBuffer(unittest.TestCase):
    """DoubleBuffer测试"""
    
    def test_create_double_buffer(self):
        """测试创建双缓冲"""
        db = DoubleBuffer(1024)
        self.assertEqual(db.buffer_size, 1024)
    
    def test_swap(self):
        """测试交换操作"""
        db = DoubleBuffer(1024)
        
        # 写入后缓冲区
        db.write_to_back(b"Hello")
        
        # 交换
        db.swap()
        
        # 从前缓冲区读取
        data = db.read_from_front(5)
        self.assertEqual(data, b"Hello")
    
    def test_front_back(self):
        """测试前后缓冲区访问"""
        db = DoubleBuffer(1024)
        
        front = db.front
        back = db.back
        
        self.assertIsNotNone(front)
        self.assertIsNotNone(back)
        self.assertNotEqual(front, back)


class TestSPSCQueue(unittest.TestCase):
    """SPSC队列测试"""
    
    def test_create_queue(self):
        """测试创建队列"""
        queue = SPSCQueue(16)
        self.assertEqual(queue.capacity, 16)
        self.assertTrue(queue.is_empty)
    
    def test_enqueue_dequeue(self):
        """测试入队出队"""
        queue = SPSCQueue(16)
        
        self.assertTrue(queue.enqueue(1))
        self.assertTrue(queue.enqueue(2))
        
        self.assertEqual(queue.dequeue(), 1)
        self.assertEqual(queue.dequeue(), 2)
    
    def test_full_empty(self):
        """测试满和空状态"""
        queue = SPSCQueue(4)  # 容量4，实际可用3
        
        queue.enqueue(1)
        queue.enqueue(2)
        queue.enqueue(3)
        
        self.assertTrue(queue.is_full())
        self.assertFalse(queue.enqueue(4))
        
        queue.dequeue()
        self.assertFalse(queue.is_full())
    
    def test_concurrent_access(self):
        """测试并发访问"""
        queue = SPSCQueue(1000)
        items_produced = []
        items_consumed = []
        stop_flag = threading.Event()
        
        def producer():
            for i in range(100):
                queue.enqueue(i)
                items_produced.append(i)
            stop_flag.set()
        
        def consumer():
            timeout = time.time() + 5.0
            while time.time() < timeout and len(items_consumed) < 100:
                item = queue.dequeue()
                if item is not None:
                    items_consumed.append(item)
                else:
                    time.sleep(0.001)
        
        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        
        t1.start()
        t2.start()
        
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        
        self.assertEqual(len(items_consumed), 100)


class TestMPSCQueue(unittest.TestCase):
    """MPSC队列测试"""
    
    def test_multiple_producers(self):
        """测试多生产者"""
        queue = MPSCQueue(100)
        
        def producer(start):
            for i in range(10):
                queue.enqueue(start + i)
        
        threads = [
            threading.Thread(target=producer, args=(i * 10,))
            for i in range(3)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        count = 0
        while not queue.is_empty():
            queue.dequeue()
            count += 1
        
        self.assertEqual(count, 30)


class TestMPMCQueue(unittest.TestCase):
    """MPMC队列测试"""
    
    def test_multiple_producers_consumers(self):
        """测试多生产者多消费者"""
        queue = MPMCQueue(100)
        consumed = []
        lock = threading.Lock()
        
        def producer(start):
            for i in range(10):
                queue.enqueue(start + i)
        
        def consumer():
            timeout = time.time() + 3.0
            count = 0
            while time.time() < timeout and count < 10:
                item = queue.dequeue()
                if item is not None:
                    with lock:
                        consumed.append(item)
                    count += 1
                else:
                    time.sleep(0.001)
        
        producers = [
            threading.Thread(target=producer, args=(i * 10,))
            for i in range(3)
        ]
        consumers = [
            threading.Thread(target=consumer)
            for _ in range(3)
        ]
        
        for t in producers + consumers:
            t.start()
        for t in producers + consumers:
            t.join(timeout=5.0)
        
        self.assertEqual(len(consumed), 30)


class TestLRU(unittest.TestCase):
    """LRU缓存测试"""
    
    def test_basic_operations(self):
        """测试基本操作"""
        lru = LRU(3)
        
        lru.put("a", 1)
        lru.put("b", 2)
        lru.put("c", 3)
        
        self.assertEqual(lru.get("a"), 1)
        self.assertEqual(lru.get("b"), 2)
        self.assertEqual(lru.get("c"), 3)
    
    def test_eviction(self):
        """测试淘汰策略"""
        lru = LRU(2)
        
        lru.put("a", 1)
        lru.put("b", 2)
        lru.put("c", 3)  # 应该淘汰a
        
        self.assertIsNone(lru.get("a"))
        self.assertEqual(lru.get("b"), 2)
        self.assertEqual(lru.get("c"), 3)
    
    def test_access_order(self):
        """测试访问顺序"""
        lru = LRU(3)
        
        lru.put("a", 1)
        lru.put("b", 2)
        lru.put("c", 3)
        
        # 访问a，使其变为最近使用
        lru.get("a")
        
        # 添加新项，应该淘汰b
        lru.put("d", 4)
        
        self.assertEqual(lru.get("a"), 1)
        self.assertIsNone(lru.get("b"))
    
    def test_hit_rate(self):
        """测试命中率"""
        lru = LRU(3)
        
        lru.put("a", 1)
        lru.get("a")  # 命中
        lru.get("b")  # 未命中
        
        self.assertEqual(lru.hit_rate, 0.5)


class TestARC(unittest.TestCase):
    """ARC缓存测试"""
    
    def test_basic_operations(self):
        """测试基本操作"""
        arc = ARC(4)
        
        arc.put("a", 1)
        arc.put("b", 2)
        arc.put("c", 3)
        arc.put("d", 4)
        
        self.assertEqual(arc.get("a"), 1)
        self.assertEqual(arc.get("b"), 2)
    
    def test_adaptive_behavior(self):
        """测试自适应行为"""
        arc = ARC(4)
        
        # 顺序访问
        for i in range(10):
            arc.put(str(i), i)
        
        # 检查缓存大小
        self.assertLessEqual(arc.size(), 4)
    
    def test_ghost_lists(self):
        """测试幽灵列表"""
        arc = ARC(4)
        
        # 填充缓存
        for i in range(4):
            arc.put(str(i), i)
        
        # 添加新项，触发淘汰
        arc.put("e", 5)
        
        # 检查统计
        stats = arc.stats
        self.assertEqual(stats["capacity"], 4)


class TestPrefetcher(unittest.TestCase):
    """预取器测试"""
    
    def test_record_access(self):
        """测试记录访问"""
        pool = BufferPool(1024, 4)
        prefetcher = Prefetcher(pool)
        
        prefetcher.record_access("a")
        prefetcher.record_access("b")
        prefetcher.record_access("a")
        
        hot_keys = prefetcher.get_hot_keys()
        self.assertEqual(hot_keys[0], "a")
    
    def test_predict_sequential(self):
        """测试顺序预测"""
        pool = BufferPool(1024, 4)
        prefetcher = Prefetcher(pool)
        
        # 模拟顺序访问
        prefetcher.record_access("1")
        prefetcher.record_access("2")
        prefetcher.record_access("3")
        
        predicted = prefetcher.predict_next()
        self.assertEqual(predicted, "4")
    
    def test_callback(self):
        """测试回调函数"""
        pool = BufferPool(1024, 4)
        prefetcher = Prefetcher(pool)
        
        called = []
        prefetcher.set_callback(lambda key: called.append(key))
        
        # 记录足够的访问来触发预测（需要至少2个）
        prefetcher.record_access("a")
        prefetcher.record_access("a")
        prefetcher.prefetch()
        
        # 预测应该返回最常访问的键 "a"
        self.assertEqual(called, ["a"])


class TestAdaptiveStrategy(unittest.TestCase):
    """自适应策略测试"""
    
    def test_create_strategy(self):
        """测试创建策略"""
        pool = BufferPool(1024, 4)
        strategy = AdaptiveStrategy(pool)
        
        self.assertEqual(strategy.level, AdaptationLevel.MODERATE)
    
    def test_update_metrics(self):
        """测试更新指标"""
        pool = BufferPool(1024, 4)
        strategy = AdaptiveStrategy(pool)
        
        strategy.update_metrics(0.8)
        self.assertEqual(strategy.stats.current_hit_rate, 0.8)
    
    def test_adjustment_levels(self):
        """测试调整级别"""
        pool = BufferPool(1024, 4)
        strategy = AdaptiveStrategy(pool)
        
        strategy.set_level(AdaptationLevel.AGGRESSIVE)
        self.assertEqual(strategy.level, AdaptationLevel.AGGRESSIVE)


class TestWatermarkManager(unittest.TestCase):
    """水位线管理测试"""
    
    def test_create_manager(self):
        """测试创建管理器"""
        wm = WatermarkManager(low=0.25, high=0.75, critical=0.90)
        
        self.assertEqual(wm.current_level, WatermarkLevel.NORMAL)
    
    def test_check_levels(self):
        """测试检查水位"""
        wm = WatermarkManager(low=0.25, high=0.75, critical=0.90)
        
        self.assertEqual(wm.check(0.1), WatermarkLevel.LOW)
        self.assertEqual(wm.check(0.5), WatermarkLevel.NORMAL)
        self.assertEqual(wm.check(0.8), WatermarkLevel.HIGH)
        self.assertEqual(wm.check(0.95), WatermarkLevel.CRITICAL)
    
    def test_callback(self):
        """测试回调函数"""
        wm = WatermarkManager(low=0.25, high=0.75, critical=0.90)
        
        called = []
        wm.set_callback(WatermarkLevel.HIGH, lambda u: called.append(u))
        
        wm.check(0.8)
        
        self.assertEqual(called, [0.8])
    
    def test_invalid_levels(self):
        """测试无效水位线配置"""
        with self.assertRaises(ValueError):
            WatermarkManager(low=0.8, high=0.5, critical=0.9)


class TestBufferMetrics(unittest.TestCase):
    """指标收集测试"""
    
    def test_record_operations(self):
        """测试记录操作"""
        metrics = BufferMetrics()
        
        metrics.record_acquire(1.5)
        metrics.record_release(0.5)
        metrics.record_hit()
        metrics.record_miss()
        
        self.assertEqual(metrics.total_acquires, 1)
        self.assertEqual(metrics.total_releases, 1)
        self.assertEqual(metrics.hit_rate, 0.5)
    
    def test_snapshot(self):
        """测试快照"""
        metrics = BufferMetrics()
        
        metrics.record_hit()
        metrics.record_miss()
        
        snapshot = metrics.take_snapshot(pool_usage=0.5)
        
        self.assertEqual(snapshot.hit_rate, 0.5)
        self.assertEqual(snapshot.pool_usage, 0.5)


class TestStreamAdapter(unittest.TestCase):
    """流适配器测试"""
    
    def test_read_from_stream(self):
        """测试从流读取"""
        adapter = StreamAdapter(chunk_size=10)
        stream = io.BytesIO(b"Hello, World! This is a test.")
        
        chunks = list(adapter.read_from_stream(stream))
        
        self.assertEqual(len(chunks), 3)
        # 统计值应该是读取的总字节数
        self.assertEqual(adapter.stats["total_read"], len(b"Hello, World! This is a test."))
    
    def test_write_to_stream(self):
        """测试写入流"""
        adapter = StreamAdapter()
        stream = io.BytesIO()
        
        adapter.write_to_stream(stream, b"Hello")
        
        stream.seek(0)
        self.assertEqual(stream.read(), b"Hello")
    
    def test_process_stream(self):
        """测试处理流"""
        adapter = StreamAdapter(chunk_size=10)
        input_stream = io.BytesIO(b"Hello, World!")
        output_stream = io.BytesIO()
        
        def processor(data):
            return data.upper()
        
        adapter.process_stream(input_stream, output_stream, processor)
        
        output_stream.seek(0)
        self.assertEqual(output_stream.read(), b"HELLO, WORLD!")


class TestCompressorAdapter(unittest.TestCase):
    """压缩器适配器测试"""
    
    def test_compress_decompress(self):
        """测试压缩解压"""
        adapter = CompressorAdapter(CompressionType.ZLIB)
        
        data = b"Hello, World! " * 100
        compressed = adapter.compress(data)
        decompressed = adapter.decompress(compressed)
        
        self.assertEqual(data, decompressed)
        self.assertLess(len(compressed), len(data))
    
    def test_compression_ratio(self):
        """测试压缩率"""
        adapter = CompressorAdapter(CompressionType.ZLIB)
        
        data = b"Hello, World! " * 100
        adapter.compress(data)
        
        self.assertLess(adapter.compression_ratio, 1.0)
        self.assertGreater(adapter.space_saved, 0)
    
    def test_gzip_compression(self):
        """测试GZIP压缩"""
        adapter = CompressorAdapter(CompressionType.GZIP)
        
        data = b"Hello, World! " * 100
        compressed = adapter.compress(data)
        decompressed = adapter.decompress(compressed)
        
        self.assertEqual(data, decompressed)
    
    def test_no_compression(self):
        """测试无压缩"""
        adapter = CompressorAdapter(CompressionType.NONE)
        
        data = b"Hello, World!"
        compressed = adapter.compress(data)
        
        self.assertEqual(data, compressed)


class TestBufferConfig(unittest.TestCase):
    """配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = BufferConfig.default()
        
        self.assertEqual(config.buffer_size, 4096)
        self.assertEqual(config.alignment, 64)
    
    def test_high_performance_config(self):
        """测试高性能配置"""
        config = BufferConfig.high_performance()
        
        self.assertEqual(config.buffer_size, 65536)
        self.assertTrue(config.prefetch_enabled)
    
    def test_low_memory_config(self):
        """测试低内存配置"""
        config = BufferConfig.low_memory()
        
        self.assertEqual(config.buffer_size, 1024)
        self.assertFalse(config.prefetch_enabled)
    
    def test_validate_config(self):
        """测试配置验证"""
        config = BufferConfig(buffer_size=-1)
        
        with self.assertRaises(ValueError):
            config.validate()


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流水线"""
        # 创建缓冲池
        pool = BufferPool(1024, 4)
        
        # 创建水位线管理器
        wm = WatermarkManager(low=0.25, high=0.75, critical=0.90)
        
        # 创建指标收集器
        metrics = BufferMetrics()
        
        # 获取缓冲区
        buffer = pool.acquire()
        
        # 写入数据
        data = b"Test data for integration test"
        buffer.write(data)
        
        # 检查水位
        usage = pool.in_use_count / pool.capacity
        level = wm.check(usage)
        
        # 记录指标
        metrics.record_acquire(1.0)
        metrics.record_hit()
        
        # 释放缓冲区
        pool.release(buffer)
        metrics.record_release(0.5)
        
        # 验证
        self.assertEqual(pool.available_count, 4)
        self.assertEqual(metrics.hit_rate, 1.0)
    
    def test_concurrent_pool_access(self):
        """测试并发池访问"""
        pool = BufferPool(1024, 8)
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    buffer = pool.acquire(timeout=1.0)
                    time.sleep(0.001)
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(pool.available_count, 8)


if __name__ == "__main__":
    unittest.main()
