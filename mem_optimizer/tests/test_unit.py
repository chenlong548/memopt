"""
mem_optimizer 单元测试

测试内存分配优化器的核心功能。
"""

import unittest
import time
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem_optimizer.core.base import (
    AllocatorType, AllocationStrategy, MemoryRegionState,
    MemoryBlock, AllocationRequest, AllocationResult,
    MemoryStatistics, AllocatorBase
)
from mem_optimizer.core.config import (
    OptimizerConfig, BuddyAllocatorConfig, SlabAllocatorConfig,
    TLSFAllocatorConfig, DefragConfig, NUMAConfig
)
from mem_optimizer.core.exceptions import (
    MemOptimizerError, AllocationError, OutOfMemoryError,
    FragmentationError, ConfigurationError
)
from mem_optimizer.core.memory_pool import MemoryPool
from mem_optimizer.allocators.buddy import BuddyAllocator
from mem_optimizer.allocators.slab import SlabAllocator
from mem_optimizer.allocators.tlsf import TLSFAllocator
from mem_optimizer.strategies.rl_selector import RLStrategySelector
from mem_optimizer.strategies.bandit import UCB1Bandit, ThompsonSamplingBandit
from mem_optimizer.defrag.defragmenter import Defragmenter
from mem_optimizer.defrag.coalescer import MemoryCoalescer
from mem_optimizer.numa.coordinator import NUMACoordinator
from mem_optimizer.monitor.monitor import MemoryMonitor


class TestMemoryBlock(unittest.TestCase):
    """内存块测试"""

    def test_create_block(self):
        """测试创建内存块"""
        block = MemoryBlock(
            address=0x1000,
            size=4096,
            state=MemoryRegionState.FREE
        )

        self.assertEqual(block.address, 0x1000)
        self.assertEqual(block.size, 4096)
        self.assertTrue(block.is_free())
        self.assertFalse(block.is_allocated())

    def test_block_contains(self):
        """测试地址包含检查"""
        block = MemoryBlock(
            address=0x1000,
            size=4096,
            state=MemoryRegionState.ALLOCATED
        )

        self.assertTrue(block.contains(0x1000))
        self.assertTrue(block.contains(0x1FFF))
        self.assertFalse(block.contains(0x0FFF))
        self.assertFalse(block.contains(0x2000))

    def test_block_can_merge(self):
        """测试块合并检查"""
        block1 = MemoryBlock(
            address=0x1000,
            size=4096,
            state=MemoryRegionState.FREE
        )

        block2 = MemoryBlock(
            address=0x2000,
            size=4096,
            state=MemoryRegionState.FREE
        )

        self.assertTrue(block1.can_merge(block2))

        block3 = MemoryBlock(
            address=0x3000,
            size=4096,
            state=MemoryRegionState.ALLOCATED
        )

        self.assertFalse(block1.can_merge(block3))


class TestAllocationRequest(unittest.TestCase):
    """分配请求测试"""

    def test_create_request(self):
        """测试创建分配请求"""
        request = AllocationRequest(
            size=1024,
            alignment=16,
            numa_node=0
        )

        self.assertEqual(request.size, 1024)
        self.assertEqual(request.alignment, 16)
        self.assertEqual(request.numa_node, 0)

    def test_default_values(self):
        """测试默认值"""
        request = AllocationRequest(size=512)

        self.assertEqual(request.alignment, 8)
        self.assertEqual(request.numa_node, -1)
        self.assertEqual(request.priority, 0)


class TestMemoryStatistics(unittest.TestCase):
    """内存统计测试"""

    def test_update_usage(self):
        """测试更新使用率"""
        stats = MemoryStatistics(total_size=1024 * 1024)

        stats.update_usage(512 * 1024, 1024 * 1024)

        self.assertEqual(stats.used_size, 512 * 1024)
        self.assertEqual(stats.free_size, 512 * 1024)
        self.assertEqual(stats.peak_usage, 512 * 1024)

        stats.update_usage(768 * 1024, 1024 * 1024)

        self.assertEqual(stats.peak_usage, 768 * 1024)

    def test_calculate_fragmentation(self):
        """测试碎片率计算"""
        stats = MemoryStatistics()

        free_blocks = [
            MemoryBlock(address=0, size=1024, state=MemoryRegionState.FREE),
            MemoryBlock(address=4096, size=512, state=MemoryRegionState.FREE),
            MemoryBlock(address=8192, size=256, state=MemoryRegionState.FREE)
        ]

        stats.calculate_fragmentation(free_blocks)

        self.assertGreater(stats.fragmentation_ratio, 0)
        self.assertLess(stats.fragmentation_ratio, 1)

    def test_get_summary(self):
        """测试获取摘要"""
        stats = MemoryStatistics(
            total_size=1024 * 1024,
            used_size=512 * 1024,
            free_size=512 * 1024,
            fragmentation_ratio=0.25
        )

        summary = stats.get_summary()

        self.assertIn('total_size_mb', summary)
        self.assertIn('usage_percent', summary)
        self.assertIn('fragmentation_ratio', summary)


class TestBuddyAllocator(unittest.TestCase):
    """Buddy分配器测试"""

    def setUp(self):
        """测试前准备"""
        self.allocator = BuddyAllocator(
            total_size=1024 * 1024,
            base_address=0
        )

    def test_allocate_small(self):
        """测试小内存分配"""
        request = AllocationRequest(size=128)
        result = self.allocator.allocate(request)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.address, 0)
        self.assertGreaterEqual(result.actual_size, 128)

    def test_allocate_large(self):
        """测试大内存分配"""
        request = AllocationRequest(size=256 * 1024)
        result = self.allocator.allocate(request)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.actual_size, 256 * 1024)

    def test_allocate_and_deallocate(self):
        """测试分配和释放"""
        request = AllocationRequest(size=1024)
        result = self.allocator.allocate(request)

        self.assertTrue(result.success)

        success = self.allocator.deallocate(result.address)
        self.assertTrue(success)

    def test_multiple_allocations(self):
        """测试多次分配"""
        addresses = []

        for i in range(10):
            request = AllocationRequest(size=1024 * (i + 1))
            result = self.allocator.allocate(request)

            self.assertTrue(result.success)
            addresses.append(result.address)

        for addr in addresses:
            self.assertTrue(self.allocator.deallocate(addr))

    def test_fragmentation_score(self):
        """测试碎片评分"""
        score = self.allocator.get_fragmentation_score()

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestSlabAllocator(unittest.TestCase):
    """Slab分配器测试"""

    def setUp(self):
        """测试前准备"""
        self.allocator = SlabAllocator(
            total_size=10 * 1024 * 1024,
            base_address=0
        )

    def test_allocate_small_object(self):
        """测试小对象分配"""
        request = AllocationRequest(size=64)
        result = self.allocator.allocate(request)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.actual_size, 64)

    def test_allocate_multiple_sizes(self):
        """测试多种大小分配"""
        sizes = [64, 256, 512, 1024]

        for size in sizes:
            request = AllocationRequest(size=size)
            result = self.allocator.allocate(request)

            self.assertTrue(result.success, f"Failed to allocate size {size}: {result.error_message}")

    def test_cache_stats(self):
        """测试缓存统计"""
        for _ in range(10):
            request = AllocationRequest(size=128)
            self.allocator.allocate(request)

        stats = self.allocator.get_cache_stats()

        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)


class TestTLSFAllocator(unittest.TestCase):
    """TLSF分配器测试"""

    def setUp(self):
        """测试前准备"""
        self.allocator = TLSFAllocator(
            total_size=1024 * 1024,
            base_address=0
        )

    def test_allocate_various_sizes(self):
        """测试各种大小分配"""
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]

        for size in sizes:
            request = AllocationRequest(size=size)
            result = self.allocator.allocate(request)

            self.assertTrue(result.success)

    def test_o1_allocation(self):
        """测试O(1)分配性能"""
        start_time = time.time()

        for _ in range(100):
            request = AllocationRequest(size=256)
            result = self.allocator.allocate(request)
            self.assertTrue(result.success)

        elapsed = time.time() - start_time

        self.assertLess(elapsed, 0.1)

    def test_bitmap_stats(self):
        """测试位图统计"""
        for _ in range(5):
            request = AllocationRequest(size=128)
            self.allocator.allocate(request)

        stats = self.allocator.get_bitmap_stats()

        self.assertIn('fl_bitmap', stats)
        self.assertIn('total_blocks', stats)


class TestMemoryPool(unittest.TestCase):
    """内存池测试"""

    def setUp(self):
        """测试前准备"""
        self.config = OptimizerConfig(
            total_memory=10 * 1024 * 1024,
            max_allocation_size=5 * 1024 * 1024,
            enable_numa=False,
            enable_defrag=False,
            enable_monitoring=False
        )
        self.pool = MemoryPool(self.config)

    def test_allocate(self):
        """测试分配"""
        result = self.pool.allocate(size=1024)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.address, 0)

    def test_deallocate(self):
        """测试释放"""
        result = self.pool.allocate(size=2048)
        self.assertTrue(result.success)

        success = self.pool.deallocate(result.address)
        self.assertTrue(success)

    def test_reallocate(self):
        """测试重新分配"""
        result = self.pool.allocate(size=1024)
        self.assertTrue(result.success)

        new_result = self.pool.reallocate(result.address, 2048)

        self.assertTrue(new_result.success)

    def test_get_stats(self):
        """测试获取统计"""
        self.pool.allocate(size=1024)
        self.pool.allocate(size=2048)

        stats = self.pool.get_stats()

        self.assertGreater(stats.allocation_count, 0)
        self.assertGreater(stats.used_size, 0)

    def test_get_snapshot(self):
        """测试获取快照"""
        self.pool.allocate(size=1024)

        snapshot = self.pool.get_snapshot()

        self.assertGreater(snapshot.total_size, 0)
        self.assertGreater(snapshot.used_size, 0)


class TestRLStrategySelector(unittest.TestCase):
    """RL策略选择器测试"""

    def setUp(self):
        """测试前准备"""
        self.selector = RLStrategySelector()

    def test_select_allocator(self):
        """测试选择分配器"""
        request = AllocationRequest(size=1024)
        context = {
            'total_size': 1024 * 1024,
            'used_size': 512 * 1024,
            'fragmentation': 0.1
        }

        allocator = self.selector.select_allocator(request, context)

        self.assertIn(allocator, [AllocatorType.BUDDY, AllocatorType.SLAB, AllocatorType.TLSF])

    def test_update_performance(self):
        """测试更新性能"""
        request = AllocationRequest(size=1024)
        context = {
            'total_size': 1024 * 1024,
            'used_size': 512 * 1024,
            'fragmentation': 0.1
        }
        self.selector.select_allocator(request, context)

        performance = {
            'success': True,
            'allocation_time': 0.001,
            'fragmentation': 0.1,
            'size': 1024
        }

        self.selector.update_performance(AllocatorType.TLSF, performance)

        stats = self.selector.get_stats()

        self.assertGreaterEqual(stats['episode_count'], 0)

    def test_get_recommendations(self):
        """测试获取推荐"""
        recommendations = self.selector.get_recommendations()

        self.assertIn(AllocatorType.BUDDY, recommendations)
        self.assertIn(AllocatorType.SLAB, recommendations)
        self.assertIn(AllocatorType.TLSF, recommendations)


class TestUCBBandit(unittest.TestCase):
    """UCB多臂老虎机测试"""

    def setUp(self):
        """测试前准备"""
        self.bandit = UCB1Bandit()

    def test_select(self):
        """测试选择"""
        for _ in range(10):
            allocator = self.bandit.select()
            self.assertIn(allocator, [AllocatorType.BUDDY, AllocatorType.SLAB, AllocatorType.TLSF])

    def test_update(self):
        """测试更新"""
        for _ in range(20):
            allocator = self.bandit.select()
            self.bandit.update(allocator, 1.0)

        stats = self.bandit.get_stats()

        self.assertGreater(stats['total_pulls'], 0)

    def test_get_best_arm(self):
        """测试获取最佳臂"""
        for _ in range(10):
            allocator = self.bandit.select()
            self.bandit.update(allocator, 1.0)

        best = self.bandit.get_best_arm()

        self.assertIn(best, [AllocatorType.BUDDY, AllocatorType.SLAB, AllocatorType.TLSF])


class TestDefragmenter(unittest.TestCase):
    """碎片整理器测试"""

    def setUp(self):
        """测试前准备"""
        self.defrag = Defragmenter()

    def test_analyze(self):
        """测试分析"""
        blocks = [
            MemoryBlock(address=0, size=1024, state=MemoryRegionState.FREE),
            MemoryBlock(address=4096, size=2048, state=MemoryRegionState.ALLOCATED),
            MemoryBlock(address=8192, size=1024, state=MemoryRegionState.FREE)
        ]

        analysis = self.defrag.analyze(blocks)

        self.assertEqual(analysis['total_blocks'], 3)
        self.assertEqual(analysis['free_blocks'], 2)
        self.assertEqual(analysis['allocated_blocks'], 1)

    def test_plan(self):
        """测试规划"""
        blocks = [
            MemoryBlock(address=0, size=1024, state=MemoryRegionState.FREE),
            MemoryBlock(address=4096, size=2048, state=MemoryRegionState.ALLOCATED),
            MemoryBlock(address=8192, size=1024, state=MemoryRegionState.FREE)
        ]

        plan = self.defrag.plan(blocks)

        self.assertIsInstance(plan, list)


class TestNUMACoordinator(unittest.TestCase):
    """NUMA协调器测试"""

    def setUp(self):
        """测试前准备"""
        self.coordinator = NUMACoordinator()

    def test_get_numa_nodes(self):
        """测试获取NUMA节点"""
        nodes = self.coordinator.get_numa_nodes()

        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)

    def test_select_node(self):
        """测试选择节点"""
        request = AllocationRequest(size=1024)

        node = self.coordinator.select_node(request)

        self.assertGreaterEqual(node, 0)

    def test_get_stats(self):
        """测试获取统计"""
        stats = self.coordinator.get_stats()

        self.assertIn('available', stats)
        self.assertIn('node_count', stats)


class TestMemoryMonitor(unittest.TestCase):
    """内存监控器测试"""

    def setUp(self):
        """测试前准备"""
        self.monitor = MemoryMonitor()

    def test_start_stop(self):
        """测试启动和停止"""
        self.monitor.start()
        time.sleep(0.1)
        self.monitor.stop()

    def test_get_current_metrics(self):
        """测试获取当前指标"""
        metrics = self.monitor.get_current_metrics()

        self.assertIsInstance(metrics, dict)

    def test_get_statistics(self):
        """测试获取统计"""
        stats = self.monitor.get_statistics()

        self.assertIn('running', stats)
        self.assertIn('history_size', stats)


class TestConfiguration(unittest.TestCase):
    """配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = OptimizerConfig()

        self.assertGreater(config.total_memory, 0)
        self.assertIsInstance(config.default_allocator, AllocatorType)

    def test_config_validation(self):
        """测试配置验证"""
        config = OptimizerConfig(
            total_memory=1024 * 1024,
            max_allocation_size=512 * 1024
        )

        self.assertTrue(config.validate())

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = OptimizerConfig()
        d = config.to_dict()

        self.assertIn('total_memory', d)
        self.assertIn('default_allocator', d)


class TestExceptions(unittest.TestCase):
    """异常测试"""

    def test_allocation_error(self):
        """测试分配错误"""
        error = AllocationError("Test error", size=1024)

        self.assertIn("Test error", str(error))
        self.assertEqual(error.size, 1024)

    def test_out_of_memory_error(self):
        """测试内存不足错误"""
        error = OutOfMemoryError(requested=1024, available=512)

        self.assertIn("Out of memory", str(error))
        self.assertEqual(error.requested, 1024)

    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError("Invalid config", config_key="test_key")

        self.assertIn("Invalid config", str(error))
        self.assertEqual(error.config_key, "test_key")


if __name__ == '__main__':
    unittest.main(verbosity=2)
