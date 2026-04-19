"""
mem_monitor 单元测试模块

测试内存监控器的各项功能。
"""

import time
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem_monitor import (
    MemoryMonitor,
    MonitorConfig,
    SamplerConfig,
    AnalyzerConfig,
    TieringConfig,
    ReporterConfig,
    MemorySnapshot,
    MonitorState,
    MonitorError,
    ConfigurationError,
)
from mem_monitor.core.config import AlertLevel
from mem_monitor.core.exceptions import (
    SamplerError,
    AnalyzerError,
    ThresholdExceededError,
)
from mem_monitor.sampler import (
    SoftwareSampler,
    TracemallocSampler,
    SampleData,
    SamplerState,
)
from mem_monitor.analyzer import (
    LifecycleAnalyzer,
    HotspotAnalyzer,
    LeakDetector,
)


class TestMonitorConfig(unittest.TestCase):
    """测试监控配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = MonitorConfig()

        self.assertEqual(config.name, "MemoryMonitor")
        self.assertTrue(config.enable_monitoring)
        self.assertTrue(config.enable_alerts)
        self.assertTrue(config.thread_safe)

    def test_config_validation(self):
        """测试配置验证"""
        config = MonitorConfig()
        self.assertTrue(config.validate())

        # 无效配置
        config.max_overhead = -1
        self.assertFalse(config.validate())

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = MonitorConfig()
        d = config.to_dict()

        self.assertIn('name', d)
        self.assertIn('enable_monitoring', d)
        self.assertIn('sampler', d)

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            'name': 'TestMonitor',
            'enable_monitoring': False,
            'max_overhead': 0.1,
        }

        config = MonitorConfig.from_dict(data)
        self.assertEqual(config.name, 'TestMonitor')
        self.assertFalse(config.enable_monitoring)
        self.assertEqual(config.max_overhead, 0.1)


class TestMemorySnapshot(unittest.TestCase):
    """测试内存快照"""

    def test_snapshot_creation(self):
        """测试快照创建"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=12345,
            rss=1024 * 1024 * 100,  # 100MB
            vms=1024 * 1024 * 200,  # 200MB
            total=1024 * 1024 * 1024,  # 1GB
        )

        self.assertEqual(snapshot.process_id, 12345)
        self.assertEqual(snapshot.rss, 1024 * 1024 * 100)

    def test_usage_ratio(self):
        """测试使用率计算"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
            rss=512 * 1024 * 1024,  # 512MB
            total=1024 * 1024 * 1024,  # 1GB
        )

        ratio = snapshot.get_usage_ratio()
        self.assertAlmostEqual(ratio, 0.5, places=2)

    def test_snapshot_to_dict(self):
        """测试快照转换为字典"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
            rss=1024,
        )

        d = snapshot.to_dict()
        self.assertIn('rss', d)
        self.assertIn('process_id', d)


class TestMemoryMonitor(unittest.TestCase):
    """测试内存监控器"""

    def test_monitor_creation(self):
        """测试监控器创建"""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        self.assertEqual(monitor.get_state(), MonitorState.CREATED)

    def test_monitor_start_stop(self):
        """测试启动和停止"""
        config = MonitorConfig()
        config.background_thread = False  # 禁用后台线程以便测试

        monitor = MemoryMonitor(config)
        monitor.start()

        self.assertEqual(monitor.get_state(), MonitorState.RUNNING)

        report = monitor.stop()
        self.assertEqual(monitor.get_state(), MonitorState.STOPPED)
        self.assertIsNotNone(report)

    def test_monitor_context_manager(self):
        """测试上下文管理器"""
        config = MonitorConfig()
        config.background_thread = False

        with MemoryMonitor(config) as monitor:
            self.assertEqual(monitor.get_state(), MonitorState.RUNNING)

        self.assertEqual(monitor.get_state(), MonitorState.STOPPED)

    def test_get_snapshot(self):
        """测试获取快照"""
        config = MonitorConfig()
        config.background_thread = False

        monitor = MemoryMonitor(config)
        monitor.start()

        snapshot = monitor.get_snapshot()
        self.assertIsInstance(snapshot, MemorySnapshot)
        self.assertGreater(snapshot.timestamp, 0)

        monitor.stop()

    def test_threshold_setting(self):
        """测试阈值设置"""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        monitor.set_threshold('memory_usage', 0.8, 'alert')

        self.assertIn('memory_usage', config.thresholds)
        self.assertEqual(config.thresholds['memory_usage'].warning, 0.8)

    def test_alert_handler(self):
        """测试告警处理器"""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        alerts_received = []

        def handler(alert):
            alerts_received.append(alert)

        monitor.add_alert_handler(handler)

        # 模拟告警
        from mem_monitor.core.monitor import Alert
        test_alert = Alert(
            level=AlertLevel.WARNING,
            metric='test',
            value=0.9,
            threshold=0.8,
            timestamp=time.time(),
            message='Test alert',
        )

        # 直接调用处理器
        for h in monitor._alert_handlers:
            h(test_alert)

        self.assertEqual(len(alerts_received), 1)

    def test_hook_registration(self):
        """测试钩子注册"""
        config = MonitorConfig()
        config.enable_hooks = True
        monitor = MemoryMonitor(config)

        called = []

        def hook(data):
            called.append(data)

        monitor.register_hook('snapshot_taken', hook)

        # 检查钩子已注册
        from mem_monitor.core.monitor import HookType
        self.assertIn(hook, monitor._hooks[HookType.SNAPSHOT_TAKEN])

        monitor.unregister_hook('snapshot_taken', hook)
        self.assertNotIn(hook, monitor._hooks[HookType.SNAPSHOT_TAKEN])

    def test_get_stats(self):
        """测试获取统计"""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        stats = monitor.get_stats()
        self.assertIn('state', stats)
        self.assertIn('history_size', stats)


class TestSampler(unittest.TestCase):
    """测试采样器"""

    def test_software_sampler(self):
        """测试软件采样器"""
        config = SamplerConfig()
        sampler = SoftwareSampler(config)

        self.assertEqual(sampler.get_state(), SamplerState.CREATED)

        sampler.start()
        self.assertEqual(sampler.get_state(), SamplerState.RUNNING)

        sampler.stop()
        self.assertEqual(sampler.get_state(), SamplerState.STOPPED)

    def test_sample_data(self):
        """测试采样数据"""
        data = SampleData(
            timestamp=time.time(),
            metrics={'rss': 1024, 'vms': 2048},
        )

        d = data.to_dict()
        self.assertIn('timestamp', d)
        self.assertIn('metrics', d)

    def test_tracemalloc_sampler(self):
        """测试tracemalloc采样器"""
        config = SamplerConfig()
        sampler = TracemallocSampler(config)

        # 检查可用性
        available = sampler.is_available()
        self.assertIsInstance(available, bool)


class TestAnalyzer(unittest.TestCase):
    """测试分析器"""

    def test_lifecycle_analyzer(self):
        """测试生命周期分析器"""
        config = AnalyzerConfig()
        analyzer = LifecycleAnalyzer(config)

        # 测试追踪分配
        obj = [1, 2, 3]
        alloc_id = analyzer.track_allocation(obj, 100)

        # 获取统计
        stats = analyzer._tracker.get_stats()
        self.assertIn('total_tracked', stats)

    def test_hotspot_analyzer(self):
        """测试热点分析器"""
        config = AnalyzerConfig()
        analyzer = HotspotAnalyzer(config)

        # 记录访问
        analyzer.record_access(0x1000, 8)
        analyzer.record_access(0x1000, 8)
        analyzer.record_access(0x2000, 8)

        # 获取热点
        hotspots = analyzer.get_hotspots(threshold=0.0)
        self.assertIsInstance(hotspots, list)

    def test_leak_detector(self):
        """测试泄漏检测器"""
        config = AnalyzerConfig()
        detector = LeakDetector(config)

        # 执行检测
        report = detector.detect()

        self.assertIsNotNone(report)
        self.assertIsInstance(report.leaks, list)


class TestExceptions(unittest.TestCase):
    """测试异常"""

    def test_monitor_error(self):
        """测试监控错误"""
        error = MonitorError("Test error", {'detail': 'value'})

        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.details['detail'], 'value')

        d = error.to_dict()
        self.assertIn('error_type', d)
        self.assertIn('message', d)

    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError("Invalid config", "test_key")

        self.assertEqual(error.details['config_key'], 'test_key')

    def test_threshold_exceeded_error(self):
        """测试阈值超限错误"""
        error = ThresholdExceededError(
            metric_name='memory_usage',
            current_value=0.95,
            threshold=0.8,
            action='alert'
        )

        self.assertEqual(error.metric_name, 'memory_usage')
        self.assertEqual(error.current_value, 0.95)


class TestIntegration(unittest.TestCase):
    """测试集成适配器"""

    def test_psutil_adapter(self):
        """测试psutil适配器"""
        from mem_monitor.integration import PsutilAdapter

        adapter = PsutilAdapter()

        if adapter.is_available():
            mem_info = adapter.get_memory_info()
            self.assertIsInstance(mem_info, dict)

            sys_mem = adapter.get_system_memory_info()
            self.assertIsNotNone(sys_mem)

    def test_tracemalloc_adapter(self):
        """测试tracemalloc适配器"""
        from mem_monitor.integration import TracemallocAdapter

        adapter = TracemallocAdapter(nframe=10)

        if adapter.is_available():
            # 启动追踪
            result = adapter.start()
            self.assertTrue(result)

            # 获取统计
            stats = adapter.get_stats()
            self.assertIn('available', stats)

            # 停止追踪
            adapter.stop()


class TestReporter(unittest.TestCase):
    """测试报告器"""

    def test_metrics_collector(self):
        """测试指标收集器"""
        from mem_monitor.reporter import MetricsCollector
        from mem_monitor.core import ReporterConfig

        config = ReporterConfig()
        collector = MetricsCollector(config)

        # 模拟快照
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
            rss=1024 * 1024,
            vms=2048 * 1024,
            total=1024 * 1024 * 1024,
        )

        metrics = collector.collect(snapshot)
        self.assertIn('memory_rss', metrics)
        self.assertIn('memory_usage_ratio', metrics)

    def test_prometheus_export(self):
        """测试Prometheus导出"""
        from mem_monitor.reporter import MetricsCollector, MetricValue, MetricType
        from mem_monitor.core import ReporterConfig

        config = ReporterConfig()
        collector = MetricsCollector(config)

        # 添加指标
        collector._metrics['test_metric'] = MetricValue(
            name='test_metric',
            value=123.45,
            metric_type=MetricType.GAUGE,
            description='Test metric'
        )

        prom = collector.to_prometheus()
        self.assertIn('test_metric', prom)


class TestConcurrency(unittest.TestCase):
    """测试并发安全性"""

    def test_concurrent_snapshot_access(self):
        """测试并发快照访问"""
        config = MonitorConfig()
        config.background_thread = False
        config.thread_safe = True
        
        monitor = MemoryMonitor(config)
        monitor.start()
        
        errors = []
        
        def take_snapshots():
            try:
                for _ in range(10):
                    snapshot = monitor.get_snapshot()
                    self.assertIsInstance(snapshot, MemorySnapshot)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程同时访问
        threads = []
        for _ in range(5):
            t = threading.Thread(target=take_snapshots)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        monitor.stop()
        
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

    def test_concurrent_history_access(self):
        """测试并发历史记录访问"""
        config = MonitorConfig()
        config.background_thread = False
        config.thread_safe = True
        
        monitor = MemoryMonitor(config)
        monitor.start()
        
        # 先采集一些快照
        for _ in range(5):
            monitor.get_snapshot()
        
        errors = []
        
        def read_history():
            try:
                for _ in range(10):
                    history = monitor.get_history()
                    self.assertIsInstance(history, list)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        def write_snapshots():
            try:
                for _ in range(10):
                    monitor.get_snapshot()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 创建读写线程
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=read_history))
            threads.append(threading.Thread(target=write_snapshots))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        monitor.stop()
        
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

    def test_concurrent_alert_access(self):
        """测试并发告警访问"""
        config = MonitorConfig()
        config.background_thread = False
        config.thread_safe = True
        config.enable_alerts = True
        
        monitor = MemoryMonitor(config)
        monitor.start()
        
        errors = []
        
        def read_alerts():
            try:
                for _ in range(10):
                    alerts = monitor.get_alerts()
                    self.assertIsInstance(alerts, list)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 创建多个读取线程
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=read_alerts))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        monitor.stop()
        
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")


class TestBoundaryConditions(unittest.TestCase):
    """测试边界条件"""

    def test_empty_history(self):
        """测试空历史记录"""
        config = MonitorConfig()
        config.background_thread = False
        
        monitor = MemoryMonitor(config)
        
        # 不采集快照，直接获取历史
        history = monitor.get_history()
        self.assertEqual(len(history), 0)
        
        # 空历史生成报告
        report = monitor.stop()
        self.assertIsNotNone(report)
        self.assertEqual(report.duration, 0)

    def test_zero_values(self):
        """测试零值处理"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
            rss=0,
            vms=0,
            total=0,
        )
        
        # 测试零值计算
        ratio = snapshot.get_usage_ratio()
        self.assertEqual(ratio, 0.0)
        
        available_ratio = snapshot.get_available_ratio()
        self.assertEqual(available_ratio, 0.0)

    def test_large_values(self):
        """测试大值处理"""
        # 测试接近系统限制的大值
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
            rss=1024 * 1024 * 1024 * 100,  # 100GB
            vms=1024 * 1024 * 1024 * 200,  # 200GB
            total=1024 * 1024 * 1024 * 1000,  # 1TB
        )
        
        ratio = snapshot.get_usage_ratio()
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1)

    def test_invalid_config_values(self):
        """测试无效配置值"""
        from mem_monitor.core.exceptions import ConfigurationError
        
        # 测试无效的max_overhead
        with self.assertRaises(ConfigurationError):
            config = MonitorConfig.from_dict({'max_overhead': -1})
        
        with self.assertRaises(ConfigurationError):
            config = MonitorConfig.from_dict({'max_overhead': 2})
        
        # 测试无效的采样间隔
        with self.assertRaises(ConfigurationError):
            config = MonitorConfig.from_dict({
                'sampler': {'interval': -1}
            })

    def test_config_type_validation(self):
        """测试配置类型验证"""
        from mem_monitor.core.exceptions import ConfigurationError
        
        # 测试非字典输入
        with self.assertRaises(ConfigurationError):
            MonitorConfig.from_dict("not a dict")
        
        # 测试无效的名称类型
        with self.assertRaises(ConfigurationError):
            MonitorConfig.from_dict({'name': 123})
        
        # 测试无效的布尔类型
        with self.assertRaises(ConfigurationError):
            MonitorConfig.from_dict({'enable_monitoring': 'yes'})

    def test_negative_timestamp(self):
        """测试负时间戳处理"""
        # 时间戳应该总是正数
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_id=1,
        )
        
        self.assertGreater(snapshot.timestamp, 0)

    def test_threshold_boundary(self):
        """测试阈值边界"""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)
        
        # 测试边界阈值
        monitor.set_threshold('memory_usage', 0.0)  # 最小值
        monitor.set_threshold('memory_usage', 1.0)  # 最大值
        
        self.assertIn('memory_usage', config.thresholds)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""

    def test_invalid_sampler_type(self):
        """测试无效采样器类型"""
        from mem_monitor.core.exceptions import ConfigurationError
        
        with self.assertRaises(ConfigurationError):
            config = MonitorConfig.from_dict({
                'sampler': {'type': 'invalid_type'}
            })

    def test_monitor_error_recovery(self):
        """测试监控器错误恢复"""
        config = MonitorConfig()
        config.background_thread = False
        
        monitor = MemoryMonitor(config)
        monitor.start()
        
        # 模拟错误情况
        monitor._stats['errors'] = 5
        
        # 监控器应该仍然可以工作
        snapshot = monitor.get_snapshot()
        self.assertIsInstance(snapshot, MemorySnapshot)
        
        monitor.stop()

    def test_hook_error_handling(self):
        """测试钩子错误处理"""
        config = MonitorConfig()
        config.enable_hooks = True
        config.background_thread = False
        
        monitor = MemoryMonitor(config)
        
        # 注册一个会抛出异常的钩子
        def bad_hook(data):
            raise ValueError("Test error")
        
        monitor.register_hook('snapshot_taken', bad_hook)
        monitor.start()
        
        # 快照应该仍然成功，即使钩子失败
        snapshot = monitor.get_snapshot()
        self.assertIsInstance(snapshot, MemorySnapshot)
        
        monitor.stop()


if __name__ == '__main__':
    unittest.main(verbosity=2)
