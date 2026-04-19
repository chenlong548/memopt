"""
mem_mapper 统计工具模块

提供性能统计和分析工具类。
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import statistics


@dataclass
class TimingRecord:
    """时间记录"""
    operation: str      # 操作名称
    start_time: float   # 开始时间
    end_time: float     # 结束时间
    duration: float     # 持续时间（秒）
    success: bool       # 是否成功
    error: Optional[str] = None  # 错误信息


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_operations: int = 0           # 总操作数
    successful_operations: int = 0      # 成功操作数
    failed_operations: int = 0          # 失败操作数
    
    total_time: float = 0.0             # 总时间（秒）
    min_time: float = float('inf')      # 最小时间
    max_time: float = 0.0               # 最大时间
    avg_time: float = 0.0               # 平均时间
    
    throughput: float = 0.0             # 吞吐量（操作/秒）
    
    # 百分位数
    p50: float = 0.0                    # 50百分位
    p90: float = 0.0                    # 90百分位
    p95: float = 0.0                    # 95百分位
    p99: float = 0.0                    # 99百分位


class Timer:
    """
    计时器类
    
    用于测量代码执行时间。
    """
    
    def __init__(self, name: str = "operation"):
        """
        初始化计时器
        
        Args:
            name: 操作名称
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: float = 0.0  # 初始化为0.0而不是None
    
    def start(self) -> 'Timer':
        """
        开始计时
        
        Returns:
            self，支持链式调用
        """
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            持续时间（秒）
        """
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time
        else:
            self.duration = 0.0
        return self.duration
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False
    
    def get_duration(self) -> float:
        """
        获取持续时间
        
        Returns:
            持续时间（秒），如果未完成则返回0.0
        """
        return self.duration


class PerformanceTracker:
    """
    性能跟踪器
    
    跟踪和统计各种操作的性能指标。
    """
    
    def __init__(self, max_records: int = 10000):
        """
        初始化性能跟踪器
        
        Args:
            max_records: 最大记录数
        """
        self.max_records = max_records
        self.records: Dict[str, List[TimingRecord]] = defaultdict(list)
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
    
    def record(self, 
               operation: str, 
               duration: float, 
               success: bool = True,
               error: Optional[str] = None):
        """
        记录操作时间
        
        Args:
            operation: 操作名称
            duration: 持续时间（秒）
            success: 是否成功
            error: 错误信息
        """
        with self._lock:
            # 创建记录
            record = TimingRecord(
                operation=operation,
                start_time=time.time() - duration,
                end_time=time.time(),
                duration=duration,
                success=success,
                error=error
            )
            
            # 添加到记录列表
            self.records[operation].append(record)
            
            # 限制记录数量
            if len(self.records[operation]) > self.max_records:
                self.records[operation] = self.records[operation][-self.max_records:]
            
            # 更新指标
            self._update_metrics(operation)
    
    def _update_metrics(self, operation: str):
        """
        更新性能指标
        
        Args:
            operation: 操作名称
        """
        records = self.records[operation]
        if not records:
            return
        
        # 计算指标
        metrics = PerformanceMetrics()
        metrics.total_operations = len(records)
        metrics.successful_operations = sum(1 for r in records if r.success)
        metrics.failed_operations = metrics.total_operations - metrics.successful_operations
        
        durations = [r.duration for r in records]
        metrics.total_time = sum(durations)
        metrics.min_time = min(durations)
        metrics.max_time = max(durations)
        metrics.avg_time = statistics.mean(durations)
        
        # 计算吞吐量
        if metrics.total_time > 0:
            metrics.throughput = metrics.total_operations / metrics.total_time
        
        # 计算百分位数
        if len(durations) >= 2:
            sorted_durations = sorted(durations)
            metrics.p50 = statistics.median(sorted_durations)
            metrics.p90 = self._percentile(sorted_durations, 0.90)
            metrics.p95 = self._percentile(sorted_durations, 0.95)
            metrics.p99 = self._percentile(sorted_durations, 0.99)
        
        self.metrics[operation] = metrics
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """
        计算百分位数
        
        Args:
            sorted_data: 已排序的数据列表
            percentile: 百分位数（0.0-1.0）
            
        Returns:
            百分位数值
        """
        if not sorted_data:
            return 0.0
        
        n = len(sorted_data)
        index = percentile * (n - 1)
        lower = int(index)
        upper = lower + 1
        
        if upper >= n:
            return sorted_data[-1]
        
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def get_metrics(self, operation: str) -> PerformanceMetrics:
        """
        获取操作的性能指标
        
        Args:
            operation: 操作名称
            
        Returns:
            性能指标，如果不存在则返回空指标
        """
        with self._lock:
            if operation in self.metrics:
                return self.metrics[operation]
            # 返回空的性能指标而不是None
            return PerformanceMetrics()
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """
        获取所有操作的性能指标
        
        Returns:
            操作名称到性能指标的映射
        """
        with self._lock:
            return dict(self.metrics)
    
    def get_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            性能摘要字典
        """
        with self._lock:
            summary = {}
            for operation, metrics in self.metrics.items():
                summary[operation] = {
                    'total_operations': metrics.total_operations,
                    'success_rate': (
                        metrics.successful_operations / metrics.total_operations
                        if metrics.total_operations > 0 else 0.0
                    ),
                    'avg_time_ms': metrics.avg_time * 1000,
                    'min_time_ms': metrics.min_time * 1000,
                    'max_time_ms': metrics.max_time * 1000,
                    'p50_ms': metrics.p50 * 1000,
                    'p90_ms': metrics.p90 * 1000,
                    'p95_ms': metrics.p95 * 1000,
                    'p99_ms': metrics.p99 * 1000,
                    'throughput_ops_per_sec': metrics.throughput,
                }
            return summary
    
    def clear(self, operation: Optional[str] = None):
        """
        清除记录
        
        Args:
            operation: 操作名称，None表示清除所有
        """
        with self._lock:
            if operation:
                self.records[operation].clear()
                if operation in self.metrics:
                    del self.metrics[operation]
            else:
                self.records.clear()
                self.metrics.clear()


class MemoryUsageTracker:
    """
    内存使用跟踪器
    
    跟踪内存映射的使用情况。
    """
    
    def __init__(self):
        """初始化内存使用跟踪器"""
        self.mappings: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def track_mapping(self, 
                     region_id: str,
                     size: int,
                     file_path: str,
                     numa_node: int = -1,
                     uses_huge_pages: bool = False):
        """
        跟踪映射
        
        Args:
            region_id: 区域ID
            size: 映射大小
            file_path: 文件路径
            numa_node: NUMA节点
            uses_huge_pages: 是否使用大页
        """
        with self._lock:
            self.mappings[region_id] = {
                'size': size,
                'file_path': file_path,
                'numa_node': numa_node,
                'uses_huge_pages': uses_huge_pages,
                'creation_time': time.time(),
                'access_count': 0,
                'last_access_time': time.time(),
            }
    
    def update_access(self, region_id: str):
        """
        更新访问记录
        
        Args:
            region_id: 区域ID
        """
        with self._lock:
            if region_id in self.mappings:
                self.mappings[region_id]['access_count'] += 1
                self.mappings[region_id]['last_access_time'] = time.time()
    
    def remove_mapping(self, region_id: str):
        """
        移除映射记录
        
        Args:
            region_id: 区域ID
        """
        with self._lock:
            if region_id in self.mappings:
                del self.mappings[region_id]
    
    def get_total_size(self) -> int:
        """
        获取总映射大小
        
        Returns:
            总映射大小（字节）
        """
        with self._lock:
            return sum(m['size'] for m in self.mappings.values())
    
    def get_numa_distribution(self) -> Dict[int, int]:
        """
        获取NUMA节点分布
        
        Returns:
            NUMA节点到映射大小的映射
        """
        with self._lock:
            distribution = defaultdict(int)
            for mapping in self.mappings.values():
                node = mapping['numa_node']
                distribution[node] += mapping['size']
            return dict(distribution)
    
    def get_huge_page_usage(self) -> Tuple[int, int]:
        """
        获取大页使用情况
        
        Returns:
            (大页映射数量, 大页映射总大小)
        """
        with self._lock:
            count = sum(1 for m in self.mappings.values() if m['uses_huge_pages'])
            size = sum(m['size'] for m in self.mappings.values() if m['uses_huge_pages'])
            return count, size
    
    def get_hot_mappings(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取热点映射
        
        Args:
            top_n: 返回前N个
            
        Returns:
            (region_id, access_count) 列表
        """
        with self._lock:
            sorted_mappings = sorted(
                self.mappings.items(),
                key=lambda x: x[1]['access_count'],
                reverse=True
            )
            return [(region_id, data['access_count']) 
                    for region_id, data in sorted_mappings[:top_n]]
    
    def get_summary(self) -> Dict:
        """
        获取使用摘要
        
        Returns:
            使用摘要字典
        """
        with self._lock:
            if not self.mappings:
                return {
                    'total_mappings': 0,
                    'total_size': 0,
                    'avg_size': 0,
                    'huge_page_mappings': 0,
                    'huge_page_size': 0,
                }
            
            total_size = sum(m['size'] for m in self.mappings.values())
            huge_page_count, huge_page_size = self.get_huge_page_usage()
            
            return {
                'total_mappings': len(self.mappings),
                'total_size': total_size,
                'avg_size': total_size / len(self.mappings),
                'huge_page_mappings': huge_page_count,
                'huge_page_size': huge_page_size,
                'numa_distribution': self.get_numa_distribution(),
            }


class AccessPatternAnalyzer:
    """
    访问模式分析器
    
    分析内存访问模式，识别顺序/随机访问。
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化访问模式分析器
        
        Args:
            window_size: 分析窗口大小
        """
        self.window_size = window_size
        self.access_history: List[Tuple[int, float]] = []  # (offset, timestamp)
        self._lock = threading.Lock()
    
    def record_access(self, offset: int):
        """
        记录访问
        
        Args:
            offset: 访问偏移
        """
        with self._lock:
            self.access_history.append((offset, time.time()))
            
            # 限制历史记录大小
            if len(self.access_history) > self.window_size:
                self.access_history = self.access_history[-self.window_size:]
    
    def analyze(self) -> Dict:
        """
        分析访问模式
        
        Returns:
            分析结果字典
        """
        with self._lock:
            if len(self.access_history) < 2:
                return {
                    'pattern': 'unknown',
                    'sequential_ratio': 0.0,
                    'random_ratio': 0.0,
                    'avg_stride': 0.0,
                }
            
            # 计算步长
            strides = []
            for i in range(1, len(self.access_history)):
                stride = abs(self.access_history[i][0] - self.access_history[i-1][0])
                strides.append(stride)
            
            # 分析步长分布
            if not strides:
                return {
                    'pattern': 'unknown',
                    'sequential_ratio': 0.0,
                    'random_ratio': 0.0,
                    'avg_stride': 0.0,
                }
            
            avg_stride = statistics.mean(strides)
            
            # 判断访问模式
            # 顺序访问：步长小且一致
            # 随机访问：步长大且不一致
            sequential_threshold = 4096 * 4  # 16KB
            
            sequential_count = sum(1 for s in strides if s <= sequential_threshold)
            sequential_ratio = sequential_count / len(strides)
            random_ratio = 1.0 - sequential_ratio
            
            # 确定主要模式
            if sequential_ratio >= 0.7:
                pattern = 'sequential'
            elif random_ratio >= 0.7:
                pattern = 'random'
            else:
                pattern = 'mixed'
            
            return {
                'pattern': pattern,
                'sequential_ratio': sequential_ratio,
                'random_ratio': random_ratio,
                'avg_stride': avg_stride,
                'stride_variance': statistics.variance(strides) if len(strides) >= 2 else 0.0,
            }
    
    def clear(self):
        """清除历史记录"""
        with self._lock:
            self.access_history.clear()


class Benchmark:
    """
    基准测试工具
    
    用于性能基准测试。
    """
    
    def __init__(self, name: str = "benchmark"):
        """
        初始化基准测试
        
        Args:
            name: 基准测试名称
        """
        self.name = name
        self.tracker = PerformanceTracker()
    
    def run(self, 
            func, 
            iterations: int = 100,
            warmup: int = 10,
            **kwargs) -> PerformanceMetrics:
        """
        运行基准测试
        
        Args:
            func: 要测试的函数
            iterations: 迭代次数
            warmup: 预热次数
            **kwargs: 传递给函数的参数
            
        Returns:
            性能指标
        """
        # 预热
        for _ in range(warmup):
            try:
                func(**kwargs)
            except Exception:
                pass
        
        # 正式测试
        for _ in range(iterations):
            timer = Timer()
            timer.start()
            
            try:
                func(**kwargs)
                duration = timer.stop()
                self.tracker.record(self.name, duration, success=True)
            except Exception as e:
                duration = timer.stop()
                self.tracker.record(self.name, duration, success=False, error=str(e))
        
        return self.tracker.get_metrics(self.name)
    
    def compare(self, 
                funcs: Dict[str, Callable],
                iterations: int = 100,
                warmup: int = 10,
                **kwargs) -> Dict[str, PerformanceMetrics]:
        """
        比较多个函数的性能
        
        Args:
            funcs: 函数名称到函数的映射
            iterations: 迭代次数
            warmup: 预热次数
            **kwargs: 传递给函数的参数
            
        Returns:
            函数名称到性能指标的映射
        """
        results = {}
        
        for name, func in funcs.items():
            benchmark = Benchmark(name)
            results[name] = benchmark.run(func, iterations, warmup, **kwargs)
        
        return results
