"""
mem_monitor 生命周期分析模块

实现对象生命周期追踪和分析。

基于GainSight论文的细粒度内存访问模式分析和数据生命周期追踪技术。
"""

import time
import weakref
import threading
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib

# 配置模块日志记录器
logger = logging.getLogger(__name__)


class ObjectState(Enum):
    """对象状态"""
    ALLOCATED = "allocated"     # 已分配
    ACCESSED = "accessed"       # 已访问
    IDLE = "idle"               # 空闲
    DEAD = "dead"               # 已死亡（等待回收）


@dataclass
class AllocationRecord:
    """
    分配记录

    记录一次内存分配的详细信息。
    """

    # 基本信息
    allocation_id: int                      # 分配ID
    timestamp: float                        # 分配时间
    size: int                               # 分配大小
    address: int = 0                        # 内存地址（如果可用）

    # 类型信息
    type_name: str = ""                     # 类型名称
    type_id: int = 0                        # 类型ID

    # 调用栈
    call_stack: List[str] = field(default_factory=list)  # 调用栈
    filename: str = ""                      # 文件名
    lineno: int = 0                         # 行号

    # 状态追踪
    state: ObjectState = ObjectState.ALLOCATED
    access_count: int = 0                   # 访问次数
    last_access_time: float = 0             # 最后访问时间

    # 生命周期
    deallocation_time: Optional[float] = None  # 释放时间
    lifetime: float = 0                     # 存活时间

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_idle_time(self) -> float:
        """获取空闲时间"""
        if self.last_access_time > 0:
            return time.time() - self.last_access_time
        return time.time() - self.timestamp

    def get_lifetime(self) -> float:
        """获取生命周期"""
        if self.deallocation_time:
            return self.deallocation_time - self.timestamp
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'allocation_id': self.allocation_id,
            'timestamp': self.timestamp,
            'size': self.size,
            'type_name': self.type_name,
            'filename': self.filename,
            'lineno': self.lineno,
            'state': self.state.value,
            'access_count': self.access_count,
            'lifetime': self.get_lifetime(),
            'idle_time': self.get_idle_time(),
        }


@dataclass
class TypeStatistics:
    """
    类型统计

    统计特定类型的内存使用情况。
    """

    type_name: str                          # 类型名称
    instance_count: int = 0                 # 实例数量
    total_size: int = 0                     # 总大小
    avg_size: float = 0.0                   # 平均大小
    max_size: int = 0                       # 最大大小
    min_size: int = float('inf')            # 最小大小

    # 生命周期统计
    avg_lifetime: float = 0.0               # 平均生命周期
    max_lifetime: float = 0.0               # 最大生命周期

    # 访问统计
    total_access_count: int = 0             # 总访问次数
    avg_access_count: float = 0.0           # 平均访问次数

    def update(self, record: AllocationRecord):
        """更新统计"""
        self.instance_count += 1
        self.total_size += record.size
        self.avg_size = self.total_size / self.instance_count
        self.max_size = max(self.max_size, record.size)
        self.min_size = min(self.min_size, record.size)

        lifetime = record.get_lifetime()
        self.avg_lifetime = (
            (self.avg_lifetime * (self.instance_count - 1) + lifetime) /
            self.instance_count
        )
        self.max_lifetime = max(self.max_lifetime, lifetime)

        self.total_access_count += record.access_count
        self.avg_access_count = self.total_access_count / self.instance_count


class ObjectTracker:
    """
    对象追踪器

    追踪Python对象的分配和释放。

    算法思路：
    1. 使用弱引用追踪对象生命周期
    2. 记录分配时的调用栈
    3. 统计各类型对象的内存使用
    4. 识别长生命周期对象

    实现要点：
    - 使用weakref.WeakSet避免阻止对象回收
    - 采样策略减少开销
    - 调用栈哈希去重
    """

    def __init__(self, max_objects: int = 100000, sample_rate: float = 1.0):
        """
        初始化对象追踪器

        Args:
            max_objects: 最大追踪对象数
            sample_rate: 采样率 (0.0-1.0)
        """
        self._max_objects = max_objects
        self._sample_rate = sample_rate

        # 分配记录
        self._allocations: Dict[int, AllocationRecord] = {}
        self._allocation_counter = 0

        # 类型统计
        self._type_stats: Dict[str, TypeStatistics] = defaultdict(
            lambda: TypeStatistics(type_name="")
        )

        # 弱引用集合
        self._tracked: Set[weakref.ref] = set()

        # 线程安全
        self._lock = threading.Lock()

        # 统计
        self._total_tracked = 0
        self._total_released = 0

    def track_allocation(self,
                        obj: Any,
                        size: int,
                        call_stack: Optional[List[str]] = None) -> Optional[int]:
        """
        追踪对象分配

        Args:
            obj: 分配的对象
            size: 对象大小
            call_stack: 调用栈

        Returns:
            Optional[int]: 分配ID，如果采样未命中则返回None
        """
        # 采样检查
        import random
        if random.random() > self._sample_rate:
            return None

        with self._lock:
            # 检查容量
            if len(self._allocations) >= self._max_objects:
                # 清理最老的记录
                self._cleanup_old_records()

            # 创建分配记录
            self._allocation_counter += 1
            allocation_id = self._allocation_counter

            # 获取类型信息
            type_name = type(obj).__name__

            # 获取调用栈
            if call_stack is None:
                call_stack = self._get_call_stack()

            record = AllocationRecord(
                allocation_id=allocation_id,
                timestamp=time.time(),
                size=size,
                type_name=type_name,
                call_stack=call_stack,
            )

            # 从调用栈提取文件名和行号
            if call_stack:
                first_frame = call_stack[0] if call_stack else ""
                if ':' in first_frame:
                    parts = first_frame.split(':')
                    record.filename = parts[0]
                    try:
                        record.lineno = int(parts[1])
                    except ValueError:
                        pass

            self._allocations[allocation_id] = record

            # 更新类型统计
            self._type_stats[type_name].type_name = type_name
            self._type_stats[type_name].update(record)

            # 创建弱引用
            try:
                ref = weakref.ref(obj, lambda r: self._on_object_released(allocation_id))
                self._tracked.add(ref)
            except TypeError:
                # 某些类型不支持弱引用
                logger.debug(f"Type '{type_name}' does not support weak references")
            except Exception as e:
                logger.warning(f"Failed to create weak reference: {e}")

            self._total_tracked += 1

            return allocation_id

    def track_access(self, allocation_id: int):
        """
        追踪对象访问

        Args:
            allocation_id: 分配ID
        """
        with self._lock:
            if allocation_id in self._allocations:
                record = self._allocations[allocation_id]
                record.access_count += 1
                record.last_access_time = time.time()
                record.state = ObjectState.ACCESSED

    def _on_object_released(self, allocation_id: int):
        """对象被释放的回调"""
        with self._lock:
            if allocation_id in self._allocations:
                record = self._allocations[allocation_id]
                record.state = ObjectState.DEAD
                record.deallocation_time = time.time()
                record.lifetime = record.get_lifetime()

                self._total_released += 1

    def _cleanup_old_records(self):
        """清理旧记录"""
        # 移除已释放的记录
        dead_ids = [
            aid for aid, record in self._allocations.items()
            if record.state == ObjectState.DEAD
        ]

        for aid in dead_ids[:len(dead_ids) // 2]:
            del self._allocations[aid]

    def _get_call_stack(self) -> List[str]:
        """获取调用栈"""
        import traceback

        stack = traceback.extract_stack()
        # 过滤掉追踪器自身的帧
        frames = [
            f"{frame.filename}:{frame.lineno} in {frame.name}"
            for frame in stack[:-2]  # 排除最后两帧
            if 'mem_monitor' not in frame.filename
        ]
        return frames

    def get_allocation(self, allocation_id: int) -> Optional[AllocationRecord]:
        """获取分配记录"""
        return self._allocations.get(allocation_id)

    def get_type_statistics(self, type_name: Optional[str] = None) -> Dict[str, TypeStatistics]:
        """获取类型统计"""
        if type_name:
            return {type_name: self._type_stats.get(type_name, TypeStatistics(type_name))}
        return dict(self._type_stats)

    def get_long_lived_objects(self,
                               min_lifetime: float = 60.0,
                               limit: int = 100) -> List[AllocationRecord]:
        """
        获取长生命周期对象

        Args:
            min_lifetime: 最小生命周期（秒）
            limit: 返回数量限制

        Returns:
            List[AllocationRecord]: 长生命周期对象列表
        """
        with self._lock:
            long_lived = [
                record for record in self._allocations.values()
                if record.get_lifetime() >= min_lifetime and
                   record.state != ObjectState.DEAD
            ]
            return sorted(long_lived, key=lambda r: r.get_lifetime(), reverse=True)[:limit]

    def get_idle_objects(self,
                        min_idle_time: float = 300.0,
                        limit: int = 100) -> List[AllocationRecord]:
        """
        获取空闲对象

        Args:
            min_idle_time: 最小空闲时间（秒）
            limit: 返回数量限制

        Returns:
            List[AllocationRecord]: 空闲对象列表
        """
        with self._lock:
            idle = [
                record for record in self._allocations.values()
                if record.get_idle_time() >= min_idle_time and
                   record.state != ObjectState.DEAD
            ]
            return sorted(idle, key=lambda r: r.get_idle_time(), reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'total_tracked': self._total_tracked,
                'total_released': self._total_released,
                'current_tracked': len(self._allocations),
                'type_count': len(self._type_stats),
                'sample_rate': self._sample_rate,
            }


class LifecycleAnalyzer:
    """
    生命周期分析器

    分析对象的生命周期模式，识别内存使用模式。

    算法思路：
    1. 追踪对象的分配和释放
    2. 分析生命周期分布
    3. 识别生命周期异常的对象
    4. 提供优化建议

    基于GainSight论文：
    - 细粒度生命周期追踪
    - 数据生命周期模式识别
    - 访问热度与生命周期的关联分析
    """

    def __init__(self, config):
        """
        初始化生命周期分析器

        Args:
            config: 分析器配置
        """
        self._config = config

        # 对象追踪器
        self._tracker = ObjectTracker(
            max_objects=config.lifecycle_max_objects,
            sample_rate=config.lifecycle_sample_rate
        )

        # 生命周期分布
        self._lifetime_distribution: Dict[str, int] = defaultdict(int)

        # 分析结果
        self._results: List[Dict[str, Any]] = []

        # 统计
        self._analysis_count = 0

    def track_allocation(self, obj: Any, size: int) -> Optional[int]:
        """
        追踪对象分配

        Args:
            obj: 分配的对象
            size: 对象大小

        Returns:
            Optional[int]: 分配ID
        """
        return self._tracker.track_allocation(obj, size)

    def track_access(self, allocation_id: int):
        """追踪对象访问"""
        self._tracker.track_access(allocation_id)

    def analyze(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        分析生命周期

        Args:
            snapshots: 内存快照列表

        Returns:
            Dict: 分析结果
        """
        self._analysis_count += 1

        # 更新生命周期分布
        self._update_lifetime_distribution()

        # 分析生命周期模式
        patterns = self._analyze_patterns()

        # 识别异常对象
        anomalies = self._identify_anomalies()

        # 生成建议
        recommendations = self._generate_recommendations()

        result = {
            'timestamp': time.time(),
            'patterns': patterns,
            'anomalies': anomalies,
            'recommendations': recommendations,
            'stats': self._tracker.get_stats(),
        }

        self._results.append(result)
        return result

    def _update_lifetime_distribution(self):
        """更新生命周期分布"""
        # 定义生命周期区间（秒）
        buckets = [
            ('transient', 0, 1),          # 瞬态：< 1秒
            ('short', 1, 10),             # 短期：1-10秒
            ('medium', 10, 60),           # 中期：10-60秒
            ('long', 60, 300),            # 长期：1-5分钟
            ('persistent', 300, float('inf'))  # 持久：> 5分钟
        ]

        # 重置分布
        self._lifetime_distribution = defaultdict(int)

        # 统计各区间
        for record in self._tracker._allocations.values():
            lifetime = record.get_lifetime()
            for name, low, high in buckets:
                if low <= lifetime < high:
                    self._lifetime_distribution[name] += 1
                    break

    def _analyze_patterns(self) -> Dict[str, Any]:
        """分析生命周期模式"""
        stats = self._tracker.get_stats()

        # 计算各类型平均生命周期
        type_lifetimes = {}
        for type_name, type_stat in self._tracker.get_type_statistics().items():
            type_lifetimes[type_name] = {
                'avg_lifetime': type_stat.avg_lifetime,
                'instance_count': type_stat.instance_count,
                'total_size': type_stat.total_size,
            }

        return {
            'distribution': dict(self._lifetime_distribution),
            'type_lifetimes': type_lifetimes,
            'total_objects': stats['current_tracked'],
        }

    def _identify_anomalies(self) -> List[Dict[str, Any]]:
        """识别异常对象"""
        anomalies = []

        # 获取长生命周期对象
        long_lived = self._tracker.get_long_lived_objects(min_lifetime=300.0, limit=50)

        for record in long_lived:
            # 检查是否异常
            is_anomaly = False
            reason = ""

            # 长生命周期但低访问
            if record.access_count < 5 and record.get_lifetime() > 300:
                is_anomaly = True
                reason = "Long-lived object with low access count"

            # 大对象
            if record.size > 1024 * 1024:  # > 1MB
                is_anomaly = True
                reason = "Large long-lived object"

            if is_anomaly:
                anomalies.append({
                    'allocation_id': record.allocation_id,
                    'type_name': record.type_name,
                    'size': record.size,
                    'lifetime': record.get_lifetime(),
                    'access_count': record.access_count,
                    'reason': reason,
                    'filename': record.filename,
                    'lineno': record.lineno,
                })

        return anomalies

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 分析分布
        total = sum(self._lifetime_distribution.values())
        if total > 0:
            persistent_ratio = self._lifetime_distribution.get('persistent', 0) / total

            if persistent_ratio > 0.5:
                recommendations.append(
                    "High ratio of persistent objects detected. "
                    "Consider using object pools or caching strategies."
                )

            transient_ratio = self._lifetime_distribution.get('transient', 0) / total
            if transient_ratio > 0.3:
                recommendations.append(
                    "High ratio of transient objects detected. "
                    "Consider reducing object creation overhead."
                )

        # 分析类型统计
        type_stats = self._tracker.get_type_statistics()
        for type_name, stats in type_stats.items():
            if stats.instance_count > 1000 and stats.avg_lifetime < 1:
                recommendations.append(
                    f"Type '{type_name}' has many short-lived instances. "
                    f"Consider using __slots__ or a custom allocator."
                )

        return recommendations

    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        stats = self._tracker.get_stats()
        return {
            'tracked_objects': stats['current_tracked'],
            'lifetime_distribution': dict(self._lifetime_distribution),
        }

    def get_results(self) -> List[Dict[str, Any]]:
        """获取分析结果"""
        return self._results.copy()

    def get_tracker(self) -> ObjectTracker:
        """获取对象追踪器"""
        return self._tracker
