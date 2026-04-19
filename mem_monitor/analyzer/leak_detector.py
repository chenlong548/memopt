"""
mem_monitor 内存泄漏检测模块

实现内存泄漏检测和分析。

基于多种检测策略：
- 增长趋势分析
- 孤立对象检测
- 引用链分析
"""

import time
import sys
import gc
import weakref
import logging
import random
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import math

# 配置模块日志记录器
logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

# 严重程度阈值常量
CRITICAL_GROWTH_RATE = 100  # 对象/秒
CRITICAL_SIZE_MB = 100      # MB
HIGH_GROWTH_RATE = 10       # 对象/秒
HIGH_SIZE_MB = 10           # MB
MEDIUM_GROWTH_RATE = 1      # 对象/秒
MEDIUM_SIZE_MB = 1          # MB

# 默认类型大小估计（字节）
DEFAULT_TYPE_SIZE = 64

# 采样相关常量
GC_SAMPLING_RATE = 0.1      # 采样率（10%的对象）
GC_SAMPLING_MIN_INTERVAL = 5.0  # 最小采样间隔（秒）
MAX_TYPES_TO_TRACK = 1000   # 最大追踪类型数

# 历史记录限制
MAX_HISTORY_SIZE = 1000
MAX_HISTORY_KEEP = 500


class LeakSeverity(Enum):
    """泄漏严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LeakCandidate:
    """
    泄漏候选

    表示一个可能的内存泄漏。
    """

    # 对象信息
    type_name: str                           # 类型名称
    estimated_count: int                     # 估计数量
    estimated_size: int                      # 估计大小

    # 泄漏指标
    growth_rate: float                       # 增长率 (对象/秒)
    growth_trend: str                        # 增长趋势: increasing, stable, decreasing

    # 严重程度
    severity: LeakSeverity                   # 严重程度
    confidence: float                        # 置信度 (0-1)

    # 来源追踪
    allocation_sites: List[str] = field(default_factory=list)  # 分配位置

    # 引用链
    reference_chain: List[str] = field(default_factory=list)   # 引用链

    # 建议
    recommendation: str = ""                 # 修复建议

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type_name': self.type_name,
            'estimated_count': self.estimated_count,
            'estimated_size': self.estimated_size,
            'growth_rate': self.growth_rate,
            'growth_trend': self.growth_trend,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'allocation_sites': self.allocation_sites[:5],
            'recommendation': self.recommendation,
        }


@dataclass
class LeakReport:
    """
    泄漏报告

    包含完整的泄漏检测结果。
    """

    timestamp: float                         # 报告时间
    duration: float                          # 检测持续时间

    # 检测到的泄漏
    leaks: List[LeakCandidate] = field(default_factory=list)

    # 汇总统计
    total_leaked_objects: int = 0            # 泄漏对象总数
    total_leaked_size: int = 0               # 泄漏内存总量

    # 按严重程度统计
    by_severity: Dict[str, int] = field(default_factory=dict)

    # 建议
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'leak_count': len(self.leaks),
            'total_leaked_objects': self.total_leaked_objects,
            'total_leaked_size': self.total_leaked_size,
            'by_severity': self.by_severity,
            'leaks': [l.to_dict() for l in self.leaks[:20]],
            'recommendations': self.recommendations,
        }


class GrowthTracker:
    """
    增长追踪器

    追踪对象数量的增长趋势。
    """

    def __init__(self, history_size: int = 100):
        """
        初始化增长追踪器

        Args:
            history_size: 历史记录大小
        """
        self._history_size = history_size
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._timestamps: deque = deque(maxlen=history_size)

    def record(self, type_counts: Dict[str, int], timestamp: float):
        """
        记录类型计数

        Args:
            type_counts: 类型到数量的映射
            timestamp: 时间戳
        """
        self._timestamps.append(timestamp)

        for type_name, count in type_counts.items():
            self._history[type_name].append(count)

    def get_growth_rate(self, type_name: str) -> Tuple[float, str]:
        """
        获取增长率

        Args:
            type_name: 类型名称

        Returns:
            Tuple[float, str]: (增长率, 趋势)
        """
        history = self._history.get(type_name)
        if not history or len(history) < 2:
            return 0.0, "unknown"

        history_list = list(history)
        timestamps = list(self._timestamps)

        if len(timestamps) < 2:
            return 0.0, "unknown"

        # 计算增长率
        time_delta = timestamps[-1] - timestamps[0]
        if time_delta <= 0:
            return 0.0, "unknown"

        count_delta = history_list[-1] - history_list[0]
        growth_rate = count_delta / time_delta

        # 判断趋势
        if len(history_list) >= 10:
            # 使用线性回归判断趋势
            trend = self._calculate_trend(history_list)
        else:
            if growth_rate > 0.1:
                trend = "increasing"
            elif growth_rate < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"

        return growth_rate, trend

    def _calculate_trend(self, values: List[int]) -> str:
        """计算趋势"""
        n = len(values)
        if n < 2:
            return "unknown"

        # 简单线性回归
        x = list(range(n))
        y = values

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"


class ReferenceAnalyzer:
    """
    引用分析器

    分析对象的引用关系，识别引用链。
    """

    def __init__(self):
        self._visited: Set[int] = set()

    def find_reference_chain(self, obj: Any, max_depth: int = 10) -> List[str]:
        """
        查找引用链

        Args:
            obj: 目标对象
            max_depth: 最大搜索深度

        Returns:
            List[str]: 引用链描述
        """
        self._visited.clear()
        chain = []
        self._dfs_referrers(obj, chain, max_depth, 0)
        return chain

    def _dfs_referrers(self,
                       obj: Any,
                       chain: List[str],
                       max_depth: int,
                       current_depth: int):
        """深度优先搜索引用者"""
        if current_depth >= max_depth:
            return

        obj_id = id(obj)
        if obj_id in self._visited:
            return

        self._visited.add(obj_id)

        try:
            referrers = gc.get_referrers(obj)
        except Exception as e:
            logger.debug(f"Failed to get referrers: {e}")
            return

        for ref in referrers[:5]:  # 限制搜索宽度
            ref_type = type(ref).__name__

            if ref_type == 'frame':
                # 栈帧引用
                try:
                    frame_info = f"frame:{ref.f_code.co_filename}:{ref.f_lineno}"
                    chain.append(frame_info)
                except Exception as e:
                    logger.debug(f"Failed to get frame info: {e}")
            elif ref_type == 'dict':
                # 字典引用
                try:
                    for key, value in ref.items():
                        if value is obj:
                            chain.append(f"dict[{repr(key)[:50]}]")
                            break
                except Exception as e:
                    logger.debug(f"Failed to iterate dict: {e}")
            elif ref_type in ('list', 'tuple'):
                # 列表/元组引用
                chain.append(f"{ref_type}[{len(ref)} items]")
            else:
                chain.append(ref_type)

            # 递归搜索
            self._dfs_referrers(ref, chain, max_depth, current_depth + 1)

            if len(chain) >= 10:
                return


class LeakDetector:
    """
    内存泄漏检测器

    检测Python程序中的内存泄漏。

    检测策略：
    1. 增长趋势分析：检测持续增长的对象类型
    2. 孤立对象检测：检测无法回收的对象
    3. 引用链分析：追踪对象的引用关系

    算法思路：
    1. 定期采样对象类型分布（使用采样策略减少开销）
    2. 使用线性回归分析增长趋势
    3. 识别增长率超过阈值的类型
    4. 分析引用链找出泄漏源
    """

    def __init__(self, config):
        """
        初始化泄漏检测器

        Args:
            config: 分析器配置
        """
        self._config = config

        # 增长追踪器
        self._growth_tracker = GrowthTracker()

        # 引用分析器
        self._ref_analyzer = ReferenceAnalyzer()

        # 检测参数
        self._growth_threshold = config.leak_growth_threshold
        self._min_age = config.leak_min_age
        self._detection_interval = config.leak_detection_interval

        # 类型大小估计
        self._type_sizes: Dict[str, int] = {}

        # 检测结果
        self._leaks: List[LeakCandidate] = []
        self._last_detection_time = 0

        # 分析结果
        self._results: List[Dict[str, Any]] = []

        # 历史数据
        self._type_history: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        
        # 采样控制 - 避免频繁调用gc.get_objects()
        self._last_gc_sample_time = 0
        self._gc_sample_interval = GC_SAMPLING_MIN_INTERVAL
        self._cached_type_counts: Optional[Dict[str, int]] = None

    def detect(self) -> LeakReport:
        """
        执行泄漏检测

        Returns:
            LeakReport: 泄漏报告
        """
        start_time = time.time()
        self._last_detection_time = start_time

        # 获取当前对象统计
        type_counts = self._get_type_counts()
        self._growth_tracker.record(type_counts, start_time)

        # 保存历史
        for type_name, count in type_counts.items():
            self._type_history[type_name].append((start_time, count))
            # 限制历史大小
            if len(self._type_history[type_name]) > MAX_HISTORY_SIZE:
                self._type_history[type_name] = self._type_history[type_name][-MAX_HISTORY_KEEP:]

        # 检测泄漏
        leaks = self._detect_leaks(type_counts)

        # 生成报告
        report = LeakReport(
            timestamp=start_time,
            duration=time.time() - start_time,
            leaks=leaks,
        )

        # 计算汇总
        report.total_leaked_objects = sum(l.estimated_count for l in leaks)
        report.total_leaked_size = sum(l.estimated_size for l in leaks)

        report.by_severity = defaultdict(int)
        for leak in leaks:
            report.by_severity[leak.severity.value] += 1

        # 生成建议
        report.recommendations = self._generate_recommendations(leaks)

        self._leaks = leaks

        return report

    def _get_type_counts(self) -> Dict[str, int]:
        """
        获取类型计数
        
        使用采样策略减少gc.get_objects()的开销：
        1. 限制采样频率
        2. 使用概率采样减少遍历对象数
        3. 缓存结果避免重复计算
        
        Returns:
            Dict[str, int]: 类型到数量的映射
        """
        current_time = time.time()
        
        # 检查是否需要重新采样
        if (self._cached_type_counts is not None and 
            current_time - self._last_gc_sample_time < self._gc_sample_interval):
            return self._cached_type_counts
        
        self._last_gc_sample_time = current_time
        type_counts: Dict[str, int] = defaultdict(int)
        
        try:
            # 获取所有对象
            all_objects = gc.get_objects()
            total_objects = len(all_objects)
            
            # 如果对象数量过大，使用采样策略
            if total_objects > 100000:
                # 采样策略：只遍历部分对象
                sample_size = max(10000, int(total_objects * GC_SAMPLING_RATE))
                step = total_objects // sample_size
                
                for i in range(0, total_objects, step):
                    try:
                        obj = all_objects[i]
                        type_name = type(obj).__name__
                        type_counts[type_name] += step  # 乘以步长估算总数
                        
                        # 估计类型大小
                        if type_name not in self._type_sizes:
                            try:
                                self._type_sizes[type_name] = sys.getsizeof(obj)
                            except Exception:
                                self._type_sizes[type_name] = DEFAULT_TYPE_SIZE
                    except Exception as e:
                        logger.debug(f"Error sampling object at index {i}: {e}")
                        continue
            else:
                # 对象数量较少时，遍历所有对象
                for obj in all_objects:
                    try:
                        type_name = type(obj).__name__
                        type_counts[type_name] += 1

                        # 估计类型大小
                        if type_name not in self._type_sizes:
                            try:
                                self._type_sizes[type_name] = sys.getsizeof(obj)
                            except Exception:
                                self._type_sizes[type_name] = DEFAULT_TYPE_SIZE
                    except Exception as e:
                        logger.debug(f"Error processing object: {e}")
                        continue
                        
            # 限制追踪的类型数量
            if len(type_counts) > MAX_TYPES_TO_TRACK:
                # 只保留数量最多的类型
                sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                type_counts = dict(sorted_types[:MAX_TYPES_TO_TRACK])
                
        except Exception as e:
            logger.error(f"Failed to get type counts: {e}", exc_info=True)
            return self._cached_type_counts or {}

        self._cached_type_counts = dict(type_counts)
        return self._cached_type_counts

    def _detect_leaks(self, type_counts: Dict[str, int]) -> List[LeakCandidate]:
        """检测泄漏"""
        leaks = []

        for type_name, count in type_counts.items():
            # 获取增长率和趋势
            growth_rate, trend = self._growth_tracker.get_growth_rate(type_name)

            # 只关注增长中的类型
            if trend != "increasing":
                continue

            # 检查增长率是否超过阈值
            if abs(growth_rate) < self._growth_threshold:
                continue

            # 估计大小
            estimated_size = count * self._type_sizes.get(type_name, DEFAULT_TYPE_SIZE)

            # 计算严重程度
            severity = self._calculate_severity(growth_rate, estimated_size)

            # 计算置信度
            confidence = self._calculate_confidence(type_name)

            # 创建泄漏候选
            leak = LeakCandidate(
                type_name=type_name,
                estimated_count=count,
                estimated_size=estimated_size,
                growth_rate=growth_rate,
                growth_trend=trend,
                severity=severity,
                confidence=confidence,
                recommendation=self._get_recommendation(type_name, trend),
            )

            leaks.append(leak)

        # 按严重程度排序
        severity_order = {
            LeakSeverity.CRITICAL: 0,
            LeakSeverity.HIGH: 1,
            LeakSeverity.MEDIUM: 2,
            LeakSeverity.LOW: 3,
        }
        leaks.sort(key=lambda l: (severity_order[l.severity], -l.estimated_size))

        return leaks

    def _calculate_severity(self, growth_rate: float, estimated_size: int) -> LeakSeverity:
        """
        计算严重程度
        
        Args:
            growth_rate: 增长率（对象/秒）
            estimated_size: 估计大小（字节）
            
        Returns:
            LeakSeverity: 严重程度
        """
        # 转换为MB
        size_mb = estimated_size / (1024 * 1024)
        
        # 基于增长率和大小判断
        if growth_rate > CRITICAL_GROWTH_RATE or size_mb > CRITICAL_SIZE_MB:
            return LeakSeverity.CRITICAL
        elif growth_rate > HIGH_GROWTH_RATE or size_mb > HIGH_SIZE_MB:
            return LeakSeverity.HIGH
        elif growth_rate > MEDIUM_GROWTH_RATE or size_mb > MEDIUM_SIZE_MB:
            return LeakSeverity.MEDIUM
        else:
            return LeakSeverity.LOW

    def _calculate_confidence(self, type_name: str) -> float:
        """计算置信度"""
        history = self._type_history.get(type_name, [])

        if len(history) < 10:
            return 0.3

        # 计算增长的稳定性
        values = [count for _, count in history[-10:]]
        if len(values) < 2:
            return 0.3

        # 检查是否持续增长
        increasing_count = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        confidence = increasing_count / (len(values) - 1)

        return confidence

    def _get_recommendation(self, type_name: str, trend: str) -> str:
        """获取修复建议"""
        recommendations = {
            'function': "Check for closures or decorators that may hold references",
            'dict': "Check for cached data that is not being cleared",
            'list': "Check for accumulating lists that should be cleared",
            'set': "Check for accumulating sets that should be cleared",
            'tuple': "Check for accumulating tuples, may be stored in containers",
            'module': "Check for module-level caches or singletons",
            'type': "Check for class-level caches or metaclass issues",
            'code': "Check for dynamically generated code",
            'cell': "Check for closures holding references",
        }

        return recommendations.get(type_name, "Investigate object lifecycle and references")

    def _generate_recommendations(self, leaks: List[LeakCandidate]) -> List[str]:
        """生成总体建议"""
        recommendations = []

        if not leaks:
            recommendations.append("No memory leaks detected")
            return recommendations

        # 按严重程度分组
        critical = [l for l in leaks if l.severity == LeakSeverity.CRITICAL]
        high = [l for l in leaks if l.severity == LeakSeverity.HIGH]

        if critical:
            recommendations.append(
                f"Critical leaks detected in: {', '.join(l.type_name for l in critical[:3])}. "
                "Immediate attention required."
            )

        if high:
            recommendations.append(
                f"High severity leaks detected in: {', '.join(l.type_name for l in high[:3])}. "
                "Review and fix recommended."
            )

        recommendations.append(
            "Use gc.collect() to force garbage collection and verify if objects are freed."
        )

        recommendations.append(
            "Check for circular references using gc.get_referrers() on suspected objects."
        )

        return recommendations

    def analyze(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        分析内存快照

        Args:
            snapshots: 内存快照列表

        Returns:
            Dict: 分析结果
        """
        # 执行泄漏检测
        report = self.detect()

        result = {
            'timestamp': report.timestamp,
            'leak_count': len(report.leaks),
            'total_leaked_size': report.total_leaked_size,
            'by_severity': report.by_severity,
            'top_leaks': [l.to_dict() for l in report.leaks[:10]],
            'recommendations': report.recommendations,
        }

        self._results.append(result)
        return result

    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return {
            'leak_count': len(self._leaks),
            'total_leaked_size': sum(l.estimated_size for l in self._leaks),
            'leak_growth_rate': sum(l.growth_rate for l in self._leaks if l.growth_rate > 0),
        }

    def get_results(self) -> List[Dict[str, Any]]:
        """获取分析结果"""
        return self._results.copy()

    def get_leaks(self) -> List[LeakCandidate]:
        """获取检测到的泄漏"""
        return self._leaks.copy()

    def find_references(self, obj: Any) -> List[str]:
        """查找对象的引用链"""
        return self._ref_analyzer.find_reference_chain(obj)
