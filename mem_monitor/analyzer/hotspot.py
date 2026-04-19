"""
mem_monitor 热点分析模块

实现内存访问热点检测和分析。

基于PROMPT框架的快速可扩展内存分析思路。
"""

import time
import math
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# ============================================================================
# 常量定义
# ============================================================================

# 时间衰减半衰期（秒）
TIME_DECAY_HALF_LIFE = 60.0

# 热度评分权重
ACCESS_RATE_WEIGHT = 0.4
TOTAL_ACCESS_WEIGHT = 0.3
TIME_DECAY_WEIGHT = 0.3

# 访问量归一化因子
ACCESS_NORMALIZATION_FACTOR = 1000

# 页面大小（字节）
DEFAULT_PAGE_SIZE = 4096

# 高热度访问频率阈值
HIGH_ACCESS_RATE_THRESHOLD = 10000

# 多线程访问阈值
HIGH_CONTENTION_THREAD_COUNT = 4

# 读写比例阈值
READ_HEAVY_RATIO = 10
WRITE_HEAVY_RATIO = 10


class AccessType(Enum):
    """访问类型"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


@dataclass
class AccessPattern:
    """
    访问模式

    描述一个内存区域的访问模式。
    """

    # 区域标识
    start_address: int                       # 起始地址
    size: int                                # 区域大小

    # 访问统计
    read_count: int = 0                      # 读次数
    write_count: int = 0                     # 写次数
    total_accesses: int = 0                  # 总访问次数

    # 时间统计
    first_access_time: float = 0             # 首次访问时间
    last_access_time: float = 0              # 最后访问时间

    # 访问频率
    access_rate: float = 0.0                 # 访问频率 (次/秒)

    # 访问者
    accessing_threads: Set[int] = field(default_factory=set)  # 访问线程

    # 热度评分
    hotness_score: float = 0.0               # 热度评分 (0-1)

    def update(self, access_type: AccessType, timestamp: float, thread_id: int = -1):
        """更新访问模式"""
        if access_type == AccessType.READ:
            self.read_count += 1
        elif access_type == AccessType.WRITE:
            self.write_count += 1

        self.total_accesses += 1

        if self.first_access_time == 0:
            self.first_access_time = timestamp
        self.last_access_time = timestamp

        if thread_id >= 0:
            self.accessing_threads.add(thread_id)

    def calculate_hotness(self, current_time: float, decay_factor: float = 0.95) -> float:
        """
        计算热度评分

        使用指数衰减模型计算热度。

        Args:
            current_time: 当前时间
            decay_factor: 衰减因子

        Returns:
            float: 热度评分 (0-1)
        """
        if self.total_accesses == 0:
            return 0.0

        # 计算时间衰减
        time_since_last = current_time - self.last_access_time
        time_decay = math.exp(-time_since_last / TIME_DECAY_HALF_LIFE)

        # 计算访问频率
        duration = self.last_access_time - self.first_access_time
        if duration > 0:
            self.access_rate = self.total_accesses / duration
        else:
            self.access_rate = self.total_accesses

        # 综合评分
        self.hotness_score = min(1.0, (
            self.access_rate * ACCESS_RATE_WEIGHT +           # 访问频率权重
            self.total_accesses / ACCESS_NORMALIZATION_FACTOR * TOTAL_ACCESS_WEIGHT +  # 总访问量权重
            time_decay * TIME_DECAY_WEIGHT                    # 时间衰减权重
        ))

        return self.hotness_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_address': hex(self.start_address),
            'size': self.size,
            'read_count': self.read_count,
            'write_count': self.write_count,
            'total_accesses': self.total_accesses,
            'access_rate': self.access_rate,
            'hotness_score': self.hotness_score,
            'thread_count': len(self.accessing_threads),
        }


@dataclass
class HotspotRegion:
    """
    热点区域

    表示一个被识别为热点的内存区域。
    """

    # 区域信息
    start_address: int                       # 起始地址
    end_address: int                         # 结束地址
    size: int                                # 区域大小

    # 热点信息
    hotness_score: float                     # 热度评分
    access_pattern: AccessPattern            # 访问模式

    # 分类
    region_type: str = "unknown"             # 区域类型 (heap, stack, mmap, etc.)

    # 建议
    recommendation: str = ""                 # 优化建议

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_address': hex(self.start_address),
            'end_address': hex(self.end_address),
            'size': self.size,
            'hotness_score': self.hotness_score,
            'access_pattern': self.access_pattern.to_dict(),
            'region_type': self.region_type,
            'recommendation': self.recommendation,
        }


class SlidingWindowCounter:
    """
    滑动窗口计数器

    使用滑动窗口算法统计访问频率。
    """

    def __init__(self, window_size: int = 1000, num_windows: int = 10):
        """
        初始化滑动窗口计数器

        Args:
            window_size: 每个窗口的大小
            num_windows: 窗口数量
        """
        self._window_size = window_size
        self._num_windows = num_windows
        self._windows: deque = deque(maxlen=num_windows)
        self._current_window: Dict[int, int] = defaultdict(int)
        self._current_count = 0

    def add(self, key: int, count: int = 1):
        """添加计数"""
        self._current_window[key] += count
        self._current_count += 1

        if self._current_count >= self._window_size:
            self._rotate_window()

    def _rotate_window(self):
        """旋转窗口"""
        self._windows.append(dict(self._current_window))
        self._current_window.clear()
        self._current_count = 0

    def get_count(self, key: int) -> int:
        """获取计数"""
        total = self._current_window.get(key, 0)
        for window in self._windows:
            total += window.get(key, 0)
        return total

    def get_top(self, n: int = 10) -> List[Tuple[int, int]]:
        """获取top N"""
        # 合并所有窗口
        merged: Dict[int, int] = defaultdict(int)
        for window in self._windows:
            for key, count in window.items():
                merged[key] += count
        for key, count in self._current_window.items():
            merged[key] += count

        # 排序
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:n]


class HotspotAnalyzer:
    """
    热点分析器

    分析内存访问热点，识别高频访问区域。

    算法思路：
    1. 使用滑动窗口统计访问频率
    2. 基于热度评分识别热点区域
    3. 分析热点区域的访问模式
    4. 提供分层优化建议

    基于PROMPT框架：
    - 插件式设计，可扩展
    - 低开销分析
    - 快速热点识别
    """

    def __init__(self, config):
        """
        初始化热点分析器

        Args:
            config: 分析器配置
        """
        self._config = config

        # 滑动窗口计数器
        self._window_counter = SlidingWindowCounter(
            window_size=config.hotspot_window_size,
            num_windows=10
        )

        # 访问模式追踪
        self._access_patterns: Dict[int, AccessPattern] = {}

        # 页面大小（用于地址分组）
        self._page_size = DEFAULT_PAGE_SIZE

        # 热点阈值
        self._hot_threshold = config.hotspot_threshold
        self._min_samples = config.hotspot_min_samples

        # 热点缓存
        self._hotspots: List[HotspotRegion] = []
        self._last_analysis_time = 0

        # 分析结果
        self._results: List[Dict[str, Any]] = []

    def record_access(self,
                     address: int,
                     size: int,
                     access_type: AccessType = AccessType.READ,
                     thread_id: int = -1):
        """
        记录访问

        Args:
            address: 内存地址
            size: 访问大小
            access_type: 访问类型
            thread_id: 线程ID
        """
        timestamp = time.time()

        # 按页对齐地址
        page_addr = address & ~(self._page_size - 1)

        # 更新滑动窗口计数器
        self._window_counter.add(page_addr)

        # 更新访问模式
        if page_addr not in self._access_patterns:
            self._access_patterns[page_addr] = AccessPattern(
                start_address=page_addr,
                size=self._page_size
            )

        self._access_patterns[page_addr].update(access_type, timestamp, thread_id)

    def analyze(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        分析热点

        Args:
            snapshots: 内存快照列表

        Returns:
            Dict: 分析结果
        """
        current_time = time.time()

        # 获取top热点页面
        top_pages = self._window_counter.get_top(100)

        # 计算热度评分
        hotspots = []
        for page_addr, count in top_pages:
            if count < self._min_samples:
                continue

            pattern = self._access_patterns.get(page_addr)
            if pattern:
                hotness = pattern.calculate_hotness(current_time)

                if hotness >= self._hot_threshold:
                    hotspot = HotspotRegion(
                        start_address=page_addr,
                        end_address=page_addr + self._page_size,
                        size=self._page_size,
                        hotness_score=hotness,
                        access_pattern=pattern,
                        region_type=self._classify_region(page_addr),
                        recommendation=self._generate_recommendation(pattern),
                    )
                    hotspots.append(hotspot)

        # 按热度排序
        hotspots.sort(key=lambda h: h.hotness_score, reverse=True)

        self._hotspots = hotspots
        self._last_analysis_time = current_time

        # 生成结果
        result = {
            'timestamp': current_time,
            'hotspot_count': len(hotspots),
            'top_hotspots': [h.to_dict() for h in hotspots[:20]],
            'access_pattern_summary': self._summarize_patterns(),
        }

        self._results.append(result)
        return result

    def _classify_region(self, address: int) -> str:
        """分类区域类型"""
        # 简化实现，实际需要查询进程内存映射
        # 通常：
        # - 低地址：代码段
        # - 中等地址：堆
        # - 高地址：栈或mmap

        if address < 0x10000000:
            return "code"
        elif address < 0x7fff00000000:
            return "heap"
        else:
            return "stack_mmap"

    def _generate_recommendation(self, pattern: AccessPattern) -> str:
        """生成优化建议"""
        recommendations = []

        # 高读频率
        if pattern.read_count > pattern.write_count * READ_HEAVY_RATIO:
            recommendations.append("Consider caching or read optimization")

        # 高写频率
        if pattern.write_count > pattern.read_count * WRITE_HEAVY_RATIO:
            recommendations.append("Consider write batching")

        # 多线程访问
        if len(pattern.accessing_threads) > HIGH_CONTENTION_THREAD_COUNT:
            recommendations.append("High contention detected, consider lock optimization")

        # 极高访问频率
        if pattern.access_rate > HIGH_ACCESS_RATE_THRESHOLD:
            recommendations.append("Very hot region, consider moving to faster memory tier")

        return "; ".join(recommendations) if recommendations else "No specific recommendation"

    def _summarize_patterns(self) -> Dict[str, Any]:
        """汇总访问模式"""
        total_reads = sum(p.read_count for p in self._access_patterns.values())
        total_writes = sum(p.write_count for p in self._access_patterns.values())
        total_accesses = sum(p.total_accesses for p in self._access_patterns.values())

        return {
            'total_pages_tracked': len(self._access_patterns),
            'total_reads': total_reads,
            'total_writes': total_writes,
            'total_accesses': total_accesses,
            'read_write_ratio': total_reads / total_writes if total_writes > 0 else 0,
        }

    def get_hotspots(self, threshold: Optional[float] = None) -> List[HotspotRegion]:
        """
        获取热点区域

        Args:
            threshold: 热度阈值，None则使用默认值

        Returns:
            List[HotspotRegion]: 热点区域列表
        """
        if threshold is None:
            return self._hotspots

        return [h for h in self._hotspots if h.hotness_score >= threshold]

    def get_access_pattern(self, address: int) -> Optional[AccessPattern]:
        """获取指定地址的访问模式"""
        page_addr = address & ~(self._page_size - 1)
        return self._access_patterns.get(page_addr)

    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return {
            'hotspot_count': len(self._hotspots),
            'tracked_pages': len(self._access_patterns),
            'top_hotspot_score': self._hotspots[0].hotness_score if self._hotspots else 0,
        }

    def get_results(self) -> List[Dict[str, Any]]:
        """获取分析结果"""
        return self._results.copy()

    def clear(self):
        """清除数据"""
        self._window_counter = SlidingWindowCounter(
            window_size=self._config.hotspot_window_size,
            num_windows=10
        )
        self._access_patterns.clear()
        self._hotspots.clear()
        self._results.clear()
