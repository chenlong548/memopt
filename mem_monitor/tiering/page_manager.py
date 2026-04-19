"""
mem_monitor 页面管理模块

实现页面热度评估和分层建议。

基于FlexMem和NOMAD论文的自适应页面分析和迁移策略。
"""

import time
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# ============================================================================
# 常量定义
# ============================================================================

# 页面大小（字节）
DEFAULT_PAGE_SIZE = 4096

# 热度评分阈值
HOT_PAGE_THRESHOLD = 0.7
VERY_HOT_THRESHOLD = 0.9
WARM_PAGE_THRESHOLD = 0.4
COLD_PAGE_THRESHOLD = 0.1

# 时间衰减半衰期（秒）
TIME_DECAY_HALF_LIFE = 300.0  # 5分钟

# 访问频率归一化因子
ACCESS_RATE_NORMALIZATION = 0.01

# 写操作权重增加因子
WRITE_WEIGHT_INCREASE = 0.5


class PageState(Enum):
    """页面状态"""
    HOT = "hot"           # 热页面
    WARM = "warm"         # 温页面
    COLD = "cold"         # 冷页面
    IDLE = "idle"         # 空闲页面
    UNKNOWN = "unknown"   # 未知状态


class PageHotness(Enum):
    """页面热度级别"""
    EXTREMELY_HOT = 5     # 极热
    VERY_HOT = 4          # 很热
    HOT = 3               # 热
    WARM = 2              # 温
    COOL = 1              # 凉
    COLD = 0              # 冷


@dataclass
class PageInfo:
    """
    页面信息

    记录单个页面的详细信息。
    """

    # 基本信息
    page_number: int                         # 页号
    virtual_address: int                     # 虚拟地址
    physical_address: int = 0                # 物理地址

    # 状态
    state: PageState = PageState.UNKNOWN
    hotness: PageHotness = PageHotness.COLD

    # 访问统计
    access_count: int = 0                    # 访问次数
    read_count: int = 0                      # 读次数
    write_count: int = 0                     # 写次数

    # 时间信息
    first_access_time: float = 0             # 首次访问时间
    last_access_time: float = 0              # 最后访问时间
    creation_time: float = field(default_factory=time.time)

    # 热度评分
    hotness_score: float = 0.0               # 热度评分 (0-1)
    access_rate: float = 0.0                 # 访问频率

    # 内存层
    current_tier: str = "unknown"            # 当前层级 (DRAM, PMEM, etc.)
    recommended_tier: str = ""               # 推荐层级

    # NUMA节点
    numa_node: int = -1                      # NUMA节点

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self, is_write: bool = False):
        """更新访问"""
        self.access_count += 1
        if is_write:
            self.write_count += 1
        else:
            self.read_count += 1

        now = time.time()
        if self.first_access_time == 0:
            self.first_access_time = now
        self.last_access_time = now

    def calculate_hotness(self, current_time: float) -> float:
        """
        计算热度评分

        使用多因素加权模型：
        - 访问频率
        - 时间衰减
        - 读/写比例

        Args:
            current_time: 当前时间

        Returns:
            float: 热度评分 (0-1)
        """
        if self.access_count == 0:
            self.hotness_score = 0.0
            self.state = PageState.IDLE
            return 0.0

        # 计算访问频率
        duration = current_time - self.creation_time
        if duration > 0:
            self.access_rate = self.access_count / duration
        else:
            self.access_rate = self.access_count

        # 时间衰减因子
        time_since_last = current_time - self.last_access_time
        time_decay = math.exp(-time_since_last / TIME_DECAY_HALF_LIFE)

        # 读写权重（写操作更重要）
        write_weight = 1.0 + (self.write_count / max(self.access_count, 1)) * WRITE_WEIGHT_INCREASE

        # 综合评分
        self.hotness_score = min(1.0, (
            self.access_rate * ACCESS_RATE_NORMALIZATION * time_decay * write_weight
        ))

        # 更新状态
        self._update_state()

        return self.hotness_score

    def _update_state(self):
        """更新页面状态"""
        if self.hotness_score >= HOT_PAGE_THRESHOLD:
            self.state = PageState.HOT
            self.hotness = PageHotness.EXTREMELY_HOT if self.hotness_score >= VERY_HOT_THRESHOLD else PageHotness.VERY_HOT
        elif self.hotness_score >= WARM_PAGE_THRESHOLD:
            self.state = PageState.WARM
            self.hotness = PageHotness.HOT if self.hotness_score >= 0.5 else PageHotness.WARM
        elif self.hotness_score >= COLD_PAGE_THRESHOLD:
            self.state = PageState.COLD
            self.hotness = PageHotness.COOL
        else:
            self.state = PageState.IDLE
            self.hotness = PageHotness.COLD

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'page_number': self.page_number,
            'state': self.state.value,
            'hotness': self.hotness.value,
            'access_count': self.access_count,
            'hotness_score': self.hotness_score,
            'access_rate': self.access_rate,
            'current_tier': self.current_tier,
            'recommended_tier': self.recommended_tier,
            'numa_node': self.numa_node,
        }


class PageTracker:
    """
    页面追踪器

    追踪页面访问模式，使用滑动窗口算法。

    算法思路：
    1. 维护多个时间窗口的访问计数
    2. 使用指数加权移动平均计算热度
    3. 自适应调整窗口大小
    """

    def __init__(self, window_size: int = 1000, num_windows: int = 10):
        """
        初始化页面追踪器

        Args:
            window_size: 每个窗口的访问事件数
            num_windows: 窗口数量
        """
        self._window_size = window_size
        self._num_windows = num_windows

        # 滑动窗口
        self._windows: List[Dict[int, int]] = [
            defaultdict(int) for _ in range(num_windows)
        ]
        self._current_window_idx = 0
        self._current_count = 0

        # 页面信息缓存
        self._pages: Dict[int, PageInfo] = {}

        # 页面大小
        self._page_size = DEFAULT_PAGE_SIZE

        # 统计
        self._total_accesses = 0
        self._total_pages_tracked = 0

    def record_access(self, address: int, count: int = 1, is_write: bool = False):
        """
        记录访问

        Args:
            address: 内存地址
            count: 访问次数
            is_write: 是否为写操作
        """
        # 计算页号
        page_number = address // self._page_size

        # 更新当前窗口
        self._windows[self._current_window_idx][page_number] += count
        self._current_count += count
        self._total_accesses += count

        # 更新页面信息
        if page_number not in self._pages:
            self._pages[page_number] = PageInfo(
                page_number=page_number,
                virtual_address=page_number * self._page_size
            )
            self._total_pages_tracked += 1

        self._pages[page_number].update_access(is_write)

        # 检查是否需要旋转窗口
        if self._current_count >= self._window_size:
            self._rotate_window()

    def _rotate_window(self):
        """旋转窗口"""
        self._current_window_idx = (self._current_window_idx + 1) % self._num_windows
        self._windows[self._current_window_idx].clear()
        self._current_count = 0

    def get_page_hotness(self, page_number: int) -> float:
        """
        获取页面热度

        Args:
            page_number: 页号

        Returns:
            float: 热度评分 (0-1)
        """
        # 计算所有窗口的加权平均
        total_weight = 0
        weighted_sum = 0

        for i, window in enumerate(self._windows):
            count = window.get(page_number, 0)
            # 近期窗口权重更高
            weight = 2 ** (self._num_windows - i - 1)
            weighted_sum += count * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # 归一化
        normalized = weighted_sum / total_weight / self._window_size
        return min(1.0, normalized)

    def get_hot_pages(self, threshold: float = 0.5, limit: int = 100) -> List[PageInfo]:
        """
        获取热页面

        Args:
            threshold: 热度阈值
            limit: 返回数量限制

        Returns:
            List[PageInfo]: 热页面列表
        """
        current_time = time.time()
        hot_pages = []

        for page in self._pages.values():
            hotness = page.calculate_hotness(current_time)
            if hotness >= threshold:
                hot_pages.append(page)

        return sorted(hot_pages, key=lambda p: p.hotness_score, reverse=True)[:limit]

    def get_cold_pages(self, threshold: float = 0.1, limit: int = 100) -> List[PageInfo]:
        """
        获取冷页面

        Args:
            threshold: 热度阈值
            limit: 返回数量限制

        Returns:
            List[PageInfo]: 冷页面列表
        """
        current_time = time.time()
        cold_pages = []

        for page in self._pages.values():
            hotness = page.calculate_hotness(current_time)
            if hotness <= threshold:
                cold_pages.append(page)

        return sorted(cold_pages, key=lambda p: p.hotness_score)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_accesses': self._total_accesses,
            'pages_tracked': self._total_pages_tracked,
            'current_window': self._current_window_idx,
            'window_size': self._window_size,
        }


class PageManager:
    """
    页面管理器

    管理页面热度评估和分层建议。

    基于FlexMem论文：
    - 自适应页面热度评估
    - 基于访问模式的分层决策
    - 低开销追踪

    基于NOMAD论文：
    - 非独占分层策略
    - 事务性迁移
    """

    def __init__(self, config):
        """
        初始化页面管理器

        Args:
            config: 分层配置
        """
        self._config = config

        # 页面追踪器
        self._tracker = PageTracker()

        # 热度阈值
        self._hot_threshold = config.hot_page_threshold
        self._cold_threshold = config.cold_page_threshold

        # 扫描间隔
        self._scan_interval = config.page_scan_interval
        self._last_scan_time = 0

        # 内存层级
        self._tiers = ['DRAM', 'PMEM', 'SSD']  # 从快到慢

        # 迁移配置
        self._enable_migration = config.enable_migration
        self._migration_batch_size = config.migration_batch_size

        # 建议
        self._recommendations: List[Dict[str, Any]] = []

    def record_access(self, address: int, count: int = 1):
        """
        记录页面访问

        Args:
            address: 内存地址
            count: 访问次数
        """
        self._tracker.record_access(address, count)

    def scan_pages(self) -> Dict[str, Any]:
        """
        扫描页面

        Returns:
            Dict: 扫描结果
        """
        current_time = time.time()
        self._last_scan_time = current_time

        # 获取热页面
        hot_pages = self._tracker.get_hot_pages(self._hot_threshold)

        # 获取冷页面
        cold_pages = self._tracker.get_cold_pages(self._cold_threshold)

        # 生成分层建议
        recommendations = []

        for page in hot_pages:
            if page.current_tier != 'DRAM':
                recommendations.append({
                    'type': 'promote',
                    'page_number': page.page_number,
                    'from_tier': page.current_tier,
                    'to_tier': 'DRAM',
                    'reason': f'Hot page (score: {page.hotness_score:.2f})',
                    'priority': int(page.hotness_score * 10),
                })

        for page in cold_pages:
            if page.current_tier == 'DRAM':
                recommendations.append({
                    'type': 'demote',
                    'page_number': page.page_number,
                    'from_tier': 'DRAM',
                    'to_tier': 'PMEM',
                    'reason': f'Cold page (score: {page.hotness_score:.2f})',
                    'priority': int((1 - page.hotness_score) * 5),
                })

        self._recommendations = recommendations

        return {
            'timestamp': current_time,
            'hot_page_count': len(hot_pages),
            'cold_page_count': len(cold_pages),
            'recommendation_count': len(recommendations),
        }

    def get_tiering_recommendations(self) -> List[Dict[str, Any]]:
        """获取分层建议"""
        # 如果距离上次扫描太久，重新扫描
        if time.time() - self._last_scan_time > self._scan_interval:
            self.scan_pages()

        return self._recommendations

    def get_page_info(self, address: int) -> Optional[PageInfo]:
        """获取页面信息"""
        page_number = address // DEFAULT_PAGE_SIZE
        return self._tracker._pages.get(page_number)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        tracker_stats = self._tracker.get_stats()

        # 统计各状态页面数量
        state_counts = defaultdict(int)
        for page in self._tracker._pages.values():
            state_counts[page.state.value] += 1

        return {
            'tracker': tracker_stats,
            'state_distribution': dict(state_counts),
            'recommendation_count': len(self._recommendations),
        }

    def clear(self):
        """清除数据"""
        self._tracker = PageTracker()
        self._recommendations.clear()
