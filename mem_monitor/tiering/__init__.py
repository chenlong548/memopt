"""
mem_monitor 分层管理层

提供内存分层管理功能。
"""

from typing import List, Dict, Any

from .page_manager import (
    PageManager,
    PageState,
    PageHotness,
    PageTracker,
)
from .numa_aware import (
    NUMAAwareManager,
    NUMATopologyInfo,
    NUMABalancer,
    MigrationPlanner,
)

__all__ = [
    # 页面管理
    'PageManager',
    'PageState',
    'PageHotness',
    'PageTracker',
    # NUMA感知
    'NUMAAwareManager',
    'NUMATopologyInfo',
    'NUMABalancer',
    'MigrationPlanner',
]


class TieringManager:
    """
    分层管理器

    整合页面管理和NUMA感知功能。
    """

    def __init__(self, config):
        self._config = config
        self._page_manager = PageManager(config)
        self._numa_manager = NUMAAwareManager(config) if config.enable_numa_aware else None
        self._recommendations: List[Dict[str, Any]] = []

    def update_page_access(self, address: int, access_count: int):
        """更新页面访问"""
        self._page_manager.record_access(address, access_count)

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """获取分层建议"""
        recommendations = []

        # 获取热/冷页面建议
        page_recommendations = self._page_manager.get_tiering_recommendations()
        recommendations.extend(page_recommendations)

        # 获取NUMA平衡建议
        if self._numa_manager:
            numa_recommendations = self._numa_manager.get_balance_recommendations()
            recommendations.extend(numa_recommendations)

        self._recommendations = recommendations
        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'page_manager': self._page_manager.get_stats(),
        }
        if self._numa_manager:
            stats['numa_manager'] = self._numa_manager.get_stats()
        return stats
