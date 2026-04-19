"""
统计信息收集器

收集和分析压缩操作的统计信息。
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
import time

from ..core.base import CompressionStats, CompressionAlgorithm

logger = logging.getLogger(__name__)


class StatisticsCollector:
    """
    统计信息收集器

    收集、分析和报告压缩操作的统计信息。
    """

    def __init__(self, max_history: int = 1000):
        """
        初始化统计收集器

        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history

        # 历史记录
        self._history: List[CompressionStats] = []

        # 算法统计
        self._algorithm_stats: Dict[CompressionAlgorithm, Dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'total_original_size': 0,
                'total_compressed_size': 0,
                'total_compression_time': 0.0,
                'total_decompression_time': 0.0,
                'avg_ratio': 0.0,
                'avg_time': 0.0,
                'best_ratio': 0.0,
                'worst_ratio': float('inf'),
            }
        )

        # 数据类型统计
        self._data_type_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'total_size': 0,
                'avg_compression_ratio': 0.0,
            }
        )

        # 时间序列数据
        self._time_series: List[Dict[str, Any]] = []

    def record(self, stats: CompressionStats):
        """
        记录压缩统计信息

        Args:
            stats: 压缩统计信息
        """
        # 添加到历史记录
        self._history.append(stats)

        # 限制历史记录大小
        if len(self._history) > self.max_history:
            self._history.pop(0)

        # 更新算法统计
        self._update_algorithm_stats(stats)

        # 更新数据类型统计
        self._update_data_type_stats(stats)

        # 添加时间序列数据
        self._time_series.append({
            'timestamp': time.time(),
            'algorithm': stats.algorithm_used.value,
            'original_size': stats.original_size,
            'compressed_size': stats.compressed_size,
            'ratio': stats.compression_ratio,
            'time': stats.compression_time,
        })

        # 限制时间序列大小
        if len(self._time_series) > self.max_history:
            self._time_series.pop(0)

    def _update_algorithm_stats(self, stats: CompressionStats):
        """更新算法统计"""
        alg = stats.algorithm_used
        alg_stats = self._algorithm_stats[alg]

        alg_stats['count'] += 1
        alg_stats['total_original_size'] += stats.original_size
        alg_stats['total_compressed_size'] += stats.compressed_size
        alg_stats['total_compression_time'] += stats.compression_time
        alg_stats['total_decompression_time'] += stats.decompression_time

        count = alg_stats['count']
        if alg_stats['total_compressed_size'] > 0:
            alg_stats['avg_ratio'] = alg_stats['total_original_size'] / alg_stats['total_compressed_size']
        elif stats.compression_ratio > 0:
            if count > 1:
                alg_stats['avg_ratio'] = (alg_stats['avg_ratio'] * (count - 1) + stats.compression_ratio) / count
            else:
                alg_stats['avg_ratio'] = stats.compression_ratio
        else:
            alg_stats['avg_ratio'] = 0.0
        alg_stats['avg_time'] = alg_stats['total_compression_time'] / count if count > 0 else 0.0

        if stats.compression_ratio > alg_stats['best_ratio']:
            alg_stats['best_ratio'] = stats.compression_ratio

        if stats.compression_ratio < alg_stats['worst_ratio']:
            alg_stats['worst_ratio'] = stats.compression_ratio

    def _update_data_type_stats(self, stats: CompressionStats):
        """更新数据类型统计"""
        # 从stats中获取数据类型（如果有）
        # 简化实现：使用'generic'
        data_type = 'generic'

        type_stats = self._data_type_stats[data_type]
        type_stats['count'] += 1
        type_stats['total_size'] += stats.original_size

        # 计算平均压缩比
        type_stats['avg_compression_ratio'] = (
            type_stats['avg_compression_ratio'] * (type_stats['count'] - 1) +
            stats.compression_ratio
        ) / type_stats['count']

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要

        Returns:
            Dict: 统计摘要信息
        """
        if not self._history:
            return {
                'total_operations': 0,
                'algorithms': {},
                'data_types': {},
            }

        # 总体统计
        total_original = sum(s.original_size for s in self._history)
        total_compressed = sum(s.compressed_size for s in self._history)
        total_time = sum(s.compression_time for s in self._history)

        summary = {
            'total_operations': len(self._history),
            'total_original_size_mb': total_original / (1024 * 1024),
            'total_compressed_size_mb': total_compressed / (1024 * 1024),
            'overall_ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'total_time_seconds': total_time,
            'avg_throughput_mbps': (total_original / (1024 * 1024)) / total_time if total_time > 0 else 0,
        }

        # 算法统计
        summary['algorithms'] = {
            alg.value: {
                'count': stats['count'],
                'avg_ratio': f"{stats['avg_ratio']:.2f}x",
                'avg_time_ms': f"{stats['avg_time'] * 1000:.2f}",
                'best_ratio': f"{stats['best_ratio']:.2f}x",
                'worst_ratio': f"{stats['worst_ratio']:.2f}x" if stats['worst_ratio'] != float('inf') else "N/A",
            }
            for alg, stats in self._algorithm_stats.items()
            if stats['count'] > 0
        }

        # 数据类型统计
        summary['data_types'] = {
            dtype: {
                'count': stats['count'],
                'total_size_mb': stats['total_size'] / (1024 * 1024),
                'avg_ratio': f"{stats['avg_compression_ratio']:.2f}x",
            }
            for dtype, stats in self._data_type_stats.items()
            if stats['count'] > 0
        }

        return summary

    def get_algorithm_ranking(self, metric: str = 'ratio') -> List[Dict[str, Any]]:
        """
        获取算法排名

        Args:
            metric: 排名指标 ('ratio' | 'speed')

        Returns:
            List: 算法排名列表
        """
        rankings = []

        for alg, stats in self._algorithm_stats.items():
            if stats['count'] == 0:
                continue

            if metric == 'ratio':
                score = stats['avg_ratio']
            elif metric == 'speed':
                score = 1.0 / stats['avg_time'] if stats['avg_time'] > 0 else 0
            else:
                score = stats['avg_ratio']

            rankings.append({
                'algorithm': alg.value,
                'score': score,
                'count': stats['count'],
            })

        # 排序
        rankings.sort(key=lambda x: x['score'], reverse=True)

        return rankings

    def get_time_series(self,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        获取时间序列数据

        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳

        Returns:
            List: 时间序列数据
        """
        if start_time is None and end_time is None:
            return self._time_series.copy()

        filtered = []
        for item in self._time_series:
            if start_time and item['timestamp'] < start_time:
                continue
            if end_time and item['timestamp'] > end_time:
                continue
            filtered.append(item)

        return filtered

    def get_recent_stats(self, n: int = 10) -> List[CompressionStats]:
        """
        获取最近的统计信息

        Args:
            n: 数量

        Returns:
            List: 最近的统计信息列表
        """
        return self._history[-n:] if self._history else []

    def clear(self):
        """清除所有统计信息"""
        self._history.clear()
        self._algorithm_stats.clear()
        self._data_type_stats.clear()
        self._time_series.clear()

    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出为字典格式

        Returns:
            Dict: 统计信息字典
        """
        return {
            'summary': self.get_summary(),
            'algorithm_ranking': self.get_algorithm_ranking(),
            'recent_stats': [
                {
                    'algorithm': s.algorithm_used.value,
                    'original_size': s.original_size,
                    'compressed_size': s.compressed_size,
                    'ratio': s.compression_ratio,
                    'time': s.compression_time,
                }
                for s in self.get_recent_stats(100)
            ],
        }
