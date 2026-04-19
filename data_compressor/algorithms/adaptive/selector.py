"""
自适应算法选择器

使用Bandit算法智能选择最优压缩算法。
"""

import logging
from typing import Dict, Any, Optional, List
import random
import math

from ...core.base import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionStats
)

logger = logging.getLogger(__name__)


class AdaptiveAlgorithmSelector:
    """
    自适应算法选择器

    使用UCB (Upper Confidence Bound) 算法进行在线学习，
    动态选择最优压缩算法。
    """

    def __init__(self, exploration_factor: float = 1.0):
        """
        初始化选择器

        Args:
            exploration_factor: 探索因子，控制探索与利用的平衡
        """
        self.exploration_factor = exploration_factor

        # 算法性能统计
        self._algorithm_stats: Dict[CompressionAlgorithm, Dict[str, Any]] = {
            algorithm: {
                'count': 0,           # 使用次数
                'total_ratio': 0.0,   # 累计压缩比
                'total_time': 0.0,    # 累计时间
                'avg_ratio': 0.0,     # 平均压缩比
                'avg_time': 0.0,      # 平均时间
                'success_rate': 0.0,  # 成功率
                'success_count': 0,   # 成功次数
            }
            for algorithm in [
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.BROTLI,
                CompressionAlgorithm.BF16_MODEL,
                CompressionAlgorithm.FP32_MODEL,
                CompressionAlgorithm.KV_CACHE,
                CompressionAlgorithm.LEXICO,
            ]
        }

        # 数据类型到算法的映射
        self._type_algorithm_map = {
            'model_weights': [
                CompressionAlgorithm.BF16_MODEL,
                CompressionAlgorithm.FP32_MODEL,
                CompressionAlgorithm.ZSTD,
            ],
            'kv_cache': [
                CompressionAlgorithm.KV_CACHE,
                CompressionAlgorithm.LEXICO,
                CompressionAlgorithm.ZSTD,
            ],
            'text': [
                CompressionAlgorithm.BROTLI,
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.LZ4,
            ],
            'numpy_array': [
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.BROTLI,
            ],
            'generic': [
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.BROTLI,
            ]
        }

        # 特征权重
        self._feature_weights = {
            'entropy': 0.3,
            'redundancy': 0.3,
            'size': 0.2,
            'data_type': 0.2,
        }

    def select(self,
              data_features: Dict[str, Any],
              config: CompressionConfig) -> CompressionAlgorithm:
        """
        选择最优算法

        Args:
            data_features: 数据特征
            config: 压缩配置

        Returns:
            CompressionAlgorithm: 选择的算法
        """
        # 1. 根据数据类型获取候选算法
        data_type = data_features.get('data_type', 'generic')
        candidates = self._get_candidate_algorithms(data_type)

        # 2. 根据性能要求过滤
        candidates = self._filter_by_requirements(candidates, config)

        # 3. 使用UCB算法选择
        selected = self._ucb_select(candidates, data_features)

        logger.debug(
            f"Selected algorithm: {selected.value} from candidates: "
            f"{[c.value for c in candidates]}"
        )

        return selected

    def update(self, algorithm: CompressionAlgorithm, performance: CompressionStats):
        """
        更新算法性能数据

        Args:
            algorithm: 算法类型
            performance: 性能统计
        """
        if algorithm not in self._algorithm_stats:
            logger.warning(f"Unknown algorithm: {algorithm.value}")
            return

        stats = self._algorithm_stats[algorithm]

        # 更新统计信息
        stats['count'] += 1
        stats['total_ratio'] += performance.compression_ratio
        stats['total_time'] += performance.compression_time
        stats['success_count'] += 1

        # 计算平均值
        stats['avg_ratio'] = stats['total_ratio'] / stats['count']
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['success_rate'] = stats['success_count'] / stats['count']

        logger.debug(
            f"Updated {algorithm.value}: avg_ratio={stats['avg_ratio']:.2f}x, "
            f"avg_time={stats['avg_time']*1000:.2f}ms"
        )

    def _get_candidate_algorithms(self, data_type: str) -> List[CompressionAlgorithm]:
        """获取候选算法列表"""
        # 映射数据类型
        type_key = self._map_data_type(data_type)

        # 获取候选算法
        candidates = self._type_algorithm_map.get(type_key, self._type_algorithm_map['generic'])

        return candidates

    def _map_data_type(self, data_type: str) -> str:
        """映射数据类型到算法选择类别"""
        mapping = {
            'model_weights': 'model_weights',
            'bf16_tensor': 'model_weights',
            'fp32_tensor': 'model_weights',
            'kv_cache': 'kv_cache',
            'text': 'text',
            'json': 'text',
            'numpy_array': 'numpy_array',
            'sparse_matrix': 'numpy_array',
            'generic': 'generic',
            'binary': 'generic',
        }

        return mapping.get(data_type, 'generic')

    def _filter_by_requirements(self,
                               candidates: List[CompressionAlgorithm],
                               config: CompressionConfig) -> List[CompressionAlgorithm]:
        """根据性能要求过滤算法"""
        filtered = []

        for algorithm in candidates:
            # 检查算法是否满足要求
            if self._meets_requirements(algorithm, config):
                filtered.append(algorithm)

        # 如果没有满足要求的算法，返回所有候选
        return filtered if filtered else candidates

    def _meets_requirements(self, algorithm: CompressionAlgorithm, config: CompressionConfig) -> bool:
        """检查算法是否满足要求"""
        # 简化实现：所有算法都满足基本要求
        # 实际实现中可以添加更复杂的检查逻辑
        return True

    def _ucb_select(self,
                   candidates: List[CompressionAlgorithm],
                   features: Dict[str, Any]) -> CompressionAlgorithm:
        """
        使用UCB算法选择

        UCB公式: score = avg_reward + c * sqrt(ln(N) / n)

        其中:
        - avg_reward: 平均奖励（压缩比）
        - c: 探索因子
        - N: 总选择次数
        - n: 该算法选择次数
        """
        # 计算总选择次数
        total_count = sum(
            self._algorithm_stats[alg]['count']
            for alg in candidates
        )

        # 如果总次数为0，随机选择
        if total_count == 0:
            return random.choice(candidates)

        best_algorithm = None
        best_score = -float('inf')

        for algorithm in candidates:
            stats = self._algorithm_stats[algorithm]

            # 如果该算法从未使用，优先探索
            if stats['count'] == 0:
                return algorithm

            # 计算UCB分数
            # 奖励 = 压缩比 - 时间惩罚
            avg_reward = stats['avg_ratio']

            # 时间惩罚：归一化到0-1范围
            # 假设理想时间为0，最差时间为1秒
            time_penalty = min(stats['avg_time'], 1.0)

            # 综合奖励
            reward = avg_reward - 0.1 * time_penalty

            # UCB项
            exploration = self.exploration_factor * math.sqrt(
                math.log(total_count) / stats['count']
            )

            # 总分数
            score = reward + exploration

            # 根据数据特征调整分数
            score = self._adjust_score_by_features(score, algorithm, features)

            if score > best_score:
                best_score = score
                best_algorithm = algorithm

        return best_algorithm

    def _adjust_score_by_features(self,
                                  score: float,
                                  algorithm: CompressionAlgorithm,
                                  features: Dict[str, Any]) -> float:
        """根据数据特征调整分数"""
        adjusted_score = score

        # 熵值调整
        entropy = features.get('entropy', 0.0)
        if entropy > 7.0:  # 高熵数据
            # 高熵数据压缩效果差，选择快速算法
            if algorithm == CompressionAlgorithm.LZ4:
                adjusted_score += 0.5

        # 冗余度调整
        redundancy = features.get('redundancy', 0.0)
        if redundancy > 0.5:  # 高冗余数据
            # 高冗余数据压缩效果好，选择高压缩比算法
            if algorithm in [CompressionAlgorithm.ZSTD, CompressionAlgorithm.BROTLI]:
                adjusted_score += 0.5

        # 数据大小调整
        size = features.get('size', 0)
        if size > 10 * 1024 * 1024:  # 大于10MB
            # 大数据优先选择快速算法
            if algorithm == CompressionAlgorithm.LZ4:
                adjusted_score += 0.3

        return adjusted_score

    def get_algorithm_stats(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        return {
            alg.value: {
                'count': stats['count'],
                'avg_ratio': f"{stats['avg_ratio']:.2f}x",
                'avg_time_ms': f"{stats['avg_time']*1000:.2f}",
                'success_rate': f"{stats['success_rate']*100:.1f}%",
            }
            for alg, stats in self._algorithm_stats.items()
            if stats['count'] > 0
        }

    def reset_stats(self):
        """重置统计信息"""
        for stats in self._algorithm_stats.values():
            stats['count'] = 0
            stats['total_ratio'] = 0.0
            stats['total_time'] = 0.0
            stats['avg_ratio'] = 0.0
            stats['avg_time'] = 0.0
            stats['success_rate'] = 0.0
            stats['success_count'] = 0
