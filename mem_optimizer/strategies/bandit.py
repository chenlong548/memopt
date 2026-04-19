"""
mem_optimizer UCB多臂老虎机

实现UCB (Upper Confidence Bound) 多臂老虎机算法。
"""

import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.base import AllocatorType
from ..core.config import BanditConfig
from ..core.exceptions import BanditError


@dataclass
class Arm:
    """老虎机臂"""
    allocator: AllocatorType
    pulls: int = 0
    rewards: float = 0.0
    avg_reward: float = 0.0
    ucb_value: float = float('inf')

    def update(self, reward: float):
        """更新臂统计"""
        self.pulls += 1
        self.rewards += reward
        self.avg_reward = self.rewards / self.pulls


@dataclass
class ContextualArm(Arm):
    """上下文老虎机臂"""
    context_rewards: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    context_pulls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update_context(self, context_key: str, reward: float):
        """更新上下文统计"""
        self.context_pulls[context_key] += 1
        self.context_rewards[context_key] += reward


class UCB1Bandit:
    """
    UCB1多臂老虎机

    实现经典的UCB1算法。
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        """
        初始化UCB1老虎机

        Args:
            config: 配置
        """
        self.config = config or BanditConfig()
        self.confidence_level = self.config.confidence_level

        self._arms: Dict[AllocatorType, Arm] = {
            AllocatorType.BUDDY: Arm(allocator=AllocatorType.BUDDY),
            AllocatorType.SLAB: Arm(allocator=AllocatorType.SLAB),
            AllocatorType.TLSF: Arm(allocator=AllocatorType.TLSF)
        }

        self._total_pulls = 0
        self._history: List[Dict[str, Any]] = []

    def _calculate_ucb(self, arm: Arm) -> float:
        """
        计算UCB值

        Args:
            arm: 臂

        Returns:
            float: UCB值
        """
        if arm.pulls == 0:
            return float('inf')

        exploration = math.sqrt(
            (self.confidence_level * math.log(self._total_pulls)) / arm.pulls
        )

        return arm.avg_reward + exploration

    def select(self) -> AllocatorType:
        """
        选择臂

        Returns:
            AllocatorType: 选择的分配器
        """
        for arm in self._arms.values():
            if arm.pulls < self.config.min_samples:
                return arm.allocator

        best_arm = None
        best_ucb = float('-inf')

        for arm in self._arms.values():
            ucb = self._calculate_ucb(arm)
            arm.ucb_value = ucb

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm.allocator if best_arm else AllocatorType.TLSF

    def update(self, allocator: AllocatorType, reward: float):
        """
        更新臂统计

        Args:
            allocator: 分配器类型
            reward: 奖励值
        """
        arm = self._arms.get(allocator)
        if arm is None:
            return

        arm.update(reward)
        self._total_pulls += 1

        if self.config.enable_decay:
            self._apply_decay()

        self._history.append({
            'allocator': allocator.value,
            'reward': reward,
            'total_pulls': self._total_pulls
        })

    def _apply_decay(self):
        """应用衰减"""
        decay = self.config.decay_rate

        for arm in self._arms.values():
            arm.rewards *= decay
            arm.pulls = max(1, int(arm.pulls * decay))
            if arm.pulls > 0:
                arm.avg_reward = arm.rewards / arm.pulls

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_pulls': self._total_pulls,
            'arms': {}
        }

        for allocator, arm in self._arms.items():
            stats['arms'][allocator.value] = {
                'pulls': arm.pulls,
                'rewards': arm.rewards,
                'avg_reward': arm.avg_reward,
                'ucb_value': arm.ucb_value
            }

        return stats

    def get_best_arm(self) -> AllocatorType:
        """
        获取最佳臂

        Returns:
            AllocatorType: 最佳分配器
        """
        best_arm = None
        best_avg = float('-inf')

        for arm in self._arms.values():
            if arm.pulls > 0 and arm.avg_reward > best_avg:
                best_avg = arm.avg_reward
                best_arm = arm

        return best_arm.allocator if best_arm else AllocatorType.TLSF

    def reset(self):
        """重置老虎机"""
        for arm in self._arms.values():
            arm.pulls = 0
            arm.rewards = 0.0
            arm.avg_reward = 0.0
            arm.ucb_value = float('inf')

        self._total_pulls = 0
        self._history.clear()


class UCBTunedBandit(UCB1Bandit):
    """
    UCB-Tuned多臂老虎机

    实现UCB-Tuned算法，使用方差估计。
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        super().__init__(config)
        self._reward_squares: Dict[AllocatorType, float] = {
            allocator: 0.0 for allocator in self._arms.keys()
        }

    def _calculate_ucb(self, arm: Arm) -> float:
        """计算UCB-Tuned值"""
        if arm.pulls == 0:
            return float('inf')

        reward_sq = self._reward_squares.get(arm.allocator, 0.0)
        variance = (reward_sq / arm.pulls) - (arm.avg_reward ** 2)
        variance = max(0, variance)

        log_term = math.log(self._total_pulls) / arm.pulls
        bound = min(0.25, variance + math.sqrt(2 * log_term))

        return arm.avg_reward + math.sqrt(self.confidence_level * log_term * bound)

    def update(self, allocator: AllocatorType, reward: float):
        """更新臂统计"""
        super().update(allocator, reward)

        self._reward_squares[allocator] = (
            self._reward_squares.get(allocator, 0.0) + reward ** 2
        )


class ThompsonSamplingBandit:
    """
    Thompson Sampling多臂老虎机

    实现Thompson Sampling算法。
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        """
        初始化Thompson Sampling老虎机

        Args:
            config: 配置
        """
        self.config = config or BanditConfig()

        self._alpha: Dict[AllocatorType, float] = {
            AllocatorType.BUDDY: 1.0,
            AllocatorType.SLAB: 1.0,
            AllocatorType.TLSF: 1.0
        }

        self._beta: Dict[AllocatorType, float] = {
            AllocatorType.BUDDY: 1.0,
            AllocatorType.SLAB: 1.0,
            AllocatorType.TLSF: 1.0
        }

        self._total_pulls = 0

    def select(self) -> AllocatorType:
        """
        选择臂

        Returns:
            AllocatorType: 选择的分配器
        """
        import random

        best_allocator = None
        best_sample = float('-inf')

        for allocator in self._alpha.keys():
            alpha = self._alpha[allocator]
            beta = self._beta[allocator]

            sample = random.betavariate(alpha, beta)

            if sample > best_sample:
                best_sample = sample
                best_allocator = allocator

        return best_allocator or AllocatorType.TLSF

    def update(self, allocator: AllocatorType, reward: float):
        """
        更新臂统计

        Args:
            allocator: 分配器类型
            reward: 奖励值 (0-1范围)
        """
        normalized_reward = max(0.0, min(1.0, reward))

        if normalized_reward > 0.5:
            self._alpha[allocator] += normalized_reward
        else:
            self._beta[allocator] += (1.0 - normalized_reward)

        self._total_pulls += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_pulls': self._total_pulls,
            'arms': {}
        }

        for allocator in self._alpha.keys():
            alpha = self._alpha[allocator]
            beta = self._beta[allocator]

            stats['arms'][allocator.value] = {
                'alpha': alpha,
                'beta': beta,
                'expected_value': alpha / (alpha + beta)
            }

        return stats

    def reset(self):
        """重置老虎机"""
        for allocator in self._alpha.keys():
            self._alpha[allocator] = 1.0
            self._beta[allocator] = 1.0

        self._total_pulls = 0


class ContextualBandit:
    """
    上下文多臂老虎机

    实现带上下文的UCB算法。
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        """
        初始化上下文老虎机

        Args:
            config: 配置
        """
        self.config = config or BanditConfig()

        self._arms: Dict[AllocatorType, ContextualArm] = {
            AllocatorType.BUDDY: ContextualArm(allocator=AllocatorType.BUDDY),
            AllocatorType.SLAB: ContextualArm(allocator=AllocatorType.SLAB),
            AllocatorType.TLSF: ContextualArm(allocator=AllocatorType.TLSF)
        }

        self._total_pulls = 0
        self._confidence_level = self.config.confidence_level

    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """生成上下文键"""
        size = context.get('size', 0)
        fragmentation = context.get('fragmentation', 0)

        size_bucket = min(int(math.log2(max(size, 1)) // 4), 5)
        frag_bucket = min(int(fragmentation * 5), 4)

        return f"{size_bucket}_{frag_bucket}"

    def select(self, context: Dict[str, Any]) -> AllocatorType:
        """
        根据上下文选择臂

        Args:
            context: 上下文信息

        Returns:
            AllocatorType: 选择的分配器
        """
        context_key = self._get_context_key(context)

        best_arm = None
        best_ucb = float('-inf')

        for arm in self._arms.values():
            context_pulls = arm.context_pulls.get(context_key, 0)

            if context_pulls < self.config.min_samples:
                return arm.allocator

            context_rewards = arm.context_rewards.get(context_key, 0.0)
            avg_reward = context_rewards / context_pulls

            exploration = math.sqrt(
                (self._confidence_level * math.log(self._total_pulls)) / context_pulls
            )

            ucb = avg_reward + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm.allocator if best_arm else AllocatorType.TLSF

    def update(self, allocator: AllocatorType, reward: float, context: Dict[str, Any]):
        """
        更新臂统计

        Args:
            allocator: 分配器类型
            reward: 奖励值
            context: 上下文信息
        """
        arm = self._arms.get(allocator)
        if arm is None:
            return

        context_key = self._get_context_key(context)

        arm.update(reward)
        arm.update_context(context_key, reward)

        self._total_pulls += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_pulls': self._total_pulls,
            'arms': {}
        }

        for allocator, arm in self._arms.items():
            stats['arms'][allocator.value] = {
                'pulls': arm.pulls,
                'rewards': arm.rewards,
                'avg_reward': arm.avg_reward,
                'context_count': len(arm.context_pulls)
            }

        return stats

    def reset(self):
        """重置老虎机"""
        for arm in self._arms.values():
            arm.pulls = 0
            arm.rewards = 0.0
            arm.avg_reward = 0.0
            arm.context_pulls.clear()
            arm.context_rewards.clear()

        self._total_pulls = 0


def create_bandit(config: Optional[BanditConfig] = None) -> Any:
    """
    创建老虎机实例

    Args:
        config: 配置

    Returns:
        老虎机实例
    """
    config = config or BanditConfig()

    algorithm = config.algorithm.lower()

    if algorithm == 'ucb1':
        return UCB1Bandit(config)
    elif algorithm == 'ucb_tuned':
        return UCBTunedBandit(config)
    elif algorithm == 'thompson':
        return ThompsonSamplingBandit(config)
    elif algorithm == 'contextual':
        return ContextualBandit(config)
    else:
        return UCB1Bandit(config)
