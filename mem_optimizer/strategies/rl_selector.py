"""
mem_optimizer RL策略选择器

实现基于强化学习的分配器选择策略。
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from ..core.base import AllocatorType, AllocationRequest, StrategySelectorBase
from ..core.config import RLSelectorConfig
from ..core.exceptions import StrategyError, RLTrainingError


@dataclass
class State:
    """状态表示"""
    size_category: int
    fragmentation_level: int
    memory_pressure: int
    allocation_pattern: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.size_category, self.fragmentation_level,
                self.memory_pressure, self.allocation_pattern)

    def to_index(self, state_space_size: int) -> int:
        return hash(self.to_tuple()) % state_space_size


@dataclass
class Action:
    """动作表示"""
    allocator: AllocatorType

    def to_index(self) -> int:
        mapping = {
            AllocatorType.BUDDY: 0,
            AllocatorType.SLAB: 1,
            AllocatorType.TLSF: 2
        }
        return mapping.get(self.allocator, 0)


@dataclass
class Experience:
    """经验记录"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


@dataclass
class QTable:
    """Q值表"""
    state_space_size: int
    action_space_size: int
    table: Dict[int, List[float]] = field(default_factory=dict)

    def get(self, state_idx: int, action_idx: int) -> float:
        if state_idx not in self.table:
            self.table[state_idx] = [0.0] * self.action_space_size
        return self.table[state_idx][action_idx]

    def set(self, state_idx: int, action_idx: int, value: float):
        if state_idx not in self.table:
            self.table[state_idx] = [0.0] * self.action_space_size
        self.table[state_idx][action_idx] = value

    def get_best_action(self, state_idx: int) -> int:
        if state_idx not in self.table:
            return 0
        values = self.table[state_idx]
        return values.index(max(values))


class RLStrategySelector(StrategySelectorBase):
    """
    强化学习策略选择器

    使用Q-Learning算法学习最优的分配器选择策略。
    """

    def __init__(self, config: Optional[RLSelectorConfig] = None):
        """
        初始化RL策略选择器

        Args:
            config: RL配置
        """
        self.config = config or RLSelectorConfig()

        self.learning_rate = self.config.learning_rate
        self.discount_factor = self.config.discount_factor
        self.exploration_rate = self.config.exploration_rate
        self.exploration_decay = self.config.exploration_decay
        self.min_exploration = self.config.min_exploration

        self.state_space_size = 1000
        self.action_space_size = 3

        self.q_table = QTable(
            state_space_size=self.state_space_size,
            action_space_size=self.action_space_size
        )

        self._experience_buffer: deque = deque(
            maxlen=self.config.history_size
        )

        self._allocator_mapping = {
            0: AllocatorType.BUDDY,
            1: AllocatorType.SLAB,
            2: AllocatorType.TLSF
        }

        self._last_state: Optional[State] = None
        self._last_action: Optional[Action] = None
        self._episode_count = 0
        self._total_reward = 0.0

    def _extract_state(self,
                      request: AllocationRequest,
                      context: Dict[str, Any]) -> State:
        """
        从请求和上下文中提取状态

        Args:
            request: 分配请求
            context: 上下文信息

        Returns:
            State: 状态
        """
        size = request.size

        if size <= 256:
            size_category = 0
        elif size <= 4096:
            size_category = 1
        elif size <= 65536:
            size_category = 2
        elif size <= 1048576:
            size_category = 3
        else:
            size_category = 4

        fragmentation = context.get('fragmentation', 0.0)
        if fragmentation < 0.1:
            frag_level = 0
        elif fragmentation < 0.3:
            frag_level = 1
        elif fragmentation < 0.5:
            frag_level = 2
        else:
            frag_level = 3

        total_size = context.get('total_size', 1)
        used_size = context.get('used_size', 0)
        memory_pressure = int((used_size / total_size) * 4) if total_size > 0 else 0
        memory_pressure = min(memory_pressure, 4)

        allocation_pattern = 0

        return State(
            size_category=size_category,
            fragmentation_level=frag_level,
            memory_pressure=memory_pressure,
            allocation_pattern=allocation_pattern
        )

    def _calculate_reward(self, performance: Dict[str, Any]) -> float:
        """
        计算奖励值

        Args:
            performance: 性能数据

        Returns:
            float: 奖励值
        """
        if not performance.get('success', False):
            return -10.0

        reward = 0.0

        allocation_time = performance.get('allocation_time', 0.0)
        if allocation_time < 0.0001:
            reward += 5.0
        elif allocation_time < 0.001:
            reward += 3.0
        elif allocation_time < 0.01:
            reward += 1.0
        else:
            reward -= 1.0

        fragmentation = performance.get('fragmentation', 0.0)
        if fragmentation < 0.1:
            reward += 3.0
        elif fragmentation < 0.3:
            reward += 1.0
        else:
            reward -= fragmentation * 5.0

        size = performance.get('size', 0)
        if size > 0:
            efficiency = 1.0 - fragmentation
            reward += efficiency * 2.0

        return reward

    def _choose_action(self, state: State) -> Action:
        """
        选择动作（epsilon-greedy策略）

        Args:
            state: 当前状态

        Returns:
            Action: 选择的动作
        """
        state_idx = state.to_index(self.state_space_size)

        if random.random() < self.exploration_rate:
            action_idx = random.randint(0, self.action_space_size - 1)
        else:
            action_idx = self.q_table.get_best_action(state_idx)

        allocator = self._allocator_mapping.get(action_idx, AllocatorType.TLSF)
        return Action(allocator=allocator)

    def _update_q_value(self,
                       state: State,
                       action: Action,
                       reward: float,
                       next_state: State):
        """
        更新Q值

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        state_idx = state.to_index(self.state_space_size)
        action_idx = action.to_index()
        next_state_idx = next_state.to_index(self.state_space_size)

        current_q = self.q_table.get(state_idx, action_idx)

        best_next_q = max([
            self.q_table.get(next_state_idx, a)
            for a in range(self.action_space_size)
        ])

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )

        self.q_table.set(state_idx, action_idx, new_q)

    def select_allocator(self,
                        request: AllocationRequest,
                        context: Dict[str, Any]) -> AllocatorType:
        """
        选择最优分配器

        Args:
            request: 分配请求
            context: 上下文信息

        Returns:
            AllocatorType: 选择的分配器类型
        """
        state = self._extract_state(request, context)

        if self._last_state is not None and self._last_action is not None:
            experience = Experience(
                state=self._last_state,
                action=self._last_action,
                reward=0.0,
                next_state=state,
                done=False
            )
            self._experience_buffer.append(experience)

        action = self._choose_action(state)

        self._last_state = state
        self._last_action = action

        return action.allocator

    def update_performance(self,
                          allocator: AllocatorType,
                          performance: Dict[str, Any]):
        """
        更新性能数据

        Args:
            allocator: 分配器类型
            performance: 性能数据
        """
        if self._last_state is None:
            return

        action = Action(allocator=allocator)
        reward = self._calculate_reward(performance)

        self._total_reward += reward
        self._episode_count += 1

        next_state = self._last_state

        self._update_q_value(self._last_state, action, reward, next_state)

        if self.config.enable_online_learning:
            self._decay_exploration()

    def _decay_exploration(self):
        """衰减探索率"""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def get_recommendations(self) -> Dict[AllocatorType, float]:
        """
        获取推荐权重

        Returns:
            Dict: 分配器到权重的映射
        """
        recommendations = {}

        for action_idx, allocator in self._allocator_mapping.items():
            total_q = 0.0
            count = 0

            for state_idx in self.q_table.table.keys():
                q_value = self.q_table.get(state_idx, action_idx)
                total_q += q_value
                count += 1

            avg_q = total_q / count if count > 0 else 0.0

            weight = 1.0 / (1.0 + math.exp(-avg_q))
            recommendations[allocator] = weight

        return recommendations

    def train_from_history(self, experiences: List[Experience]):
        """
        从历史经验训练

        Args:
            experiences: 经验列表
        """
        for exp in experiences:
            self._update_q_value(exp.state, exp.action, exp.reward, exp.next_state)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'episode_count': self._episode_count,
            'total_reward': self._total_reward,
            'average_reward': self._total_reward / max(self._episode_count, 1),
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table.table),
            'experience_buffer_size': len(self._experience_buffer)
        }

    def reset(self):
        """重置学习器"""
        self.q_table = QTable(
            state_space_size=self.state_space_size,
            action_space_size=self.action_space_size
        )
        self._experience_buffer.clear()
        self._last_state = None
        self._last_action = None
        self._episode_count = 0
        self._total_reward = 0.0
        self.exploration_rate = self.config.exploration_rate

    def save_state(self) -> Dict[str, Any]:
        """
        保存状态

        Returns:
            Dict: 状态字典
        """
        return {
            'q_table': dict(self.q_table.table),
            'exploration_rate': self.exploration_rate,
            'episode_count': self._episode_count,
            'total_reward': self._total_reward
        }

    def load_state(self, state: Dict[str, Any]):
        """
        加载状态

        Args:
            state: 状态字典
        """
        self.q_table.table = state.get('q_table', {})
        self.exploration_rate = state.get('exploration_rate', self.config.exploration_rate)
        self._episode_count = state.get('episode_count', 0)
        self._total_reward = state.get('total_reward', 0.0)
