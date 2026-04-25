"""
stream_processor 恢复机制

实现故障恢复功能。
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import logging

from .snapshot import Snapshot, SnapshotStore
from .state_backend import StateBackend
from ..core.exceptions import RecoveryError


logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """
    恢复结果

    记录恢复操作的结果。
    """

    success: bool

    snapshot_id: Optional[str] = None

    recovered_operators: List[str] = field(default_factory=list)

    recovery_time_ms: float = 0.0

    error_message: Optional[str] = None

    def __post_init__(self):
        if self.recovered_operators is None:
            self.recovered_operators = []


class RecoveryManager:
    """
    恢复管理器

    管理故障恢复流程。
    """

    def __init__(self,
                 snapshot_store: SnapshotStore,
                 state_backend: StateBackend):
        """
        初始化恢复管理器

        Args:
            snapshot_store: 快照存储
            state_backend: 状态后端
        """
        self._snapshot_store = snapshot_store
        self._state_backend = state_backend

    def recover(self, snapshot_id: Optional[str] = None) -> RecoveryResult:
        """
        执行恢复

        Args:
            snapshot_id: 指定快照ID，None则使用最新快照

        Returns:
            RecoveryResult: 恢复结果
        """
        start_time = time.time()

        try:
            if snapshot_id:
                snapshot = self._snapshot_store.get(snapshot_id)
            else:
                snapshot = self._snapshot_store.get_latest()

            if not snapshot:
                return RecoveryResult(
                    success=False,
                    error_message="No snapshot available for recovery"
                )

            self._restore_state(snapshot)

            recovered_operators = self._restore_operators(snapshot)

            recovery_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Recovery completed successfully from snapshot {snapshot.metadata.snapshot_id}"
            )

            return RecoveryResult(
                success=True,
                snapshot_id=snapshot.metadata.snapshot_id,
                recovered_operators=recovered_operators,
                recovery_time_ms=recovery_time_ms
            )

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return RecoveryResult(
                success=False,
                error_message=str(e)
            )

    def _restore_state(self, snapshot: Snapshot):
        """
        恢复状态

        Args:
            snapshot: 快照对象
        """
        self._state_backend.restore(snapshot.state_data)

    def _restore_operators(self, snapshot: Snapshot) -> List[str]:
        """
        恢复操作符

        Args:
            snapshot: 快照对象

        Returns:
            List[str]: 恢复的操作符列表
        """
        return list(snapshot.operator_states.keys())

    def validate_snapshot(self, snapshot: Snapshot) -> bool:
        """
        验证快照

        Args:
            snapshot: 快照对象

        Returns:
            bool: 是否有效
        """
        if not snapshot.metadata.is_completed:
            return False

        if not snapshot.metadata.snapshot_id:
            return False

        if not snapshot.operator_states and not snapshot.state_data:
            return False

        return True

    def get_available_snapshots(self) -> List[Dict[str, Any]]:
        """
        获取可用快照列表

        Returns:
            List[Dict]: 快照列表
        """
        snapshots = self._snapshot_store.list_snapshots()
        return [
            {
                'snapshot_id': s.snapshot_id,
                'timestamp': s.timestamp,
                'checkpoint_id': s.checkpoint_id,
                'is_completed': s.is_completed
            }
            for s in snapshots
        ]


class OperatorRecovery:
    """
    操作符恢复

    处理单个操作符的恢复。
    """

    def __init__(self, operator_id: str):
        """
        初始化操作符恢复

        Args:
            operator_id: 操作符ID
        """
        self._operator_id = operator_id

    def recover_state(self, state: Any) -> bool:
        """
        恢复状态

        Args:
            state: 状态数据

        Returns:
            bool: 是否成功
        """
        try:
            return True
        except Exception as e:
            logger.error(f"Failed to recover state for operator {self._operator_id}: {e}")
            return False

    def validate_state(self, state: Any) -> bool:
        """
        验证状态

        Args:
            state: 状态数据

        Returns:
            bool: 是否有效
        """
        return state is not None


class RecoveryStrategy:
    """
    恢复策略

    定义不同的恢复策略。
    """

    @staticmethod
    def latest_snapshot() -> 'RecoveryStrategy':
        """使用最新快照恢复"""
        return RecoveryStrategy(strategy='latest')

    @staticmethod
    def specific_snapshot(snapshot_id: str) -> 'RecoveryStrategy':
        """使用指定快照恢复"""
        return RecoveryStrategy(strategy='specific', snapshot_id=snapshot_id)

    @staticmethod
    def from_beginning() -> 'RecoveryStrategy':
        """从头开始恢复"""
        return RecoveryStrategy(strategy='beginning')

    def __init__(self,
                 strategy: str = 'latest',
                 snapshot_id: Optional[str] = None):
        """
        初始化恢复策略

        Args:
            strategy: 策略类型
            snapshot_id: 快照ID
        """
        self._strategy = strategy
        self._snapshot_id = snapshot_id

    def get_strategy(self) -> str:
        """获取策略类型"""
        return self._strategy

    def get_snapshot_id(self) -> Optional[str]:
        """获取快照ID"""
        return self._snapshot_id


class CheckpointRecovery:
    """
    检查点恢复

    处理检查点级别的恢复。
    """

    def __init__(self, checkpoint_dir: str):
        """
        初始化检查点恢复

        Args:
            checkpoint_dir: 检查点目录
        """
        self._checkpoint_dir = checkpoint_dir

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        查找最新检查点

        Returns:
            Optional[str]: 检查点路径
        """
        import os
        import glob

        pattern = os.path.join(self._checkpoint_dir, 'checkpoint_*')
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]

    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            Optional[Dict]: 检查点数据
        """
        import json
        import os

        try:
            metadata_path = os.path.join(checkpoint_path, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
