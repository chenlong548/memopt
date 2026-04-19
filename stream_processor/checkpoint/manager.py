"""
stream_processor 检查点管理器

管理检查点的创建和协调。
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import time
import threading
import logging

from .snapshot import Snapshot, SnapshotBuilder, SnapshotStore
from .state_backend import StateBackend, MemoryStateBackend
from .recovery import RecoveryManager, RecoveryResult
from ..core.exceptions import CheckpointError
from ..core.execution_context import ExecutionContext


logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """
    检查点配置

    定义检查点的配置参数。
    """

    checkpoint_interval: float = 60.0

    checkpoint_timeout: float = 600.0

    max_concurrent_checkpoints: int = 1

    min_pause_between_checkpoints: float = 0.0

    tolerate_failure_number: int = 0

    enable_unaligned_checkpoints: bool = False


@dataclass
class CheckpointResult:
    """
    检查点结果

    记录检查点操作的结果。
    """

    checkpoint_id: int

    success: bool

    snapshot_id: Optional[str] = None

    duration_ms: float = 0.0

    size_bytes: int = 0

    num_operators: int = 0

    error_message: Optional[str] = None


class CheckpointManager:
    """
    检查点管理器

    协调检查点的创建和管理。
    """

    def __init__(self,
                 job_id: str,
                 config: Optional[CheckpointConfig] = None,
                 state_backend: Optional[StateBackend] = None):
        """
        初始化检查点管理器

        Args:
            job_id: 任务ID
            config: 检查点配置
            state_backend: 状态后端
        """
        self._job_id = job_id
        self._config = config or CheckpointConfig()
        self._state_backend = state_backend or MemoryStateBackend()
        self._snapshot_store = SnapshotStore()
        self._recovery_manager = RecoveryManager(self._snapshot_store, self._state_backend)

        self._checkpoint_id_counter = 0
        self._last_checkpoint_time = 0.0
        self._pending_checkpoints: Dict[int, SnapshotBuilder] = {}
        self._completed_checkpoints: List[int] = []

        self._lock = threading.Lock()
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._running = False

        self._checkpoint_callbacks: List[Callable[[CheckpointResult], None]] = []

    def start(self):
        """启动检查点管理器"""
        if self._running:
            return

        self._running = True
        self._checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
        self._checkpoint_thread.start()

        logger.info(f"Checkpoint manager started for job {self._job_id}")

    def stop(self):
        """停止检查点管理器"""
        self._running = False
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5.0)
            self._checkpoint_thread = None

        logger.info(f"Checkpoint manager stopped for job {self._job_id}")

    def _checkpoint_loop(self):
        """检查点循环"""
        while self._running:
            try:
                current_time = time.time()

                if current_time - self._last_checkpoint_time >= self._config.checkpoint_interval:
                    self.trigger_checkpoint()
                    self._last_checkpoint_time = current_time

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")

    def trigger_checkpoint(self) -> CheckpointResult:
        """
        触发检查点

        Returns:
            CheckpointResult: 检查点结果
        """
        start_time = time.time()

        with self._lock:
            self._checkpoint_id_counter += 1
            checkpoint_id = self._checkpoint_id_counter

        try:
            logger.info(f"Triggering checkpoint {checkpoint_id}")

            snapshot = self._create_snapshot(checkpoint_id)

            self._snapshot_store.store(snapshot)

            self._completed_checkpoints.append(checkpoint_id)

            duration_ms = (time.time() - start_time) * 1000

            result = CheckpointResult(
                checkpoint_id=checkpoint_id,
                success=True,
                snapshot_id=snapshot.metadata.snapshot_id,
                duration_ms=duration_ms,
                size_bytes=snapshot.metadata.size_bytes,
                num_operators=len(snapshot.metadata.operator_ids)
            )

            self._notify_callbacks(result)

            logger.info(f"Checkpoint {checkpoint_id} completed in {duration_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"Checkpoint {checkpoint_id} failed: {e}")

            result = CheckpointResult(
                checkpoint_id=checkpoint_id,
                success=False,
                error_message=str(e)
            )

            self._notify_callbacks(result)

            return result

    def _create_snapshot(self, checkpoint_id: int) -> Snapshot:
        """
        创建快照

        Args:
            checkpoint_id: 检查点ID

        Returns:
            Snapshot: 快照对象
        """
        builder = SnapshotBuilder(
            job_id=self._job_id,
            checkpoint_id=checkpoint_id
        )

        state_data = self._state_backend.snapshot()
        for key, value in state_data.items():
            builder.add_state_data(key, value)

        return builder.build()

    def restore(self, snapshot_id: Optional[str] = None) -> RecoveryResult:
        """
        恢复状态

        Args:
            snapshot_id: 快照ID

        Returns:
            RecoveryResult: 恢复结果
        """
        return self._recovery_manager.recover(snapshot_id)

    def register_callback(self, callback: Callable[[CheckpointResult], None]):
        """
        注册回调函数

        Args:
            callback: 回调函数
        """
        self._checkpoint_callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[CheckpointResult], None]):
        """
        注销回调函数

        Args:
            callback: 回调函数
        """
        if callback in self._checkpoint_callbacks:
            self._checkpoint_callbacks.remove(callback)

    def _notify_callbacks(self, result: CheckpointResult):
        """
        通知回调函数

        Args:
            result: 检查点结果
        """
        for callback in self._checkpoint_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Checkpoint callback error: {e}")

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """
        获取检查点历史

        Returns:
            List[Dict]: 检查点历史
        """
        snapshots = self._snapshot_store.list_snapshots()
        return [
            {
                'checkpoint_id': s.checkpoint_id,
                'snapshot_id': s.snapshot_id,
                'timestamp': s.timestamp,
                'duration_ms': s.duration_ms,
                'size_bytes': s.size_bytes,
                'is_completed': s.is_completed
            }
            for s in snapshots
        ]

    def get_latest_checkpoint_id(self) -> Optional[int]:
        """
        获取最新检查点ID

        Returns:
            Optional[int]: 检查点ID
        """
        if not self._completed_checkpoints:
            return None
        return self._completed_checkpoints[-1]

    def get_state_backend(self) -> StateBackend:
        """
        获取状态后端

        Returns:
            StateBackend: 状态后端
        """
        return self._state_backend

    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running


class CheckpointCoordinator:
    """
    检查点协调器

    协调分布式检查点。
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        初始化检查点协调器

        Args:
            checkpoint_manager: 检查点管理器
        """
        self._checkpoint_manager = checkpoint_manager
        self._pending_acknowledgments: Dict[int, set] = {}
        self._lock = threading.Lock()

    def initiate_checkpoint(self, operator_ids: List[str]) -> int:
        """
        发起检查点

        Args:
            operator_ids: 操作符ID列表

        Returns:
            int: 检查点ID
        """
        result = self._checkpoint_manager.trigger_checkpoint()

        if result.success:
            with self._lock:
                self._pending_acknowledgments[result.checkpoint_id] = set(operator_ids)

        return result.checkpoint_id

    def acknowledge_checkpoint(self, checkpoint_id: int, operator_id: str):
        """
        确认检查点

        Args:
            checkpoint_id: 检查点ID
            operator_id: 操作符ID
        """
        with self._lock:
            if checkpoint_id in self._pending_acknowledgments:
                self._pending_acknowledgments[checkpoint_id].discard(operator_id)

                if not self._pending_acknowledgments[checkpoint_id]:
                    del self._pending_acknowledgments[checkpoint_id]

    def is_checkpoint_complete(self, checkpoint_id: int) -> bool:
        """
        检查点是否完成

        Args:
            checkpoint_id: 检查点ID

        Returns:
            bool: 是否完成
        """
        with self._lock:
            return checkpoint_id not in self._pending_acknowledgments
