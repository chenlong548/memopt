"""
stream_processor 容错层

提供检查点和恢复机制。
"""

from .state_backend import (
    StateBackend,
    StateValue,
    MemoryStateBackend,
    FileSystemStateBackend,
    KeyValueState,
    ListState,
)

from .snapshot import (
    Snapshot,
    SnapshotMetadata,
    SnapshotBuilder,
    SnapshotStore,
)

from .recovery import (
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    OperatorRecovery,
    CheckpointRecovery,
)

from .manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointResult,
    CheckpointCoordinator,
)

__all__ = [
    'StateBackend',
    'StateValue',
    'MemoryStateBackend',
    'FileSystemStateBackend',
    'KeyValueState',
    'ListState',
    'Snapshot',
    'SnapshotMetadata',
    'SnapshotBuilder',
    'SnapshotStore',
    'RecoveryManager',
    'RecoveryResult',
    'RecoveryStrategy',
    'OperatorRecovery',
    'CheckpointRecovery',
    'CheckpointManager',
    'CheckpointConfig',
    'CheckpointResult',
    'CheckpointCoordinator',
]
