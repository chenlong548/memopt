"""
stream_processor 快照机制

实现状态快照功能。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import json
import hashlib


@dataclass
class SnapshotMetadata:
    """
    快照元数据

    记录快照的基本信息。
    """

    snapshot_id: str

    timestamp: float

    checkpoint_id: int

    job_id: str

    operator_ids: List[str] = field(default_factory=list)

    size_bytes: int = 0

    duration_ms: float = 0.0

    is_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'checkpoint_id': self.checkpoint_id,
            'job_id': self.job_id,
            'operator_ids': self.operator_ids,
            'size_bytes': self.size_bytes,
            'duration_ms': self.duration_ms,
            'is_completed': self.is_completed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotMetadata':
        """从字典创建"""
        return cls(
            snapshot_id=data['snapshot_id'],
            timestamp=data['timestamp'],
            checkpoint_id=data['checkpoint_id'],
            job_id=data['job_id'],
            operator_ids=data.get('operator_ids', []),
            size_bytes=data.get('size_bytes', 0),
            duration_ms=data.get('duration_ms', 0.0),
            is_completed=data.get('is_completed', False)
        )


@dataclass
class Snapshot:
    """
    快照

    存储完整的快照数据。
    """

    metadata: SnapshotMetadata

    state_data: Dict[str, Any] = field(default_factory=dict)

    operator_states: Dict[str, Any] = field(default_factory=dict)

    def get_operator_state(self, operator_id: str) -> Optional[Any]:
        """
        获取操作符状态

        Args:
            operator_id: 操作符ID

        Returns:
            Optional[Any]: 操作符状态
        """
        return self.operator_states.get(operator_id)

    def set_operator_state(self, operator_id: str, state: Any):
        """
        设置操作符状态

        Args:
            operator_id: 操作符ID
            state: 状态数据
        """
        self.operator_states[operator_id] = state

    def to_bytes(self) -> bytes:
        """
        序列化为字节

        Returns:
            bytes: 序列化数据
        """
        data = {
            'metadata': self.metadata.to_dict(),
            'state_data': self.state_data,
            'operator_states': self.operator_states
        }
        return json.dumps(data).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Snapshot':
        """
        从字节反序列化

        Args:
            data: 字节数据

        Returns:
            Snapshot: 快照对象
        """
        obj = json.loads(data.decode('utf-8'))
        return cls(
            metadata=SnapshotMetadata.from_dict(obj['metadata']),
            state_data=obj.get('state_data', {}),
            operator_states=obj.get('operator_states', {})
        )

    def compute_checksum(self) -> str:
        """
        计算校验和

        Returns:
            str: 校验和
        """
        data = self.to_bytes()
        return hashlib.md5(data).hexdigest()


class SnapshotBuilder:
    """
    快照构建器

    构建快照对象。
    """

    def __init__(self, job_id: str, checkpoint_id: int):
        """
        初始化快照构建器

        Args:
            job_id: 任务ID
            checkpoint_id: 检查点ID
        """
        self._job_id = job_id
        self._checkpoint_id = checkpoint_id
        self._start_time = time.time()
        self._operator_states: Dict[str, Any] = {}
        self._state_data: Dict[str, Any] = {}

    def add_operator_state(self, operator_id: str, state: Any) -> 'SnapshotBuilder':
        """
        添加操作符状态

        Args:
            operator_id: 操作符ID
            state: 状态数据

        Returns:
            SnapshotBuilder: 构建器
        """
        self._operator_states[operator_id] = state
        return self

    def add_state_data(self, key: str, value: Any) -> 'SnapshotBuilder':
        """
        添加状态数据

        Args:
            key: 键
            value: 值

        Returns:
            SnapshotBuilder: 构建器
        """
        self._state_data[key] = value
        return self

    def build(self) -> Snapshot:
        """
        构建快照

        Returns:
            Snapshot: 快照对象
        """
        snapshot_id = self._generate_snapshot_id()

        duration_ms = (time.time() - self._start_time) * 1000

        size_bytes = len(json.dumps({
            'state_data': self._state_data,
            'operator_states': self._operator_states
        }).encode('utf-8'))

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            timestamp=self._start_time,
            checkpoint_id=self._checkpoint_id,
            job_id=self._job_id,
            operator_ids=list(self._operator_states.keys()),
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            is_completed=True
        )

        return Snapshot(
            metadata=metadata,
            state_data=self._state_data,
            operator_states=self._operator_states
        )

    def _generate_snapshot_id(self) -> str:
        """生成快照ID"""
        data = f"{self._job_id}_{self._checkpoint_id}_{self._start_time}"
        return hashlib.md5(data.encode()).hexdigest()


class SnapshotStore:
    """
    快照存储

    管理快照的存储和检索。
    """

    def __init__(self, max_snapshots: int = 10):
        """
        初始化快照存储

        Args:
            max_snapshots: 最大快照数量
        """
        self._max_snapshots = max_snapshots
        self._snapshots: Dict[str, Snapshot] = {}
        self._snapshot_list: List[str] = []

    def store(self, snapshot: Snapshot):
        """
        存储快照

        Args:
            snapshot: 快照对象
        """
        snapshot_id = snapshot.metadata.snapshot_id

        self._snapshots[snapshot_id] = snapshot
        self._snapshot_list.append(snapshot_id)

        while len(self._snapshot_list) > self._max_snapshots:
            old_id = self._snapshot_list.pop(0)
            del self._snapshots[old_id]

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        """
        获取快照

        Args:
            snapshot_id: 快照ID

        Returns:
            Optional[Snapshot]: 快照对象
        """
        return self._snapshots.get(snapshot_id)

    def get_latest(self) -> Optional[Snapshot]:
        """
        获取最新快照

        Returns:
            Optional[Snapshot]: 最新快照
        """
        if not self._snapshot_list:
            return None
        return self._snapshots.get(self._snapshot_list[-1])

    def list_snapshots(self) -> List[SnapshotMetadata]:
        """
        列出所有快照

        Returns:
            List[SnapshotMetadata]: 快照元数据列表
        """
        return [
            self._snapshots[sid].metadata
            for sid in self._snapshot_list
            if sid in self._snapshots
        ]

    def delete(self, snapshot_id: str) -> bool:
        """
        删除快照

        Args:
            snapshot_id: 快照ID

        Returns:
            bool: 是否成功
        """
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            self._snapshot_list.remove(snapshot_id)
            return True
        return False

    def clear(self):
        """清空所有快照"""
        self._snapshots.clear()
        self._snapshot_list.clear()
