"""
stream_processor 状态后端

定义状态存储机制。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import threading
import json
import os


@dataclass
class StateValue:
    """
    状态值

    存储状态数据及其元数据。
    """

    value: Any

    timestamp: float

    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateValue':
        """从字典创建"""
        return cls(
            value=data['value'],
            timestamp=data['timestamp'],
            version=data.get('version', 1)
        )


class StateBackend(ABC):
    """
    状态后端基类

    定义状态存储的标准接口。
    """

    @abstractmethod
    def get(self, key: str) -> Optional[StateValue]:
        """
        获取状态

        Args:
            key: 状态键

        Returns:
            Optional[StateValue]: 状态值
        """
        pass

    @abstractmethod
    def put(self, key: str, value: Any) -> bool:
        """
        存储状态

        Args:
            key: 状态键
            value: 状态值

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除状态

        Args:
            key: 状态键

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def list_keys(self) -> list:
        """
        列出所有键

        Returns:
            list: 键列表
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有状态"""
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        创建快照

        Returns:
            Dict: 快照数据
        """
        pass

    @abstractmethod
    def restore(self, snapshot: Dict[str, Any]):
        """
        从快照恢复

        Args:
            snapshot: 快照数据
        """
        pass


class MemoryStateBackend(StateBackend):
    """
    内存状态后端

    将状态存储在内存中。
    """

    def __init__(self):
        """初始化内存状态后端"""
        self._state: Dict[str, StateValue] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[StateValue]:
        """获取状态"""
        with self._lock:
            return self._state.get(key)

    def put(self, key: str, value: Any) -> bool:
        """存储状态"""
        import time
        with self._lock:
            self._state[key] = StateValue(
                value=value,
                timestamp=time.time()
            )
            return True

    def delete(self, key: str) -> bool:
        """删除状态"""
        with self._lock:
            if key in self._state:
                del self._state[key]
                return True
            return False

    def list_keys(self) -> list:
        """列出所有键"""
        with self._lock:
            return list(self._state.keys())

    def clear(self):
        """清空所有状态"""
        with self._lock:
            self._state.clear()

    def snapshot(self) -> Dict[str, Any]:
        """创建快照"""
        with self._lock:
            return {
                key: value.to_dict()
                for key, value in self._state.items()
            }

    def restore(self, snapshot: Dict[str, Any]):
        """从快照恢复"""
        with self._lock:
            self._state.clear()
            for key, value_dict in snapshot.items():
                self._state[key] = StateValue.from_dict(value_dict)


class FileSystemStateBackend(StateBackend):
    """
    文件系统状态后端

    将状态持久化到文件系统。
    """

    def __init__(self, base_path: str):
        """
        初始化文件系统状态后端

        Args:
            base_path: 基础路径
        """
        self._base_path = base_path
        self._state_file = os.path.join(base_path, 'state.json')
        self._state: Dict[str, StateValue] = {}
        self._lock = threading.Lock()

        os.makedirs(base_path, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """加载状态"""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value_dict in data.items():
                        self._state[key] = StateValue.from_dict(value_dict)
            except Exception:
                self._state = {}

    def _save_state(self):
        """保存状态"""
        data = {
            key: value.to_dict()
            for key, value in self._state.items()
        }
        with open(self._state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def get(self, key: str) -> Optional[StateValue]:
        """获取状态"""
        with self._lock:
            return self._state.get(key)

    def put(self, key: str, value: Any) -> bool:
        """存储状态"""
        import time
        with self._lock:
            self._state[key] = StateValue(
                value=value,
                timestamp=time.time()
            )
            self._save_state()
            return True

    def delete(self, key: str) -> bool:
        """删除状态"""
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._save_state()
                return True
            return False

    def list_keys(self) -> list:
        """列出所有键"""
        with self._lock:
            return list(self._state.keys())

    def clear(self):
        """清空所有状态"""
        with self._lock:
            self._state.clear()
            self._save_state()

    def snapshot(self) -> Dict[str, Any]:
        """创建快照"""
        with self._lock:
            return {
                key: value.to_dict()
                for key, value in self._state.items()
            }

    def restore(self, snapshot: Dict[str, Any]):
        """从快照恢复"""
        with self._lock:
            self._state.clear()
            for key, value_dict in snapshot.items():
                self._state[key] = StateValue.from_dict(value_dict)
            self._save_state()


class KeyValueState:
    """
    键值状态

    提供键值对状态访问。
    """

    def __init__(self, backend: StateBackend, namespace: str = 'default'):
        """
        初始化键值状态

        Args:
            backend: 状态后端
            namespace: 命名空间
        """
        self._backend = backend
        self._namespace = namespace

    def _get_full_key(self, key: str) -> str:
        """获取完整键"""
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        获取值

        Args:
            key: 键

        Returns:
            Optional[Any]: 值
        """
        state_value = self._backend.get(self._get_full_key(key))
        return state_value.value if state_value else None

    def put(self, key: str, value: Any):
        """
        存储值

        Args:
            key: 键
            value: 值
        """
        self._backend.put(self._get_full_key(key), value)

    def delete(self, key: str):
        """
        删除值

        Args:
            key: 键
        """
        self._backend.delete(self._get_full_key(key))

    def contains(self, key: str) -> bool:
        """
        是否包含键

        Args:
            key: 键

        Returns:
            bool: 是否包含
        """
        return self._backend.get(self._get_full_key(key)) is not None


class ListState:
    """
    列表状态

    提供列表状态访问。
    """

    def __init__(self, backend: StateBackend, namespace: str = 'default'):
        """
        初始化列表状态

        Args:
            backend: 状态后端
            namespace: 命名空间
        """
        self._backend = backend
        self._namespace = namespace
        self._key = f"{namespace}:list"

    def add(self, value: Any):
        """
        添加元素

        Args:
            value: 元素值
        """
        current = self.get()
        if current is None:
            current = []
        current.append(value)
        self._backend.put(self._key, current)

    def get(self) -> Optional[list]:
        """
        获取列表

        Returns:
            Optional[list]: 列表
        """
        state_value = self._backend.get(self._key)
        return state_value.value if state_value else None

    def update(self, values: list):
        """
        更新列表

        Args:
            values: 新列表
        """
        self._backend.put(self._key, values)

    def clear(self):
        """清空列表"""
        self._backend.delete(self._key)
