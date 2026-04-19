"""
stream_processor 序列化工具

提供数据序列化和反序列化功能。
"""

from typing import Any, Dict, Optional
import json
import pickle
import struct
from abc import ABC, abstractmethod

from ..core.exceptions import SerializationError, DeserializationError


class Serializer(ABC):
    """
    序列化器基类

    定义序列化的标准接口。
    """

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """
        序列化对象

        Args:
            obj: 对象

        Returns:
            bytes: 序列化数据
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """
        反序列化数据

        Args:
            data: 序列化数据

        Returns:
            Any: 对象
        """
        pass


class JsonSerializer(Serializer):
    """
    JSON序列化器

    使用JSON格式进行序列化。
    """

    def serialize(self, obj: Any) -> bytes:
        """序列化对象"""
        try:
            return json.dumps(obj, ensure_ascii=False).encode('utf-8')
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}") from e

    def deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            raise DeserializationError(f"JSON deserialization failed: {e}") from e


class PickleSerializer(Serializer):
    """
    Pickle序列化器

    使用Pickle格式进行序列化。

    警告：Pickle反序列化可能存在安全风险，仅用于可信数据源。
    对于不可信数据，请使用JSONSerializer。
    """

    # 安全模式下允许的模块和类
    SAFE_MODULES = frozenset({
        'builtins', 'collections', 'datetime', 'decimal', 
        'collections.abc', 'enum', 'typing'
    })
    
    SAFE_CLASSES = frozenset({
        'dict', 'list', 'tuple', 'set', 'frozenset',
        'str', 'int', 'float', 'bool', 'bytes', 'bytearray',
        'NoneType', 'type', 'complex',
        'date', 'datetime', 'time', 'timedelta',
        'Decimal', 'OrderedDict', 'defaultdict', 'Counter',
        'Enum', 'IntEnum'
    })

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL, safe_mode: bool = True):
        """
        初始化Pickle序列化器

        Args:
            protocol: Pickle协议版本
            safe_mode: 安全模式，启用后将限制可反序列化的类型（默认启用）
        """
        self._protocol = protocol
        self._safe_mode = safe_mode

    def serialize(self, obj: Any) -> bytes:
        """序列化对象"""
        try:
            # 输入验证：检查对象大小，防止序列化过大的对象
            data = pickle.dumps(obj, protocol=self._protocol)
            max_size = 100 * 1024 * 1024  # 100MB
            if len(data) > max_size:
                raise SerializationError(
                    f"Serialized data too large: {len(data)} bytes (max: {max_size})"
                )
            return data
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {e}") from e

    def deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            # 输入验证
            if not isinstance(data, bytes):
                raise DeserializationError("Input must be bytes")
            
            if len(data) == 0:
                raise DeserializationError("Input data is empty")
            
            # 防止反序列化过大的数据
            max_size = 100 * 1024 * 1024  # 100MB
            if len(data) > max_size:
                raise DeserializationError(
                    f"Input data too large: {len(data)} bytes (max: {max_size})"
                )
            
            if self._safe_mode:
                import io
                class RestrictedUnpickler(pickle.Unpickler):
                    def find_class(inner_self, module, name):
                        # 检查模块和类是否在白名单中
                        if module not in PickleSerializer.SAFE_MODULES:
                            if name not in PickleSerializer.SAFE_CLASSES:
                                raise DeserializationError(
                                    f"Unsafe deserialization blocked: {module}.{name}. "
                                    f"If you trust this data, use safe_mode=False"
                                )
                        return super().find_class(module, name)
                
                return RestrictedUnpickler(io.BytesIO(data)).load()
            return pickle.loads(data)
        except DeserializationError:
            raise
        except Exception as e:
            raise DeserializationError(f"Pickle deserialization failed: {e}") from e


class BinarySerializer(Serializer):
    """
    二进制序列化器

    使用二进制格式进行序列化。
    """

    def __init__(self, fallback_to_pickle: bool = False):
        """
        初始化二进制序列化器

        Args:
            fallback_to_pickle: 是否回退到pickle序列化（不推荐，存在安全风险）
        """
        self._fallback_to_pickle = fallback_to_pickle

    def serialize(self, obj: Any) -> bytes:
        """序列化对象"""
        try:
            if isinstance(obj, bytes):
                return obj
            elif isinstance(obj, str):
                return obj.encode('utf-8')
            elif isinstance(obj, int):
                return struct.pack('q', obj)
            elif isinstance(obj, float):
                return struct.pack('d', obj)
            elif self._fallback_to_pickle:
                # 使用安全的pickle序列化器
                safe_pickle = PickleSerializer(safe_mode=True)
                return safe_pickle.serialize(obj)
            else:
                raise SerializationError(
                    f"Cannot serialize type {type(obj).__name__} in binary mode. "
                    f"Use JSON or enable fallback_to_pickle (not recommended for untrusted data)"
                )
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"Binary serialization failed: {e}") from e

    def deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            # 输入验证
            if not isinstance(data, bytes):
                raise DeserializationError("Input must be bytes")
            return data
        except DeserializationError:
            raise
        except Exception as e:
            raise DeserializationError(f"Binary deserialization failed: {e}") from e


class CompositeSerializer(Serializer):
    """
    复合序列化器

    根据类型选择序列化器。
    """

    def __init__(self, default_serializer: str = 'json'):
        """
        初始化复合序列化器

        Args:
            default_serializer: 默认序列化器（推荐使用 'json' 以确保安全）
        """
        self._serializers: Dict[str, Serializer] = {
            'json': JsonSerializer(),
            'pickle': PickleSerializer(safe_mode=True),
            'binary': BinarySerializer()
        }
        self._default_serializer = default_serializer

    def register_serializer(self, name: str, serializer: Serializer):
        """
        注册序列化器

        Args:
            name: 序列化器名称
            serializer: 序列化器
        """
        self._serializers[name] = serializer

    def serialize(self, obj: Any, serializer_name: Optional[str] = None) -> bytes:
        """序列化对象"""
        name = serializer_name or self._default_serializer

        if name not in self._serializers:
            raise SerializationError(f"Unknown serializer: {name}")

        serializer = self._serializers[name]
        data = serializer.serialize(obj)

        header = struct.pack('I', len(name)) + name.encode('utf-8')

        return header + data

    def deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            name_length = struct.unpack('I', data[:4])[0]
            name = data[4:4 + name_length].decode('utf-8')
            content = data[4 + name_length:]

            if name not in self._serializers:
                raise DeserializationError(f"Unknown serializer: {name}")

            serializer = self._serializers[name]
            return serializer.deserialize(content)

        except Exception as e:
            raise DeserializationError(f"Composite deserialization failed: {e}") from e


class SerializationContext:
    """
    序列化上下文

    管理序列化配置和状态。
    """

    def __init__(self, default_serializer: str = 'pickle'):
        """
        初始化序列化上下文

        Args:
            default_serializer: 默认序列化器
        """
        self._composite_serializer = CompositeSerializer()
        self._default_serializer = default_serializer

    def serialize(self, obj: Any, serializer_name: Optional[str] = None) -> bytes:
        """序列化对象"""
        name = serializer_name or self._default_serializer
        return self._composite_serializer.serialize(obj, name)

    def deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        return self._composite_serializer.deserialize(data)

    def register_serializer(self, name: str, serializer: Serializer):
        """注册序列化器"""
        self._composite_serializer.register_serializer(name, serializer)


def serialize(obj: Any, format: str = 'json') -> bytes:
    """
    便捷序列化函数

    Args:
        obj: 对象
        format: 格式（推荐使用 'json' 以确保安全）

    Returns:
        bytes: 序列化数据
    """
    serializers = {
        'json': JsonSerializer(),
        'pickle': PickleSerializer(safe_mode=True),
        'binary': BinarySerializer()
    }

    if format not in serializers:
        raise SerializationError(f"Unknown format: {format}")

    return serializers[format].serialize(obj)


def deserialize(data: bytes, format: str = 'json') -> Any:
    """
    便捷反序列化函数

    Args:
        data: 序列化数据
        format: 格式（推荐使用 'json' 以确保安全）

    Returns:
        Any: 对象
    """
    serializers = {
        'json': JsonSerializer(),
        'pickle': PickleSerializer(safe_mode=True),
        'binary': BinarySerializer()
    }

    if format not in serializers:
        raise DeserializationError(f"Unknown format: {format}")

    return serializers[format].deserialize(data)
