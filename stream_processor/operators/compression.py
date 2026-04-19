"""
stream_processor 压缩操作符

与data_compressor模块集成的压缩操作符。
"""

from typing import Any, List, Optional
import time

from .base import OneInputOperator, OperatorConfig, OperatorType, OperatorState
from ..core.record import Record
from ..core.execution_context import ExecutionContext
from ..core.exceptions import OperatorError

try:
    from data_compressor import (
        DataCompressor,
        CompressionConfig,
        CompressionAlgorithm,
        CompressionLevel,
        CompressionStats,
        CompressedData,
    )
    HAS_DATA_COMPRESSOR = True
except ImportError:
    HAS_DATA_COMPRESSOR = False
    DataCompressor = None
    CompressionConfig = None
    CompressionAlgorithm = None
    CompressionLevel = None
    CompressionStats = None
    CompressedData = None


class CompressionOperator(OneInputOperator):
    """
    压缩操作符

    对流数据进行压缩处理。
    """

    def __init__(self,
                 name: str,
                 algorithm: str = 'auto',
                 level: int = -1,
                 parallelism: int = 1):
        """
        初始化压缩操作符

        Args:
            name: 操作符名称
            algorithm: 压缩算法 ('auto' | 'zstd' | 'lz4' | 'brotli')
            level: 压缩级别 (-1表示自动)
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.COMPRESSION,
            parallelism=parallelism
        )
        super().__init__(config)
        self._algorithm = algorithm
        self._level = level
        self._compressor: Optional[DataCompressor] = None
        self._stats_list: List[CompressionStats] = []

        if not HAS_DATA_COMPRESSOR:
            raise ImportError(
                "data_compressor module is required for CompressionOperator. "
                "Please install it first."
            )

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            data = self._serialize_value(record.value)

            compressed = self._compressor.compress(data)

            self._stats_list.append(compressed.stats)

            compressed_record = Record(
                value={
                    'compressed_data': compressed.data,
                    'algorithm': compressed.algorithm.value,
                    'original_size': compressed.original_size,
                    'compressed_size': compressed.compressed_size,
                    'metadata': {
                        'key': record.key,
                        'timestamp': record.timestamp,
                        'headers': record.headers
                    }
                },
                key=record.key,
                timestamp=record.timestamp,
                headers=record.headers.copy()
            )

            return [compressed_record]

        except Exception as e:
            raise OperatorError(f"Compression failed: {e}") from e

    def _serialize_value(self, value: Any) -> bytes:
        """序列化值（安全模式）"""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return value.encode('utf-8')
        else:
            # 使用安全的序列化方式
            import json
            try:
                return json.dumps(value, ensure_ascii=False).encode('utf-8')
            except (TypeError, ValueError):
                # 如果JSON序列化失败，使用安全的pickle
                import pickle
                from ..utils.serialization import PickleSerializer
                serializer = PickleSerializer(safe_mode=True)
                return serializer.serialize(value)

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

        try:
            algo = CompressionAlgorithm(self._algorithm)
            level = CompressionLevel(self._level) if self._level > 0 else CompressionLevel.AUTO

            compression_config = CompressionConfig(
                algorithm=algo,
                level=level
            )

            self._compressor = DataCompressor(compression_config)
            self._stats_list.clear()

        except Exception as e:
            raise OperatorError(f"Failed to initialize compressor: {e}") from e

    def close(self):
        """关闭操作符"""
        self._compressor = None
        self._stats_list.clear()
        self.set_state(OperatorState.COMPLETED)

    def get_compression_stats(self) -> List[Any]:
        """
        获取压缩统计信息

        Returns:
            List: 压缩统计信息列表
        """
        return self._stats_list.copy()


class DecompressionOperator(OneInputOperator):
    """
    解压操作符

    对压缩数据进行解压处理。
    """

    def __init__(self,
                 name: str,
                 parallelism: int = 1):
        """
        初始化解压操作符

        Args:
            name: 操作符名称
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.COMPRESSION,
            parallelism=parallelism
        )
        super().__init__(config)
        self._compressor: Optional[DataCompressor] = None

        if not HAS_DATA_COMPRESSOR:
            raise ImportError(
                "data_compressor module is required for DecompressionOperator. "
                "Please install it first."
            )

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        try:
            if not isinstance(record.value, dict):
                raise OperatorError("Record value must be a dict with compressed data")

            compressed_data = record.value.get('compressed_data')
            algorithm = record.value.get('algorithm')
            original_size = record.value.get('original_size')
            metadata = record.value.get('metadata', {})

            compressed = CompressedData(
                data=compressed_data,
                algorithm=CompressionAlgorithm(algorithm),
                level=CompressionLevel.BALANCED,
                original_size=original_size,
                compressed_size=len(compressed_data)
            )

            decompressed = self._compressor.decompress(compressed)

            value = self._deserialize_value(decompressed)

            decompressed_record = Record(
                value=value,
                key=metadata.get('key'),
                timestamp=metadata.get('timestamp', record.timestamp),
                headers=metadata.get('headers', record.headers.copy())
            )

            return [decompressed_record]

        except Exception as e:
            raise OperatorError(f"Decompression failed: {e}") from e

    def _deserialize_value(self, data: bytes) -> Any:
        """反序列化值（安全模式）"""
        try:
            import json
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return data

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)
        self._compressor = DataCompressor()

    def close(self):
        """关闭操作符"""
        self._compressor = None
        self.set_state(OperatorState.COMPLETED)


class StreamCompressionOperator(OneInputOperator):
    """
    流式压缩操作符

    对流数据进行批量压缩。
    """

    def __init__(self,
                 name: str,
                 batch_size: int = 100,
                 algorithm: str = 'auto',
                 level: int = -1,
                 parallelism: int = 1):
        """
        初始化流式压缩操作符

        Args:
            name: 操作符名称
            batch_size: 批量大小
            algorithm: 压缩算法
            level: 压缩级别
            parallelism: 并行度
        """
        config = OperatorConfig(
            name=name,
            operator_type=OperatorType.COMPRESSION,
            parallelism=parallelism
        )
        super().__init__(config)
        self._batch_size = batch_size
        self._algorithm = algorithm
        self._level = level
        self._compressor: Optional[DataCompressor] = None
        self._batch: List[Record] = []

        if not HAS_DATA_COMPRESSOR:
            raise ImportError(
                "data_compressor module is required for StreamCompressionOperator. "
                "Please install it first."
            )

    def process_element(self, record: Record) -> List[Record]:
        """处理元素"""
        self._batch.append(record)

        if len(self._batch) >= self._batch_size:
            return self._flush_batch()

        return []

    def _flush_batch(self) -> List[Record]:
        """刷新批量数据"""
        if not self._batch:
            return []

        try:
            # 使用安全的序列化方式
            import json
            from ..utils.serialization import PickleSerializer
            
            batch_data_list = [r.to_dict() for r in self._batch]
            
            # 优先使用JSON序列化
            try:
                batch_data = json.dumps(batch_data_list, ensure_ascii=False).encode('utf-8')
            except (TypeError, ValueError):
                # 如果JSON序列化失败，使用安全的pickle
                serializer = PickleSerializer(safe_mode=True)
                batch_data = serializer.serialize(batch_data_list)

            compressed = self._compressor.compress(batch_data)

            result_record = Record(
                value={
                    'compressed_batch': compressed.data,
                    'algorithm': compressed.algorithm.value,
                    'batch_size': len(self._batch),
                    'original_size': compressed.original_size,
                    'compressed_size': compressed.compressed_size
                }
            )

            self._batch.clear()

            return [result_record]

        except Exception as e:
            self._batch.clear()
            raise OperatorError(f"Batch compression failed: {e}") from e

    def open(self, context: ExecutionContext):
        """打开操作符"""
        self.set_context(context)
        self.set_state(OperatorState.RUNNING)

        algo = CompressionAlgorithm(self._algorithm)
        level = CompressionLevel(self._level) if self._level > 0 else CompressionLevel.AUTO

        compression_config = CompressionConfig(
            algorithm=algo,
            level=level
        )

        self._compressor = DataCompressor(compression_config)
        self._batch.clear()

    def close(self):
        """关闭操作符"""
        if self._batch:
            self._flush_batch()

        self._compressor = None
        self._batch.clear()
        self.set_state(OperatorState.COMPLETED)
