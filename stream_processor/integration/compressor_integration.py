"""
stream_processor 压缩集成

与data_compressor模块集成。
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

from ..core.record import Record
from ..core.exceptions import OperatorError

try:
    from data_compressor import (
        DataCompressor,
        StreamCompressor,
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
    StreamCompressor = None
    CompressionConfig = None
    CompressionAlgorithm = None
    CompressionLevel = None
    CompressionStats = None
    CompressedData = None


logger = logging.getLogger(__name__)


@dataclass
class CompressionIntegrationConfig:
    """
    压缩集成配置

    定义压缩集成的配置参数。
    """

    algorithm: str = 'auto'

    level: int = -1

    enable_streaming: bool = True

    chunk_size: int = 1024 * 1024

    enable_metrics: bool = True


class CompressionIntegration:
    """
    压缩集成

    提供与data_compressor模块的集成功能。
    """

    def __init__(self, config: Optional[CompressionIntegrationConfig] = None):
        """
        初始化压缩集成

        Args:
            config: 压缩集成配置
        """
        if not HAS_DATA_COMPRESSOR:
            raise ImportError(
                "data_compressor module is required. "
                "Please install it first."
            )

        self._config = config or CompressionIntegrationConfig()
        self._compressor: Optional[DataCompressor] = None  # type: ignore
        self._stream_compressor: Optional[StreamCompressor] = None  # type: ignore
        self._stats: List[CompressionStats] = []  # type: ignore

        self._init_compressor()

    def _init_compressor(self):
        """初始化压缩器"""
        try:
            algo = CompressionAlgorithm(self._config.algorithm)  # type: ignore
            level = CompressionLevel(self._config.level) if self._config.level > 0 else CompressionLevel.AUTO  # type: ignore

            compression_config = CompressionConfig(
                algorithm=algo,  # type: ignore
                level=level,  # type: ignore
                enable_streaming=self._config.enable_streaming,  # type: ignore
                chunk_size=self._config.chunk_size  # type: ignore
            )  # type: ignore

            self._compressor = DataCompressor(compression_config)  # type: ignore

            if self._config.enable_streaming:
                self._stream_compressor = StreamCompressor(compression_config)  # type: ignore

            logger.info("Compression integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize compression integration: {e}")
            raise

    def compress_record(self, record: Record) -> Record:
        """
        压缩记录

        Args:
            record: 数据记录

        Returns:
            Record: 压缩后的记录
        """
        try:
            data = self._serialize(record.value)

            compressed = self._compressor.compress(data)  # type: ignore

            if self._config.enable_metrics:
                self._stats.append(compressed.stats)

            return Record(
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

        except Exception as e:
            logger.error(f"Failed to compress record: {e}")
            raise OperatorError(f"Compression failed: {e}") from e

    def decompress_record(self, record: Record) -> Record:
        """
        解压记录

        Args:
            record: 压缩记录

        Returns:
            Record: 解压后的记录
        """
        try:
            if not isinstance(record.value, dict):
                raise OperatorError("Record value must be a dict with compressed data")

            compressed_data = record.value.get('compressed_data')
            algorithm = record.value.get('algorithm')
            original_size = record.value.get('original_size')
            metadata = record.value.get('metadata', {})

            compressed = CompressedData(
                data=compressed_data,  # type: ignore
                algorithm=CompressionAlgorithm(algorithm),  # type: ignore
                level=CompressionLevel.BALANCED,  # type: ignore
                original_size=original_size,  # type: ignore
                compressed_size=len(compressed_data)  # type: ignore
            )  # type: ignore

            decompressed = self._compressor.decompress(compressed)  # type: ignore

            value = self._deserialize(decompressed)

            return Record(
                value=value,
                key=metadata.get('key'),
                timestamp=metadata.get('timestamp', record.timestamp),
                headers=metadata.get('headers', record.headers.copy())
            )

        except Exception as e:
            logger.error(f"Failed to decompress record: {e}")
            raise OperatorError(f"Decompression failed: {e}") from e

    def compress_batch(self, records: List[Record]) -> Record:
        """
        批量压缩

        Args:
            records: 记录列表

        Returns:
            Record: 压缩后的记录
        """
        try:
            import pickle
            batch_data = pickle.dumps([r.to_dict() for r in records])

            compressed = self._compressor.compress(batch_data)  # type: ignore

            if self._config.enable_metrics:
                self._stats.append(compressed.stats)

            return Record(
                value={
                    'compressed_batch': compressed.data,
                    'algorithm': compressed.algorithm.value,
                    'batch_size': len(records),
                    'original_size': compressed.original_size,
                    'compressed_size': compressed.compressed_size
                }
            )

        except Exception as e:
            logger.error(f"Failed to compress batch: {e}")
            raise OperatorError(f"Batch compression failed: {e}") from e

    def decompress_batch(self, record: Record) -> List[Record]:
        """
        批量解压

        Args:
            record: 压缩记录

        Returns:
            List[Record]: 解压后的记录列表
        """
        try:
            if not isinstance(record.value, dict):
                raise OperatorError("Record value must be a dict with compressed batch")

            compressed_data = record.value.get('compressed_batch')
            algorithm = record.value.get('algorithm')
            original_size = record.value.get('original_size')

            compressed = CompressedData(
                data=compressed_data,  # type: ignore
                algorithm=CompressionAlgorithm(algorithm),  # type: ignore
                level=CompressionLevel.BALANCED,  # type: ignore
                original_size=original_size,  # type: ignore
                compressed_size=len(compressed_data)  # type: ignore
            )  # type: ignore

            decompressed = self._compressor.decompress(compressed)  # type: ignore

            import pickle
            records_data = pickle.loads(decompressed)

            return [Record.from_dict(data) for data in records_data]

        except Exception as e:
            logger.error(f"Failed to decompress batch: {e}")
            raise OperatorError(f"Batch decompression failed: {e}") from e

    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return value.encode('utf-8')
        else:
            import pickle
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            import pickle
            return pickle.loads(data)
        except:
            try:
                return data.decode('utf-8')
            except:
                return data

    def get_stats(self) -> List[Dict[str, Any]]:
        """
        获取统计信息

        Returns:
            List[Dict]: 统计信息列表
        """
        if not self._stats:
            return []

        return [
            {
                'original_size': stat.original_size,
                'compressed_size': stat.compressed_size,
                'compression_ratio': stat.compression_ratio,
                'compression_time': stat.compression_time,
                'algorithm': stat.algorithm_used.value
            }
            for stat in self._stats
        ]

    def clear_stats(self):
        """清空统计信息"""
        self._stats.clear()

    def analyze_data(self, data: bytes) -> Dict[str, Any]:
        """
        分析数据

        Args:
            data: 数据

        Returns:
            Dict: 分析结果
        """
        return self._compressor.analyze(data)  # type: ignore

    def benchmark_algorithms(self, data: bytes) -> Dict[str, Any]:
        """
        基准测试算法

        Args:
            data: 测试数据

        Returns:
            Dict: 测试结果
        """
        return self._compressor.benchmark(data)  # type: ignore
