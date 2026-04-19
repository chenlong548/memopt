"""
data_compressor 核心基础接口

定义压缩器和解压器的基础接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import time


class CompressionAlgorithm(Enum):
    """压缩算法类型"""
    ZSTD = "zstd"           # Zstandard - 最佳平衡
    LZ4 = "lz4"             # LZ4 - 最快速度
    BROTLI = "brotli"       # Brotli - 最佳压缩比
    BF16_MODEL = "bf16"     # BF16模型专用
    FP32_MODEL = "fp32"     # FP32模型专用
    KV_CACHE = "kv_cache"   # KV Cache专用
    LEXICO = "lexico"       # Lexico字典压缩
    AUTO = "auto"           # 自动选择


class DataType(Enum):
    """数据类型"""
    GENERIC = "generic"             # 通用数据
    MODEL_WEIGHTS = "model_weights" # 模型权重
    BF16_TENSOR = "bf16_tensor"     # BF16张量
    FP32_TENSOR = "fp32_tensor"     # FP32张量
    KV_CACHE = "kv_cache"           # KV Cache
    SPARSE_MATRIX = "sparse_matrix" # 稀疏矩阵
    TEXT = "text"                   # 文本数据
    BINARY = "binary"               # 二进制数据
    JSON = "json"                   # JSON数据
    NUMPY_ARRAY = "numpy_array"     # NumPy数组


class CompressionLevel(Enum):
    """压缩级别"""
    FASTEST = 1          # 最快速度
    FAST = 3             # 快速
    BALANCED = 5         # 平衡
    HIGH = 7             # 高压缩比
    MAXIMUM = 9          # 最大压缩比
    AUTO = -1            # 自动选择


@dataclass
class CompressionConfig:
    """
    压缩配置

    定义压缩操作的配置选项。
    """

    algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO

    level: CompressionLevel = CompressionLevel.BALANCED

    data_type: DataType = DataType.GENERIC

    enable_parallel: bool = True
    enable_streaming: bool = False
    chunk_size: int = 1024 * 1024
    num_workers: int = 4

    max_memory_usage: int = 100 * 1024 * 1024
    use_memory_mapping: bool = False

    preserve_structure: bool = True
    enable_deduplication: bool = True
    enable_delta_encoding: bool = False

    enable_gpu: bool = False
    gpu_device: int = 0

    enable_stats: bool = True
    verbose: bool = False

    def __post_init__(self):
        """验证配置参数"""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
        if self.max_memory_usage <= 0:
            raise ValueError(f"max_memory_usage must be positive, got {self.max_memory_usage}")


@dataclass
class CompressionStats:
    """
    压缩统计信息

    记录压缩操作的性能指标。
    """

    # 基本指标
    original_size: int = 0                # 原始大小（字节）
    compressed_size: int = 0              # 压缩后大小（字节）
    compression_ratio: float = 0.0        # 压缩比

    # 时间指标
    compression_time: float = 0.0         # 压缩时间（秒）
    decompression_time: float = 0.0       # 解压时间（秒）
    throughput_mbps: float = 0.0          # 吞吐量（MB/s）

    # 算法信息
    algorithm_used: CompressionAlgorithm = CompressionAlgorithm.AUTO
    level_used: CompressionLevel = CompressionLevel.BALANCED

    # 内存指标
    peak_memory_usage: int = 0            # 峰值内存使用（字节）

    # 高级指标
    entropy: float = 0.0                  # 数据熵
    redundancy: float = 0.0               # 冗余度
    dedup_ratio: float = 0.0              # 去重比例

    # 时间戳
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def calculate_ratio(self):
        """计算压缩比"""
        if self.original_size > 0:
            self.compression_ratio = self.original_size / self.compressed_size

    def calculate_throughput(self):
        """计算吞吐量"""
        if self.compression_time > 0:
            self.throughput_mbps = (self.original_size / (1024 * 1024)) / self.compression_time

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'original_size_mb': self.original_size / (1024 * 1024),
            'compressed_size_mb': self.compressed_size / (1024 * 1024),
            'compression_ratio': f"{self.compression_ratio:.2f}x",
            'compression_time_ms': self.compression_time * 1000,
            'throughput_mbps': f"{self.throughput_mbps:.2f}",
            'algorithm': self.algorithm_used.value,
            'level': self.level_used.value,
            'peak_memory_mb': self.peak_memory_usage / (1024 * 1024),
            'entropy': f"{self.entropy:.4f}",
            'redundancy': f"{self.redundancy:.4f}"
        }


@dataclass
class CompressedData:
    """
    压缩数据容器

    存储压缩后的数据及其元数据。
    """

    # 压缩数据
    data: bytes

    # 元数据
    algorithm: CompressionAlgorithm
    level: CompressionLevel
    original_size: int
    compressed_size: int

    # 数据类型信息
    data_type: DataType = DataType.GENERIC
    shape: Optional[Tuple[int, ...]] = None      # 数组形状
    dtype: Optional[str] = None                   # 数据类型

    # 校验信息
    checksum: Optional[str] = None                # 校验和
    version: str = "1.0"                          # 格式版本

    # 统计信息
    stats: Optional[CompressionStats] = None

    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return {
            'algorithm': self.algorithm.value,
            'level': self.level.value,
            'original_size': self.original_size,
            'compressed_size': self.compressed_size,
            'data_type': self.data_type.value,
            'shape': self.shape,
            'dtype': self.dtype,
            'checksum': self.checksum,
            'version': self.version
        }


class CompressorBase(ABC):
    """
    压缩器基类

    定义压缩器的标准接口。
    """

    @abstractmethod
    def compress(self,
                 data: Union[bytes, bytearray, memoryview],
                 config: Optional[CompressionConfig] = None) -> CompressedData:
        """
        压缩数据

        Args:
            data: 待压缩数据
            config: 压缩配置

        Returns:
            CompressedData: 压缩后的数据容器
        """
        pass

    @abstractmethod
    def decompress(self, compressed: CompressedData) -> bytes:
        """
        解压数据

        Args:
            compressed: 压缩数据容器

        Returns:
            bytes: 解压后的数据
        """
        pass

    @abstractmethod
    def get_algorithm(self) -> CompressionAlgorithm:
        """
        获取算法类型

        Returns:
            CompressionAlgorithm: 算法类型
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取算法能力

        Returns:
            Dict: 算法能力描述
        """
        pass


class StreamCompressorBase(ABC):
    """
    流式压缩器基类

    定义流式压缩的标准接口。
    """

    @abstractmethod
    def compress_stream(self,
                       input_stream: Any,
                       output_stream: Any,
                       config: Optional[CompressionConfig] = None) -> CompressionStats:
        """
        流式压缩

        Args:
            input_stream: 输入流
            output_stream: 输出流
            config: 压缩配置

        Returns:
            CompressionStats: 压缩统计信息
        """
        pass

    @abstractmethod
    def decompress_stream(self,
                         input_stream: Any,
                         output_stream: Any) -> CompressionStats:
        """
        流式解压

        Args:
            input_stream: 输入流
            output_stream: 输出流

        Returns:
            CompressionStats: 解压统计信息
        """
        pass


class DataTypeDetectorBase(ABC):
    """
    数据类型检测器基类

    定义数据类型检测的标准接口。
    """

    @abstractmethod
    def detect(self, data: Union[bytes, bytearray, memoryview]) -> DataType:
        """
        检测数据类型

        Args:
            data: 待检测数据

        Returns:
            DataType: 数据类型
        """
        pass

    @abstractmethod
    def analyze(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """
        分析数据特征

        Args:
            data: 待分析数据

        Returns:
            Dict: 数据特征分析结果
        """
        pass


class AlgorithmSelectorBase(ABC):
    """
    算法选择器基类

    定义算法选择的标准接口。
    """

    @abstractmethod
    def select(self,
              data_features: Dict[str, Any],
              config: CompressionConfig) -> CompressionAlgorithm:
        """
        选择最优算法

        Args:
            data_features: 数据特征
            config: 压缩配置

        Returns:
            CompressionAlgorithm: 选择的算法
        """
        pass

    @abstractmethod
    def update(self,
              algorithm: CompressionAlgorithm,
              performance: CompressionStats):
        """
        更新算法性能数据（用于在线学习）

        Args:
            algorithm: 算法类型
            performance: 性能统计
        """
        pass
