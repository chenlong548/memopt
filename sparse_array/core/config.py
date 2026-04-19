"""
sparse_array 配置模块

定义稀疏数组的配置选项和枚举类型。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np


class SparseFormat(Enum):
    """稀疏存储格式"""
    CSR = "csr"           # 压缩稀疏行格式
    CSC = "csc"           # 压缩稀疏列格式
    COO = "coo"           # 坐标格式
    BCSR = "bcsr"         # 块CSR格式
    BITMAP = "bitmap"     # 位图格式
    AUTO = "auto"         # 自动选择


class ComputeBackend(Enum):
    """计算后端"""
    CPU = "cpu"           # CPU计算
    GPU_CUSPARSE = "gpu_cusparse"  # cuSPARSE GPU加速
    GPU_TENSOR_CORE = "gpu_tensor_core"  # Tensor Core优化
    AUTO = "auto"         # 自动选择


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"         # 无压缩
    LOW_RANK = "low_rank" # 块低秩压缩
    HSS = "hss"           # 层次半可分矩阵
    AUTO = "auto"         # 自动选择


class SelectionStrategy(Enum):
    """格式选择策略"""
    RULE_BASED = "rule_based"       # 基于规则
    ML_BASED = "ml_based"           # 基于机器学习
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    AUTO = "auto"                   # 自动选择


@dataclass
class SparseArrayConfig:
    """
    稀疏数组配置

    定义稀疏数组操作的配置选项。
    """

    # 存储格式
    format: SparseFormat = SparseFormat.AUTO

    # 计算后端
    backend: ComputeBackend = ComputeBackend.AUTO

    # GPU配置
    enable_gpu: bool = False
    gpu_device: int = 0

    # 压缩配置
    compression: CompressionType = CompressionType.NONE
    compression_threshold: float = 0.3  # 压缩阈值（稀疏度）
    low_rank_threshold: float = 1e-6    # 低秩截断阈值

    # 格式选择配置
    selection_strategy: SelectionStrategy = SelectionStrategy.RULE_BASED

    # 稀疏阈值
    sparsity_threshold: float = 0.5  # 稀疏度阈值，低于此值使用稀疏格式

    # 块大小（用于BCSR）
    block_size: Tuple[int, int] = (4, 4)

    # 内存限制
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB

    # 并行配置
    enable_parallel: bool = True
    num_threads: int = 4

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 100

    # 统计配置
    enable_stats: bool = True
    verbose: bool = False

    def __post_init__(self):
        """验证配置参数"""
        if not 0.0 <= self.sparsity_threshold <= 1.0:
            raise ValueError(f"sparsity_threshold must be in [0, 1], got {self.sparsity_threshold}")

        if not 0.0 <= self.compression_threshold <= 1.0:
            raise ValueError(f"compression_threshold must be in [0, 1], got {self.compression_threshold}")

        if self.block_size[0] <= 0 or self.block_size[1] <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

        if self.num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {self.num_threads}")

    def get_format_metadata(self) -> Dict[str, Any]:
        """获取格式元数据"""
        return {
            'format': self.format.value,
            'backend': self.backend.value,
            'compression': self.compression.value,
            'block_size': self.block_size,
            'sparsity_threshold': self.sparsity_threshold
        }


@dataclass
class SparseArrayStats:
    """
    稀疏数组统计信息

    记录稀疏数组操作的性能指标。
    """

    # 基本指标
    shape: Tuple[int, ...] = (0,)
    nnz: int = 0                          # 非零元素数量
    density: float = 0.0                  # 密度
    sparsity: float = 1.0                 # 稀疏度

    # 内存指标
    memory_usage: int = 0                 # 内存使用（字节）
    compression_ratio: float = 1.0        # 压缩比

    # 格式信息
    format_used: SparseFormat = SparseFormat.CSR
    backend_used: ComputeBackend = ComputeBackend.CPU

    # 性能指标
    operation_time: float = 0.0           # 操作时间（秒）
    throughput: float = 0.0               # 吞吐量

    # 压缩指标
    compression_type: CompressionType = CompressionType.NONE
    rank_approximation: int = 0           # 近似秩

    def calculate_density(self):
        """计算密度"""
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim
        if total_elements > 0:
            self.density = self.nnz / total_elements
            self.sparsity = 1.0 - self.density

    def calculate_compression_ratio(self, dense_size: int):
        """计算压缩比"""
        if dense_size > 0:
            self.compression_ratio = dense_size / self.memory_usage

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'shape': self.shape,
            'nnz': self.nnz,
            'density': f"{self.density:.6f}",
            'sparsity': f"{self.sparsity:.6f}",
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'compression_ratio': f"{self.compression_ratio:.2f}x",
            'format': self.format_used.value,
            'backend': self.backend_used.value,
            'operation_time_ms': self.operation_time * 1000
        }


@dataclass
class FormatFeatures:
    """
    格式特征

    用于格式选择的特征向量。
    """

    # 稀疏性特征
    sparsity: float = 0.0                 # 稀疏度
    density: float = 1.0                  # 密度

    # 结构特征
    bandwidth: int = 0                    # 带宽
    bandwidth_ratio: float = 0.0          # 带宽比
    diagonal_ratio: float = 0.0           # 对角线元素比例
    block_structure_score: float = 0.0    # 块结构评分

    # 分布特征
    row_nnz_variance: float = 0.0         # 行非零元素方差
    col_nnz_variance: float = 0.0         # 列非零元素方差
    max_row_nnz: int = 0                  # 最大行非零元素数
    max_col_nnz: int = 0                  # 最大列非零元素数
    is_regular: bool = False              # 是否规则分布

    # 形状特征
    aspect_ratio: float = 1.0             # 长宽比
    total_elements: int = 0               # 总元素数

    # 操作特征
    expected_operation: str = "spmv"      # 预期操作类型

    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.sparsity,
            self.density,
            self.bandwidth,
            self.diagonal_ratio,
            self.block_structure_score,
            self.row_nnz_variance,
            self.col_nnz_variance,
            self.max_row_nnz,
            self.max_col_nnz,
            self.aspect_ratio
        ])

    def get_summary(self) -> Dict[str, Any]:
        """获取特征摘要"""
        return {
            'sparsity': f"{self.sparsity:.4f}",
            'bandwidth': self.bandwidth,
            'block_structure_score': f"{self.block_structure_score:.4f}",
            'aspect_ratio': f"{self.aspect_ratio:.2f}",
            'expected_operation': self.expected_operation
        }
