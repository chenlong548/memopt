"""
配置管理模块

提供缓冲区管理器的配置类和默认配置。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BufferConfig:
    """缓冲区配置类"""
    
    # 基本配置
    buffer_size: int = 4096  # 默认缓冲区大小 4KB
    alignment: int = 64  # 内存对齐，默认64字节缓存行对齐
    
    # 缓冲池配置
    pool_size: int = 16  # 缓冲池中缓冲区数量
    pool_timeout: float = 5.0  # 获取缓冲区超时时间（秒）
    
    # 环形缓冲区配置
    ring_capacity: int = 65536  # 环形缓冲区默认容量 64KB
    
    # 水位线配置（百分比）
    watermark_low: float = 0.25  # 低水位线
    watermark_high: float = 0.75  # 高水位线
    watermark_critical: float = 0.90  # 临界水位线
    
    # 预取配置
    prefetch_enabled: bool = True  # 是否启用预取
    prefetch_size: int = 3  # 预取窗口大小
    
    # 替换策略配置
    cache_capacity: int = 1024  # 缓存容量
    replacement_policy: str = "arc"  # 替换策略: "lru" 或 "arc"
    
    # 监控配置
    metrics_enabled: bool = True  # 是否启用指标收集
    metrics_interval: float = 1.0  # 指标收集间隔（秒）
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(f"alignment must be power of 2, got {self.alignment}")
        
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {self.pool_size}")
        
        if self.ring_capacity <= 0 or (self.ring_capacity & (self.ring_capacity - 1)) != 0:
            raise ValueError(f"ring_capacity must be power of 2, got {self.ring_capacity}")
        
        if not (0 < self.watermark_low < self.watermark_high < self.watermark_critical <= 1.0):
            raise ValueError(
                f"Invalid watermark levels: low={self.watermark_low}, "
                f"high={self.watermark_high}, critical={self.watermark_critical}"
            )
        
        if self.replacement_policy not in ("lru", "arc"):
            raise ValueError(f"replacement_policy must be 'lru' or 'arc', got {self.replacement_policy}")
    
    @classmethod
    def default(cls) -> "BufferConfig":
        """获取默认配置"""
        return cls()
    
    @classmethod
    def high_performance(cls) -> "BufferConfig":
        """获取高性能配置"""
        return cls(
            buffer_size=65536,  # 64KB
            alignment=64,
            pool_size=32,
            ring_capacity=1048576,  # 1MB
            prefetch_enabled=True,
            prefetch_size=5,
            cache_capacity=4096,
            metrics_enabled=True,
        )
    
    @classmethod
    def low_memory(cls) -> "BufferConfig":
        """获取低内存配置"""
        return cls(
            buffer_size=1024,  # 1KB
            alignment=64,
            pool_size=4,
            ring_capacity=4096,  # 4KB
            prefetch_enabled=False,
            cache_capacity=256,
            metrics_enabled=False,
        )
