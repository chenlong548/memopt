"""
stream_processor 集成层

提供与外部模块的集成功能。
"""

from .compressor_integration import (
    CompressionIntegration,
    CompressionIntegrationConfig,
)

from .memory_integration import (
    MemoryIntegration,
    MemoryIntegrationConfig,
    BufferPool,
)

__all__ = [
    'CompressionIntegration',
    'CompressionIntegrationConfig',
    'MemoryIntegration',
    'MemoryIntegrationConfig',
    'BufferPool',
]
