"""
mem_optimizer 集成模块

提供与其他模块的集成功能。
"""

from .mem_mapper import MemMapperIntegration, MappedMemoryInfo
from .data_compressor import DataCompressorIntegration, CompressedMemoryBlock

__all__ = [
    'MemMapperIntegration',
    'MappedMemoryInfo',
    'DataCompressorIntegration',
    'CompressedMemoryBlock'
]
