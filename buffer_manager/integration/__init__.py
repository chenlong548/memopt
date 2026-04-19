"""
Integration模块 - 集成层

提供与其他模块的适配器。
"""

from .stream_adapter import StreamAdapter
from .compressor_adapter import CompressorAdapter

__all__ = [
    "StreamAdapter",
    "CompressorAdapter",
]
