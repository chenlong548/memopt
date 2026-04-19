"""
集成层 (Integration Layer)

提供与其他模块的集成适配器。
"""

from .stream_adapter import StreamAdapter
from .compressor_adapter import CompressorAdapter
from .optimizer_adapter import OptimizerAdapter

__all__ = [
    "StreamAdapter",
    "CompressorAdapter",
    "OptimizerAdapter",
]
