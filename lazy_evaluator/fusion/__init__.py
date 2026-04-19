"""
融合优化层 (Fusion Layer)

提供流操作融合和管道优化功能。
"""

from .stream_fusion import StreamFusion
from .pipeline import LazyPipeline

__all__ = [
    "StreamFusion",
    "LazyPipeline",
]
