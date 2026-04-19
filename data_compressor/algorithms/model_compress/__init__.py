"""
模型压缩算法模块
"""

from .bf16_compress import BF16ModelCompressor
from .fp32_compress import FP32ModelCompressor

__all__ = [
    'BF16ModelCompressor',
    'FP32ModelCompressor',
]
