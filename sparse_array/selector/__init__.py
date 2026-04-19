"""
sparse_array 格式选择模块

提供自动格式选择功能。
"""

from .features import (
    extract_features,
    analyze_sparsity,
    analyze_structure,
    analyze_distribution
)
from .auto_select import (
    select_format,
    select_format_ml,
    FormatSelector,
    recommend_format
)

__all__ = [
    'extract_features',
    'analyze_sparsity',
    'analyze_structure',
    'analyze_distribution',
    'select_format',
    'select_format_ml',
    'FormatSelector',
    'recommend_format'
]
