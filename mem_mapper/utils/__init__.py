"""
mem_mapper 工具模块

提供对齐和统计等工具函数。
"""

from .alignment import (
    PAGE_SIZE_4KB,
    PAGE_SIZE_2MB,
    PAGE_SIZE_1GB,
    PGD_ALIGNMENT,
    align_up,
    align_down,
    is_aligned,
    align_to_page,
    align_to_huge_page,
    align_to_pgd,
    calculate_padding,
    calculate_pages,
    calculate_huge_pages,
    find_optimal_alignment,
    align_offset_and_size,
    get_alignment_waste,
    get_alignment_efficiency,
    is_power_of_two,
    next_power_of_two,
    previous_power_of_two,
    align_address_to_page,
    is_address_page_aligned,
    calculate_page_range,
    format_size,
    parse_size,
)

from .stats import (
    TimingRecord,
    PerformanceMetrics,
    Timer,
    PerformanceTracker,
    MemoryUsageTracker,
    AccessPatternAnalyzer,
    Benchmark,
)

__all__ = [
    # 对齐常量
    'PAGE_SIZE_4KB',
    'PAGE_SIZE_2MB',
    'PAGE_SIZE_1GB',
    'PGD_ALIGNMENT',
    
    # 对齐函数
    'align_up',
    'align_down',
    'is_aligned',
    'align_to_page',
    'align_to_huge_page',
    'align_to_pgd',
    'calculate_padding',
    'calculate_pages',
    'calculate_huge_pages',
    'find_optimal_alignment',
    'align_offset_and_size',
    'get_alignment_waste',
    'get_alignment_efficiency',
    'is_power_of_two',
    'next_power_of_two',
    'previous_power_of_two',
    'align_address_to_page',
    'is_address_page_aligned',
    'calculate_page_range',
    'format_size',
    'parse_size',
    
    # 统计类
    'TimingRecord',
    'PerformanceMetrics',
    'Timer',
    'PerformanceTracker',
    'MemoryUsageTracker',
    'AccessPatternAnalyzer',
    'Benchmark',
]
