"""
mem_mapper 对齐工具模块

提供内存对齐相关的工具函数。
"""

from typing import Tuple
from ..core.exceptions import AlignmentError


# 常用页面大小常量
PAGE_SIZE_4KB = 4 * 1024           # 4KB - 标准页面
PAGE_SIZE_2MB = 2 * 1024 * 1024    # 2MB - 大页
PAGE_SIZE_1GB = 1024 * 1024 * 1024 # 1GB - 超大页

# PGD对齐大小（用于共享页表）
PGD_ALIGNMENT = 512 * 1024 * 1024 * 1024  # 512GB


def align_up(value: int, alignment: int) -> int:
    """
    向上对齐到指定边界
    
    Args:
        value: 需要对齐的值
        alignment: 对齐边界（必须是2的幂）
        
    Returns:
        对齐后的值
        
    Raises:
        AlignmentError: 对齐边界不是2的幂时抛出
    """
    if alignment <= 0:
        raise AlignmentError("Alignment must be positive", value, alignment)
    
    if (alignment & (alignment - 1)) != 0:
        raise AlignmentError("Alignment must be power of 2", value, alignment)
    
    return (value + alignment - 1) & ~(alignment - 1)


def align_down(value: int, alignment: int) -> int:
    """
    向下对齐到指定边界
    
    Args:
        value: 需要对齐的值
        alignment: 对齐边界（必须是2的幂）
        
    Returns:
        对齐后的值
        
    Raises:
        AlignmentError: 对齐边界不是2的幂时抛出
    """
    if alignment <= 0:
        raise AlignmentError("Alignment must be positive", value, alignment)
    
    if (alignment & (alignment - 1)) != 0:
        raise AlignmentError("Alignment must be power of 2", value, alignment)
    
    return value & ~(alignment - 1)


def is_aligned(value: int, alignment: int) -> bool:
    """
    检查值是否已对齐
    
    Args:
        value: 需要检查的值
        alignment: 对齐边界
        
    Returns:
        是否已对齐
    """
    if alignment <= 0:
        return False
    
    return (value & (alignment - 1)) == 0


def align_to_page(size: int, page_size: int = PAGE_SIZE_4KB) -> int:
    """
    将大小对齐到页面边界
    
    Args:
        size: 原始大小
        page_size: 页面大小
        
    Returns:
        对齐后的大小
    """
    return align_up(size, page_size)


def align_to_huge_page(size: int, huge_page_size: int = PAGE_SIZE_2MB) -> int:
    """
    将大小对齐到大页边界
    
    Args:
        size: 原始大小
        huge_page_size: 大页大小
        
    Returns:
        对齐后的大小
    """
    return align_up(size, huge_page_size)


def align_to_pgd(size: int) -> int:
    """
    将大小对齐到PGD边界（512GB）
    
    用于共享页表映射。
    
    Args:
        size: 原始大小
        
    Returns:
        对齐后的大小
    """
    return align_up(size, PGD_ALIGNMENT)


def calculate_padding(value: int, alignment: int) -> int:
    """
    计算对齐所需的填充大小
    
    Args:
        value: 需要对齐的值
        alignment: 对齐边界
        
    Returns:
        填充大小
    """
    if is_aligned(value, alignment):
        return 0
    
    aligned = align_up(value, alignment)
    return aligned - value


def calculate_pages(size: int, page_size: int = PAGE_SIZE_4KB) -> int:
    """
    计算所需的页面数量
    
    Args:
        size: 数据大小
        page_size: 页面大小
        
    Returns:
        页面数量
    """
    if size <= 0:
        return 0
    
    return (size + page_size - 1) // page_size


def calculate_huge_pages(size: int, huge_page_size: int = PAGE_SIZE_2MB) -> int:
    """
    计算所需的大页数量
    
    Args:
        size: 数据大小
        huge_page_size: 大页大小
        
    Returns:
        大页数量
    """
    return calculate_pages(size, huge_page_size)


def find_optimal_alignment(size: int, 
                          min_alignment: int = PAGE_SIZE_4KB,
                          max_alignment: int = PAGE_SIZE_1GB) -> int:
    """
    找到最优的对齐大小
    
    根据数据大小选择合适的对齐边界，
    平衡内存利用率和性能。
    
    Args:
        size: 数据大小
        min_alignment: 最小对齐边界
        max_alignment: 最大对齐边界
        
    Returns:
        推荐的对齐大小
    """
    if size <= 0:
        return min_alignment
    
    # 根据大小选择对齐边界
    if size >= 1024 * 1024 * 1024:  # >= 1GB
        # 使用1GB大页
        return min(max_alignment, PAGE_SIZE_1GB)
    elif size >= 2 * 1024 * 1024:  # >= 2MB
        # 使用2MB大页
        return PAGE_SIZE_2MB
    else:
        # 使用标准页面
        return min_alignment


def align_offset_and_size(offset: int, 
                         size: int, 
                         alignment: int) -> Tuple[int, int, int]:
    """
    对齐偏移和大小，返回调整后的值
    
    Args:
        offset: 原始偏移
        size: 原始大小
        alignment: 对齐边界
        
    Returns:
        (aligned_offset, aligned_size, adjustment)
        - aligned_offset: 对齐后的偏移
        - aligned_size: 对齐后的大小
        - adjustment: 偏移调整量（可能为负）
    """
    # 向下对齐偏移
    aligned_offset = align_down(offset, alignment)
    
    # 计算偏移调整量
    adjustment = aligned_offset - offset
    
    # 调整大小（加上偏移调整）
    adjusted_size = size - adjustment
    
    # 向上对齐大小
    aligned_size = align_up(adjusted_size, alignment)
    
    return aligned_offset, aligned_size, adjustment


def get_alignment_waste(size: int, alignment: int) -> int:
    """
    计算对齐造成的内存浪费
    
    Args:
        size: 原始大小
        alignment: 对齐边界
        
    Returns:
        浪费的内存大小
    """
    aligned = align_up(size, alignment)
    return aligned - size


def get_alignment_efficiency(size: int, alignment: int) -> float:
    """
    计算对齐效率
    
    Args:
        size: 原始大小
        alignment: 对齐边界
        
    Returns:
        效率百分比（0.0-1.0）
    """
    if size <= 0:
        return 0.0
    
    aligned = align_up(size, alignment)
    return size / aligned


def is_power_of_two(value: int) -> bool:
    """
    检查值是否为2的幂
    
    Args:
        value: 需要检查的值
        
    Returns:
        是否为2的幂
    """
    return value > 0 and (value & (value - 1)) == 0


def next_power_of_two(value: int) -> int:
    """
    获取大于等于指定值的最小2的幂
    
    Args:
        value: 输入值
        
    Returns:
        大于等于value的最小2的幂
    """
    if value <= 0:
        return 1
    
    # 如果已经是2的幂，直接返回
    if is_power_of_two(value):
        return value
    
    # 计算下一个2的幂
    value -= 1
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    value |= value >> 16
    value |= value >> 32
    value += 1
    
    return value


def previous_power_of_two(value: int) -> int:
    """
    获取小于等于指定值的最大2的幂
    
    Args:
        value: 输入值
        
    Returns:
        小于等于value的最大2的幂
    """
    if value <= 0:
        return 1
    
    # 如果已经是2的幂，直接返回
    if is_power_of_two(value):
        return value
    
    # 计算前一个2的幂
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    value |= value >> 16
    value |= value >> 32
    
    return value - (value >> 1)


def align_address_to_page(addr: int, page_size: int = PAGE_SIZE_4KB) -> int:
    """
    将地址对齐到页面边界
    
    Args:
        addr: 内存地址
        page_size: 页面大小
        
    Returns:
        对齐后的地址
    """
    return align_down(addr, page_size)


def is_address_page_aligned(addr: int, page_size: int = PAGE_SIZE_4KB) -> bool:
    """
    检查地址是否页面对齐
    
    Args:
        addr: 内存地址
        page_size: 页面大小
        
    Returns:
        是否对齐
    """
    return is_aligned(addr, page_size)


def calculate_page_range(addr: int, 
                        size: int, 
                        page_size: int = PAGE_SIZE_4KB) -> Tuple[int, int]:
    """
    计算地址范围覆盖的页面范围
    
    Args:
        addr: 起始地址
        size: 大小
        page_size: 页面大小
        
    Returns:
        (start_page, end_page) 页面范围
    """
    start_page = addr // page_size
    end_addr = addr + size
    end_page = (end_addr + page_size - 1) // page_size
    
    return start_page, end_page


def format_size(size: int) -> str:
    """
    格式化大小为可读字符串
    
    Args:
        size: 大小（字节）
        
    Returns:
        格式化后的字符串
    """
    current_size: float = float(size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(current_size) < 1024.0:
            return f"{current_size:.2f} {unit}"
        current_size /= 1024.0
    return f"{current_size:.2f} PB"


def parse_size(size_str: str) -> int:
    """
    解析大小字符串为字节数
    
    Args:
        size_str: 大小字符串（如 "4KB", "2MB", "1GB"）
        
    Returns:
        字节数
        
    Raises:
        ValueError: 格式无效时抛出
    """
    size_str = size_str.strip().upper()
    
    # 单位映射
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024,
    }
    
    # 解析数字和单位
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")
    
    # 如果没有单位，假设为字节
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")
