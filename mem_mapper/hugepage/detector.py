"""
mem_mapper 大页检测模块

提供大页检测和配置功能。
"""

import os
import sys
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..core.exceptions import HugePageError, HugePageNotAvailableError
from ..utils.alignment import PAGE_SIZE_2MB, PAGE_SIZE_1GB


@dataclass
class HugePageInfo:
    """
    大页信息
    
    表示单个大页大小的详细信息。
    """
    
    page_size: int           # 页面大小（字节）
    total_pages: int = 0     # 总页面数
    free_pages: int = 0      # 空闲页面数
    reserved_pages: int = 0  # 保留页面数
    surplus_pages: int = 0   # 剩余页面数
    
    def get_total_size(self) -> int:
        """
        获取总大小
        
        Returns:
            总大小（字节）
        """
        return self.total_pages * self.page_size
    
    def get_free_size(self) -> int:
        """
        获取空闲大小
        
        Returns:
            空闲大小（字节）
        """
        return self.free_pages * self.page_size
    
    def get_used_pages(self) -> int:
        """
        获取已使用页面数
        
        Returns:
            已使用页面数
        """
        return self.total_pages - self.free_pages
    
    def get_usage_ratio(self) -> float:
        """
        获取使用率
        
        Returns:
            使用率（0.0-1.0）
        """
        if self.total_pages == 0:
            return 0.0
        return self.get_used_pages() / self.total_pages


@dataclass
class HugePageConfig:
    """
    大页配置
    
    大页分配和使用的配置信息。
    """
    
    supported_sizes: List[int] = field(default_factory=list)  # 支持的大页大小
    default_size: int = PAGE_SIZE_2MB  # 默认大页大小
    min_size: int = PAGE_SIZE_2MB      # 最小大页大小
    max_size: int = PAGE_SIZE_1GB      # 最大大页大小
    
    def is_size_supported(self, size: int) -> bool:
        """
        检查大小是否支持
        
        Args:
            size: 页面大小
            
        Returns:
            是否支持
        """
        return size in self.supported_sizes
    
    def get_nearest_size(self, size: int) -> int:
        """
        获取最接近的支持大小
        
        Args:
            size: 请求的大小
            
        Returns:
            最接近的支持大小
        """
        if not self.supported_sizes:
            return self.default_size
        
        # 找到最接近的大小
        nearest = self.supported_sizes[0]
        min_diff = abs(size - nearest)
        
        for supported_size in self.supported_sizes:
            diff = abs(size - supported_size)
            if diff < min_diff:
                min_diff = diff
                nearest = supported_size
        
        return nearest


class HugePageDetector:
    """
    大页检测器
    
    检测系统支持的大页配置和状态。
    """
    
    def __init__(self):
        """初始化大页检测器"""
        self._config = None
        self._page_info = {}
        self._cached = False
    
    def detect(self) -> HugePageConfig:
        """
        检测大页配置
        
        Returns:
            大页配置
        """
        if self._cached and self._config is not None:
            return self._config
        
        # 根据平台选择检测方法
        if sys.platform.startswith('linux'):
            self._config = self._detect_linux()
        elif sys.platform == 'win32':
            self._config = self._detect_windows()
        else:
            # 不支持的平台，返回默认配置
            self._config = self._create_default_config()
        
        self._cached = True
        return self._config
    
    def _detect_linux(self) -> HugePageConfig:
        """
        检测Linux系统的大页配置
        
        Returns:
            大页配置
        """
        config = HugePageConfig()
        config.supported_sizes = []
        
        # 方法1：检查/proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                
                # 检查Hugepagesize
                match = re.search(r'Hugepagesize:\s+(\d+)\s+kB', meminfo)
                if match:
                    size_kb = int(match.group(1))
                    size_bytes = size_kb * 1024
                    if size_bytes not in config.supported_sizes:
                        config.supported_sizes.append(size_bytes)
                    config.default_size = size_bytes
        except IOError:
            pass
        
        # 方法2：检查/sys/kernel/mm/hugepages目录
        hugepages_dir = '/sys/kernel/mm/hugepages'
        if os.path.exists(hugepages_dir):
            try:
                entries = os.listdir(hugepages_dir)
                for entry in entries:
                    if entry.startswith('hugepages-'):
                        # 格式: hugepages-2048kB
                        match = re.search(r'hugepages-(\d+)kB', entry)
                        if match:
                            size_kb = int(match.group(1))
                            size_bytes = size_kb * 1024
                            if size_bytes not in config.supported_sizes:
                                config.supported_sizes.append(size_bytes)
            except OSError:
                pass
        
        # 排序支持的大小
        config.supported_sizes.sort()
        
        # 更新最小和最大值
        if config.supported_sizes:
            config.min_size = config.supported_sizes[0]
            config.max_size = config.supported_sizes[-1]
        
        # 检测每个大小的详细信息
        for size in config.supported_sizes:
            info = self._get_hugepage_info_linux(size)
            if info:
                self._page_info[size] = info
        
        return config
    
    def _get_hugepage_info_linux(self, page_size: int) -> Optional[HugePageInfo]:
        """
        获取Linux系统指定大页大小的信息
        
        Args:
            page_size: 页面大小（字节）
            
        Returns:
            大页信息
        """
        # 转换为kB用于路径
        size_kb = page_size // 1024
        hugepages_dir = f'/sys/kernel/mm/hugepages/hugepages-{size_kb}kB'
        
        if not os.path.exists(hugepages_dir):
            return None
        
        info = HugePageInfo(page_size=page_size)
        
        # 读取各种计数
        try:
            # 总页面数
            nr_hugepages_path = os.path.join(hugepages_dir, 'nr_hugepages')
            if os.path.exists(nr_hugepages_path):
                with open(nr_hugepages_path, 'r') as f:
                    info.total_pages = int(f.read().strip())
            
            # 空闲页面数
            free_hugepages_path = os.path.join(hugepages_dir, 'free_hugepages')
            if os.path.exists(free_hugepages_path):
                with open(free_hugepages_path, 'r') as f:
                    info.free_pages = int(f.read().strip())
            
            # 保留页面数
            resv_hugepages_path = os.path.join(hugepages_dir, 'resv_hugepages')
            if os.path.exists(resv_hugepages_path):
                with open(resv_hugepages_path, 'r') as f:
                    info.reserved_pages = int(f.read().strip())
            
            # 剩余页面数
            surplus_hugepages_path = os.path.join(hugepages_dir, 'surplus_hugepages')
            if os.path.exists(surplus_hugepages_path):
                with open(surplus_hugepages_path, 'r') as f:
                    info.surplus_pages = int(f.read().strip())
            
        except (IOError, OSError, ValueError):
            pass
        
        return info
    
    def _detect_windows(self) -> HugePageConfig:
        """
        检测Windows系统的大页配置
        
        Returns:
            大页配置
        """
        config = HugePageConfig()
        
        try:
            import ctypes
            
            # 获取最小大页大小
            kernel32 = ctypes.windll.kernel32
            
            GetLargePageMinimum = kernel32.GetLargePageMinimum
            GetLargePageMinimum.argtypes = []
            GetLargePageMinimum.restype = ctypes.c_size_t
            
            min_large_page = GetLargePageMinimum()
            
            if min_large_page > 0:
                config.supported_sizes = [min_large_page]
                config.default_size = min_large_page
                config.min_size = min_large_page
                config.max_size = min_large_page
                
                # 创建大页信息
                info = HugePageInfo(page_size=min_large_page)
                # Windows没有直接获取大页数量的API
                # 这里设置为0，实际使用时需要检查
                self._page_info[min_large_page] = info
            
        except (AttributeError, OSError):
            # 不支持大页，使用默认配置
            pass
        
        return config
    
    def _create_default_config(self) -> HugePageConfig:
        """
        创建默认配置
        
        Returns:
            默认大页配置
        """
        config = HugePageConfig()
        config.supported_sizes = [PAGE_SIZE_2MB]
        config.default_size = PAGE_SIZE_2MB
        config.min_size = PAGE_SIZE_2MB
        config.max_size = PAGE_SIZE_2MB
        
        # 创建默认信息
        info = HugePageInfo(page_size=PAGE_SIZE_2MB)
        self._page_info[PAGE_SIZE_2MB] = info
        
        return config
    
    def get_page_info(self, page_size: int) -> Optional[HugePageInfo]:
        """
        获取指定大页大小的信息
        
        Args:
            page_size: 页面大小
            
        Returns:
            大页信息
        """
        # 刷新信息
        if sys.platform.startswith('linux'):
            info = self._get_hugepage_info_linux(page_size)
            if info:
                self._page_info[page_size] = info
        
        return self._page_info.get(page_size)
    
    def get_all_page_info(self) -> Dict[int, HugePageInfo]:
        """
        获取所有大页信息
        
        Returns:
            页面大小到大页信息的映射
        """
        # 刷新所有信息
        config = self.detect()
        for size in config.supported_sizes:
            self.get_page_info(size)
        
        return self._page_info.copy()
    
    def is_huge_page_available(self, page_size: Optional[int] = None) -> bool:
        """
        检查大页是否可用
        
        Args:
            page_size: 页面大小，None表示检查任意大小
            
        Returns:
            是否可用
        """
        config = self.detect()
        
        if page_size is not None:
            # 检查指定大小
            if page_size not in config.supported_sizes:
                return False
            
            info = self.get_page_info(page_size)
            return info is not None and info.free_pages > 0
        else:
            # 检查任意大小
            for size in config.supported_sizes:
                info = self.get_page_info(size)
                if info and info.free_pages > 0:
                    return True
            return False
    
    def recommend_page_size(self, size: int) -> int:
        """
        推荐大页大小
        
        根据需要映射的大小推荐最优的大页大小。
        
        Args:
            size: 需要映射的大小（字节）
            
        Returns:
            推荐的大页大小
        """
        config = self.detect()
        
        if not config.supported_sizes:
            return PAGE_SIZE_2MB  # 默认2MB
        
        # 从大到小检查
        for page_size in sorted(config.supported_sizes, reverse=True):
            info = self.get_page_info(page_size)
            
            # 检查是否有足够的大页
            if info and info.free_pages > 0:
                # 计算需要的大页数量
                pages_needed = (size + page_size - 1) // page_size
                
                # 如果有足够的大页，推荐这个大小
                if info.free_pages >= pages_needed:
                    return page_size
        
        # 如果没有足够的大页，返回最小支持的大小
        return config.min_size
    
    def can_allocate(self, size: int, page_size: Optional[int] = None) -> bool:
        """
        检查是否可以分配指定大小的大页
        
        Args:
            size: 需要分配的大小（字节）
            page_size: 大页大小，None表示自动选择
            
        Returns:
            是否可以分配
        """
        config = self.detect()
        
        if page_size is None:
            page_size = self.recommend_page_size(size)
        
        if page_size not in config.supported_sizes:
            return False
        
        info = self.get_page_info(page_size)
        if not info:
            return False
        
        # 计算需要的大页数量
        pages_needed = (size + page_size - 1) // page_size
        
        return info.free_pages >= pages_needed
    
    def refresh(self) -> HugePageConfig:
        """
        刷新大页信息
        
        Returns:
            最新的配置
        """
        self._cached = False
        self._page_info.clear()
        return self.detect()
    
    def get_summary(self) -> Dict:
        """
        获取大页摘要信息
        
        Returns:
            摘要信息字典
        """
        config = self.detect()
        page_info = self.get_all_page_info()
        
        summary = {
            'supported_sizes': config.supported_sizes,
            'default_size': config.default_size,
            'min_size': config.min_size,
            'max_size': config.max_size,
            'pages': {}
        }
        
        for size, info in page_info.items():
            summary['pages'][size] = {
                'total_pages': info.total_pages,
                'free_pages': info.free_pages,
                'used_pages': info.get_used_pages(),
                'total_size': info.get_total_size(),
                'free_size': info.get_free_size(),
                'usage_ratio': info.get_usage_ratio(),
            }
        
        return summary


# 全局大页检测器实例
_global_detector = None


def get_hugepage_detector() -> HugePageDetector:
    """
    获取全局大页检测器
    
    Returns:
        大页检测器实例
    """
    global _global_detector
    
    if _global_detector is None:
        _global_detector = HugePageDetector()
    
    return _global_detector


def get_hugepage_config() -> HugePageConfig:
    """
    获取大页配置（全局函数）
    
    Returns:
        大页配置
    """
    return get_hugepage_detector().detect()


def is_hugepage_available(page_size: Optional[int] = None) -> bool:
    """
    检查大页是否可用（全局函数）
    
    Args:
        page_size: 页面大小
        
    Returns:
        是否可用
    """
    return get_hugepage_detector().is_huge_page_available(page_size)
