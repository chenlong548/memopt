"""
mem_mapper 平台抽象基类模块

定义了跨平台内存映射操作的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class PlatformBase(ABC):
    """
    平台抽象层基类
    
    定义了所有平台必须实现的内存映射操作接口。
    支持Linux和Windows平台的统一抽象。
    """
    
    @abstractmethod
    def mmap(self, 
             addr: Optional[int], 
             length: int, 
             prot: int, 
             flags: int, 
             fd: int, 
             offset: int) -> int:
        """
        创建内存映射
        
        Args:
            addr: 建议的映射地址（None表示由系统选择）
            length: 映射长度（字节）
            prot: 保护标志（PROT_READ | PROT_WRITE | PROT_EXEC）
            flags: 映射标志（MAP_SHARED | MAP_PRIVATE | MAP_ANONYMOUS等）
            fd: 文件描述符（匿名映射时为-1）
            offset: 文件偏移量
            
        Returns:
            映射的起始地址
            
        Raises:
            MMapError: 映射失败时抛出
        """
        pass
    
    @abstractmethod
    def munmap(self, addr: int, length: int) -> int:
        """
        解除内存映射
        
        Args:
            addr: 映射起始地址
            length: 映射长度
            
        Returns:
            0表示成功
            
        Raises:
            MUnmapError: 解除映射失败时抛出
        """
        pass
    
    @abstractmethod
    def madvise(self, addr: int, length: int, advice: int) -> int:
        """
        提供内存访问建议
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            advice: 建议类型（MADV_NORMAL, MADV_RANDOM, MADV_SEQUENTIAL等）
            
        Returns:
            0表示成功
            
        Raises:
            MAdviseError: 建议设置失败时抛出
        """
        pass
    
    @abstractmethod
    def mprotect(self, addr: int, length: int, prot: int) -> int:
        """
        设置内存保护
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            prot: 保护标志
            
        Returns:
            0表示成功
            
        Raises:
            MProtectError: 保护设置失败时抛出
        """
        pass
    
    @abstractmethod
    def msync(self, addr: int, length: int, flags: int) -> int:
        """
        同步内存到存储设备
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            flags: 同步标志（MS_SYNC, MS_ASYNC, MS_INVALIDATE）
            
        Returns:
            0表示成功
            
        Raises:
            MSyncError: 同步失败时抛出
        """
        pass
    
    @abstractmethod
    def mlock(self, addr: int, length: int) -> int:
        """
        锁定内存（防止被交换到磁盘）
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            
        Returns:
            0表示成功
            
        Raises:
            MLockError: 锁定失败时抛出
        """
        pass
    
    @abstractmethod
    def munlock(self, addr: int, length: int) -> int:
        """
        解锁内存
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            
        Returns:
            0表示成功
            
        Raises:
            MUnlockError: 解锁失败时抛出
        """
        pass
    
    @abstractmethod
    def get_page_size(self) -> int:
        """
        获取系统页面大小
        
        Returns:
            页面大小（字节）
        """
        pass
    
    @abstractmethod
    def get_huge_page_sizes(self) -> List[int]:
        """
        获取系统支持的大页大小列表
        
        Returns:
            大页大小列表（字节），如[2097152, 1073741824]表示2MB和1GB
        """
        pass
    
    @abstractmethod
    def is_numa_available(self) -> bool:
        """
        检查NUMA是否可用
        
        Returns:
            NUMA是否可用
        """
        pass
    
    @abstractmethod
    def get_numa_node_count(self) -> int:
        """
        获取NUMA节点数量
        
        Returns:
            NUMA节点数量
        """
        pass
    
    @abstractmethod
    def get_available_memory(self) -> int:
        """
        获取可用内存大小
        
        Returns:
            可用内存大小（字节）
        """
        pass
    
    @abstractmethod
    def get_total_memory(self) -> int:
        """
        获取总内存大小
        
        Returns:
            总内存大小（字节）
        """
        pass
    
    def align_to_page(self, size: int, page_size: Optional[int] = None) -> int:
        """
        将大小对齐到页面边界
        
        Args:
            size: 原始大小
            page_size: 页面大小（None则使用系统页面大小）
            
        Returns:
            对齐后的大小
        """
        if page_size is None:
            page_size = self.get_page_size()
        return (size + page_size - 1) & ~(page_size - 1)
    
    def is_page_aligned(self, addr: int, page_size: Optional[int] = None) -> bool:
        """
        检查地址是否页面对齐
        
        Args:
            addr: 内存地址
            page_size: 页面大小（None则使用系统页面大小）
            
        Returns:
            是否对齐
        """
        if page_size is None:
            page_size = self.get_page_size()
        return (addr & (page_size - 1)) == 0
    
    def get_platform_name(self) -> str:
        """
        获取平台名称
        
        Returns:
            平台名称字符串
        """
        return self.__class__.__name__.replace('Platform', '').lower()
    
    @staticmethod
    def get_prot_read() -> int:
        """获取读保护标志"""
        return 0x1  # PROT_READ
    
    @staticmethod
    def get_prot_write() -> int:
        """获取写保护标志"""
        return 0x2  # PROT_WRITE
    
    @staticmethod
    def get_prot_exec() -> int:
        """获取执行保护标志"""
        return 0x4  # PROT_EXEC
    
    @staticmethod
    def get_prot_none() -> int:
        """获取无访问权限标志"""
        return 0x0  # PROT_NONE
    
    @staticmethod
    def get_map_shared() -> int:
        """获取共享映射标志"""
        return 0x01  # MAP_SHARED
    
    @staticmethod
    def get_map_private() -> int:
        """获取私有映射标志"""
        return 0x02  # MAP_PRIVATE
    
    @staticmethod
    def get_map_anonymous() -> int:
        """获取匿名映射标志"""
        return 0x20  # MAP_ANONYMOUS
    
    @staticmethod
    def get_map_fixed() -> int:
        """获取固定地址映射标志"""
        return 0x10  # MAP_FIXED
    
    @staticmethod
    def get_ms_sync() -> int:
        """获取同步刷新标志"""
        return 0x0  # MS_SYNC
    
    @staticmethod
    def get_ms_async() -> int:
        """获取异步刷新标志"""
        return 0x1  # MS_ASYNC
    
    @staticmethod
    def get_ms_invalidate() -> int:
        """获取失效标志"""
        return 0x2  # MS_INVALIDATE


class PlatformFactory:
    """
    平台工厂类
    
    根据当前操作系统创建对应的平台实例。
    """
    
    _instance: Optional[PlatformBase] = None
    
    @classmethod
    def create(cls) -> PlatformBase:
        """
        创建平台实例
        
        Returns:
            当前平台的PlatformBase实例
            
        Raises:
            PlatformNotSupportedError: 不支持的平台
        """
        if cls._instance is not None:
            return cls._instance
        
        import sys
        
        if sys.platform.startswith('linux'):
            from .linux import LinuxPlatform
            cls._instance = LinuxPlatform()
        elif sys.platform == 'win32':
            from .windows import WindowsPlatform
            cls._instance = WindowsPlatform()
        else:
            from ..core.exceptions import PlatformNotSupportedError
            raise PlatformNotSupportedError(sys.platform)
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """重置单例实例（用于测试）"""
        cls._instance = None
