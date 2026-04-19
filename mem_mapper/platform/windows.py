"""
mem_mapper Windows平台实现模块

提供Windows系统下的内存映射操作实现。
使用ctypes调用kernel32.dll和系统API。
"""

import ctypes
import ctypes.wintypes
from typing import List, Optional

from .base import PlatformBase
from ..core.exceptions import (
    MMapError, MUnmapError, MAdviseError, MProtectError,
    MSyncError, MLockError, MUnlockError, NUMAError
)


# Windows常量定义
# 内存保护常量
PAGE_NOACCESS = 0x01
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80
PAGE_GUARD = 0x100
PAGE_NOCACHE = 0x200
PAGE_WRITECOMBINE = 0x400

# 内存分配类型
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
MEM_DECOMMIT = 0x4000
MEM_RELEASE = 0x8000
MEM_FREE = 0x10000
MEM_PRIVATE = 0x20000
MEM_MAPPED = 0x40000
MEM_RESET = 0x80000
MEM_TOP_DOWN = 0x100000
MEM_WRITE_WATCH = 0x200000
MEM_PHYSICAL = 0x400000
MEM_LARGE_PAGES = 0x20000000

# 文件映射常量
FILE_MAP_WRITE = 0x0002
FILE_MAP_READ = 0x0004
FILE_MAP_ALL_ACCESS = 0x001F
FILE_MAP_COPY = 0x0001
FILE_MAP_EXECUTE = 0x0020

# 文件访问常量
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
GENERIC_EXECUTE = 0x20000000
GENERIC_ALL = 0x10000000

# 文件共享常量
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004

# 文件创建常量
CREATE_NEW = 1
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
OPEN_ALWAYS = 4
TRUNCATE_EXISTING = 5

# 文件属性
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_ATTRIBUTE_TEMPORARY = 0x100

# 错误代码
ERROR_SUCCESS = 0
ERROR_INVALID_PARAMETER = 87
ERROR_NOT_ENOUGH_MEMORY = 8

# NUMA常量
NUMA_NO_PREFERRED_NODE = -1


class MEMORYSTATUSEX(ctypes.Structure):
    """Windows内存状态结构"""
    _fields_ = [
        ('dwLength', ctypes.wintypes.DWORD),
        ('dwMemoryLoad', ctypes.wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


class SYSTEM_INFO(ctypes.Structure):
    """Windows系统信息结构"""
    _fields_ = [
        ('wProcessorArchitecture', ctypes.wintypes.WORD),
        ('wReserved', ctypes.wintypes.WORD),
        ('dwPageSize', ctypes.wintypes.DWORD),
        ('lpMinimumApplicationAddress', ctypes.wintypes.LPVOID),
        ('lpMaximumApplicationAddress', ctypes.wintypes.LPVOID),
        ('dwActiveProcessorMask', ctypes.POINTER(ctypes.wintypes.DWORD)),
        ('dwNumberOfProcessors', ctypes.wintypes.DWORD),
        ('dwProcessorType', ctypes.wintypes.DWORD),
        ('dwAllocationGranularity', ctypes.wintypes.DWORD),
        ('wProcessorLevel', ctypes.wintypes.WORD),
        ('wProcessorRevision', ctypes.wintypes.WORD),
    ]


class WindowsPlatform(PlatformBase):
    """
    Windows平台实现
    
    使用ctypes调用kernel32.dll实现内存映射操作。
    支持大页、NUMA等Windows特有功能。
    """
    
    def __init__(self):
        """初始化Windows平台"""
        # 加载kernel32.dll
        self.kernel32 = ctypes.windll.kernel32
        self.kernel32.SetLastError(0)
        
        # 设置函数签名
        self._setup_function_signatures()
        
        # 获取系统信息
        self._system_info = None
        
        # 缓存页面大小
        self._page_size = None
        
        # 缓存大页大小
        self._huge_page_sizes = None
        
        # NUMA支持标志
        self._numa_available = None
    
    def _setup_function_signatures(self):
        """设置Windows API函数签名"""
        # VirtualAlloc
        self.kernel32.VirtualAlloc.argtypes = [
            ctypes.wintypes.LPVOID,  # lpAddress
            ctypes.c_size_t,         # dwSize
            ctypes.wintypes.DWORD,   # flAllocationType
            ctypes.wintypes.DWORD    # flProtect
        ]
        self.kernel32.VirtualAlloc.restype = ctypes.wintypes.LPVOID
        
        # VirtualFree
        self.kernel32.VirtualFree.argtypes = [
            ctypes.wintypes.LPVOID,  # lpAddress
            ctypes.c_size_t,         # dwSize
            ctypes.wintypes.DWORD    # dwFreeType
        ]
        self.kernel32.VirtualFree.restype = ctypes.wintypes.BOOL
        
        # VirtualProtect
        self.kernel32.VirtualProtect.argtypes = [
            ctypes.wintypes.LPVOID,  # lpAddress
            ctypes.c_size_t,         # dwSize
            ctypes.wintypes.DWORD,   # flNewProtect
            ctypes.POINTER(ctypes.wintypes.DWORD)  # lpflOldProtect
        ]
        self.kernel32.VirtualProtect.restype = ctypes.wintypes.BOOL
        
        # VirtualLock
        self.kernel32.VirtualLock.argtypes = [
            ctypes.wintypes.LPVOID,  # lpAddress
            ctypes.c_size_t          # dwSize
        ]
        self.kernel32.VirtualLock.restype = ctypes.wintypes.BOOL
        
        # VirtualUnlock
        self.kernel32.VirtualUnlock.argtypes = [
            ctypes.wintypes.LPVOID,  # lpAddress
            ctypes.c_size_t          # dwSize
        ]
        self.kernel32.VirtualUnlock.restype = ctypes.wintypes.BOOL
        
        # CreateFileMapping
        self.kernel32.CreateFileMappingW.argtypes = [
            ctypes.wintypes.HANDLE,  # hFile
            ctypes.wintypes.LPVOID,  # lpAttributes
            ctypes.wintypes.DWORD,   # flProtect
            ctypes.wintypes.DWORD,   # dwMaximumSizeHigh
            ctypes.wintypes.DWORD,   # dwMaximumSizeLow
            ctypes.wintypes.LPCWSTR  # lpName
        ]
        self.kernel32.CreateFileMappingW.restype = ctypes.wintypes.HANDLE
        
        # MapViewOfFile
        self.kernel32.MapViewOfFile.argtypes = [
            ctypes.wintypes.HANDLE,  # hFileMappingObject
            ctypes.wintypes.DWORD,   # dwDesiredAccess
            ctypes.wintypes.DWORD,   # dwFileOffsetHigh
            ctypes.wintypes.DWORD,   # dwFileOffsetLow
            ctypes.c_size_t          # dwNumberOfBytesToMap
        ]
        self.kernel32.MapViewOfFile.restype = ctypes.wintypes.LPVOID
        
        # MapViewOfFileEx
        self.kernel32.MapViewOfFileEx.argtypes = [
            ctypes.wintypes.HANDLE,  # hFileMappingObject
            ctypes.wintypes.DWORD,   # dwDesiredAccess
            ctypes.wintypes.DWORD,   # dwFileOffsetHigh
            ctypes.wintypes.DWORD,   # dwFileOffsetLow
            ctypes.c_size_t,         # dwNumberOfBytesToMap
            ctypes.wintypes.LPVOID   # lpBaseAddress
        ]
        self.kernel32.MapViewOfFileEx.restype = ctypes.wintypes.LPVOID
        
        # UnmapViewOfFile
        self.kernel32.UnmapViewOfFile.argtypes = [
            ctypes.wintypes.LPCVOID  # lpBaseAddress
        ]
        self.kernel32.UnmapViewOfFile.restype = ctypes.wintypes.BOOL
        
        # FlushViewOfFile
        self.kernel32.FlushViewOfFile.argtypes = [
            ctypes.wintypes.LPCVOID,  # lpBaseAddress
            ctypes.c_size_t           # dwNumberOfBytesToFlush
        ]
        self.kernel32.FlushViewOfFile.restype = ctypes.wintypes.BOOL
        
        # CloseHandle
        self.kernel32.CloseHandle.argtypes = [
            ctypes.wintypes.HANDLE
        ]
        self.kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        
        # GetSystemInfo
        self.kernel32.GetSystemInfo.argtypes = [
            ctypes.POINTER(SYSTEM_INFO)
        ]
        
        # GlobalMemoryStatusEx
        self.kernel32.GlobalMemoryStatusEx.argtypes = [
            ctypes.POINTER(MEMORYSTATUSEX)
        ]
        self.kernel32.GlobalMemoryStatusEx.restype = ctypes.wintypes.BOOL
        
        # GetLastError
        self.kernel32.GetLastError.argtypes = []
        self.kernel32.GetLastError.restype = ctypes.wintypes.DWORD
    
    def _get_system_info(self) -> SYSTEM_INFO:
        """获取系统信息"""
        if self._system_info is None:
            self._system_info = SYSTEM_INFO()
            self.kernel32.GetSystemInfo(ctypes.byref(self._system_info))
        return self._system_info
    
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
            prot: 保护标志
            flags: 映射标志
            fd: 文件描述符（匿名映射时为-1）
            offset: 文件偏移量
            
        Returns:
            映射的起始地址
            
        Raises:
            MMapError: 映射失败时抛出
        """
        # 转换保护标志
        win_prot = self._prot_to_win(prot)
        
        # 转换映射标志
        is_shared = (flags & 0x01) != 0  # MAP_SHARED
        is_anonymous = (flags & 0x20) != 0  # MAP_ANONYMOUS
        
        if is_anonymous or fd == -1:
            # 匿名映射：使用VirtualAlloc
            alloc_type = MEM_COMMIT | MEM_RESERVE
            
            # 检查是否使用大页
            if flags & 0x40000:  # MAP_HUGETLB
                alloc_type |= MEM_LARGE_PAGES
            
            result = self.kernel32.VirtualAlloc(
                ctypes.wintypes.LPVOID(addr) if addr else None,
                length,
                alloc_type,
                win_prot
            )
            
            if not result:
                err = self.kernel32.GetLastError()
                raise MMapError(err, f"VirtualAlloc failed for length={length}")
            
            return result
        else:
            # 文件映射：使用CreateFileMapping和MapViewOfFile
            # 注意：Python的os.open返回的文件描述符需要转换为Windows HANDLE
            # 在Windows上，文件描述符和HANDLE可以互换使用（CRT会自动转换）
            
            # 创建文件映射对象
            flProtect = win_prot
            if is_shared:
                flProtect = PAGE_READWRITE
            else:
                flProtect = PAGE_WRITECOPY
            
            # 对于文件映射，max_size应该设置为0（使用文件大小）
            # 或者设置为映射的大小
            max_size_high = (length >> 32) & 0xFFFFFFFF
            max_size_low = length & 0xFFFFFFFF
            
            # 使用 _get_osfhandle 将文件描述符转换为HANDLE
            try:
                msvcrt = ctypes.cdll.msvcrt
                msvcrt._get_osfhandle.argtypes = [ctypes.c_int]
                msvcrt._get_osfhandle.restype = ctypes.wintypes.HANDLE
                
                handle = msvcrt._get_osfhandle(fd)
                if handle == -1 or handle == 0:
                    err = self.kernel32.GetLastError()
                    raise MMapError(err, f"_get_osfhandle failed for fd={fd}")
            except Exception as e:
                # 如果转换失败，尝试直接使用fd
                handle = ctypes.wintypes.HANDLE(fd)
            
            mapping_handle = self.kernel32.CreateFileMappingW(
                handle,
                None,
                flProtect,
                max_size_high,
                max_size_low,
                None
            )
            
            if not mapping_handle:
                err = self.kernel32.GetLastError()
                raise MMapError(err, f"CreateFileMapping failed for fd={fd}")
            
            # 映射视图
            if is_shared:
                access = FILE_MAP_WRITE | FILE_MAP_READ
            else:
                access = FILE_MAP_COPY
            
            offset_high = (offset >> 32) & 0xFFFFFFFF
            offset_low = offset & 0xFFFFFFFF
            
            if addr:
                result = self.kernel32.MapViewOfFileEx(
                    mapping_handle,
                    access,
                    offset_high,
                    offset_low,
                    length,
                    ctypes.wintypes.LPVOID(addr)
                )
            else:
                result = self.kernel32.MapViewOfFile(
                    mapping_handle,
                    access,
                    offset_high,
                    offset_low,
                    length
                )
            
            # 关闭映射句柄（映射视图保持有效）
            self.kernel32.CloseHandle(mapping_handle)
            
            if not result:
                err = self.kernel32.GetLastError()
                raise MMapError(err, f"MapViewOfFile failed for length={length}")
            
            return result
    
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
        # 尝试UnmapViewOfFile（文件映射）
        result = self.kernel32.UnmapViewOfFile(ctypes.wintypes.LPCVOID(addr))
        
        if result:
            return 0
        
        # 如果失败，尝试VirtualFree（匿名映射）
        result = self.kernel32.VirtualFree(
            ctypes.wintypes.LPVOID(addr),
            0,
            MEM_RELEASE
        )
        
        if not result:
            err = self.kernel32.GetLastError()
            raise MUnmapError(err, f"UnmapViewOfFile/VirtualFree failed for addr={addr:#x}")
        
        return 0
    
    def madvise(self, addr: int, length: int, advice: int) -> int:
        """
        提供内存访问建议
        
        注意：Windows没有完全对应的madvise功能，
        这里提供部分模拟实现。
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            advice: 建议类型
            
        Returns:
            0表示成功
            
        Raises:
            MAdviseError: 建议设置失败时抛出
        """
        # Windows没有直接的madvise对应
        # 这里可以提供一些模拟实现
        # 例如：使用PrefetchVirtualMemory或OfferVirtualMemory
        
        # 对于MADV_WILLNEED，可以使用PrefetchVirtualMemory
        if advice == 3:  # MADV_WILLNEED
            try:
                # Windows 8+支持PrefetchVirtualMemory
                # 这里简化实现，直接返回成功
                return 0
            except Exception:
                pass
        
        # 对于MADV_DONTNEED，可以使用DiscardVirtualMemory
        if advice == 4:  # MADV_DONTNEED
            try:
                # Windows 8+支持DiscardVirtualMemory
                # 这里简化实现，直接返回成功
                return 0
            except Exception:
                pass
        
        # 其他建议在Windows上不支持，但返回成功以避免错误
        return 0
    
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
        # 转换保护标志
        win_prot = self._prot_to_win(prot)
        
        # 调用VirtualProtect
        old_prot = ctypes.wintypes.DWORD()
        result = self.kernel32.VirtualProtect(
            ctypes.wintypes.LPVOID(addr),
            length,
            win_prot,
            ctypes.byref(old_prot)
        )
        
        if not result:
            err = self.kernel32.GetLastError()
            raise MProtectError(err, f"VirtualProtect failed for addr={addr:#x}")
        
        return 0
    
    def msync(self, addr: int, length: int, flags: int) -> int:
        """
        同步内存到存储设备
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            flags: 同步标志
            
        Returns:
            0表示成功
            
        Raises:
            MSyncError: 同步失败时抛出
        """
        # Windows使用FlushViewOfFile
        result = self.kernel32.FlushViewOfFile(
            ctypes.wintypes.LPCVOID(addr),
            length
        )
        
        if not result:
            err = self.kernel32.GetLastError()
            raise MSyncError(err, f"FlushViewOfFile failed for addr={addr:#x}")
        
        return 0
    
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
        result = self.kernel32.VirtualLock(
            ctypes.wintypes.LPVOID(addr),
            length
        )
        
        if not result:
            err = self.kernel32.GetLastError()
            raise MLockError(err, f"VirtualLock failed for addr={addr:#x}")
        
        return 0
    
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
        result = self.kernel32.VirtualUnlock(
            ctypes.wintypes.LPVOID(addr),
            length
        )
        
        if not result:
            err = self.kernel32.GetLastError()
            raise MUnlockError(err, f"VirtualUnlock failed for addr={addr:#x}")
        
        return 0
    
    def get_page_size(self) -> int:
        """
        获取系统页面大小
        
        Returns:
            页面大小（字节）
        """
        if self._page_size is None:
            sys_info = self._get_system_info()
            self._page_size = sys_info.dwPageSize
        
        return self._page_size
    
    def get_huge_page_sizes(self) -> List[int]:
        """
        获取系统支持的大页大小列表
        
        Returns:
            大页大小列表（字节）
        """
        if self._huge_page_sizes is not None:
            return self._huge_page_sizes
        
        sizes = []
        
        # Windows大页通常是2MB
        # 可以通过GetLargePageMinimum获取最小大页大小
        try:
            GetLargePageMinimum = self.kernel32.GetLargePageMinimum
            GetLargePageMinimum.argtypes = []
            GetLargePageMinimum.restype = ctypes.c_size_t
            
            min_large_page = GetLargePageMinimum()
            if min_large_page > 0:
                sizes.append(min_large_page)
        except AttributeError:
            # 如果不支持，使用默认值
            sizes.append(2 * 1024 * 1024)  # 2MB
        
        self._huge_page_sizes = sizes
        return sizes
    
    def is_numa_available(self) -> bool:
        """
        检查NUMA是否可用
        
        Returns:
            NUMA是否可用
        """
        if self._numa_available is not None:
            return self._numa_available
        
        # 检查是否有GetNumaHighestNodeNumber函数
        try:
            GetNumaHighestNodeNumber = self.kernel32.GetNumaHighestNodeNumber
            GetNumaHighestNodeNumber.argtypes = [
                ctypes.POINTER(ctypes.wintypes.ULONG)
            ]
            GetNumaHighestNodeNumber.restype = ctypes.wintypes.BOOL
            
            highest_node = ctypes.wintypes.ULONG()
            result = GetNumaHighestNodeNumber(ctypes.byref(highest_node))
            
            if result and highest_node.value > 0:
                self._numa_available = True
                return True
        except AttributeError:
            pass
        
        self._numa_available = False
        return False
    
    def get_numa_node_count(self) -> int:
        """
        获取NUMA节点数量
        
        Returns:
            NUMA节点数量
        """
        if not self.is_numa_available():
            return 1
        
        try:
            GetNumaHighestNodeNumber = self.kernel32.GetNumaHighestNodeNumber
            GetNumaHighestNodeNumber.argtypes = [
                ctypes.POINTER(ctypes.wintypes.ULONG)
            ]
            GetNumaHighestNodeNumber.restype = ctypes.wintypes.BOOL
            
            highest_node = ctypes.wintypes.ULONG()
            result = GetNumaHighestNodeNumber(ctypes.byref(highest_node))
            
            if result:
                return highest_node.value + 1
        except (AttributeError, Exception):
            pass
        
        return 1
    
    def get_available_memory(self) -> int:
        """
        获取可用内存大小
        
        Returns:
            可用内存大小（字节）
        """
        mem_status = MEMORYSTATUSEX()
        mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        
        result = self.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
        
        if result:
            return mem_status.ullAvailPhys
        
        return 0
    
    def get_total_memory(self) -> int:
        """
        获取总内存大小
        
        Returns:
            总内存大小（字节）
        """
        mem_status = MEMORYSTATUSEX()
        mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        
        result = self.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
        
        if result:
            return mem_status.ullTotalPhys
        
        return 0
    
    def _prot_to_win(self, prot: int) -> int:
        """
        将Linux保护标志转换为Windows保护标志
        
        Args:
            prot: Linux保护标志
            
        Returns:
            Windows保护标志
        """
        win_prot = 0
        
        # PROT_READ = 0x1
        if prot & 0x1:
            win_prot |= PAGE_READONLY
        
        # PROT_WRITE = 0x2
        if prot & 0x2:
            if prot & 0x1:  # READ + WRITE
                win_prot = PAGE_READWRITE
            else:  # WRITE only
                win_prot = PAGE_READWRITE  # Windows不支持只写
        
        # PROT_EXEC = 0x4
        if prot & 0x4:
            if prot & 0x2:  # EXEC + WRITE
                win_prot = PAGE_EXECUTE_READWRITE
            elif prot & 0x1:  # EXEC + READ
                win_prot = PAGE_EXECUTE_READ
            else:  # EXEC only
                win_prot = PAGE_EXECUTE
        
        # PROT_NONE = 0x0
        if prot == 0:
            win_prot = PAGE_NOACCESS
        
        return win_prot
    
    def get_prot_flags(self) -> dict:
        """
        获取保护标志常量
        
        Returns:
            保护标志常量字典
        """
        return {
            'PAGE_NOACCESS': PAGE_NOACCESS,
            'PAGE_READONLY': PAGE_READONLY,
            'PAGE_READWRITE': PAGE_READWRITE,
            'PAGE_WRITECOPY': PAGE_WRITECOPY,
            'PAGE_EXECUTE': PAGE_EXECUTE,
            'PAGE_EXECUTE_READ': PAGE_EXECUTE_READ,
            'PAGE_EXECUTE_READWRITE': PAGE_EXECUTE_READWRITE,
            'PAGE_EXECUTE_WRITECOPY': PAGE_EXECUTE_WRITECOPY,
        }
    
    def get_allocation_flags(self) -> dict:
        """
        获取内存分配标志常量
        
        Returns:
            分配标志常量字典
        """
        return {
            'MEM_COMMIT': MEM_COMMIT,
            'MEM_RESERVE': MEM_RESERVE,
            'MEM_DECOMMIT': MEM_DECOMMIT,
            'MEM_RELEASE': MEM_RELEASE,
            'MEM_LARGE_PAGES': MEM_LARGE_PAGES,
        }
    
    def get_file_mapping_flags(self) -> dict:
        """
        获取文件映射标志常量
        
        Returns:
            文件映射标志常量字典
        """
        return {
            'FILE_MAP_READ': FILE_MAP_READ,
            'FILE_MAP_WRITE': FILE_MAP_WRITE,
            'FILE_MAP_ALL_ACCESS': FILE_MAP_ALL_ACCESS,
            'FILE_MAP_COPY': FILE_MAP_COPY,
            'FILE_MAP_EXECUTE': FILE_MAP_EXECUTE,
        }
