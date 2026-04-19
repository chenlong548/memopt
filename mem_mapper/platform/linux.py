"""
mem_mapper Linux平台实现模块

提供Linux系统下的内存映射操作实现。
使用ctypes调用libc.so.6和系统调用。
"""

import ctypes
import ctypes.util
import os
import errno
from typing import List, Optional

from .base import PlatformBase
from ..core.exceptions import (
    MMapError, MUnmapError, MAdviseError, MProtectError,
    MSyncError, MLockError, MUnlockError, NUMAError
)


# Linux保护标志常量
PROT_NONE = 0x0
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4

# Linux映射标志常量
MAP_SHARED = 0x01
MAP_PRIVATE = 0x02
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20
MAP_HUGETLB = 0x40000
MAP_HUGE_SHIFT = 26
MAP_LOCKED = 0x2000
MAP_NORESERVE = 0x4000
MAP_POPULATE = 0x8000
MAP_NONBLOCK = 0x10000

# madvise建议常量
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4
MADV_REMOVE = 9
MADV_DONTFORK = 10
MADV_DOFORK = 11
MADV_MERGEABLE = 12
MADV_UNMERGEABLE = 13
MADV_HUGEPAGE = 14
MADV_NOHUGEPAGE = 15
MADV_DONTDUMP = 16
MADV_DODUMP = 17

# msync标志常量
MS_SYNC = 0x4
MS_ASYNC = 0x1
MS_INVALIDATE = 0x2

# mbind策略常量
MPOL_DEFAULT = 0
MPOL_BIND = 1
MPOL_INTERLEAVE = 2
MPOL_PREFERRED = 3
MPOL_LOCAL = 4

# mbind标志常量
MPOL_MF_STRICT = 1 << 0
MPOL_MF_MOVE = 1 << 1
MPOL_MF_MOVE_ALL = 1 << 2


class LinuxPlatform(PlatformBase):
    """
    Linux平台实现
    
    使用ctypes调用libc.so.6实现内存映射操作。
    支持NUMA、大页、madvise等Linux特有功能。
    """
    
    def __init__(self):
        """初始化Linux平台"""
        # 加载libc库
        libc_name = ctypes.util.find_library('c')
        if not libc_name:
            libc_name = 'libc.so.6'
        
        try:
            self.libc = ctypes.CDLL(libc_name, use_errno=True)
        except OSError as e:
            raise RuntimeError(f"Failed to load libc: {e}")
        
        # 设置函数签名
        self._setup_function_signatures()
        
        # 缓存页面大小
        self._page_size = None
        
        # 缓存大页大小
        self._huge_page_sizes = None
        
        # NUMA支持标志
        self._numa_available = None
    
    def _setup_function_signatures(self):
        """设置libc函数签名"""
        # mmap
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t,  # length
            ctypes.c_int,     # prot
            ctypes.c_int,     # flags
            ctypes.c_int,     # fd
            ctypes.c_long     # offset
        ]
        self.libc.mmap.restype = ctypes.c_void_p
        
        # munmap
        self.libc.munmap.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t   # length
        ]
        self.libc.munmap.restype = ctypes.c_int
        
        # madvise
        self.libc.madvise.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t,  # length
            ctypes.c_int      # advice
        ]
        self.libc.madvise.restype = ctypes.c_int
        
        # mprotect
        self.libc.mprotect.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t,  # length
            ctypes.c_int      # prot
        ]
        self.libc.mprotect.restype = ctypes.c_int
        
        # msync
        self.libc.msync.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t,  # length
            ctypes.c_int      # flags
        ]
        self.libc.msync.restype = ctypes.c_int
        
        # mlock
        self.libc.mlock.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t   # length
        ]
        self.libc.mlock.restype = ctypes.c_int
        
        # munlock
        self.libc.munlock.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t   # length
        ]
        self.libc.munlock.restype = ctypes.c_int
        
        # getpagesize
        if hasattr(self.libc, 'getpagesize'):
            self.libc.getpagesize.argtypes = []
            self.libc.getpagesize.restype = ctypes.c_int
    
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
        # 转换地址
        addr_ptr = ctypes.c_void_p(addr) if addr is not None else None
        
        # 调用mmap
        result = self.libc.mmap(addr_ptr, length, prot, flags, fd, offset)
        
        # 检查错误
        if result == ctypes.c_void_p(-1).value:
            err = ctypes.get_errno()
            raise MMapError(err, f"mmap failed for length={length}, flags={flags:#x}")
        
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
        result = self.libc.munmap(ctypes.c_void_p(addr), length)
        
        if result != 0:
            err = ctypes.get_errno()
            raise MUnmapError(err, f"munmap failed for addr={addr:#x}, length={length}")
        
        return 0
    
    def madvise(self, addr: int, length: int, advice: int) -> int:
        """
        提供内存访问建议
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            advice: 建议类型
            
        Returns:
            0表示成功
            
        Raises:
            MAdviseError: 建议设置失败时抛出
        """
        result = self.libc.madvise(ctypes.c_void_p(addr), length, advice)
        
        if result != 0:
            err = ctypes.get_errno()
            # madvise可能返回EINVAL但不影响功能，只记录警告
            if err == errno.EINVAL:
                import warnings
                warnings.warn(f"madvise returned EINVAL for addr={addr:#x}, advice={advice}")
                return 0
            raise MAdviseError(err, f"madvise failed for addr={addr:#x}, advice={advice}")
        
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
        result = self.libc.mprotect(ctypes.c_void_p(addr), length, prot)
        
        if result != 0:
            err = ctypes.get_errno()
            raise MProtectError(err, f"mprotect failed for addr={addr:#x}, prot={prot:#x}")
        
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
        result = self.libc.msync(ctypes.c_void_p(addr), length, flags)
        
        if result != 0:
            err = ctypes.get_errno()
            raise MSyncError(err, f"msync failed for addr={addr:#x}, length={length}")
        
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
        result = self.libc.mlock(ctypes.c_void_p(addr), length)
        
        if result != 0:
            err = ctypes.get_errno()
            raise MLockError(err, f"mlock failed for addr={addr:#x}, length={length}")
        
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
        result = self.libc.munlock(ctypes.c_void_p(addr), length)
        
        if result != 0:
            err = ctypes.get_errno()
            raise MUnlockError(err, f"munlock failed for addr={addr:#x}, length={length}")
        
        return 0
    
    def get_page_size(self) -> int:
        """
        获取系统页面大小
        
        Returns:
            页面大小（字节）
        """
        if self._page_size is None:
            if hasattr(self.libc, 'getpagesize'):
                self._page_size = self.libc.getpagesize()
            elif hasattr(os, 'sysconf'):
                try:
                    # 从系统获取（仅Unix系统）
                    self._page_size = os.sysconf('SC_PAGESIZE')  # type: ignore[attr-defined]
                except (ValueError, OSError):
                    # sysconf调用失败，使用默认值
                    self._page_size = 4096
            else:
                # 默认页面大小
                self._page_size = 4096
        
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
        
        # 检查/proc/meminfo中的大页信息
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                
                # 检查2MB大页
                if 'Hugepagesize:' in meminfo:
                    for line in meminfo.split('\n'):
                        if line.startswith('Hugepagesize:'):
                            # 格式: Hugepagesize:       2048 kB
                            parts = line.split()
                            if len(parts) >= 2:
                                size_kb = int(parts[1])
                                sizes.append(size_kb * 1024)  # 转换为字节
                            break
                
                # 检查是否有其他大页大小（通过hugepages-目录）
                import os as os_module
                hugepages_dir = '/sys/kernel/mm/hugepages'
                if os_module.path.exists(hugepages_dir):
                    for entry in os_module.listdir(hugepages_dir):
                        if entry.startswith('hugepages-'):
                            # 格式: hugepages-2048kB
                            size_str = entry.replace('hugepages-', '').replace('kB', '')
                            try:
                                size_kb = int(size_str)
                                size_bytes = size_kb * 1024
                                if size_bytes not in sizes:
                                    sizes.append(size_bytes)
                            except ValueError:
                                pass
        except (IOError, OSError):
            pass
        
        # 如果没有找到，使用默认值
        if not sizes:
            sizes = [2 * 1024 * 1024]  # 默认2MB
        
        # 排序（从小到大）
        sizes.sort()
        
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
        
        # 检查/proc或/sys中的NUMA信息
        try:
            # 检查是否有多个NUMA节点
            numa_dir = '/sys/devices/system/node'
            if os.path.exists(numa_dir):
                nodes = [d for d in os.listdir(numa_dir) if d.startswith('node')]
                self._numa_available = len(nodes) > 1
                return self._numa_available
            
            # 检查/proc中的NUMA信息
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # 如果有physical id字段，可能有多个NUMA节点
                if 'physical id' in cpuinfo:
                    self._numa_available = True
                    return self._numa_available
        except (IOError, OSError):
            pass
        
        self._numa_available = False
        return self._numa_available
    
    def get_numa_node_count(self) -> int:
        """
        获取NUMA节点数量
        
        Returns:
            NUMA节点数量
        """
        if not self.is_numa_available():
            return 1
        
        try:
            numa_dir = '/sys/devices/system/node'
            if os.path.exists(numa_dir):
                nodes = [d for d in os.listdir(numa_dir) if d.startswith('node')]
                return len(nodes)
        except (IOError, OSError):
            pass
        
        return 1
    
    def get_available_memory(self) -> int:
        """
        获取可用内存大小
        
        Returns:
            可用内存大小（字节）
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemAvailable:'):
                        # 格式: MemAvailable:    12345678 kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024  # 转换为字节
                    elif line.startswith('MemFree:'):
                        # 备用：使用MemFree
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (IOError, OSError):
            pass
        
        # 如果无法读取，返回0
        return 0
    
    def get_total_memory(self) -> int:
        """
        获取总内存大小
        
        Returns:
            总内存大小（字节）
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        # 格式: MemTotal:       12345678 kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024  # 转换为字节
        except (IOError, OSError):
            pass
        
        # 如果无法读取，返回0
        return 0
    
    def mbind(self, 
              addr: int, 
              length: int, 
              mode: int, 
              nodemask: List[int], 
              maxnode: int, 
              flags: int) -> int:
        """
        绑定内存到NUMA节点（Linux特有）
        
        Args:
            addr: 内存起始地址
            length: 内存长度
            mode: NUMA策略（MPOL_BIND, MPOL_INTERLEAVE等）
            nodemask: NUMA节点掩码列表
            maxnode: 最大节点数
            flags: 标志（MPOL_MF_STRICT等）
            
        Returns:
            0表示成功
            
        Raises:
            NUMAError: 绑定失败时抛出
        """
        # 尝试加载libnuma或使用syscall
        try:
            # 构建节点掩码
            # nodemask是一个位图，每个bit代表一个节点
            mask_size = (maxnode + 63) // 64  # 64位为单位
            nodemask_array = (ctypes.c_ulong * mask_size)()
            
            for node in nodemask:
                if node < maxnode:
                    idx = node // 64
                    bit = node % 64
                    nodemask_array[idx] |= (1 << bit)
            
            # 使用syscall
            # syscall number for mbind: __NR_mbind
            # 在x86_64上是237
            import sys
            if sys.maxsize > 2**32:  # 64位系统
                syscall_num = 237
            else:  # 32位系统
                syscall_num = 274
            
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            syscall.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_ulong),
                              ctypes.c_ulong, ctypes.c_int]
            syscall.restype = ctypes.c_int
            
            result = syscall(
                syscall_num,
                ctypes.c_void_p(addr),
                length,
                mode,
                nodemask_array,
                maxnode,
                flags
            )
            
            if result != 0:
                err = ctypes.get_errno()
                raise NUMAError(f"mbind failed: {os.strerror(err)}")
            
            return 0
            
        except Exception as e:
            raise NUMAError(f"mbind failed: {e}")
    
    def get_mmap_flags(self) -> dict:
        """
        获取mmap标志常量
        
        Returns:
            标志常量字典
        """
        return {
            'MAP_SHARED': MAP_SHARED,
            'MAP_PRIVATE': MAP_PRIVATE,
            'MAP_FIXED': MAP_FIXED,
            'MAP_ANONYMOUS': MAP_ANONYMOUS,
            'MAP_HUGETLB': MAP_HUGETLB,
            'MAP_HUGE_SHIFT': MAP_HUGE_SHIFT,
            'MAP_LOCKED': MAP_LOCKED,
            'MAP_NORESERVE': MAP_NORESERVE,
            'MAP_POPULATE': MAP_POPULATE,
            'MAP_NONBLOCK': MAP_NONBLOCK,
        }
    
    def get_prot_flags(self) -> dict:
        """
        获取保护标志常量
        
        Returns:
            保护标志常量字典
        """
        return {
            'PROT_NONE': PROT_NONE,
            'PROT_READ': PROT_READ,
            'PROT_WRITE': PROT_WRITE,
            'PROT_EXEC': PROT_EXEC,
        }
    
    def get_madvise_flags(self) -> dict:
        """
        获取madvise标志常量
        
        Returns:
            madvise标志常量字典
        """
        return {
            'MADV_NORMAL': MADV_NORMAL,
            'MADV_RANDOM': MADV_RANDOM,
            'MADV_SEQUENTIAL': MADV_SEQUENTIAL,
            'MADV_WILLNEED': MADV_WILLNEED,
            'MADV_DONTNEED': MADV_DONTNEED,
            'MADV_REMOVE': MADV_REMOVE,
            'MADV_DONTFORK': MADV_DONTFORK,
            'MADV_DOFORK': MADV_DOFORK,
            'MADV_MERGEABLE': MADV_MERGEABLE,
            'MADV_UNMERGEABLE': MADV_UNMERGEABLE,
            'MADV_HUGEPAGE': MADV_HUGEPAGE,
            'MADV_NOHUGEPAGE': MADV_NOHUGEPAGE,
            'MADV_DONTDUMP': MADV_DONTDUMP,
            'MADV_DODUMP': MADV_DODUMP,
        }
    
    def get_msync_flags(self) -> dict:
        """
        获取msync标志常量
        
        Returns:
            msync标志常量字典
        """
        return {
            'MS_SYNC': MS_SYNC,
            'MS_ASYNC': MS_ASYNC,
            'MS_INVALIDATE': MS_INVALIDATE,
        }
