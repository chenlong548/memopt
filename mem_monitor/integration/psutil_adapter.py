"""
mem_monitor psutil适配器模块

提供与psutil库的集成，获取系统和进程内存信息。
"""

import time
import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# 配置模块日志记录器
logger = logging.getLogger(__name__)


@dataclass
class ProcessMemoryInfo:
    """
    进程内存信息

    记录单个进程的内存使用情况。
    """

    pid: int                                 # 进程ID

    # 基本内存指标
    rss: int = 0                             # 驻留集大小
    vms: int = 0                             # 虚拟内存大小
    shared: int = 0                          # 共享内存
    text: int = 0                            # 代码段
    data: int = 0                            # 数据段
    lib: int = 0                             # 库
    dirty: int = 0                           # 脏页

    # 内存百分比
    percent: float = 0.0                     # 内存使用百分比

    # 时间戳
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pid': self.pid,
            'rss': self.rss,
            'vms': self.vms,
            'shared': self.shared,
            'text': self.text,
            'data': self.data,
            'percent': self.percent,
            'rss_mb': self.rss / (1024 * 1024),
            'vms_mb': self.vms / (1024 * 1024),
        }


@dataclass
class SystemMemoryInfo:
    """
    系统内存信息

    记录系统级别的内存使用情况。
    """

    # 基本指标
    total: int = 0                           # 总内存
    available: int = 0                       # 可用内存
    used: int = 0                            # 已用内存
    free: int = 0                            # 空闲内存

    # 百分比
    percent: float = 0.0                     # 使用百分比

    # 交换内存
    swap_total: int = 0                      # 交换总大小
    swap_used: int = 0                       # 交换已用
    swap_free: int = 0                       # 交换空闲
    swap_percent: float = 0.0                # 交换使用百分比

    # 时间戳
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total': self.total,
            'available': self.available,
            'used': self.used,
            'free': self.free,
            'percent': self.percent,
            'total_gb': self.total / (1024 * 1024 * 1024),
            'available_gb': self.available / (1024 * 1024 * 1024),
            'swap_total': self.swap_total,
            'swap_used': self.swap_used,
            'swap_percent': self.swap_percent,
        }


class PsutilAdapter:
    """
    psutil适配器

    提供与psutil库的集成，获取系统和进程内存信息。

    功能：
    1. 获取当前进程内存使用
    2. 获取系统内存状态
    3. 获取CPU和内存关联信息
    4. 获取进程列表和内存使用
    """

    def __init__(self):
        """初始化psutil适配器"""
        self._psutil = None
        self._available = False
        self._process = None

        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._available = True
        except ImportError as e:
            logger.debug(f"psutil not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize psutil adapter: {e}")

    def is_available(self) -> bool:
        """检查psutil是否可用"""
        return self._available

    def get_memory_info(self) -> Dict[str, Any]:
        """
        获取当前进程内存信息

        Returns:
            Dict: 内存信息字典
        """
        if not self._available:
            return {}

        try:
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()
            
            sys_mem = self._psutil.virtual_memory()
            
            return {
                'rss': mem_info.rss,
                'vms': mem_info.vms,
                'shared': getattr(mem_info, 'shared', 0),
                'text': getattr(mem_info, 'text', 0),
                'data': getattr(mem_info, 'data', 0),
                'percent': mem_percent,
                'available': sys_mem.available,
                'total': sys_mem.total,
            }

        except Exception as e:
            logger.debug(f"Failed to get memory info: {e}")
            return {}

    def get_process_memory_info(self, pid: Optional[int] = None) -> Optional[ProcessMemoryInfo]:
        """
        获取指定进程的内存信息

        Args:
            pid: 进程ID，None则使用当前进程

        Returns:
            Optional[ProcessMemoryInfo]: 进程内存信息
        """
        if not self._available:
            return None

        try:
            if pid is None:
                process = self._process
            else:
                process = self._psutil.Process(pid)

            mem_info = process.memory_info()
            mem_percent = process.memory_percent()

            return ProcessMemoryInfo(
                pid=process.pid,
                rss=mem_info.rss,
                vms=mem_info.vms,
                shared=getattr(mem_info, 'shared', 0),
                text=getattr(mem_info, 'text', 0),
                data=getattr(mem_info, 'data', 0),
                lib=getattr(mem_info, 'lib', 0),
                dirty=getattr(mem_info, 'dirty', 0),
                percent=mem_percent,
            )

        except self._psutil.NoSuchProcess:
            logger.debug(f"Process {pid} not found")
            return None
        except self._psutil.AccessDenied:
            logger.debug(f"Access denied to process {pid}")
            return None
        except Exception as e:
            logger.debug(f"Failed to get process memory info: {e}")
            return None

    def get_system_memory_info(self) -> Optional[SystemMemoryInfo]:
        """
        获取系统内存信息

        Returns:
            Optional[SystemMemoryInfo]: 系统内存信息
        """
        if not self._available:
            return None

        try:
            virtual_mem = self._psutil.virtual_memory()
            swap = self._psutil.swap_memory()

            return SystemMemoryInfo(
                total=virtual_mem.total,
                available=virtual_mem.available,
                used=virtual_mem.used,
                free=virtual_mem.free,
                percent=virtual_mem.percent,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_free=swap.free,
                swap_percent=swap.percent,
            )

        except Exception as e:
            logger.debug(f"Failed to get system memory info: {e}")
            return None

    def get_top_memory_processes(self, n: int = 10) -> List[ProcessMemoryInfo]:
        """
        获取内存使用最高的进程

        Args:
            n: 返回数量

        Returns:
            List[ProcessMemoryInfo]: 进程列表
        """
        if not self._available:
            return []

        processes = []

        try:
            for proc in self._psutil.process_iter(['pid', 'memory_info', 'memory_percent']):
                try:
                    mem_info = proc.info['memory_info']
                    if mem_info:
                        processes.append(ProcessMemoryInfo(
                            pid=proc.info['pid'],
                            rss=mem_info.rss,
                            vms=mem_info.vms,
                            percent=proc.info.get('memory_percent', 0),
                        ))
                except (KeyError, AttributeError) as e:
                    logger.debug(f"Failed to get process info: {e}")
                except Exception as e:
                    logger.debug(f"Unexpected error getting process info: {e}")

        except Exception as e:
            logger.warning(f"Failed to iterate processes: {e}")

        # 按RSS排序
        processes.sort(key=lambda p: p.rss, reverse=True)
        return processes[:n]

    def get_cpu_memory_affinity(self) -> Dict[int, List[int]]:
        """
        获取CPU和内存节点的亲和性

        Returns:
            Dict[int, List[int]]: CPU到内存节点的映射
        """
        if not self._available:
            return {}

        try:
            # 获取NUMA节点信息
            cpu_affinity = self._process.cpu_affinity()

            # 简化实现：假设所有CPU都可以访问所有内存
            return {cpu: [0] for cpu in cpu_affinity}

        except (AttributeError, self._psutil.AccessDenied) as e:
            logger.debug(f"Failed to get CPU affinity: {e}")
            return {}
        except Exception as e:
            logger.debug(f"Unexpected error getting CPU affinity: {e}")
            return {}

    def get_io_stats(self) -> Dict[str, Any]:
        """
        获取IO统计

        Returns:
            Dict: IO统计信息
        """
        if not self._available:
            return {}

        try:
            io_counters = self._process.io_counters()

            return {
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
            }

        except (AttributeError, self._psutil.AccessDenied) as e:
            logger.debug(f"Failed to get IO stats: {e}")
            return {}
        except Exception as e:
            logger.debug(f"Unexpected error getting IO stats: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'available': self._available,
            'process_memory': self.get_memory_info(),
            'system_memory': self.get_system_memory_info().to_dict() if self.is_available() else {},
        }
