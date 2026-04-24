"""
mem_monitor 硬件采样器模块

实现基于硬件性能计数器的内存采样。
支持Intel PEBS和AMD IBS。
"""

import time
import os
import sys
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base import SamplerBase, SampleData, SamplerState, MemoryAccessEvent

# 配置模块日志记录器
logger = logging.getLogger(__name__)


@dataclass
class HardwareCounterConfig:
    """硬件计数器配置"""

    event_name: str            # 事件名称
    period: int                # 采样周期
    precise: int = 0           # 精确级别 (0-3)
    sample_type: int = 0       # 采样类型
    branch_sample_type: int = 0  # 分支采样类型


class HardwareSampler(SamplerBase):
    """
    硬件采样器基类

    提供硬件性能计数器采样的基础功能。
    """

    def __init__(self, config):
        super().__init__(config)
        self._perf_fd: Optional[int] = None
        self._mmap_buffer: Optional[bytes] = None
        self._counter_config: Optional[HardwareCounterConfig] = None
        self._platform_supported = self._check_platform()
        self._resources_released = False

    def _check_platform(self) -> bool:
        """检查平台是否支持"""
        # 硬件采样主要支持Linux
        return sys.platform.startswith('linux')

    def start(self) -> None:
        """启动采样"""
        if not self._platform_supported:
            self._state = SamplerState.ERROR
            logger.warning("Hardware sampling not supported on this platform")
            return

        self._state = SamplerState.INITIALIZING

        try:
            self._setup_counters()
            self._enable_counters()
            self._start_time = time.time()
            self._state = SamplerState.RUNNING
            self._resources_released = False
        except Exception as e:
            self._state = SamplerState.ERROR
            self._record_error(e)
            logger.error(f"Failed to start hardware sampler: {e}", exc_info=True)

    def stop(self) -> None:
        """停止采样"""
        if self._perf_fd is not None:
            try:
                self._disable_counters()
                self._cleanup_counters()
            except Exception as e:
                logger.warning(f"Error during hardware sampler cleanup: {e}")
            finally:
                self._perf_fd = None
                self._mmap_buffer = None
                self._resources_released = True

        self._state = SamplerState.STOPPED
    
    def __del__(self):
        """析构函数，确保资源释放"""
        if not self._resources_released and self._perf_fd is not None:
            try:
                self._cleanup_counters()
            except Exception as e:
                logger.debug(f"Error during hardware sampler cleanup in __del__: {e}")
            finally:
                self._perf_fd = None
                self._mmap_buffer = None

    def _setup_counters(self) -> None:
        """设置硬件计数器"""
        # 子类实现
        pass

    def _enable_counters(self) -> None:
        """启用计数器"""
        # 子类实现
        pass

    def _disable_counters(self) -> None:
        """禁用计数器"""
        # 子类实现
        pass

    def _cleanup_counters(self) -> None:
        """清理计数器资源"""
        # 子类实现
        pass

    def sample(self) -> SampleData:
        """执行采样"""
        if self._state != SamplerState.RUNNING:
            return SampleData(timestamp=time.time())

        try:
            events = self._read_events()
            metrics = self._compute_metrics(events)

            # 确保events是List[Dict[str, Any]]类型
            event_dicts: List[Dict[str, Any]] = []
            for e in events:
                if hasattr(e, '__dict__'):
                    event_dicts.append(e.__dict__)
                elif isinstance(e, dict):
                    event_dicts.append(e)
                else:
                    event_dicts.append({})
            
            data = SampleData(
                timestamp=time.time(),
                metrics=metrics,
                events=event_dicts
            )

            self._add_to_buffer(data)
            return data

        except Exception as e:
            self._record_error(e)
            return SampleData(timestamp=time.time())

    def _read_events(self) -> List[Any]:
        """读取事件"""
        # 子类实现
        return []

    def _compute_metrics(self, events: List[Any]) -> Dict[str, Any]:
        """计算指标"""
        if not events:
            return {}

        # 计算基本统计
        total_accesses = len(events)
        read_count = sum(1 for e in events if e.access_type == 'read')
        write_count = sum(1 for e in events if e.access_type == 'write')

        # 计算延迟统计
        latencies = [e.latency for e in events if e.latency > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # 计算NUMA分布
        numa_distribution: Dict[int, int] = {}
        for event in events:
            if event.numa_node >= 0:
                numa_distribution[event.numa_node] = numa_distribution.get(event.numa_node, 0) + 1

        return {
            'total_accesses': total_accesses,
            'read_count': read_count,
            'write_count': write_count,
            'avg_latency_ns': avg_latency,
            'numa_distribution': numa_distribution,
            'sample_rate': total_accesses / self._config.sample_interval if self._config.sample_interval > 0 else 0,
        }

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._platform_supported and self._check_permissions()

    def _check_permissions(self) -> bool:
        """检查权限"""
        # 需要检查perf_event_paranoid
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/sys/kernel/perf_event_paranoid', 'r') as f:
                    value = int(f.read().strip())
                    # 值越小权限越高，-1表示完全开放
                    return value <= 1
            except FileNotFoundError:
                logger.debug("perf_event_paranoid file not found")
                return False
            except (ValueError, IOError) as e:
                logger.debug(f"Failed to read perf_event_paranoid: {e}")
                return False
            except Exception as e:
                logger.warning(f"Unexpected error checking perf permissions: {e}")
                return False
        return False


class PEBSSampler(HardwareSampler):
    """
    Intel PEBS采样器

    使用Intel Precise Event-Based Sampling进行精确的内存访问采样。

    PEBS (Precise Event-Based Sampling) 是Intel处理器提供的精确事件采样机制，
    可以精确捕获内存访问事件的地址、延迟等信息。

    算法思路：
    1. 配置PEBS事件（如mem_inst_retired.all）
    2. 设置采样周期
    3. 读取PEBS记录，包含精确的内存地址和延迟
    4. 聚合分析访问模式
    """

    # PEBS事件类型
    PEBS_EVENTS = {
        'mem_inst_retired.all': 0x05,        # 所有内存指令
        'mem_inst_retired.loads': 0x01,      # 加载指令
        'mem_inst_retired.stores': 0x02,     # 存储指令
        'mem_inst_retired.loads_stores': 0x03,  # 加载和存储
        'mem_load_retired.l1_hit': 0x01,     # L1命中
        'mem_load_retired.l1_miss': 0x08,    # L1未命中
        'mem_load_retired.l2_hit': 0x02,     # L2命中
        'mem_load_retired.l2_miss': 0x10,    # L2未命中
        'mem_load_retired.l3_hit': 0x04,     # L3命中
        'mem_load_retired.l3_miss': 0x20,    # L3未命中
    }

    def __init__(self, config):
        super().__init__(config)
        self._pebs_event = config.pebs_event
        self._pebs_period = config.pebs_period
        self._precise_level = 3  # PEBS精确级别

        # PEBS记录缓冲区
        self._pebs_buffer: List[Dict[str, Any]] = []

        # 访问模式追踪
        self._access_pattern: Dict[int, int] = {}  # address -> count

    def _setup_counters(self) -> None:
        """设置PEBS计数器"""
        # 检查事件是否支持
        if self._pebs_event not in self.PEBS_EVENTS:
            raise ValueError(f"Unsupported PEBS event: {self._pebs_event}")

        # 配置硬件计数器
        self._counter_config = HardwareCounterConfig(
            event_name=self._pebs_event,
            period=self._pebs_period,
            precise=self._precise_level,
            sample_type=self._get_sample_type()
        )

        # 实际的perf_event_open调用
        # 这里使用简化的实现
        self._setup_perf_event()

    def _get_sample_type(self) -> int:
        """获取采样类型标志"""
        # PERF_SAMPLE_IP | PERF_SAMPLE_ADDR | PERF_SAMPLE_DATA_SRC |
        # PERF_SAMPLE_WEIGHT | PERF_SAMPLE_TRANSACTION
        return (1 << 0) | (1 << 3) | (1 << 10) | (1 << 14) | (1 << 17)

    def _setup_perf_event(self):
        """设置perf事件"""
        # 使用perf_event_open系统调用
        # 这里提供Python实现框架
        try:
            import ctypes
            import ctypes.util

            # 加载libc
            libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

            # perf_event_attr结构体
            class PerfEventAttr(ctypes.Structure):
                _fields_ = [
                    ('type', ctypes.c_uint32),
                    ('size', ctypes.c_uint32),
                    ('config', ctypes.c_uint64),
                    ('sample_period', ctypes.c_uint64),
                    ('sample_type', ctypes.c_uint64),
                    ('read_format', ctypes.c_uint64),
                    ('flags', ctypes.c_uint64),
                    ('wakeup_events', ctypes.c_uint32),
                    ('bp_type', ctypes.c_uint32),
                    ('config1', ctypes.c_uint64),
                    ('config2', ctypes.c_uint64),
                    ('branch_sample_type', ctypes.c_uint64),
                    ('sample_regs_user', ctypes.c_uint64),
                    ('sample_stack_user', ctypes.c_uint32),
                    ('clockid', ctypes.c_int32),
                    ('sample_regs_intr', ctypes.c_uint64),
                    ('aux_watermark', ctypes.c_uint32),
                    ('sample_max_stack', ctypes.c_uint16),
                    ('__reserved_2', ctypes.c_uint16),
                ]

            attr = PerfEventAttr()
            attr.type = 0  # PERF_TYPE_HARDWARE
            attr.size = ctypes.sizeof(PerfEventAttr)
            attr.config = self.PEBS_EVENTS.get(self._pebs_event, 0)
            attr.sample_period = self._pebs_period
            attr.sample_type = self._get_sample_type()
            attr.precise_ip = self._precise_level

            # perf_event_open系统调用
            # fd = syscall(__NR_perf_event_open, &attr, -1, 0, -1, 0)
            # 这里简化处理

        except ImportError as e:
            # 如果无法设置perf事件，使用模拟模式
            self._use_simulation = True
            logger.debug(f"ctypes not available, using simulation mode: {e}")
        except Exception as e:
            self._use_simulation = True
            logger.warning(f"Failed to setup perf event, using simulation mode: {e}")

    def _read_events(self) -> List[Any]:
        """读取PEBS事件"""
        events = []

        # 从PEBS缓冲区读取记录
        for record in self._pebs_buffer:
            event = MemoryAccessEvent(
                timestamp=record.get('timestamp', time.time()),
                address=record.get('address', 0),
                size=record.get('size', 8),
                access_type=record.get('access_type', 'read'),
                latency=record.get('latency', 0),
                numa_node=record.get('numa_node', -1),
                cpu=record.get('cpu', -1),
            )
            events.append(event)

            # 更新访问模式
            page_addr = event.address & ~0xFFF  # 页对齐
            self._access_pattern[page_addr] = self._access_pattern.get(page_addr, 0) + 1

        # 清空缓冲区
        self._pebs_buffer.clear()

        return events

    def get_access_pattern(self) -> Dict[int, int]:
        """获取访问模式"""
        return self._access_pattern.copy()

    def get_hot_pages(self, threshold: int = 100) -> List[int]:
        """获取热页面"""
        return [addr for addr, count in self._access_pattern.items() if count >= threshold]


class IBSSampler(HardwareSampler):
    """
    AMD IBS采样器

    使用AMD Instruction-Based Sampling进行指令级采样。

    IBS (Instruction-Based Sampling) 是AMD处理器提供的指令采样机制，
    可以捕获指令执行和内存访问的详细信息。

    算法思路：
    1. 配置IBS Op采样或Fetch采样
    2. 设置采样周期
    3. 读取IBS记录，包含指令地址、数据地址、延迟等
    4. 分析指令级内存访问模式
    """

    # IBS配置
    IBS_OP_ENABLE = 0x00010000
    IBS_FETCH_ENABLE = 0x00020000

    def __init__(self, config):
        super().__init__(config)
        self._ibs_op_enable = config.ibs_op_enable
        self._ibs_fetch_enable = config.ibs_fetch_enable
        self._ibs_max_cnt = 0x100000  # 最大计数

        # IBS记录缓冲区
        self._ibs_buffer: List[Dict[str, Any]] = []

    def _setup_counters(self) -> None:
        """设置IBS计数器"""
        # 配置IBS MSRs
        # 这里使用模拟实现
        self._counter_config = HardwareCounterConfig(
            event_name='ibs',
            period=self._ibs_max_cnt,
            precise=2
        )

    def _enable_counters(self) -> None:
        """启用IBS"""
        # 写入MSR启用IBS
        # 实际实现需要内核模块或perf支持
        pass

    def _read_events(self) -> List[Any]:
        """读取IBS事件"""
        events = []

        for record in self._ibs_buffer:
            event = MemoryAccessEvent(
                timestamp=record.get('timestamp', time.time()),
                address=record.get('data_address', 0),
                size=record.get('size', 8),
                access_type=record.get('access_type', 'read'),
                latency=record.get('latency', 0),
                cpu=record.get('cpu', -1),
            )
            events.append(event)

        self._ibs_buffer.clear()
        return events

    def sample(self) -> SampleData:
        """执行采样"""
        if self._state != SamplerState.RUNNING:
            return SampleData(timestamp=time.time())

        # 模拟IBS采样数据
        # 实际实现需要从内核读取IBS记录
        events = self._simulate_ibs_sample()
        metrics = self._compute_metrics(events)

        # 确保events是List[Dict[str, Any]]类型
        event_dicts: List[Dict[str, Any]] = []
        for e in events:
            if hasattr(e, '__dict__'):
                event_dicts.append(e.__dict__)
            elif isinstance(e, dict):
                event_dicts.append(e)
            else:
                event_dicts.append({})
        
        data = SampleData(
            timestamp=time.time(),
            metrics=metrics,
            events=event_dicts
        )

        self._add_to_buffer(data)
        return data

    def _simulate_ibs_sample(self) -> List[Any]:
        """模拟IBS采样（用于不支持IBS的环境）"""
        # 返回空列表，实际实现需要真实的IBS支持
        return []


class IntelPTSampler(HardwareSampler):
    """
    Intel PT (Processor Trace) 采样器

    使用Intel处理器追踪技术进行控制流和内存访问分析。

    Intel PT可以追踪完整的控制流，结合内存访问信息进行深度分析。
    """

    def __init__(self, config):
        super().__init__(config)
        self._pt_buffer_size = 4 * 1024 * 1024  # 4MB缓冲区
        self._trace_data: bytes = b''

    def _setup_counters(self) -> None:
        """设置Intel PT"""
        # 配置PT追踪
        pass

    def _read_events(self) -> List[Any]:
        """从PT追踪数据中提取内存事件"""
        # 解析PT数据包，提取内存访问信息
        return []

    def get_trace_data(self) -> bytes:
        """获取原始追踪数据"""
        return self._trace_data
