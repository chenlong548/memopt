"""
mem_monitor 软件采样器模块

实现基于软件的内存采样方法。
支持tracemalloc、perf、eBPF等。
"""

import time
import sys
import os
import traceback
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager

from .base import SamplerBase, SampleData, SamplerState, MemoryAccessEvent


class SoftwareSampler(SamplerBase):
    """
    软件采样器基类

    提供基于软件的内存采样基础功能。
    """

    def __init__(self, config):
        super().__init__(config)
        self._sampling_active = False

    def start(self) -> None:
        """启动采样"""
        self._state = SamplerState.RUNNING
        self._start_time = time.time()
        self._sampling_active = True

    def stop(self) -> None:
        """停止采样"""
        self._sampling_active = False
        self._state = SamplerState.STOPPED

    def sample(self) -> SampleData:
        """执行采样"""
        # 基础实现，子类覆盖
        return SampleData(timestamp=time.time())


class TracemallocSampler(SoftwareSampler):
    """
    Tracemalloc采样器

    使用Python的tracemalloc模块进行内存分配追踪。

    算法思路：
    1. 启用tracemalloc追踪
    2. 定期获取内存快照
    3. 比较快照差异，识别分配热点
    4. 统计各代码路径的内存分配量

    优点：
    - 纯Python实现，跨平台
    - 可以获取完整的调用栈
    - 精确追踪Python对象

    缺点：
    - 有一定性能开销
    - 只能追踪Python对象，不能追踪C扩展分配
    """

    def __init__(self, config):
        super().__init__(config)
        self._frames = config.tracemalloc_frames
        self._snapshot = None
        self._previous_snapshot = None
        self._allocation_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'size': 0}
        )
        self._top_allocations: List[Tuple[str, int, int]] = []

    def start(self) -> None:
        """启动tracemalloc采样"""
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start(self._frames)

        self._snapshot = tracemalloc.take_snapshot()
        self._previous_snapshot = None
        self._state = SamplerState.RUNNING
        self._start_time = time.time()

    def stop(self) -> None:
        """停止采样"""
        import tracemalloc

        if tracemalloc.is_tracing():
            tracemalloc.stop()

        self._state = SamplerState.STOPPED

    def sample(self) -> SampleData:
        """执行采样"""
        import tracemalloc

        if not tracemalloc.is_tracing():
            return SampleData(timestamp=time.time())

        try:
            # 保存前一个快照
            self._previous_snapshot = self._snapshot

            # 获取新快照
            self._snapshot = tracemalloc.take_snapshot()

            # 计算差异
            metrics = self._compute_metrics()

            data = SampleData(
                timestamp=time.time(),
                metrics=metrics,
                metadata={'frames': self._frames}
            )

            self._add_to_buffer(data)
            return data

        except Exception as e:
            self._record_error(e)
            return SampleData(timestamp=time.time())

    def _compute_metrics(self) -> Dict[str, Any]:
        """计算指标"""
        import tracemalloc

        metrics = {}

        # 获取当前内存使用
        current_mem = tracemalloc.get_traced_memory()
        metrics['heap_size'] = current_mem[1]  # 峰值
        metrics['heap_used'] = current_mem[0]  # 当前

        # 获取统计信息
        stats = tracemalloc.get_tracemalloc_memory()
        metrics['tracemalloc_overhead'] = stats

        # 如果有前一个快照，计算差异
        if self._previous_snapshot is not None:
            diff = self._snapshot.compare_to(self._previous_snapshot, 'traceback')

            total_diff_size = 0
            total_diff_count = 0

            for stat in diff[:100]:  # 只处理前100个
                size_diff = stat.size_diff
                count_diff = stat.count_diff

                total_diff_size += size_diff
                total_diff_count += count_diff

                # 记录分配热点
                if stat.traceback:
                    trace_str = str(stat.traceback[0])
                    self._allocation_stats[trace_str]['count'] += count_diff
                    self._allocation_stats[trace_str]['size'] += size_diff

            metrics['allocation_diff_size'] = total_diff_size
            metrics['allocation_diff_count'] = total_diff_count

        # 获取top分配
        top_stats = self._snapshot.statistics('traceback')[:10]
        self._top_allocations = [
            (str(stat.traceback[0]) if stat.traceback else 'unknown',
             stat.size, stat.count)
            for stat in top_stats
        ]

        metrics['top_allocation_count'] = len(self._top_allocations)

        return metrics

    def get_top_allocations(self, n: int = 10) -> List[Tuple[str, int, int]]:
        """
        获取top N分配

        Args:
            n: 返回数量

        Returns:
            List[Tuple]: (traceback, size, count) 列表
        """
        return self._top_allocations[:n]

    def get_allocation_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取分配统计"""
        return dict(self._allocation_stats)

    def get_current_data(self) -> Dict[str, Any]:
        """获取当前数据"""
        data = super().get_current_data()
        data['top_allocations'] = self._top_allocations[:10]
        return data

    def is_available(self) -> bool:
        """检查是否可用"""
        try:
            import tracemalloc
            return True
        except ImportError:
            return False


class PerfSampler(SoftwareSampler):
    """
    Perf采样器

    使用Linux perf子系统进行内存采样。

    算法思路：
    1. 通过perf_event_open系统调用创建性能计数器
    2. 配置内存相关事件（如cache-misses, page-faults）
    3. 定期读取计数器值
    4. 分析内存性能指标

    支持的事件：
    - cache-references: 缓存引用
    - cache-misses: 缓存未命中
    - LLC-loads: 最后一级缓存加载
    - LLC-stores: 最后一级缓存存储
    - page-faults: 页错误
    """

    # 性能事件定义
    PERF_EVENTS = {
        'cache-references': 0,
        'cache-misses': 1,
        'LLC-loads': 2,
        'LLC-stores': 3,
        'page-faults': 4,
        'major-faults': 5,
        'minor-faults': 6,
    }

    def __init__(self, config):
        super().__init__(config)
        self._event_fds: Dict[str, int] = {}
        self._event_values: Dict[str, int] = {}
        self._platform_supported = sys.platform.startswith('linux')

    def start(self) -> None:
        """启动perf采样"""
        if not self._platform_supported:
            self._state = SamplerState.ERROR
            return

        try:
            self._setup_perf_events()
            self._state = SamplerState.RUNNING
            self._start_time = time.time()
        except Exception as e:
            self._state = SamplerState.ERROR
            self._record_error(e)

    def _setup_perf_events(self):
        """设置perf事件"""
        # 尝试读取/proc文件系统获取内存信息
        # 作为perf的替代方案
        pass

    def stop(self) -> None:
        """停止采样"""
        for fd in self._event_fds.values():
            try:
                os.close(fd)
            except Exception:
                pass
        self._event_fds.clear()
        self._state = SamplerState.STOPPED

    def sample(self) -> SampleData:
        """执行采样"""
        if self._state != SamplerState.RUNNING:
            return SampleData(timestamp=time.time())

        try:
            metrics = self._read_perf_events()

            # 添加/proc/meminfo数据
            metrics.update(self._read_proc_meminfo())

            # 添加/proc/self/status数据
            metrics.update(self._read_proc_status())

            data = SampleData(
                timestamp=time.time(),
                metrics=metrics
            )

            self._add_to_buffer(data)
            return data

        except Exception as e:
            self._record_error(e)
            return SampleData(timestamp=time.time())

    def _read_perf_events(self) -> Dict[str, Any]:
        """读取perf事件"""
        metrics = {}

        # 如果有真实的perf文件描述符，读取它们
        for event_name, fd in self._event_fds.items():
            try:
                # 读取计数器值
                value = os.read(fd, 8)
                metrics[event_name] = int.from_bytes(value, 'little')
            except Exception:
                pass

        return metrics

    def _read_proc_meminfo(self) -> Dict[str, Any]:
        """读取/proc/meminfo"""
        metrics = {}

        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1]) * 1024  # kB to bytes

                        if key == 'MemTotal':
                            metrics['system_total'] = value
                        elif key == 'MemFree':
                            metrics['system_free'] = value
                        elif key == 'MemAvailable':
                            metrics['system_available'] = value
                        elif key == 'Buffers':
                            metrics['buffers'] = value
                        elif key == 'Cached':
                            metrics['cached'] = value
        except Exception:
            pass

        return metrics

    def _read_proc_status(self) -> Dict[str, Any]:
        """读取/proc/self/status"""
        metrics = {}

        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        metrics['rss'] = int(line.split()[1]) * 1024
                    elif line.startswith('VmSize:'):
                        metrics['vms'] = int(line.split()[1]) * 1024
                    elif line.startswith('VmData:'):
                        metrics['data'] = int(line.split()[1]) * 1024
                    elif line.startswith('VmStk:'):
                        metrics['stack'] = int(line.split()[1]) * 1024
        except Exception:
            pass

        return metrics

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._platform_supported


class EBPFSampler(SoftwareSampler):
    """
    eBPF采样器

    使用eBPF (extended Berkeley Packet Filter) 进行内核级内存追踪。

    算法思路：
    1. 加载eBPF程序到内核
    2. 在内存分配/释放函数上设置探针
    3. 收集内存事件到perf buffer
    4. 用户空间程序读取并分析事件

    支持的追踪点：
    - kmalloc/kfree: 内核内存分配
    - __do_page_fault: 页错误
    - brk: 堆扩展
    - mmap/munmap: 内存映射

    优点：
    - 极低开销
    - 可以追踪内核和用户空间
    - 实时性强

    缺点：
    - 需要root权限
    - 仅支持Linux
    - 需要bcc工具链
    """

    def __init__(self, config):
        super().__init__(config)
        self._bpf_program = config.ebpf_program
        self._bpf = None
        self._events: List[Dict[str, Any]] = []
        self._platform_supported = self._check_platform()

    def _check_platform(self) -> bool:
        """检查平台支持"""
        if not sys.platform.startswith('linux'):
            return False

        # 检查bcc是否可用
        try:
            from bcc import BPF
            return True
        except ImportError:
            return False

    def start(self) -> None:
        """启动eBPF采样"""
        if not self._platform_supported:
            self._state = SamplerState.ERROR
            return

        try:
            self._load_bpf_program()
            self._state = SamplerState.RUNNING
            self._start_time = time.time()
        except Exception as e:
            self._state = SamplerState.ERROR
            self._record_error(e)

    def _load_bpf_program(self):
        """加载eBPF程序"""
        try:
            from bcc import BPF

            # 默认的内存追踪程序
            bpf_program = self._bpf_program or self._get_default_bpf_program()

            self._bpf = BPF(text=bpf_program)

            # 设置perf buffer回调
            self._bpf['events'].open_perf_buffer(self._handle_event)

        except Exception as e:
            raise

    def _get_default_bpf_program(self) -> str:
        """获取默认eBPF程序"""
        return """
        #include <uapi/linux/ptrace.h>
        #include <linux/sched.h>

        struct event_t {
            u32 pid;
            u64 addr;
            u64 size;
            u64 timestamp;
            char comm[16];
        };

        BPF_PERF_OUTPUT(events);

        int trace_alloc(struct pt_regs *ctx, size_t size) {
            struct event_t event = {};
            u64 pid_tgid = bpf_get_current_pid_tgid();

            event.pid = pid_tgid >> 32;
            event.size = size;
            event.timestamp = bpf_ktime_get_ns();
            bpf_get_current_comm(&event.comm, sizeof(event.comm));

            events.perf_submit(ctx, &event, sizeof(event));
            return 0;
        }
        """

    def _handle_event(self, cpu, data, size):
        """处理eBPF事件"""
        try:
            from bcc import BPF

            event = self._bpf['events'].event(data)
            self._events.append({
                'pid': event.pid,
                'addr': event.addr,
                'size': event.size,
                'timestamp': event.timestamp,
                'comm': event.comm.decode('utf-8', errors='ignore'),
            })
        except Exception:
            pass

    def stop(self) -> None:
        """停止采样"""
        if self._bpf:
            del self._bpf
            self._bpf = None
        self._state = SamplerState.STOPPED

    def sample(self) -> SampleData:
        """执行采样"""
        if self._state != SamplerState.RUNNING:
            return SampleData(timestamp=time.time())

        try:
            # 轮询eBPF事件
            if self._bpf:
                self._bpf.perf_buffer_poll(timeout=100)

            # 计算指标
            metrics = self._compute_metrics()

            data = SampleData(
                timestamp=time.time(),
                metrics=metrics,
                events=self._events.copy()
            )

            self._events.clear()
            self._add_to_buffer(data)
            return data

        except Exception as e:
            self._record_error(e)
            return SampleData(timestamp=time.time())

    def _compute_metrics(self) -> Dict[str, Any]:
        """计算指标"""
        if not self._events:
            return {}

        total_size = sum(e['size'] for e in self._events)
        event_count = len(self._events)

        # 按进程分组
        by_process: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {'count': 0, 'size': 0}
        )
        for event in self._events:
            pid = event['pid']
            by_process[pid]['count'] += 1
            by_process[pid]['size'] += event['size']

        return {
            'total_events': event_count,
            'total_size': total_size,
            'avg_size': total_size / event_count if event_count > 0 else 0,
            'process_count': len(by_process),
            'top_processes': sorted(
                [(pid, data['size']) for pid, data in by_process.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._platform_supported


class SamplingProfiler(SoftwareSampler):
    """
    采样分析器

    使用采样方法进行内存分析，平衡精度和开销。

    算法思路：
    1. 按固定间隔采样当前内存状态
    2. 记录调用栈信息
    3. 聚合分析内存使用模式
    4. 识别内存热点
    """

    def __init__(self, config):
        super().__init__(config)
        self._stack_samples: List[Tuple[str, int]] = []
        self._sample_rate = config.sample_interval

    def sample(self) -> SampleData:
        """执行采样"""
        # 获取当前调用栈
        stack = self._get_current_stack()

        # 获取当前内存使用
        import sys
        mem_usage = sys.getsizeof(None)  # 基础开销

        self._stack_samples.append((stack, mem_usage))

        metrics = {
            'sample_count': len(self._stack_samples),
            'current_stack': stack,
        }

        data = SampleData(
            timestamp=time.time(),
            metrics=metrics
        )

        self._add_to_buffer(data)
        return data

    def _get_current_stack(self) -> str:
        """获取当前调用栈"""
        stack = traceback.extract_stack()
        # 只保留最近的几帧
        frames = stack[-5:-1] if len(stack) > 5 else stack[:-1]
        return ' -> '.join(f"{f.filename}:{f.lineno}" for f in frames)

    def get_stack_summary(self) -> Dict[str, int]:
        """获取调用栈摘要"""
        summary: Dict[str, int] = defaultdict(int)
        for stack, _ in self._stack_samples:
            summary[stack] += 1
        return dict(summary)
