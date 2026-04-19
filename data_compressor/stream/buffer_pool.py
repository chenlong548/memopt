"""
缓冲池管理

管理流式压缩的内存缓冲区。
"""

import logging
import threading
from typing import Optional, List
import time

logger = logging.getLogger(__name__)


class BufferPool:
    """
    缓冲池

    管理固定大小的内存缓冲区，支持并发访问。
    """

    def __init__(self,
                 buffer_size: int = 1024 * 1024,  # 1MB
                 num_buffers: int = 4):
        """
        初始化缓冲池

        Args:
            buffer_size: 每个缓冲区的大小（字节）
            num_buffers: 缓冲区数量
        """
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers

        # 缓冲区列表
        self._buffers: List[bytearray] = [
            bytearray(buffer_size) for _ in range(num_buffers)
        ]

        # 可用缓冲区索引
        self._available: List[int] = list(range(num_buffers))

        # 锁
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # 统计信息
        self._stats = {
            'total_acquisitions': 0,
            'total_releases': 0,
            'total_wait_time': 0.0,
            'peak_usage': 0,
        }

        logger.info(
            f"BufferPool initialized: {num_buffers} buffers x {buffer_size} bytes"
        )

    def acquire(self, timeout: Optional[float] = None) -> Optional[bytearray]:
        """
        获取一个缓冲区

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            bytearray: 缓冲区，超时返回None
        """
        start_time = time.time()

        with self._condition:
            # 等待可用缓冲区
            while not self._available:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning("Buffer acquisition timeout")
                        return None

                    # 计算剩余等待时间
                    remaining = timeout - elapsed
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

            # 获取缓冲区索引
            index = self._available.pop(0)
            buffer = self._buffers[index]

            # 更新统计
            self._stats['total_acquisitions'] += 1
            self._stats['total_wait_time'] += time.time() - start_time
            self._stats['peak_usage'] = max(
                self._stats['peak_usage'],
                self.num_buffers - len(self._available)
            )

            logger.debug(f"Buffer {index} acquired")

            return buffer

    def release(self, buffer: bytearray):
        """
        释放缓冲区

        Args:
            buffer: 要释放的缓冲区
        """
        with self._condition:
            # 查找缓冲区索引
            for i, buf in enumerate(self._buffers):
                if buf is buffer:
                    # 添加回可用列表
                    self._available.append(i)

                    # 清空缓冲区
                    # buf[:] = b'\x00' * len(buf)  # 可选：清空缓冲区

                    # 更新统计
                    self._stats['total_releases'] += 1

                    # 通知等待的线程
                    self._condition.notify()

                    logger.debug(f"Buffer {i} released")
                    return

            logger.warning("Attempted to release unknown buffer")

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            dict: 统计信息
        """
        with self._lock:
            return {
                'buffer_size': self.buffer_size,
                'num_buffers': self.num_buffers,
                'available_buffers': len(self._available),
                'in_use_buffers': self.num_buffers - len(self._available),
                'total_acquisitions': self._stats['total_acquisitions'],
                'total_releases': self._stats['total_releases'],
                'avg_wait_time_ms': (
                    self._stats['total_wait_time'] * 1000 /
                    self._stats['total_acquisitions']
                ) if self._stats['total_acquisitions'] > 0 else 0,
                'peak_usage': self._stats['peak_usage'],
            }

    def resize(self, new_size: int):
        """
        调整缓冲区大小

        Args:
            new_size: 新的缓冲区大小
        """
        with self._lock:
            # 重新创建缓冲区
            self._buffers = [
                bytearray(new_size) for _ in range(self.num_buffers)
            ]
            self.buffer_size = new_size

            logger.info(f"BufferPool resized to {new_size} bytes per buffer")

    def clear(self):
        """清空所有缓冲区"""
        with self._lock:
            for buffer in self._buffers:
                buffer[:] = b'\x00' * len(buffer)

            logger.debug("All buffers cleared")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.clear()
        return False


class BufferView:
    """
    缓冲区视图

    提供对缓冲区的只读或读写视图。
    """

    def __init__(self,
                 buffer: bytearray,
                 offset: int = 0,
                 size: Optional[int] = None):
        """
        初始化缓冲区视图

        Args:
            buffer: 底层缓冲区
            offset: 起始偏移
            size: 视图大小，None表示到缓冲区末尾
        """
        self.buffer = buffer
        self.offset = offset
        self.size = size if size is not None else len(buffer) - offset

        # 边界检查
        if offset < 0 or offset >= len(buffer):
            raise ValueError(f"Invalid offset: {offset}")

        if self.size < 0 or offset + self.size > len(buffer):
            raise ValueError(f"Invalid size: {self.size}")

    def read(self, size: Optional[int] = None) -> bytes:
        """
        读取数据

        Args:
            size: 读取大小，None表示读取全部

        Returns:
            bytes: 读取的数据
        """
        read_size = size if size is not None else self.size
        read_size = min(read_size, self.size)

        return bytes(self.buffer[self.offset:self.offset + read_size])

    def write(self, data: bytes, offset: int = 0) -> int:
        """
        写入数据

        Args:
            data: 要写入的数据
            offset: 相对于视图起始的偏移

        Returns:
            int: 实际写入的字节数
        """
        write_offset = self.offset + offset
        write_size = min(len(data), self.size - offset)

        if write_size <= 0:
            return 0

        self.buffer[write_offset:write_offset + write_size] = data[:write_size]

        return write_size

    def __len__(self) -> int:
        """返回视图大小"""
        return self.size

    def __getitem__(self, index: int) -> int:
        """索引访问"""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        return self.buffer[self.offset + index]

    def __setitem__(self, index: int, value: int):
        """索引设置"""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        self.buffer[self.offset + index] = value
