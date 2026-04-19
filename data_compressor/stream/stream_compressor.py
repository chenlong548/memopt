"""
流式压缩器

支持大规模数据的流式压缩处理。
"""

import logging
from typing import Optional, BinaryIO, Callable, Any
import threading
import queue
import time

from ..core.base import (
    CompressionConfig,
    CompressionStats,
    CompressionAlgorithm
)
from ..core.exceptions import StreamError
from .buffer_pool import BufferPool
from .chunk_manager import ChunkManager

logger = logging.getLogger(__name__)


class StreamCompressor:
    """
    流式压缩器

    支持大规模数据的流式压缩，避免内存溢出。
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        初始化流式压缩器

        Args:
            config: 压缩配置
        """
        self.config = config or CompressionConfig()

        self.buffer_pool = BufferPool(
            buffer_size=self.config.chunk_size,
            num_buffers=4
        )

        self.chunk_manager = ChunkManager(chunk_size=self.config.chunk_size)

        self.stats = CompressionStats()

        self._stop_flag = threading.Event()
        self._error = None
        self._num_workers = 4

    def compress_stream(self,
                       input_stream: BinaryIO,
                       output_stream: BinaryIO,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> CompressionStats:
        """
        流式压缩

        Args:
            input_stream: 输入流
            output_stream: 输出流
            progress_callback: 进度回调函数 (current_bytes, total_bytes)

        Returns:
            CompressionStats: 压缩统计信息

        Raises:
            StreamError: 流处理错误时抛出
        """
        logger.info("Starting stream compression")

        self.stats = CompressionStats()
        self.stats.start_time = time.time()
        self._stop_flag.clear()
        self._error = None

        try:
            input_stream.seek(0, 2)
            total_size = input_stream.tell()
            input_stream.seek(0)

            processed_bytes = 0
            chunk_index = 0
            first_chunk = True
            header_written = False

            while not self._stop_flag.is_set():
                chunk = self._read_chunk(input_stream)

                if not chunk:
                    break

                compressed_chunk = self._compress_chunk(chunk, chunk_index)
                
                if first_chunk:
                    self._write_header(output_stream, total_size, self.stats.algorithm_used)
                    header_written = True
                    first_chunk = False

                self._write_chunk(output_stream, compressed_chunk, chunk_index)

                processed_bytes += len(chunk)
                chunk_index += 1

                if progress_callback:
                    progress_callback(processed_bytes, total_size)

                self.stats.original_size += len(chunk)
                self.stats.compressed_size += len(compressed_chunk)
            
            if not header_written:
                self._write_header(output_stream, total_size, self.config.algorithm)

            # 写入结束标记
            self._write_footer(output_stream)

            # 完成统计
            self.stats.end_time = time.time()
            self.stats.compression_time = self.stats.end_time - self.stats.start_time
            self.stats.calculate_ratio()
            self.stats.calculate_throughput()

            logger.info(
                f"Stream compression completed: {self.stats.original_size} -> "
                f"{self.stats.compressed_size} bytes ({self.stats.compression_ratio:.2f}x)"
            )

            return self.stats

        except Exception as e:
            logger.error(f"Stream compression failed: {e}")
            self._error = e
            raise StreamError(f"Stream compression failed: {e}") from e

    def decompress_stream(self,
                         input_stream: BinaryIO,
                         output_stream: BinaryIO,
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> CompressionStats:
        """
        流式解压

        Args:
            input_stream: 输入流
            output_stream: 输出流
            progress_callback: 进度回调函数

        Returns:
            CompressionStats: 解压统计信息

        Raises:
            StreamError: 流处理错误时抛出
        """
        logger.info("Starting stream decompression")

        self.stats = CompressionStats()
        self.stats.start_time = time.time()
        self._stop_flag.clear()
        self._error = None

        try:
            header = self._read_header(input_stream)
            total_size = header['original_size']
            
            algorithm_str = header.get('algorithm', 'zstd')
            self.stats.algorithm_used = CompressionAlgorithm(algorithm_str)

            processed_bytes = 0
            chunk_index = 0

            while not self._stop_flag.is_set():
                compressed_chunk = self._read_compressed_chunk(input_stream)

                if not compressed_chunk:
                    break

                # 解压数据块
                chunk = self._decompress_chunk(compressed_chunk, chunk_index)

                # 写入解压块
                output_stream.write(chunk)

                # 更新统计
                processed_bytes += len(chunk)
                chunk_index += 1

                # 更新进度
                if progress_callback:
                    progress_callback(processed_bytes, total_size)

                # 更新统计信息
                self.stats.original_size += len(chunk)
                self.stats.compressed_size += len(compressed_chunk)

            # 完成统计
            self.stats.end_time = time.time()
            self.stats.decompression_time = self.stats.end_time - self.stats.start_time

            logger.info(
                f"Stream decompression completed: {self.stats.compressed_size} -> "
                f"{self.stats.original_size} bytes"
            )

            return self.stats

        except Exception as e:
            logger.error(f"Stream decompression failed: {e}")
            self._error = e
            raise StreamError(f"Stream decompression failed: {e}") from e

    def compress_parallel(self,
                         input_stream: BinaryIO,
                         output_stream: BinaryIO,
                         num_workers: int = 4) -> CompressionStats:
        """
        并行流式压缩

        Args:
            input_stream: 输入流
            output_stream: 输出流
            num_workers: 工作线程数

        Returns:
            CompressionStats: 压缩统计信息
        """
        logger.info(f"Starting parallel stream compression with {num_workers} workers")

        self._num_workers = num_workers
        self.stats = CompressionStats()
        self.stats.start_time = time.time()
        self._stop_flag.clear()
        self._error = None

        try:
            input_stream.seek(0, 2)
            total_size = input_stream.tell()
            input_stream.seek(0)

            self._write_header(output_stream, total_size)

            # 创建队列
            read_queue = queue.Queue(maxsize=num_workers * 2)
            write_queue = queue.Queue(maxsize=num_workers * 2)

            # 启动读取线程
            read_thread = threading.Thread(
                target=self._reader_thread,
                args=(input_stream, read_queue)
            )
            read_thread.start()

            # 启动压缩线程
            compress_threads = []
            for i in range(num_workers):
                t = threading.Thread(
                    target=self._compressor_thread,
                    args=(read_queue, write_queue, i)
                )
                t.start()
                compress_threads.append(t)

            # 启动写入线程
            write_thread = threading.Thread(
                target=self._writer_thread,
                args=(output_stream, write_queue)
            )
            write_thread.start()

            # 等待完成
            read_thread.join()

            for t in compress_threads:
                t.join()

            write_thread.join()

            # 写入结束标记
            self._write_footer(output_stream)

            # 完成统计
            self.stats.end_time = time.time()
            self.stats.compression_time = self.stats.end_time - self.stats.start_time
            self.stats.calculate_ratio()
            self.stats.calculate_throughput()

            logger.info(f"Parallel stream compression completed")

            return self.stats

        except Exception as e:
            logger.error(f"Parallel stream compression failed: {e}")
            self._error = e
            raise StreamError(f"Parallel stream compression failed: {e}") from e

    def _read_chunk(self, stream: BinaryIO) -> bytes:
        """读取数据块"""
        buffer = self.buffer_pool.acquire()
        try:
            data = stream.read(self.config.chunk_size)
            return data
        finally:
            self.buffer_pool.release(buffer)

    def _compress_chunk(self, chunk: bytes, index: int) -> bytes:
        """压缩数据块"""
        # 获取压缩算法
        from ..core.compressor import DataCompressor
        compressor = DataCompressor(self.config)

        # 压缩
        compressed = compressor.compress(chunk, self.config)

        # 更新统计
        self.stats.algorithm_used = compressed.algorithm

        return compressed.data

    def _decompress_chunk(self, compressed: bytes, index: int) -> bytes:
        """解压数据块"""
        # 获取压缩算法
        from ..core.compressor import DataCompressor
        from ..core.base import CompressedData

        compressor = DataCompressor(self.config)

        # 创建压缩数据容器
        compressed_data = CompressedData(
            data=compressed,
            algorithm=self.stats.algorithm_used,
            level=self.config.level,
            original_size=0,  # 未知
            compressed_size=len(compressed)
        )

        # 解压
        return compressor.decompress(compressed_data)

    def _write_header(self, stream: BinaryIO, original_size: int, algorithm: Optional[CompressionAlgorithm] = None):
        """写入头部信息"""
        import struct

        stream.write(b'DCOMP')

        stream.write(struct.pack('I', 1))

        stream.write(struct.pack('Q', original_size))

        stream.write(struct.pack('I', self.config.chunk_size))

        algo = algorithm if algorithm else self.config.algorithm
        stream.write(algo.value.encode('utf-8').ljust(32, b'\x00'))

        stream.write(struct.pack('i', self.config.level.value))

    def _read_header(self, stream: BinaryIO) -> dict:
        """读取头部信息"""
        import struct

        # 魔数
        magic = stream.read(5)
        if magic != b'DCOMP':
            raise StreamError("Invalid stream format")

        # 版本
        version = struct.unpack('I', stream.read(4))[0]

        # 原始大小
        original_size = struct.unpack('Q', stream.read(8))[0]

        # 块大小
        chunk_size = struct.unpack('I', stream.read(4))[0]

        # 算法
        algorithm_bytes = stream.read(32)
        algorithm = algorithm_bytes.rstrip(b'\x00').decode('utf-8')

        # 压缩级别
        level = struct.unpack('i', stream.read(4))[0]

        return {
            'version': version,
            'original_size': original_size,
            'chunk_size': chunk_size,
            'algorithm': algorithm,
            'level': level
        }

    def _write_chunk(self, stream: BinaryIO, chunk: bytes, index: int):
        """写入压缩块"""
        import struct

        # 块大小
        stream.write(struct.pack('I', len(chunk)))

        # 块索引
        stream.write(struct.pack('I', index))

        # 块数据
        stream.write(chunk)

    def _read_compressed_chunk(self, stream: BinaryIO) -> Optional[bytes]:
        """读取压缩块"""
        import struct

        # 读取块大小
        size_bytes = stream.read(4)
        if not size_bytes:
            return None

        size = struct.unpack('I', size_bytes)[0]

        # 检查是否为结束标记
        if size == 0:
            return None

        # 读取块索引
        index = struct.unpack('I', stream.read(4))[0]

        # 读取块数据
        chunk = stream.read(size)

        return chunk

    def _write_footer(self, stream: BinaryIO):
        """写入结束标记"""
        import struct
        stream.write(struct.pack('I', 0))  # 大小为0表示结束

    def _reader_thread(self, stream: BinaryIO, queue: queue.Queue):
        """读取线程"""
        try:
            chunk_index = 0
            while not self._stop_flag.is_set():
                chunk = self._read_chunk(stream)
                if not chunk:
                    break
                queue.put((chunk_index, chunk))
                chunk_index += 1

            for _ in range(self._num_workers):
                queue.put((None, None))

        except Exception as e:
            logger.error(f"Reader thread failed: {e}")
            self._error = e

    def _compressor_thread(self, read_queue: queue.Queue, write_queue: queue.Queue, worker_id: int):
        """压缩线程"""
        try:
            while not self._stop_flag.is_set():
                chunk_index, chunk = read_queue.get()

                if chunk is None:
                    write_queue.put((None, None))
                    break

                compressed = self._compress_chunk(chunk, chunk_index)
                write_queue.put((chunk_index, compressed))
                
                self.stats.original_size += len(chunk)
                self.stats.compressed_size += len(compressed)

        except Exception as e:
            logger.error(f"Compressor thread {worker_id} failed: {e}")
            self._error = e

    def _writer_thread(self, stream: BinaryIO, queue: queue.Queue):
        """写入线程"""
        try:
            chunks = {}
            next_index = 0
            end_signals = 0

            while not self._stop_flag.is_set():
                chunk_index, chunk = queue.get()

                if chunk is None:
                    end_signals += 1
                    if end_signals >= self._num_workers:
                        break
                    continue

                chunks[chunk_index] = chunk

                while next_index in chunks:
                    self._write_chunk(stream, chunks[next_index], next_index)
                    del chunks[next_index]
                    next_index += 1

        except Exception as e:
            logger.error(f"Writer thread failed: {e}")
            self._error = e

    def stop(self):
        """停止流处理"""
        self._stop_flag.set()

    def get_stats(self) -> CompressionStats:
        """获取统计信息"""
        return self.stats
