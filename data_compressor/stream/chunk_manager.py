"""
分块管理器

管理流式压缩的数据分块。
"""

import logging
import struct
from typing import List, Dict, Any, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


class ChunkManager:
    """
    分块管理器

    管理数据分块的创建、索引和重组。
    """

    def __init__(self,
                 chunk_size: int = 1024 * 1024,  # 1MB
                 enable_dedup: bool = True,
                 enable_checksum: bool = True):
        """
        初始化分块管理器

        Args:
            chunk_size: 分块大小（字节）
            enable_dedup: 是否启用去重
            enable_checksum: 是否启用校验和
        """
        self.chunk_size = chunk_size
        self.enable_dedup = enable_dedup
        self.enable_checksum = enable_checksum

        # 分块索引
        self._chunks: Dict[int, Dict[str, Any]] = {}

        # 去重索引（哈希 -> 分块ID）
        self._dedup_index: Dict[str, int] = {}

        # 统计信息
        self._stats = {
            'total_chunks': 0,
            'total_bytes': 0,
            'dedup_chunks': 0,
            'dedup_bytes': 0,
        }

        logger.info(
            f"ChunkManager initialized: chunk_size={chunk_size}, "
            f"dedup={enable_dedup}, checksum={enable_checksum}"
        )

    def create_chunk(self,
                    data: bytes,
                    chunk_index: int,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建分块

        Args:
            data: 分块数据
            chunk_index: 分块索引
            metadata: 元数据

        Returns:
            Dict: 分块信息
        """
        # 计算校验和
        checksum = None
        if self.enable_checksum:
            checksum = self._calculate_checksum(data)

        # 检查去重
        dedup_id = None
        if self.enable_dedup and checksum:
            if checksum in self._dedup_index:
                dedup_id = self._dedup_index[checksum]
                self._stats['dedup_chunks'] += 1
                self._stats['dedup_bytes'] += len(data)

        # 创建分块信息
        chunk_info = {
            'index': chunk_index,
            'size': len(data),
            'checksum': checksum,
            'dedup_id': dedup_id,
            'metadata': metadata or {},
            'created_at': self._get_timestamp(),
        }

        # 如果不是去重分块，存储数据
        if dedup_id is None:
            chunk_info['data'] = data

            # 添加到去重索引
            if self.enable_dedup and checksum:
                self._dedup_index[checksum] = chunk_index

        # 存储分块信息
        self._chunks[chunk_index] = chunk_info

        # 更新统计
        self._stats['total_chunks'] += 1
        self._stats['total_bytes'] += len(data)

        logger.debug(
            f"Chunk {chunk_index} created: size={len(data)}, "
            f"checksum={checksum[:8] if checksum else 'N/A'}..., "
            f"dedup={dedup_id is not None}"
        )

        return chunk_info

    def get_chunk(self, chunk_index: int) -> Optional[bytes]:
        """
        获取分块数据

        Args:
            chunk_index: 分块索引

        Returns:
            bytes: 分块数据，不存在返回None
        """
        if chunk_index not in self._chunks:
            return None

        chunk_info = self._chunks[chunk_index]

        # 如果是去重分块，从原始分块获取数据
        if chunk_info.get('dedup_id') is not None:
            original_index = chunk_info['dedup_id']
            if original_index in self._chunks:
                return self._chunks[original_index].get('data')
            else:
                logger.warning(f"Dedup source chunk {original_index} not found")
                return None

        return chunk_info.get('data')

    def get_chunk_info(self, chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        获取分块信息

        Args:
            chunk_index: 分块索引

        Returns:
            Dict: 分块信息，不存在返回None
        """
        return self._chunks.get(chunk_index)

    def verify_chunk(self, chunk_index: int) -> bool:
        """
        验证分块完整性

        Args:
            chunk_index: 分块索引

        Returns:
            bool: 验证是否通过
        """
        if chunk_index not in self._chunks:
            return False

        chunk_info = self._chunks[chunk_index]

        # 如果没有校验和，跳过验证
        if not self.enable_checksum or chunk_info.get('checksum') is None:
            return True

        # 获取数据
        data = self.get_chunk(chunk_index)
        if data is None:
            return False

        # 计算并比较校验和
        actual_checksum = self._calculate_checksum(data)
        expected_checksum = chunk_info['checksum']

        if actual_checksum != expected_checksum:
            logger.error(
                f"Chunk {chunk_index} checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
            return False

        return True

    def split_data(self, data: bytes) -> List[Tuple[int, bytes]]:
        """
        分割数据为多个分块

        Args:
            data: 原始数据

        Returns:
            List: [(分块索引, 分块数据), ...]
        """
        chunks = []
        offset = 0
        chunk_index = 0

        while offset < len(data):
            # 提取分块
            chunk_data = data[offset:offset + self.chunk_size]
            
            # 创建并存储分块
            self.create_chunk(chunk_data, chunk_index)
            
            chunks.append((chunk_index, chunk_data))

            offset += self.chunk_size
            chunk_index += 1

        return chunks

    def merge_chunks(self, chunk_indices: List[int]) -> bytes:
        """
        合并多个分块

        Args:
            chunk_indices: 分块索引列表

        Returns:
            bytes: 合并后的数据
        """
        merged = bytearray()

        for index in chunk_indices:
            chunk_data = self.get_chunk(index)
            if chunk_data is None:
                logger.warning(f"Chunk {index} not found during merge")
                continue

            merged.extend(chunk_data)

        return bytes(merged)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'chunk_size': self.chunk_size,
            'total_chunks': self._stats['total_chunks'],
            'total_bytes': self._stats['total_bytes'],
            'total_bytes_mb': self._stats['total_bytes'] / (1024 * 1024),
            'dedup_chunks': self._stats['dedup_chunks'],
            'dedup_bytes': self._stats['dedup_bytes'],
            'dedup_ratio': (
                self._stats['dedup_bytes'] / self._stats['total_bytes']
                if self._stats['total_bytes'] > 0 else 0
            ),
            'unique_chunks': len(self._chunks),
            'dedup_index_size': len(self._dedup_index),
        }

    def clear(self):
        """清除所有分块"""
        self._chunks.clear()
        self._dedup_index.clear()

        # 重置统计
        self._stats = {
            'total_chunks': 0,
            'total_bytes': 0,
            'dedup_chunks': 0,
            'dedup_bytes': 0,
        }

        logger.debug("All chunks cleared")

    def _calculate_checksum(self, data: bytes) -> str:
        """
        计算校验和

        Args:
            data: 数据

        Returns:
            str: 校验和（十六进制字符串）
        """
        return hashlib.sha256(data).hexdigest()

    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()

    def serialize_chunk_info(self, chunk_index: int) -> bytes:
        """
        序列化分块信息

        Args:
            chunk_index: 分块索引

        Returns:
            bytes: 序列化后的数据
        """
        chunk_info = self.get_chunk_info(chunk_index)
        if chunk_info is None:
            return b''

        result = b''

        # 分块索引
        result += struct.pack('I', chunk_info['index'])

        # 分块大小
        result += struct.pack('I', chunk_info['size'])

        # 校验和
        checksum = chunk_info.get('checksum', '')
        checksum_bytes = checksum.encode('utf-8')
        result += struct.pack('I', len(checksum_bytes))
        result += checksum_bytes

        # 去重ID
        dedup_id = chunk_info.get('dedup_id')
        result += struct.pack('i', dedup_id if dedup_id is not None else -1)

        return result

    def deserialize_chunk_info(self, data: bytes) -> Dict[str, Any]:
        """
        反序列化分块信息

        Args:
            data: 序列化数据

        Returns:
            Dict: 分块信息
        """
        offset = 0

        # 分块索引
        chunk_index = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # 分块大小
        chunk_size = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # 校验和
        checksum_len = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        checksum = data[offset:offset+checksum_len].decode('utf-8')
        offset += checksum_len

        # 去重ID
        dedup_id = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4

        return {
            'index': chunk_index,
            'size': chunk_size,
            'checksum': checksum,
            'dedup_id': dedup_id if dedup_id >= 0 else None,
        }
