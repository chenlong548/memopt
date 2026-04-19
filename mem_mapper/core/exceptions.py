"""
mem_mapper 异常定义模块

定义了内存映射工具中使用的所有异常类。
"""

import os
from typing import Optional


class MemMapperError(Exception):
    """
    mem_mapper基础异常类
    
    所有mem_mapper相关的异常都继承自此类。
    """
    
    def __init__(self, message: str, errno: Optional[int] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            errno: 系统错误码（可选）
        """
        self.message = message
        self.errno = errno
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.errno is not None:
            return f"{self.message} (errno={self.errno})"
        return self.message


class MMapError(MemMapperError):
    """
    内存映射错误
    
    当mmap操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化映射错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"mmap failed: {self.message}", errno)


class MUnmapError(MemMapperError):
    """
    内存解除映射错误
    
    当munmap操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化解除映射错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"munmap failed: {self.message}", errno)


class MAdviseError(MemMapperError):
    """
    内存建议错误
    
    当madvise操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化内存建议错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"madvise failed: {self.message}", errno)


class MProtectError(MemMapperError):
    """
    内存保护错误
    
    当mprotect操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化内存保护错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"mprotect failed: {self.message}", errno)


class MSyncError(MemMapperError):
    """
    内存同步错误
    
    当msync操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化内存同步错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"msync failed: {self.message}", errno)


class MLockError(MemMapperError):
    """
    内存锁定错误
    
    当mlock操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化内存锁定错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"mlock failed: {self.message}", errno)


class MUnlockError(MemMapperError):
    """
    内存解锁错误
    
    当munlock操作失败时抛出此异常。
    """
    
    def __init__(self, errno: int, message: Optional[str] = None):
        """
        初始化内存解锁错误
        
        Args:
            errno: 系统错误码
            message: 自定义错误消息（可选）
        """
        self.errno = errno
        self.message = message or os.strerror(errno)
        super().__init__(f"munlock failed: {self.message}", errno)


class NUMAError(MemMapperError):
    """
    NUMA操作错误
    
    当NUMA相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, node: Optional[int] = None):
        """
        初始化NUMA错误
        
        Args:
            message: 错误消息
            node: 相关的NUMA节点ID（可选）
        """
        self.node = node
        full_message = f"NUMA error: {message}"
        if node is not None:
            full_message += f" (node={node})"
        super().__init__(full_message)


class NUMANotSupportedError(NUMAError):
    """
    NUMA不支持错误
    
    当系统不支持NUMA时抛出此异常。
    """
    
    def __init__(self, message: str = "NUMA is not supported on this system"):
        super().__init__(message)


class NUMABindingError(NUMAError):
    """
    NUMA绑定错误
    
    当内存绑定到NUMA节点失败时抛出此异常。
    """
    
    def __init__(self, node: int, errno: Optional[int] = None):
        """
        初始化NUMA绑定错误
        
        Args:
            node: 目标NUMA节点ID
            errno: 系统错误码（可选）
        """
        message = f"Failed to bind memory to NUMA node {node}"
        if errno is not None:
            message += f": {os.strerror(errno)}"
        super().__init__(message, node)
        self.errno = errno


class HugePageError(MemMapperError):
    """
    大页操作错误
    
    当大页相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, page_size: Optional[int] = None):
        """
        初始化大页错误
        
        Args:
            message: 错误消息
            page_size: 相关的大页大小（可选）
        """
        self.page_size = page_size
        full_message = f"Huge page error: {message}"
        if page_size is not None:
            full_message += f" (page_size={page_size})"
        super().__init__(full_message)


class HugePageNotAvailableError(HugePageError):
    """
    大页不可用错误
    
    当请求的大页大小不可用时抛出此异常。
    """
    
    def __init__(self, page_size: int):
        """
        初始化大页不可用错误
        
        Args:
            page_size: 请求的大页大小
        """
        super().__init__(f"Huge page size {page_size} is not available", page_size)


class HugePagePoolExhaustedError(HugePageError):
    """
    大页池耗尽错误
    
    当大页池中没有足够的大页时抛出此异常。
    """
    
    def __init__(self, requested: int, available: int, page_size: int):
        """
        初始化大页池耗尽错误
        
        Args:
            requested: 请求的大页数量
            available: 可用的大页数量
            page_size: 大页大小
        """
        self.requested = requested
        self.available = available
        super().__init__(
            f"Huge page pool exhausted: requested {requested}, available {available}",
            page_size
        )


class GPUMappingError(MemMapperError):
    """
    GPU映射错误
    
    当GPU内存映射相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, device_id: Optional[int] = None):
        """
        初始化GPU映射错误
        
        Args:
            message: 错误消息
            device_id: GPU设备ID（可选）
        """
        self.device_id = device_id
        full_message = f"GPU mapping error: {message}"
        if device_id is not None:
            full_message += f" (device={device_id})"
        super().__init__(full_message)


class GPUNotAvailableError(GPUMappingError):
    """
    GPU不可用错误
    
    当GPU不可用时抛出此异常。
    """
    
    def __init__(self, message: str = "GPU is not available"):
        super().__init__(message)


class GPUOutOfMemoryError(GPUMappingError):
    """
    GPU内存不足错误
    
    当GPU内存不足时抛出此异常。
    """
    
    def __init__(self, requested: int, available: int, device_id: int):
        """
        初始化GPU内存不足错误
        
        Args:
            requested: 请求的内存大小
            available: 可用的内存大小
            device_id: GPU设备ID
        """
        self.requested = requested
        self.available = available
        super().__init__(
            f"GPU out of memory: requested {requested}, available {available}",
            device_id
        )


class RegionError(MemMapperError):
    """
    映射区域错误
    
    当映射区域相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, region_id: Optional[str] = None):
        """
        初始化映射区域错误
        
        Args:
            message: 错误消息
            region_id: 映射区域ID（可选）
        """
        self.region_id = region_id
        full_message = f"Region error: {message}"
        if region_id is not None:
            full_message += f" (region_id={region_id})"
        super().__init__(full_message)


class RegionNotFoundError(RegionError):
    """
    映射区域未找到错误
    
    当指定的映射区域不存在时抛出此异常。
    """
    
    def __init__(self, region_id: str):
        """
        初始化映射区域未找到错误
        
        Args:
            region_id: 映射区域ID
        """
        super().__init__(f"Region not found", region_id)


class RegionAlreadyExistsError(RegionError):
    """
    映射区域已存在错误
    
    当尝试创建已存在的映射区域时抛出此异常。
    """
    
    def __init__(self, region_id: str):
        """
        初始化映射区域已存在错误
        
        Args:
            region_id: 映射区域ID
        """
        super().__init__(f"Region already exists", region_id)


class ConfigError(MemMapperError):
    """
    配置错误
    
    当配置相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        """
        初始化配置错误
        
        Args:
            message: 错误消息
            config_key: 相关的配置键（可选）
        """
        self.config_key = config_key
        full_message = f"Configuration error: {message}"
        if config_key is not None:
            full_message += f" (key={config_key})"
        super().__init__(full_message)


class PlatformError(MemMapperError):
    """
    平台错误
    
    当平台相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, platform: Optional[str] = None):
        """
        初始化平台错误
        
        Args:
            message: 错误消息
            platform: 平台名称（可选）
        """
        self.platform = platform
        full_message = f"Platform error: {message}"
        if platform is not None:
            full_message += f" (platform={platform})"
        super().__init__(full_message)


class PlatformNotSupportedError(PlatformError):
    """
    平台不支持错误
    
    当当前平台不支持某项功能时抛出此异常。
    """
    
    def __init__(self, platform: str):
        """
        初始化平台不支持错误
        
        Args:
            platform: 不支持的平台名称
        """
        super().__init__(f"Platform not supported", platform)


class FileError(MemMapperError):
    """
    文件错误
    
    当文件相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None, errno: Optional[int] = None):
        """
        初始化文件错误
        
        Args:
            message: 错误消息
            file_path: 文件路径（可选）
            errno: 系统错误码（可选）
        """
        self.file_path = file_path
        self.errno = errno
        
        # 清理敏感信息
        try:
            from ..utils.security import get_security_config, ErrorSanitizer
            security_config = get_security_config()
            sanitizer = ErrorSanitizer(security_config)
            
            # 清理消息和路径
            sanitized_message = sanitizer.sanitize_error_message(message)
            sanitized_path = sanitizer.sanitize_path(file_path) if file_path else None
            
            full_message = f"File error: {sanitized_message}"
            if sanitized_path is not None:
                full_message += f" (path={sanitized_path})"
            if errno is not None:
                full_message += f": {os.strerror(errno)}"
            
            super().__init__(full_message, errno)
        except Exception:
            # 如果安全工具不可用，使用原始消息
            full_message = f"File error: {message}"
            if file_path is not None:
                full_message += f" (path={file_path})"
            if errno is not None:
                full_message += f": {os.strerror(errno)}"
            super().__init__(full_message, errno)


class FileNotFoundError(FileError):
    """
    文件未找到错误
    
    当指定的文件不存在时抛出此异常。
    """
    
    def __init__(self, file_path: str):
        """
        初始化文件未找到错误
        
        Args:
            file_path: 文件路径
        """
        super().__init__("File not found", file_path)


class FilePermissionError(FileError):
    """
    文件权限错误
    
    当没有足够的权限访问文件时抛出此异常。
    """
    
    def __init__(self, file_path: str, errno: Optional[int] = None):
        """
        初始化文件权限错误
        
        Args:
            file_path: 文件路径
            errno: 系统错误码（可选）
        """
        super().__init__("Permission denied", file_path, errno)


class AlignmentError(MemMapperError):
    """
    对齐错误
    
    当对齐相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, value: Optional[int] = None, alignment: Optional[int] = None):
        """
        初始化对齐错误
        
        Args:
            message: 错误消息
            value: 需要对齐的值（可选）
            alignment: 对齐大小（可选）
        """
        self.value = value
        self.alignment = alignment
        full_message = f"Alignment error: {message}"
        if value is not None and alignment is not None:
            full_message += f" (value={value}, alignment={alignment})"
        super().__init__(full_message)


class PrefetchError(MemMapperError):
    """
    预取错误
    
    当预取相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, region_id: Optional[str] = None):
        """
        初始化预取错误
        
        Args:
            message: 错误消息
            region_id: 映射区域ID（可选）
        """
        self.region_id = region_id
        full_message = f"Prefetch error: {message}"
        if region_id is not None:
            full_message += f" (region_id={region_id})"
        super().__init__(full_message)


class LifecycleError(MemMapperError):
    """
    生命周期错误
    
    当生命周期管理相关操作失败时抛出此异常。
    """
    
    def __init__(self, message: str, region_id: Optional[str] = None):
        """
        初始化生命周期错误
        
        Args:
            message: 错误消息
            region_id: 映射区域ID（可选）
        """
        self.region_id = region_id
        full_message = f"Lifecycle error: {message}"
        if region_id is not None:
            full_message += f" (region_id={region_id})"
        super().__init__(full_message)
