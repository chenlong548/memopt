"""
mem_mapper 安全工具模块

提供安全相关的验证、限制和防护功能。
"""

import os
import stat
import logging
from typing import Optional, List, Set, Tuple
from dataclasses import dataclass


# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """
    安全配置
    
    定义安全相关的配置选项。
    """
    
    # 路径安全配置
    allowed_base_dirs: Optional[List[str]] = None  # 允许的基础目录列表（None表示不限制）
    max_path_length: int = 4096  # 最大路径长度
    allow_symlinks: bool = False  # 是否允许符号链接
    allow_absolute_paths: bool = True  # 是否允许绝对路径
    
    # 资源限制配置
    max_mapping_size: int = 1024 * 1024 * 1024  # 单个映射最大大小（1GB）
    max_total_mapping_size: int = 16 * 1024 * 1024 * 1024  # 总映射最大大小（16GB）
    max_mappings: int = 10000  # 最大映射数量
    
    # 文件权限配置
    check_file_permissions: bool = True  # 是否检查文件权限
    require_owner_match: bool = False  # 是否要求文件所有者匹配
    
    # 信息泄露防护
    sanitize_error_messages: bool = True  # 是否清理错误消息中的敏感信息
    hide_memory_addresses: bool = True  # 是否隐藏内存地址
    hide_file_paths: bool = False  # 是否隐藏文件路径（可能影响调试）
    
    def __post_init__(self):
        """初始化后处理"""
        if self.allowed_base_dirs is not None:
            # 规范化允许的目录路径
            self.allowed_base_dirs = [
                os.path.abspath(os.path.normpath(d)) 
                for d in self.allowed_base_dirs
            ]


class PathValidator:
    """
    路径验证器
    
    验证文件路径的安全性，防止路径遍历攻击。
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        初始化路径验证器
        
        Args:
            config: 安全配置
        """
        self.config = config or SecurityConfig()
    
    def validate_path(self, path: str) -> Tuple[bool, str, Optional[str]]:
        """
        验证路径安全性
        
        Args:
            path: 要验证的路径
            
        Returns:
            (是否有效, 规范化路径, 错误消息)
        """
        try:
            # 1. 检查路径长度
            if len(path) > self.config.max_path_length:
                return False, path, f"Path length exceeds maximum ({self.config.max_path_length})"
            
            # 2. 检查路径是否为空
            if not path or not path.strip():
                return False, path, "Path is empty"
            
            # 3. 检查是否包含空字节
            if '\x00' in path:
                return False, path, "Path contains null byte"
            
            # 4. 规范化路径
            normalized_path = os.path.normpath(path)
            
            # 5. 检查是否允许绝对路径
            if not self.config.allow_absolute_paths and os.path.isabs(normalized_path):
                return False, normalized_path, "Absolute paths are not allowed"
            
            # 6. 检查路径遍历
            if self._has_path_traversal(normalized_path):
                return False, normalized_path, "Path traversal detected"
            
            # 7. 检查符号链接
            if os.path.exists(normalized_path):
                real_path = os.path.realpath(normalized_path)
                
                # 检查是否是符号链接
                if os.path.islink(normalized_path) and not self.config.allow_symlinks:
                    return False, normalized_path, "Symbolic links are not allowed"
                
                # 使用真实路径进行后续检查
                check_path = real_path
            else:
                check_path = normalized_path
            
            # 8. 检查是否在允许的目录内
            if self.config.allowed_base_dirs:
                if not self._is_in_allowed_dirs(check_path):
                    return False, normalized_path, "Path is not in allowed directories"
            
            # 9. 获取绝对路径
            abs_path = os.path.abspath(check_path)
            
            return True, abs_path, None
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False, path, f"Path validation failed: {str(e)}"
    
    def _has_path_traversal(self, path: str) -> bool:
        """
        检查路径是否包含遍历序列
        
        Args:
            path: 路径
            
        Returns:
            是否包含遍历序列
        """
        # 检查常见的路径遍历模式
        dangerous_patterns = [
            '..',
            '~',
            '$',
        ]
        
        path_lower = path.lower()
        for pattern in dangerous_patterns:
            if pattern in path_lower:
                # 允许在文件名中包含这些字符，但不允许作为路径分隔
                parts = path.split(os.sep)
                for part in parts:
                    if part == '..':
                        return True
        
        return False
    
    def _is_in_allowed_dirs(self, path: str) -> bool:
        """
        检查路径是否在允许的目录内
        
        Args:
            path: 路径
            
        Returns:
            是否在允许的目录内
        """
        if not self.config.allowed_base_dirs:
            return True
        
        abs_path = os.path.abspath(path)
        
        for allowed_dir in self.config.allowed_base_dirs:
            # 确保路径在允许的目录下
            if abs_path.startswith(allowed_dir + os.sep) or abs_path == allowed_dir:
                return True
        
        return False


class PermissionChecker:
    """
    权限检查器
    
    检查文件和目录的访问权限。
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        初始化权限检查器
        
        Args:
            config: 安全配置
        """
        self.config = config or SecurityConfig()
    
    def check_file_permission(self, 
                             path: str, 
                             mode: str) -> Tuple[bool, Optional[str]]:
        """
        检查文件权限
        
        Args:
            path: 文件路径
            mode: 访问模式 ('readonly', 'readwrite', 'writecopy')
            
        Returns:
            (是否有权限, 错误消息)
        """
        if not self.config.check_file_permissions:
            return True, None
        
        try:
            # 获取文件状态
            file_stat = os.stat(path)
            
            # 检查文件类型
            if not stat.S_ISREG(file_stat.st_mode):
                return False, "Not a regular file"
            
            # 检查权限
            if mode == 'readonly':
                # 检查读权限
                if not os.access(path, os.R_OK):
                    return False, "No read permission"
                    
            elif mode == 'readwrite':
                # 检查读写权限
                if not os.access(path, os.R_OK | os.W_OK):
                    return False, "No read/write permission"
                    
            elif mode == 'writecopy':
                # 检查读权限
                if not os.access(path, os.R_OK):
                    return False, "No read permission"
            
            # 检查所有者匹配（如果要求）
            if self.config.require_owner_match:
                # os.getuid() 仅在Unix系统上可用
                current_uid = getattr(os, 'getuid', lambda: -1)()
                if current_uid != -1 and file_stat.st_uid != current_uid:
                    return False, "File owner does not match current user"
            
            return True, None
            
        except OSError as e:
            return False, f"Permission check failed: {e}"
    
    def check_directory_permission(self, 
                                  path: str, 
                                  require_write: bool = False) -> Tuple[bool, Optional[str]]:
        """
        检查目录权限
        
        Args:
            path: 目录路径
            require_write: 是否需要写权限
            
        Returns:
            (是否有权限, 错误消息)
        """
        try:
            # 获取目录路径
            dir_path = os.path.dirname(path) if os.path.isfile(path) else path
            
            # 检查目录是否存在
            if not os.path.exists(dir_path):
                return False, f"Directory does not exist: {dir_path}"
            
            # 检查是否是目录
            if not os.path.isdir(dir_path):
                return False, f"Not a directory: {dir_path}"
            
            # 检查权限
            if require_write:
                if not os.access(dir_path, os.R_OK | os.W_OK | os.X_OK):
                    return False, "No read/write/execute permission on directory"
            else:
                if not os.access(dir_path, os.R_OK | os.X_OK):
                    return False, "No read/execute permission on directory"
            
            return True, None
            
        except OSError as e:
            return False, f"Directory permission check failed: {e}"


class ResourceLimiter:
    """
    资源限制器
    
    限制资源使用，防止资源耗尽攻击。
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        初始化资源限制器
        
        Args:
            config: 安全配置
        """
        self.config = config or SecurityConfig()
        
        # 当前资源使用情况
        self._current_mappings = 0
        self._current_total_size = 0
        
        # 线程锁
        import threading
        self._lock = threading.Lock()
    
    def check_mapping_limit(self, 
                           size: int, 
                           current_count: int,
                           current_total: int) -> Tuple[bool, Optional[str]]:
        """
        检查映射限制
        
        Args:
            size: 请求的映射大小
            current_count: 当前映射数量
            current_total: 当前总映射大小
            
        Returns:
            (是否允许, 错误消息)
        """
        with self._lock:
            # 检查单个映射大小
            if size > self.config.max_mapping_size:
                return False, (
                    f"Mapping size ({size}) exceeds maximum "
                    f"({self.config.max_mapping_size})"
                )
            
            # 检查映射数量
            if current_count >= self.config.max_mappings:
                return False, (
                    f"Maximum mappings ({self.config.max_mappings}) reached"
                )
            
            # 检查总映射大小
            new_total = current_total + size
            if new_total > self.config.max_total_mapping_size:
                return False, (
                    f"Total mapping size ({new_total}) would exceed maximum "
                    f"({self.config.max_total_mapping_size})"
                )
            
            return True, None
    
    def check_file_size(self, 
                       file_size: int, 
                       offset: int, 
                       requested_size: int) -> Tuple[bool, Optional[str]]:
        """
        检查文件大小限制
        
        Args:
            file_size: 文件大小
            offset: 偏移量
            requested_size: 请求的映射大小
            
        Returns:
            (是否允许, 错误消息)
        """
        # 检查偏移量是否有效
        if offset < 0:
            return False, f"Invalid offset: {offset}"
        
        if offset >= file_size:
            return False, f"Offset ({offset}) exceeds file size ({file_size})"
        
        # 计算实际映射大小
        if requested_size == 0:
            actual_size = file_size - offset
        else:
            actual_size = requested_size
        
        # 检查映射范围
        if offset + actual_size > file_size:
            return False, (
                f"Mapping range exceeds file size: "
                f"offset={offset}, size={actual_size}, file_size={file_size}"
            )
        
        # 检查大小限制
        if actual_size > self.config.max_mapping_size:
            return False, (
                f"Mapping size ({actual_size}) exceeds maximum "
                f"({self.config.max_mapping_size})"
            )
        
        return True, None


class ErrorSanitizer:
    """
    错误消息清理器
    
    清理错误消息中的敏感信息。
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        初始化错误消息清理器
        
        Args:
            config: 安全配置
        """
        self.config = config or SecurityConfig()
    
    def sanitize_error_message(self, 
                              message: str,
                              hide_path: Optional[bool] = None,
                              hide_address: Optional[bool] = None) -> str:
        """
        清理错误消息
        
        Args:
            message: 原始错误消息
            hide_path: 是否隐藏路径（None使用配置）
            hide_address: 是否隐藏地址（None使用配置）
            
        Returns:
            清理后的错误消息
        """
        if not self.config.sanitize_error_messages:
            return message
        
        hide_path = hide_path if hide_path is not None else self.config.hide_file_paths
        hide_address = hide_address if hide_address is not None else self.config.hide_memory_addresses
        
        sanitized = message
        
        # 隐藏文件路径
        if hide_path:
            import re
            # 替换文件路径模式
            sanitized = re.sub(
                r'(path|file|File)[:=]\s*[^\s,)]+',
                r'\1: <REDACTED>',
                sanitized
            )
            sanitized = re.sub(
                r'(/[^\s:]+)',
                r'<PATH>',
                sanitized
            )
        
        # 隐藏内存地址
        if hide_address:
            import re
            # 替换十六进制地址
            sanitized = re.sub(
                r'0x[0-9a-fA-F]+',
                '<ADDRESS>',
                sanitized
            )
            # 替换addr=后面的地址
            sanitized = re.sub(
                r'addr=\s*0x[0-9a-fA-F]+',
                'addr=<ADDRESS>',
                sanitized
            )
        
        return sanitized
    
    def sanitize_address(self, address: int) -> str:
        """
        清理内存地址
        
        Args:
            address: 内存地址
            
        Returns:
            清理后的地址字符串
        """
        if self.config.hide_memory_addresses:
            return "<ADDRESS>"
        else:
            return hex(address)
    
    def sanitize_path(self, path: str) -> str:
        """
        清理文件路径
        
        Args:
            path: 文件路径
            
        Returns:
            清理后的路径字符串
        """
        if self.config.hide_file_paths:
            # 只显示文件名
            return os.path.basename(path) if path else "<PATH>"
        else:
            return path


class FileDescriptorTracker:
    """
    文件描述符跟踪器
    
    跟踪文件描述符的使用，防止泄漏。
    """
    
    def __init__(self):
        """初始化文件描述符跟踪器"""
        self._open_fds: Set[int] = set()
        import threading
        self._lock = threading.Lock()
    
    def register(self, fd: int):
        """
        注册文件描述符
        
        Args:
            fd: 文件描述符
        """
        with self._lock:
            self._open_fds.add(fd)
            logger.debug(f"Registered file descriptor: {fd}")
    
    def unregister(self, fd: int):
        """
        注销文件描述符
        
        Args:
            fd: 文件描述符
        """
        with self._lock:
            self._open_fds.discard(fd)
            logger.debug(f"Unregistered file descriptor: {fd}")
    
    def close(self, fd: int) -> bool:
        """
        关闭文件描述符
        
        Args:
            fd: 文件描述符
            
        Returns:
            是否成功关闭
        """
        try:
            os.close(fd)
            self.unregister(fd)
            return True
        except OSError as e:
            logger.error(f"Failed to close file descriptor {fd}: {e}")
            self.unregister(fd)  # 即使失败也从跟踪中移除
            return False
    
    def close_all(self):
        """关闭所有跟踪的文件描述符"""
        with self._lock:
            fds_to_close = list(self._open_fds)
        
        for fd in fds_to_close:
            self.close(fd)
    
    def get_open_fds(self) -> Set[int]:
        """
        获取所有打开的文件描述符
        
        Returns:
            打开的文件描述符集合
        """
        with self._lock:
            return self._open_fds.copy()
    
    def check_leaks(self) -> List[int]:
        """
        检查文件描述符泄漏
        
        Returns:
            可能泄漏的文件描述符列表
        """
        with self._lock:
            leaked = []
            for fd in self._open_fds:
                try:
                    # 检查文件描述符是否仍然有效
                    os.fstat(fd)
                except OSError:
                    # 文件描述符无效，可能已泄漏
                    leaked.append(fd)
            return leaked


# 全局安全配置实例
_global_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """
    获取全局安全配置
    
    Returns:
        安全配置实例
    """
    global _global_security_config
    
    if _global_security_config is None:
        _global_security_config = SecurityConfig()
    
    return _global_security_config


def set_security_config(config: SecurityConfig):
    """
    设置全局安全配置
    
    Args:
        config: 安全配置
    """
    global _global_security_config
    _global_security_config = config


# 全局文件描述符跟踪器
_global_fd_tracker: Optional[FileDescriptorTracker] = None


def get_fd_tracker() -> FileDescriptorTracker:
    """
    获取全局文件描述符跟踪器
    
    Returns:
        文件描述符跟踪器实例
    """
    global _global_fd_tracker
    
    if _global_fd_tracker is None:
        _global_fd_tracker = FileDescriptorTracker()
    
    return _global_fd_tracker
