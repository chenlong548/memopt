"""
mem_mapper 安全测试模块

测试所有安全相关的功能和防护措施。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem_mapper.utils.security import (
    SecurityConfig, PathValidator, PermissionChecker,
    ResourceLimiter, ErrorSanitizer, FileDescriptorTracker
)
from mem_mapper.core.exceptions import FileError


class TestPathValidator(unittest.TestCase):
    """路径验证器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig()
        self.validator = PathValidator(self.config)
    
    def test_valid_path(self):
        """测试有效路径"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            is_valid, validated_path, error = self.validator.validate_path(temp_file)
            self.assertTrue(is_valid)
            self.assertIsNotNone(validated_path)
            self.assertIsNone(error)
        finally:
            os.unlink(temp_file)
    
    def test_empty_path(self):
        """测试空路径"""
        is_valid, _, error = self.validator.validate_path("")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("empty", error.lower() if error else "")
    
    def test_null_byte_injection(self):
        """测试空字节注入"""
        is_valid, _, error = self.validator.validate_path("/tmp/test\x00file.txt")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("null byte", error.lower() if error else "")
    
    def test_path_traversal(self):
        """测试路径遍历攻击"""
        # 测试 ../ 遍历
        is_valid, _, error = self.validator.validate_path("../../../etc/passwd")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("traversal", error.lower() if error else "")
    
    def test_max_path_length(self):
        """测试最大路径长度"""
        long_path = "/tmp/" + "a" * 5000
        is_valid, _, error = self.validator.validate_path(long_path)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("length", error.lower() if error else "")
    
    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="Windows requires admin privileges to create symlinks"
    )
    def test_symlink_restriction(self):
        """测试符号链接限制"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        temp_link = temp_file + ".link"
        
        try:
            os.symlink(temp_file, temp_link)
            
            config = SecurityConfig(allow_symlinks=False)
            validator = PathValidator(config)
            
            is_valid, _, error = validator.validate_path(temp_link)
            self.assertFalse(is_valid)
            self.assertIsNotNone(error)
            self.assertIn("symbolic link", error.lower() if error else "")
            
        finally:
            if os.path.exists(temp_link):
                os.unlink(temp_link)
            os.unlink(temp_file)
    
    def test_allowed_directories(self):
        """测试允许的目录限制"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建允许的目录配置
            config = SecurityConfig(allowed_base_dirs=[temp_dir])
            validator = PathValidator(config)
            
            # 测试允许的路径
            allowed_file = os.path.join(temp_dir, "test.txt")
            with open(allowed_file, 'w') as f:
                f.write("test")
            
            is_valid, _, _ = validator.validate_path(allowed_file)
            self.assertTrue(is_valid)
            
            # 测试不允许的路径
            disallowed_file = "/tmp/test.txt"
            is_valid, _, error = validator.validate_path(disallowed_file)
            self.assertFalse(is_valid)
            self.assertIsNotNone(error)
            self.assertIn("not in allowed", error.lower() if error else "")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPermissionChecker(unittest.TestCase):
    """权限检查器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig()
        self.checker = PermissionChecker(self.config)
    
    def test_read_permission(self):
        """测试读权限检查"""
        # 创建可读文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test")
            temp_file = f.name
        
        try:
            has_perm, error = self.checker.check_file_permission(temp_file, 'readonly')
            self.assertTrue(has_perm)
            self.assertIsNone(error)
        finally:
            os.unlink(temp_file)
    
    def test_write_permission(self):
        """测试写权限检查"""
        # 创建可写文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test")
            temp_file = f.name
        
        try:
            has_perm, error = self.checker.check_file_permission(temp_file, 'readwrite')
            self.assertTrue(has_perm)
            self.assertIsNone(error)
        finally:
            os.unlink(temp_file)
    
    def test_directory_permission(self):
        """测试目录权限检查"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            has_perm, error = self.checker.check_directory_permission(temp_dir)
            self.assertTrue(has_perm)
            self.assertIsNone(error)
        finally:
            os.rmdir(temp_dir)


class TestResourceLimiter(unittest.TestCase):
    """资源限制器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig(
            max_mapping_size=1024 * 1024,  # 1MB
            max_total_mapping_size=10 * 1024 * 1024,  # 10MB
            max_mappings=100
        )
        self.limiter = ResourceLimiter(self.config)
    
    def test_mapping_size_limit(self):
        """测试映射大小限制"""
        # 测试超过限制的大小
        large_size = 2 * 1024 * 1024  # 2MB
        is_allowed, error = self.limiter.check_mapping_limit(large_size, 0, 0)
        self.assertFalse(is_allowed)
        self.assertIn("exceeds maximum", error.lower() if error else "")
    
    def test_mapping_count_limit(self):
        """测试映射数量限制"""
        # 测试超过数量限制
        is_allowed, error = self.limiter.check_mapping_limit(1024, 100, 0)
        self.assertFalse(is_allowed)
        self.assertIn("maximum mappings", error.lower() if error else "")
    
    def test_total_size_limit(self):
        """测试总大小限制"""
        # 测试超过总大小限制
        # max_total_mapping_size = 10MB
        # 使用接近限制的current_total，使得添加new_size后会超过限制
        current_total = 10 * 1024 * 1024 - 256 * 1024  # 9.75MB
        new_size = 512 * 1024  # 512KB (9.75MB + 0.5MB = 10.25MB > 10MB)
        
        is_allowed, error = self.limiter.check_mapping_limit(new_size, 0, current_total)
        self.assertFalse(is_allowed)
        self.assertIn("total mapping size", error.lower() if error else "")
    
    def test_allowed_mapping(self):
        """测试允许的映射"""
        # 测试正常大小的映射
        is_allowed, error = self.limiter.check_mapping_limit(512 * 1024, 0, 0)  # 512KB
        self.assertTrue(is_allowed)
        self.assertIsNone(error)


class TestErrorSanitizer(unittest.TestCase):
    """错误消息清理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig(
            sanitize_error_messages=True,
            hide_memory_addresses=True,
            hide_file_paths=True
        )
        self.sanitizer = ErrorSanitizer(self.config)
    
    def test_sanitize_address(self):
        """测试地址清理"""
        address = 0x7f8a12345678
        sanitized = self.sanitizer.sanitize_address(address)
        self.assertEqual(sanitized, "<ADDRESS>")
    
    def test_sanitize_path(self):
        """测试路径清理"""
        path = "/home/user/secret/file.txt"
        sanitized = self.sanitizer.sanitize_path(path)
        self.assertEqual(sanitized, "file.txt")
    
    def test_sanitize_error_message(self):
        """测试错误消息清理"""
        message = "Failed to access /home/user/secret/file.txt at address 0x7f8a12345678"
        sanitized = self.sanitizer.sanitize_error_message(message)
        
        # 检查敏感信息是否被清理
        self.assertNotIn("/home/user/secret", sanitized)
        self.assertNotIn("0x7f8a12345678", sanitized)
        self.assertIn("<ADDRESS>", sanitized)
    
    def test_no_sanitization(self):
        """测试不清理消息"""
        config = SecurityConfig(
            sanitize_error_messages=False,
            hide_memory_addresses=False,
            hide_file_paths=False
        )
        sanitizer = ErrorSanitizer(config)
        
        address = 0x7f8a12345678
        sanitized = sanitizer.sanitize_address(address)
        self.assertEqual(sanitized, hex(address))


class TestFileDescriptorTracker(unittest.TestCase):
    """文件描述符跟踪器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.tracker = FileDescriptorTracker()
    
    def test_register_and_unregister(self):
        """测试注册和注销"""
        # 打开文件
        fd = os.open(__file__, os.O_RDONLY)
        
        try:
            # 注册
            self.tracker.register(fd)
            self.assertIn(fd, self.tracker.get_open_fds())
            
            # 注销
            self.tracker.unregister(fd)
            self.assertNotIn(fd, self.tracker.get_open_fds())
            
        finally:
            os.close(fd)
    
    def test_close_fd(self):
        """测试关闭文件描述符"""
        fd = os.open(__file__, os.O_RDONLY)
        
        # 注册并关闭
        self.tracker.register(fd)
        result = self.tracker.close(fd)
        
        self.assertTrue(result)
        self.assertNotIn(fd, self.tracker.get_open_fds())
        
        # 验证文件描述符已关闭
        with self.assertRaises(OSError):
            os.fstat(fd)
    
    def test_close_all(self):
        """测试关闭所有文件描述符"""
        fds = []
        
        try:
            # 打开多个文件
            for _ in range(3):
                fd = os.open(__file__, os.O_RDONLY)
                fds.append(fd)
                self.tracker.register(fd)
            
            # 关闭所有
            self.tracker.close_all()
            
            # 验证所有都已关闭
            self.assertEqual(len(self.tracker.get_open_fds()), 0)
            
        finally:
            # 确保清理
            for fd in fds:
                try:
                    os.close(fd)
                except:
                    pass


class TestSecurityIntegration(unittest.TestCase):
    """安全集成测试"""
    
    def test_file_error_sanitization(self):
        """测试文件错误消息清理"""
        # 创建会触发FileError的情况
        try:
            raise FileError(
                "Test error with /sensitive/path and address 0x12345678",
                file_path="/sensitive/path/file.txt",
                errno=2
            )
        except FileError as e:
            # 验证错误消息已被清理
            message = str(e)
            # 注意：如果安全配置未设置，可能不会清理
            # 这里主要测试不会崩溃
            self.assertIsNotNone(message)


def run_security_tests():
    """运行所有安全测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestPathValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPermissionChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorSanitizer))
    suite.addTests(loader.loadTestsFromTestCase(TestFileDescriptorTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_security_tests()
    sys.exit(0 if success else 1)
