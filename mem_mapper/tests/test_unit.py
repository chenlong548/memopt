"""
mem_mapper 单元测试模块

测试内存映射工具的核心功能。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mem_mapper import (
    MemoryMapper, MapperConfig, MappedRegion,
    ProtectionFlags, MappingType, NUMAPolicy,
    MemMapperError, MMapError, FileError, ConfigError,
    get_version, create_mapper
)
from mem_mapper.utils.alignment import (
    PAGE_SIZE_4KB, PAGE_SIZE_2MB, PAGE_SIZE_1GB,
    align_to_page, align_to_huge_page, is_aligned
)
from mem_mapper.utils.stats import Timer, PerformanceTracker
from mem_mapper.utils.security import (
    SecurityConfig, PathValidator, PermissionChecker,
    ResourceLimiter, ErrorSanitizer, FileDescriptorTracker
)


class TestMapperConfig(unittest.TestCase):
    """配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MapperConfig()
        self.assertIsNotNone(config)
        self.assertTrue(config.use_numa)
        self.assertTrue(config.use_huge_pages)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MapperConfig(
            use_numa=False,
            use_huge_pages=False,
            use_prefetch=False
        )
        self.assertFalse(config.use_numa)
        self.assertFalse(config.use_huge_pages)
        self.assertFalse(config.use_prefetch)
    
    def test_config_dataclass(self):
        """测试配置是dataclass"""
        from dataclasses import asdict
        config = MapperConfig(use_numa=False)
        d = asdict(config)
        self.assertIsInstance(d, dict)
        self.assertFalse(d['use_numa'])


class TestMemoryMapper(unittest.TestCase):
    """内存映射器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_file = None
        self.test_file_size = 4096
        self.mapper = None
    
    def _create_test_file(self, size=4096):
        """创建测试文件"""
        fd, path = tempfile.mkstemp(suffix='.bin')
        with os.fdopen(fd, 'wb') as f:
            f.write(b'X' * size)
        return path
    
    def tearDown(self):
        """测试后清理"""
        if self.mapper:
            self.mapper.cleanup()
            self.mapper = None
        if self.test_file and os.path.exists(self.test_file):
            os.unlink(self.test_file)
    
    def test_create_mapper(self):
        """测试创建映射器"""
        self.mapper = MemoryMapper()
        self.assertIsNotNone(self.mapper)
    
    def test_create_mapper_with_config(self):
        """测试带配置创建映射器"""
        config = MapperConfig(use_numa=False, use_huge_pages=False)
        self.mapper = MemoryMapper(config)
        self.assertIsNotNone(self.mapper)
        self.assertEqual(self.mapper.config.use_numa, False)
    
    def test_map_file_readonly(self):
        """测试只读映射"""
        self.test_file = self._create_test_file()
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertIsNotNone(region)
        self.assertEqual(region.size, self.test_file_size)
        self.assertTrue(region.is_readable())
        
        self.mapper.unmap(region)
    
    def test_map_file_readwrite(self):
        """测试读写映射"""
        self.test_file = self._create_test_file()
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='readwrite')
        self.assertIsNotNone(region)
        self.assertTrue(region.is_readable())
        self.assertTrue(region.is_writable())
        
        self.mapper.unmap(region)
    
    def test_map_file_writecopy(self):
        """测试写时复制映射"""
        self.test_file = self._create_test_file()
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='writecopy')
        self.assertIsNotNone(region)
        self.assertTrue(region.is_readable())
        self.assertTrue(region.is_writable())
        self.assertTrue(region.is_private())
        
        self.mapper.unmap(region)
    
    def test_map_file_with_size(self):
        """测试指定大小映射"""
        self.test_file = self._create_test_file(8192)
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='readonly', size=4096)
        self.assertIsNotNone(region)
        self.assertEqual(region.size, 4096)
        
        self.mapper.unmap(region)
    
    def test_unmap_region(self):
        """测试解除映射"""
        self.test_file = self._create_test_file()
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertIsNotNone(region)
        
        self.mapper.unmap(region)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        self.test_file = self._create_test_file()
        
        with MemoryMapper() as mapper:
            region = mapper.map_file(self.test_file, mode='readonly')
            self.assertIsNotNone(region)
            mapper.unmap(region)
    
    def test_get_stats(self):
        """测试获取统计"""
        self.mapper = MemoryMapper()
        stats = self.mapper.get_stats()
        self.assertIsNotNone(stats)
        self.assertIn('registry', stats)
    
    def test_get_all_regions(self):
        """测试获取所有映射区域"""
        self.test_file = self._create_test_file()
        self.mapper = MemoryMapper()
        
        region = self.mapper.map_file(self.test_file, mode='readonly')
        all_regions = self.mapper.get_all_regions()
        self.assertEqual(len(all_regions), 1)
        
        self.mapper.unmap(region)


class TestMappedRegion(unittest.TestCase):
    """映射区域测试"""
    
    def setUp(self):
        """测试前准备"""
        fd, path = tempfile.mkstemp(suffix='.bin')
        with os.fdopen(fd, 'wb') as f:
            f.write(b'Y' * 4096)
        self.test_file = path
        self.mapper = MemoryMapper()
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'region'):
            try:
                self.mapper.unmap(self.region)
            except:
                pass
        self.mapper.cleanup()
        if os.path.exists(self.test_file):
            os.unlink(self.test_file)
    
    def test_region_size(self):
        """测试区域大小"""
        self.region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertEqual(self.region.size, 4096)
    
    def test_region_address(self):
        """测试区域地址"""
        self.region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertGreater(self.region.base_address, 0)
    
    def test_region_properties(self):
        """测试区域属性"""
        self.region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertGreater(self.region.aligned_size, 0)
        self.assertGreaterEqual(self.region.get_ref_count(), 1)
    
    def test_region_contains(self):
        """测试区域包含检查"""
        self.region = self.mapper.map_file(self.test_file, mode='readonly')
        self.assertTrue(self.region.contains(self.region.base_address))
        self.assertFalse(self.region.contains(0))


class TestAlignment(unittest.TestCase):
    """对齐工具测试"""
    
    def test_page_size_constants(self):
        """测试页面大小常量"""
        self.assertEqual(PAGE_SIZE_4KB, 4096)
        self.assertEqual(PAGE_SIZE_2MB, 2 * 1024 * 1024)
        self.assertEqual(PAGE_SIZE_1GB, 1024 * 1024 * 1024)
    
    def test_align_to_page(self):
        """测试页面对齐"""
        self.assertEqual(align_to_page(0, 4096), 0)
        self.assertEqual(align_to_page(1, 4096), 4096)
        self.assertEqual(align_to_page(4096, 4096), 4096)
        self.assertEqual(align_to_page(4097, 4096), 8192)
        self.assertEqual(align_to_page(5000, 4096), 8192)
    
    def test_align_to_huge_page(self):
        """测试大页对齐"""
        huge_size = 2 * 1024 * 1024
        self.assertEqual(align_to_huge_page(0, huge_size), 0)
        self.assertEqual(align_to_huge_page(huge_size, huge_size), huge_size)
        self.assertEqual(align_to_huge_page(huge_size + 1, huge_size), 2 * huge_size)
    
    def test_is_aligned(self):
        """测试对齐检查"""
        self.assertTrue(is_aligned(0, 4096))
        self.assertTrue(is_aligned(4096, 4096))
        self.assertTrue(is_aligned(8192, 4096))
        self.assertFalse(is_aligned(1, 4096))
        self.assertFalse(is_aligned(4095, 4096))
        self.assertFalse(is_aligned(4097, 4096))


class TestTimer(unittest.TestCase):
    """计时器测试"""
    
    def test_timer_start_stop(self):
        """测试计时器开始停止"""
        timer = Timer()
        timer.start()
        elapsed = timer.stop()
        self.assertGreaterEqual(elapsed, 0)
    
    def test_timer_duration(self):
        """测试计时器持续时间"""
        import time
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        self.assertGreaterEqual(elapsed, 0.01)


class TestPerformanceTracker(unittest.TestCase):
    """性能跟踪器测试"""
    
    def test_record_operation(self):
        """测试记录操作"""
        tracker = PerformanceTracker()
        tracker.record('test_op', 0.1, success=True)
        tracker.record('test_op', 0.2, success=True)
        
        summary = tracker.get_summary()
        self.assertIsNotNone(summary)
    
    def test_get_summary(self):
        """测试获取摘要"""
        tracker = PerformanceTracker()
        for i in range(5):
            tracker.record('op', 0.01 * i, success=True)
        
        summary = tracker.get_summary()
        self.assertIn('op', summary)


class TestPathValidator(unittest.TestCase):
    """路径验证器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig()
        self.validator = PathValidator(self.config)
    
    def test_valid_path(self):
        """测试有效路径"""
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
        """测试路径遍历"""
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


class TestPermissionChecker(unittest.TestCase):
    """权限检查器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SecurityConfig()
        self.checker = PermissionChecker(self.config)
    
    def test_read_permission(self):
        """测试读权限"""
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
        """测试写权限"""
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
        """测试目录权限"""
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
            max_mapping_size=1024 * 1024,
            max_total_mapping_size=10 * 1024 * 1024,
            max_mappings=100
        )
        self.limiter = ResourceLimiter(self.config)
    
    def test_mapping_size_limit(self):
        """测试映射大小限制"""
        large_size = 2 * 1024 * 1024
        is_allowed, error = self.limiter.check_mapping_limit(large_size, 0, 0)
        self.assertFalse(is_allowed)
        self.assertIn("exceeds maximum", error.lower() if error else "")
    
    def test_mapping_count_limit(self):
        """测试映射数量限制"""
        is_allowed, error = self.limiter.check_mapping_limit(1024, 100, 0)
        self.assertFalse(is_allowed)
        self.assertIn("maximum mappings", error.lower() if error else "")
    
    def test_total_size_limit(self):
        """测试总大小限制"""
        current_total = 10 * 1024 * 1024 - 256 * 1024
        new_size = 512 * 1024
        
        is_allowed, error = self.limiter.check_mapping_limit(new_size, 0, current_total)
        self.assertFalse(is_allowed)
        self.assertIn("total mapping size", error.lower() if error else "")
    
    def test_allowed_mapping(self):
        """测试允许的映射"""
        is_allowed, error = self.limiter.check_mapping_limit(512 * 1024, 0, 0)
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
        
        self.assertNotIn("/home/user/secret", sanitized)
        self.assertNotIn("0x7f8a12345678", sanitized)


class TestFileDescriptorTracker(unittest.TestCase):
    """文件描述符跟踪器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.tracker = FileDescriptorTracker()
    
    def test_register_and_unregister(self):
        """测试注册和注销"""
        fd = os.open(__file__, os.O_RDONLY)
        
        try:
            self.tracker.register(fd)
            self.assertIn(fd, self.tracker.get_open_fds())
            
            self.tracker.unregister(fd)
            self.assertNotIn(fd, self.tracker.get_open_fds())
        finally:
            os.close(fd)
    
    def test_close_fd(self):
        """测试关闭文件描述符"""
        fd = os.open(__file__, os.O_RDONLY)
        
        self.tracker.register(fd)
        result = self.tracker.close(fd)
        
        self.assertTrue(result)
        self.assertNotIn(fd, self.tracker.get_open_fds())
    
    def test_close_all(self):
        """测试关闭所有"""
        fds = []
        
        try:
            for _ in range(3):
                fd = os.open(__file__, os.O_RDONLY)
                fds.append(fd)
                self.tracker.register(fd)
            
            self.tracker.close_all()
            self.assertEqual(len(self.tracker.get_open_fds()), 0)
        finally:
            for fd in fds:
                try:
                    os.close(fd)
                except:
                    pass


class TestExceptions(unittest.TestCase):
    """异常测试"""
    
    def test_mem_mapper_error(self):
        """测试基础异常"""
        with self.assertRaises(MemMapperError):
            raise MemMapperError("Test error")
    
    def test_file_error(self):
        """测试文件异常"""
        try:
            raise FileError("File not found", file_path="/test/path", errno=2)
        except FileError as e:
            self.assertIn("File not found", str(e))
    
    def test_config_error(self):
        """测试配置异常"""
        with self.assertRaises(ConfigError):
            raise ConfigError("Invalid config")
    
    def test_mmap_error(self):
        """测试映射异常"""
        with self.assertRaises(MMapError):
            raise MMapError(errno=12, message="Mapping failed")


class TestVersion(unittest.TestCase):
    """版本测试"""
    
    def test_get_version(self):
        """测试获取版本"""
        version = get_version()
        self.assertIsNotNone(version)
        self.assertIsInstance(version, str)
    
    def test_create_mapper(self):
        """测试创建映射器工厂函数"""
        mapper = create_mapper()
        self.assertIsNotNone(mapper)
        self.assertIsInstance(mapper, MemoryMapper)
        mapper.cleanup()


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        fd, path = tempfile.mkstemp(suffix='.bin')
        with os.fdopen(fd, 'wb') as f:
            f.write(b'Test data' * 100)
        
        try:
            config = MapperConfig(use_numa=False, use_huge_pages=False)
            mapper = MemoryMapper(config)
            
            region = mapper.map_file(path, mode='readonly')
            self.assertIsNotNone(region)
            self.assertEqual(region.size, 900)
            
            mapper.unmap(region)
            mapper.cleanup()
        finally:
            os.unlink(path)
    
    def test_multiple_mappings(self):
        """测试多映射"""
        files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix='.bin')
            with os.fdopen(fd, 'wb') as f:
                f.write(b'Data' * (i + 1) * 100)
            files.append(path)
        
        try:
            mapper = MemoryMapper()
            regions = []
            
            for path in files:
                region = mapper.map_file(path, mode='readonly')
                regions.append(region)
            
            self.assertEqual(len(regions), 3)
            
            for region in regions:
                mapper.unmap(region)
            
            mapper.cleanup()
        finally:
            for path in files:
                os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
