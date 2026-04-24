"""
sparse_array 单元测试

测试稀疏数组模块的核心功能。
"""

import unittest
import numpy as np
import tempfile
import os
import threading
import concurrent.futures

# 添加父目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sparse_array import SparseArray, SparseArrayConfig, SparseFormat
from sparse_array.formats import CSRFormat, CSCFormat, COOFormat, BCSRFormat, BitmapFormat
from sparse_array.ops import spmv, spmm, add, multiply, sum, norm, divide
from sparse_array.ops.transform import save_sparse, load_sparse, FileOperationError


class TestSparseArrayCreation(unittest.TestCase):
    """测试稀疏数组创建"""

    def test_from_dense(self):
        """测试从密集数组创建"""
        dense = np.random.rand(100, 100)
        dense[dense < 0.9] = 0

        sparse = SparseArray.from_dense(dense)

        self.assertEqual(sparse.shape, dense.shape)
        self.assertGreater(sparse.nnz, 0)
        np.testing.assert_array_almost_equal(sparse.to_dense(), dense)

    def test_from_coo(self):
        """测试从COO坐标创建"""
        rows = np.array([0, 1, 2])
        cols = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        sparse = SparseArray.from_coo((3, 3), rows, cols, data)

        self.assertEqual(sparse.nnz, 3)
        self.assertEqual(sparse[0, 0], 1.0)
        self.assertEqual(sparse[1, 1], 2.0)

    def test_from_csr(self):
        """测试从CSR格式创建"""
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 1, 2, 3])

        sparse = SparseArray.from_csr((3, 3), data, indices, indptr)

        self.assertEqual(sparse.nnz, 3)
        self.assertEqual(sparse.format, 'csr')

    def test_zeros(self):
        """测试创建全零矩阵"""
        sparse = SparseArray.zeros((10, 10))

        self.assertEqual(sparse.nnz, 0)
        self.assertEqual(sparse.shape, (10, 10))

    def test_identity(self):
        """测试创建单位矩阵"""
        sparse = SparseArray.identity(5)

        self.assertEqual(sparse.nnz, 5)
        np.testing.assert_array_almost_equal(sparse.to_dense(), np.eye(5))

    def test_random(self):
        """测试创建随机稀疏矩阵"""
        sparse = SparseArray.random((100, 100), density=0.1, random_state=42)

        self.assertGreater(sparse.nnz, 0)
        expected_nnz = 100 * 100 * 0.1
        self.assertAlmostEqual(sparse.nnz / expected_nnz, 1.0, places=1)


class TestFormatConversion(unittest.TestCase):
    """测试格式转换"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(50, 50)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)

    def test_csr_conversion(self):
        """测试CSR转换"""
        csr = self.sparse.to_csr()

        self.assertEqual(csr.format, 'csr')
        np.testing.assert_array_almost_equal(csr.to_dense(), self.sparse.to_dense())

    def test_csc_conversion(self):
        """测试CSC转换"""
        csc = self.sparse.to_csc()

        self.assertEqual(csc.format, 'csc')
        np.testing.assert_array_almost_equal(csc.to_dense(), self.sparse.to_dense())

    def test_coo_conversion(self):
        """测试COO转换"""
        coo = self.sparse.to_coo()

        self.assertEqual(coo.format, 'coo')
        np.testing.assert_array_almost_equal(coo.to_dense(), self.sparse.to_dense())

    def test_round_trip(self):
        """测试往返转换"""
        csr = self.sparse.to_csr()
        csc = csr.to_csc()
        coo = csc.to_coo()
        back = coo.to_csr()

        np.testing.assert_array_almost_equal(back.to_dense(), self.sparse.to_dense())


class TestArithmeticOperations(unittest.TestCase):
    """测试算术运算"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(50, 50)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)
        self.dense = dense

    def test_add_scalar(self):
        """测试加标量"""
        result = self.sparse + 1.0
        expected = self.dense + 1.0

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_multiply_scalar(self):
        """测试乘标量"""
        result = self.sparse * 2.0
        expected = self.dense * 2.0

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_add_sparse(self):
        """测试稀疏加法"""
        other = SparseArray.from_dense(self.dense)
        result = self.sparse + other
        expected = self.dense + self.dense

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_multiply_sparse(self):
        """测试稀疏元素级乘法"""
        other = SparseArray.from_dense(self.dense)
        result = self.sparse * other
        expected = self.dense * self.dense

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_negate(self):
        """测试取负"""
        result = -self.sparse
        expected = -self.dense

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_sum(self):
        """测试求和"""
        result = sum(self.sparse)
        expected = np.sum(self.dense)

        self.assertAlmostEqual(result, float(expected), places=5)  # type: ignore


class TestLinearAlgebra(unittest.TestCase):
    """测试线性代数运算"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(100, 100)
        dense[dense < 0.9] = 0
        self.sparse = SparseArray.from_dense(dense)
        self.dense = dense

    def test_spmv(self):
        """测试稀疏矩阵-向量乘法"""
        x = np.random.rand(100)

        result = spmv(self.sparse, x)
        expected = self.dense @ x

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_spmm_dense(self):
        """测试稀疏矩阵-密集矩阵乘法"""
        B = np.random.rand(100, 50)

        result = spmm(self.sparse, B)
        expected = self.dense @ B

        np.testing.assert_array_almost_equal(result, expected, decimal=5)  # type: ignore

    def test_matmul_operator(self):
        """测试矩阵乘法运算符"""
        x = np.random.rand(100)

        result = self.sparse @ x
        expected = self.dense @ x

        np.testing.assert_array_almost_equal(result, expected, decimal=5)  # type: ignore

    def test_transpose(self):
        """测试转置"""
        result = self.sparse.T
        expected = self.dense.T

        np.testing.assert_array_almost_equal(result.to_dense(), expected, decimal=5)

    def test_norm_frobenius(self):
        """测试Frobenius范数"""
        result = norm(self.sparse, 'fro')
        expected = np.linalg.norm(self.dense, 'fro')

        self.assertAlmostEqual(result, expected, places=5)


class TestIndexing(unittest.TestCase):
    """测试索引操作"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(50, 50)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)
        self.dense = dense

    def test_single_element(self):
        """测试单个元素访问"""
        for _ in range(10):
            i = np.random.randint(0, 50)
            j = np.random.randint(0, 50)

            self.assertEqual(self.sparse[i, j], self.dense[i, j])

    def test_row_access(self):
        """测试行访问"""
        row = self.sparse[5]
        expected = self.dense[5]

        np.testing.assert_array_almost_equal(row, expected)

    def test_slice(self):
        """测试切片"""
        result = self.sparse[10:20, 10:20]
        expected = self.dense[10:20, 10:20]

        np.testing.assert_array_almost_equal(result.to_dense(), expected)

    def test_set_element(self):
        """测试设置元素"""
        sparse = self.sparse.copy()
        sparse[0, 0] = 999.0

        self.assertEqual(sparse[0, 0], 999.0)


class TestSerialization(unittest.TestCase):
    """测试序列化"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(50, 50)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)

    def test_save_load_npz(self):
        """测试NPZ格式保存和加载"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        try:
            self.sparse.save(path, format='npz')
            loaded = SparseArray.load(path)

            self.assertEqual(loaded.shape, self.sparse.shape)
            self.assertEqual(loaded.nnz, self.sparse.nnz)
            np.testing.assert_array_almost_equal(loaded.to_dense(), self.sparse.to_dense())
        finally:
            os.unlink(path)

    def test_save_load_mtx(self):
        """测试MTX格式保存和加载"""
        with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False) as f:
            path = f.name

        try:
            self.sparse.save(path, format='mtx')
            loaded = SparseArray.load(path, format='mtx')

            self.assertEqual(loaded.shape, self.sparse.shape)
            np.testing.assert_array_almost_equal(loaded.to_dense(), self.sparse.to_dense())
        finally:
            os.unlink(path)


class TestFormatFeatures(unittest.TestCase):
    """测试格式特性"""

    def test_csr_row_access(self):
        """测试CSR行访问效率"""
        dense = np.eye(100)
        config = SparseArrayConfig(format=SparseFormat.CSR)
        sparse = SparseArray.from_dense(dense, config=config)

        # 获取行应该很快
        row = sparse[50]
        expected = dense[50]
        np.testing.assert_array_almost_equal(row, expected)

    def test_csc_col_access(self):
        """测试CSC列访问效率"""
        dense = np.eye(100)
        config = SparseArrayConfig(format=SparseFormat.CSC)
        sparse = SparseArray.from_dense(dense, config=config)

        # 获取列应该很快
        col = sparse[:, 50]
        expected = dense[:, 50]
        np.testing.assert_array_almost_equal(col, expected)

    def test_bcsr_block_structure(self):
        """测试BCSR块结构"""
        # 创建块对角矩阵
        dense = np.zeros((100, 100))
        for i in range(0, 100, 4):
            dense[i:i+4, i:i+4] = np.random.rand(4, 4)

        config = SparseArrayConfig(format=SparseFormat.BCSR)
        sparse = SparseArray.from_dense(dense, config=config)

        np.testing.assert_array_almost_equal(sparse.to_dense(), dense)


class TestShapeValidation(unittest.TestCase):
    """测试shape参数验证"""

    def test_invalid_shape_none(self):
        """测试shape为None"""
        with self.assertRaises(ValueError):
            SparseArray(shape=None)  # type: ignore

    def test_invalid_shape_empty(self):
        """测试shape为空元组"""
        with self.assertRaises(ValueError):
            SparseArray(shape=())

    def test_invalid_shape_negative(self):
        """测试shape包含负数"""
        with self.assertRaises(ValueError):
            SparseArray(shape=(-1, 10))

    def test_invalid_shape_zero(self):
        """测试shape包含零"""
        with self.assertRaises(ValueError):
            SparseArray(shape=(0, 10))

    def test_invalid_shape_non_integer(self):
        """测试shape包含非整数"""
        with self.assertRaises(ValueError):
            SparseArray(shape=(10.5, 10))  # type: ignore

    def test_valid_shape(self):
        """测试有效的shape"""
        sparse = SparseArray(shape=(10, 20))
        self.assertEqual(sparse.shape, (10, 20))


class TestDivisionByZero(unittest.TestCase):
    """测试除零检查"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(10, 10)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)

    def test_divide_by_zero_scalar(self):
        """测试除以零标量"""
        with self.assertRaises(ZeroDivisionError):
            divide(self.sparse, 0)

    def test_divide_by_zero_with_fill_value(self):
        """测试使用fill_value处理除零"""
        result = divide(self.sparse, 0, fill_value=np.inf)
        dense_result = result.to_dense()
        self.assertTrue(np.all(np.isinf(dense_result)))

    def test_divide_by_array_with_zeros(self):
        """测试除以包含零的数组"""
        divisor = np.zeros((10, 10))
        with self.assertRaises(ZeroDivisionError):
            divide(self.sparse, divisor)

    def test_normal_division(self):
        """测试正常除法"""
        result = divide(self.sparse, 2.0)
        expected = self.sparse.to_dense() / 2.0
        np.testing.assert_array_almost_equal(result.to_dense(), expected)


class TestThreadSafety(unittest.TestCase):
    """测试线程安全"""

    def test_random_thread_safety(self):
        """测试随机生成的线程安全性"""
        results = []
        errors = []

        def create_random(seed):
            try:
                sparse = SparseArray.random((100, 100), density=0.1, random_state=seed)
                results.append((seed, sparse.nnz))
            except Exception as e:
                errors.append((seed, str(e)))

        # 创建多个线程
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_random, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 检查没有错误
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 10)

    def test_random_reproducibility(self):
        """测试随机生成的可重复性"""
        sparse1 = SparseArray.random((100, 100), density=0.1, random_state=42)
        sparse2 = SparseArray.random((100, 100), density=0.1, random_state=42)

        np.testing.assert_array_equal(sparse1.to_dense(), sparse2.to_dense())


class TestFileOperationExceptions(unittest.TestCase):
    """测试文件操作异常处理"""

    def setUp(self):
        """设置测试数据"""
        dense = np.random.rand(10, 10)
        dense[dense < 0.8] = 0
        self.sparse = SparseArray.from_dense(dense)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            SparseArray.load('/nonexistent/path/file.npz')

    def test_save_to_invalid_path(self):
        """测试保存到无效路径"""
        # 在Windows上，尝试保存到无效路径
        with self.assertRaises(FileOperationError):
            save_sparse(self.sparse, '/invalid\x00path/test.npz', format='npz')

    def test_load_invalid_format(self):
        """测试加载无效格式文件"""
        with tempfile.NamedTemporaryFile(suffix='.invalid', delete=False) as f:
            path = f.name
            f.write(b'invalid content')

        try:
            with self.assertRaises(FileOperationError):
                SparseArray.load(path, format='auto')
        finally:
            os.unlink(path)


class TestBitmapBoundsCheck(unittest.TestCase):
    """测试Bitmap边界检查"""

    def test_bitmap_index_out_of_bounds(self):
        """测试Bitmap索引越界"""
        dense = np.eye(5)
        config = SparseArrayConfig(format=SparseFormat.BITMAP)
        sparse = SparseArray.from_dense(dense, config=config)

        # 测试行越界
        with self.assertRaises(IndexError):
            _ = sparse[10, 0]

        # 测试列越界
        with self.assertRaises(IndexError):
            _ = sparse[0, 10]

    def test_bitmap_negative_index(self):
        """测试Bitmap负索引"""
        dense = np.eye(5)
        config = SparseArrayConfig(format=SparseFormat.BITMAP)
        sparse = SparseArray.from_dense(dense, config=config)

        # 测试负索引
        self.assertEqual(sparse[-1, -1], 1.0)
        self.assertEqual(sparse[-5, 0], 1.0)


class TestVectorizedToDense(unittest.TestCase):
    """测试向量化to_dense方法"""

    def test_csr_to_dense_large(self):
        """测试CSR大矩阵to_dense"""
        dense = np.random.rand(1000, 1000)
        dense[dense < 0.9] = 0
        config = SparseArrayConfig(format=SparseFormat.CSR)
        sparse = SparseArray.from_dense(dense, config=config)

        result = sparse.to_dense()
        np.testing.assert_array_almost_equal(result, dense)

    def test_csc_to_dense_large(self):
        """测试CSC大矩阵to_dense"""
        dense = np.random.rand(1000, 1000)
        dense[dense < 0.9] = 0
        config = SparseArrayConfig(format=SparseFormat.CSC)
        sparse = SparseArray.from_dense(dense, config=config)

        result = sparse.to_dense()
        np.testing.assert_array_almost_equal(result, dense)

    def test_bitmap_to_dense_large(self):
        """测试Bitmap大矩阵to_dense"""
        dense = np.eye(100)
        config = SparseArrayConfig(format=SparseFormat.BITMAP)
        sparse = SparseArray.from_dense(dense, config=config)

        result = sparse.to_dense()
        np.testing.assert_array_almost_equal(result, dense)


class TestTensorCoreCache(unittest.TestCase):
    """测试TensorCore缓存限制"""

    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        from sparse_array.gpu.tensor_core import TensorCoreOptimizer, DEFAULT_MAX_CACHE_SIZE

        optimizer = TensorCoreOptimizer(max_cache_size=5)

        # 创建多个稀疏矩阵
        for i in range(10):
            dense = np.random.rand(10, 10)
            dense[dense < 0.8] = 0
            sparse = SparseArray.from_dense(dense)
            optimizer.optimize(sparse)

        # 缓存大小不应超过限制
        self.assertLessEqual(optimizer.get_cache_size(), 5)

    def test_cache_clear(self):
        """测试缓存清除"""
        from sparse_array.gpu.tensor_core import TensorCoreOptimizer

        optimizer = TensorCoreOptimizer(max_cache_size=10)

        dense = np.random.rand(10, 10)
        dense[dense < 0.8] = 0
        sparse = SparseArray.from_dense(dense)
        optimizer.optimize(sparse)

        self.assertGreater(optimizer.get_cache_size(), 0)
        optimizer.clear_cache()
        self.assertEqual(optimizer.get_cache_size(), 0)


def test_sparse_array_creation():
    """运行创建测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSparseArrayCreation)
    unittest.TextTestRunner(verbosity=2).run(suite)


def test_format_conversion():
    """运行格式转换测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFormatConversion)
    unittest.TextTestRunner(verbosity=2).run(suite)


def test_arithmetic_operations():
    """运行算术运算测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestArithmeticOperations)
    unittest.TextTestRunner(verbosity=2).run(suite)


def test_linear_algebra():
    """运行线性代数测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearAlgebra)
    unittest.TextTestRunner(verbosity=2).run(suite)


def test_indexing():
    """运行索引测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIndexing)
    unittest.TextTestRunner(verbosity=2).run(suite)


def test_serialization():
    """运行序列化测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSerialization)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_all_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSparseArrayCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestArithmeticOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearAlgebra))
    suite.addTests(loader.loadTestsFromTestCase(TestIndexing))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestShapeValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDivisionByZero))
    suite.addTests(loader.loadTestsFromTestCase(TestThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestFileOperationExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestBitmapBoundsCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedToDense))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorCoreCache))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_all_tests()
