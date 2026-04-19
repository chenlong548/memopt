"""
data_compressor 核心压缩器

实现自适应压缩的核心逻辑。
"""

import logging
from typing import Union, Optional, Dict, Any
import time

from .base import (
    CompressorBase,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionLevel,
    CompressionStats,
    CompressedData,
    DataType
)
from .exceptions import CompressionError, DecompressionError
from ..detection.type_detector import DataTypeDetector
from ..algorithms.adaptive.selector import AdaptiveAlgorithmSelector
from ..algorithms.adaptive.feature_extractor import FeatureExtractor
from ..utils.statistics import StatisticsCollector
from ..utils.validation import DataValidator

logger = logging.getLogger(__name__)


class DataCompressor(CompressorBase):
    """
    数据压缩器主类

    提供自适应压缩功能，自动选择最优算法。
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        初始化数据压缩器

        Args:
            config: 压缩配置，None则使用默认配置
        """
        self.config = config or CompressionConfig()

        # 初始化组件
        self.type_detector = DataTypeDetector()
        self.algorithm_selector = AdaptiveAlgorithmSelector()
        self.feature_extractor = FeatureExtractor()
        self.stats_collector = StatisticsCollector()
        self.validator = DataValidator()

        # 算法注册表
        self._algorithm_registry: Dict[CompressionAlgorithm, CompressorBase] = {}

        # 初始化算法
        self._init_algorithms()

        logger.info("DataCompressor initialized successfully")

    def _init_algorithms(self):
        """初始化压缩算法"""
        # 延迟导入以避免循环依赖
        from ..algorithms.zstd_wrapper import ZstdCompressor
        from ..algorithms.lz4_wrapper import LZ4Compressor
        from ..algorithms.brotli_wrapper import BrotliCompressor
        from ..algorithms.model_compress.bf16_compress import BF16ModelCompressor
        from ..algorithms.model_compress.fp32_compress import FP32ModelCompressor
        from ..algorithms.kv_cache.zsmerge import ZSMergeCompressor
        from ..algorithms.kv_cache.lexico import LexicoCompressor

        # 注册算法
        self._algorithm_registry = {
            CompressionAlgorithm.ZSTD: ZstdCompressor(),
            CompressionAlgorithm.LZ4: LZ4Compressor(),
            CompressionAlgorithm.BROTLI: BrotliCompressor(),
            CompressionAlgorithm.BF16_MODEL: BF16ModelCompressor(),
            CompressionAlgorithm.FP32_MODEL: FP32ModelCompressor(),
            CompressionAlgorithm.KV_CACHE: ZSMergeCompressor(),
            CompressionAlgorithm.LEXICO: LexicoCompressor(),
        }

    def compress(self,
                data: Union[bytes, bytearray, memoryview],
                config: Optional[CompressionConfig] = None) -> CompressedData:
        """
        压缩数据

        Args:
            data: 待压缩数据
            config: 压缩配置

        Returns:
            CompressedData: 压缩后的数据容器

        Raises:
            CompressionError: 压缩失败时抛出
        """
        # 使用提供的配置或默认配置
        actual_config = config or self.config

        # 开始统计
        stats = CompressionStats()
        stats.start_time = time.time()

        try:
            # 1. 数据验证
            self.validator.validate(data)

            # 2. 数据类型检测（如果需要）
            if actual_config.data_type == DataType.GENERIC:
                detected_type = self.type_detector.detect(data)
                actual_config.data_type = detected_type
                logger.debug(f"Detected data type: {detected_type.value}")

            # 3. 提取数据特征
            features = self.feature_extractor.extract(data, actual_config.data_type)

            # 4. 选择最优算法（如果需要）
            if actual_config.algorithm == CompressionAlgorithm.AUTO:
                selected_algorithm = self.algorithm_selector.select(features, actual_config)
                actual_config.algorithm = selected_algorithm
                logger.debug(f"Selected algorithm: {selected_algorithm.value}")

            # 5. 获取算法实例
            compressor = self._get_compressor(actual_config.algorithm)

            # 6. 执行压缩
            compressed = compressor.compress(data, actual_config)

            # 7. 更新统计信息
            stats.original_size = len(data)
            stats.compressed_size = len(compressed.data)
            stats.algorithm_used = actual_config.algorithm
            stats.level_used = actual_config.level
            stats.entropy = features.get('entropy', 0.0)
            stats.redundancy = features.get('redundancy', 0.0)

            # 8. 更新压缩数据的统计信息
            compressed.stats = stats

            # 9. 更新算法选择器（在线学习）
            if self.config.enable_stats:
                self.algorithm_selector.update(actual_config.algorithm, stats)
                self.stats_collector.record(stats)

            logger.info(
                f"Compression completed: {stats.original_size} -> {stats.compressed_size} bytes "
                f"({stats.compression_ratio:.2f}x) using {actual_config.algorithm.value}"
            )

            return compressed

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Compression failed: {e}") from e

        finally:
            stats.end_time = time.time()
            stats.compression_time = stats.end_time - stats.start_time
            stats.calculate_ratio()
            stats.calculate_throughput()

    def decompress(self, compressed: CompressedData) -> bytes:
        """
        解压数据

        Args:
            compressed: 压缩数据容器

        Returns:
            bytes: 解压后的数据

        Raises:
            DecompressionError: 解压失败时抛出
        """
        stats = CompressionStats()
        stats.start_time = time.time()

        try:
            # 获取算法实例
            compressor = self._get_compressor(compressed.algorithm)

            # 执行解压
            decompressed = compressor.decompress(compressed)

            # 更新统计
            stats.original_size = compressed.original_size
            stats.compressed_size = compressed.compressed_size
            stats.algorithm_used = compressed.algorithm
            stats.level_used = compressed.level

            logger.info(
                f"Decompression completed: {compressed.compressed_size} -> {len(decompressed)} bytes"
            )

            return decompressed

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise DecompressionError(f"Decompression failed: {e}") from e

        finally:
            stats.end_time = time.time()
            stats.decompression_time = stats.end_time - stats.start_time

    def _get_compressor(self, algorithm: CompressionAlgorithm) -> CompressorBase:
        """
        获取压缩算法实例

        Args:
            algorithm: 算法类型

        Returns:
            CompressorBase: 压缩器实例

        Raises:
            CompressionError: 算法不存在时抛出
        """
        if algorithm not in self._algorithm_registry:
            raise CompressionError(f"Unsupported algorithm: {algorithm.value}")

        return self._algorithm_registry[algorithm]

    def get_algorithm(self) -> CompressionAlgorithm:
        """
        获取算法类型

        Returns:
            CompressionAlgorithm: 算法类型
        """
        return CompressionAlgorithm.AUTO

    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取算法能力

        Returns:
            Dict: 算法能力描述
        """
        capabilities = {
            'supported_algorithms': [alg.value for alg in self._algorithm_registry.keys()],
            'features': {
                'adaptive_selection': True,
                'type_detection': True,
                'streaming': True,
                'parallel_compression': True,
                'gpu_acceleration': False
            },
            'performance': {
                'max_compression_ratio': '10x',
                'max_throughput': '500 MB/s',
                'supported_data_types': [dt.value for dt in DataType]
            }
        }

        return capabilities

    def analyze(self, data: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
        """
        分析数据特征

        Args:
            data: 待分析数据

        Returns:
            Dict: 分析结果
        """
        # 检测数据类型
        data_type = self.type_detector.detect(data)

        # 提取特征
        features = self.feature_extractor.extract(data, data_type)

        # 推荐算法
        recommended = self.algorithm_selector.select(features, self.config)

        return {
            'data_type': data_type.value,
            'features': features,
            'recommended_algorithm': recommended.value,
            'estimated_compression_ratio': features.get('estimated_ratio', 1.0)
        }

    def benchmark(self,
                 data: Union[bytes, bytearray, memoryview],
                 algorithms: Optional[list] = None) -> Dict[str, CompressionStats]:
        """
        基准测试多个算法

        Args:
            data: 测试数据
            algorithms: 要测试的算法列表，None则测试所有算法

        Returns:
            Dict: 算法名称到统计信息的映射
        """
        if algorithms is None:
            algorithms = list(self._algorithm_registry.keys())

        results = {}

        for algorithm in algorithms:
            try:
                config = CompressionConfig(algorithm=algorithm)
                compressed = self.compress(data, config)
                results[algorithm.value] = compressed.stats

            except Exception as e:
                logger.warning(f"Benchmark failed for {algorithm.value}: {e}")
                results[algorithm.value] = None

        return results

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要

        Returns:
            Dict: 统计摘要
        """
        return self.stats_collector.get_summary()
