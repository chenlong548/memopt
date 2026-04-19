"""
memopt 测试模块

包含集成测试和端到端测试。
"""

from .test_integration import (
    TestIntegration,
    TestMemMapperDataCompressorIntegration,
    TestStreamProcessorBufferManagerIntegration,
    TestMemOptimizerMemMonitorIntegration,
    TestSparseArrayLazyEvaluatorIntegration,
    TestFullPipeline,
)

__all__ = [
    "TestIntegration",
    "TestMemMapperDataCompressorIntegration",
    "TestStreamProcessorBufferManagerIntegration",
    "TestMemOptimizerMemMonitorIntegration",
    "TestSparseArrayLazyEvaluatorIntegration",
    "TestFullPipeline",
]
