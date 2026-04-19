"""
mem_monitor 异常定义模块

定义内存监控器的所有异常类型。
"""

from typing import Optional, Any


class MonitorError(Exception):
    """
    监控器基础异常

    所有监控相关异常的基类。
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            details: 额外详情
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ConfigurationError(MonitorError):
    """
    配置错误

    配置参数无效或不完整时抛出。
    """

    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {'config_key': config_key} if config_key else {}
        super().__init__(message, details)


class SamplerError(MonitorError):
    """
    采样器错误

    采样过程中发生错误时抛出。
    """

    def __init__(self, message: str, sampler_type: Optional[str] = None):
        details = {'sampler_type': sampler_type} if sampler_type else {}
        super().__init__(message, details)


class AnalyzerError(MonitorError):
    """
    分析器错误

    分析过程中发生错误时抛出。
    """

    def __init__(self, message: str, analyzer_type: Optional[str] = None):
        details = {'analyzer_type': analyzer_type} if analyzer_type else {}
        super().__init__(message, details)


class ReporterError(MonitorError):
    """
    报告器错误

    报告生成过程中发生错误时抛出。
    """

    def __init__(self, message: str, report_type: Optional[str] = None):
        details = {'report_type': report_type} if report_type else {}
        super().__init__(message, details)


class IntegrationError(MonitorError):
    """
    集成错误

    与外部模块集成时发生错误时抛出。
    """

    def __init__(self, message: str, module_name: Optional[str] = None):
        details = {'module_name': module_name} if module_name else {}
        super().__init__(message, details)


class ThresholdExceededError(MonitorError):
    """
    阈值超限错误

    监控指标超过阈值时抛出。
    """

    def __init__(self,
                 metric_name: str,
                 current_value: float,
                 threshold: float,
                 action: str = 'alert'):
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold = threshold
        self.action = action

        details = {
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'action': action
        }
        super().__init__(
            f"Metric '{metric_name}' exceeded threshold: {current_value} > {threshold}",
            details
        )


class SamplingNotSupportedError(SamplerError):
    """
    采样不支持错误

    当前平台或环境不支持请求的采样方式时抛出。
    """

    def __init__(self, sampler_type: str, reason: str = ""):
        message = f"Sampling type '{sampler_type}' is not supported"
        if reason:
            message += f": {reason}"
        super().__init__(message, sampler_type)


class AnalysisTimeoutError(AnalyzerError):
    """
    分析超时错误

    分析操作超时时抛出。
    """

    def __init__(self, analyzer_type: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Analysis '{analyzer_type}' timed out after {timeout_seconds}s",
            analyzer_type
        )
