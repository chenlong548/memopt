"""
mem_optimizer 异常定义模块

定义内存分配优化器中使用的所有异常类。
"""

from typing import Optional


class MemOptimizerError(Exception):
    """
    mem_optimizer基础异常类

    所有mem_optimizer相关的异常都继承自此类。
    """

    def __init__(self, message: str, error_code: Optional[int] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            error_code: 错误码（可选）
        """
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    def __str__(self) -> str:
        if self.error_code is not None:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AllocationError(MemOptimizerError):
    """
    内存分配错误

    当内存分配失败时抛出此异常。
    """

    def __init__(self, message: str, size: Optional[int] = None, error_code: Optional[int] = None):
        """
        初始化分配错误

        Args:
            message: 错误消息
            size: 请求的大小（可选）
            error_code: 错误码（可选）
        """
        self.size = size
        full_message = f"Allocation failed: {message}"
        if size is not None:
            full_message += f" (size={size})"
        super().__init__(full_message, error_code)


class OutOfMemoryError(AllocationError):
    """
    内存不足错误

    当没有足够的可用内存时抛出此异常。
    """

    def __init__(self, requested: int, available: int):
        """
        初始化内存不足错误

        Args:
            requested: 请求的大小
            available: 可用的大小
        """
        self.requested = requested
        self.available = available
        super().__init__(
            f"Out of memory: requested {requested}, available {available}",
            size=requested,
            error_code=1
        )


class FragmentationError(MemOptimizerError):
    """
    碎片错误

    当碎片化导致无法分配时抛出此异常。
    """

    def __init__(self, message: str, fragmentation_ratio: Optional[float] = None):
        """
        初始化碎片错误

        Args:
            message: 错误消息
            fragmentation_ratio: 碎片率（可选）
        """
        self.fragmentation_ratio = fragmentation_ratio
        full_message = f"Fragmentation error: {message}"
        if fragmentation_ratio is not None:
            full_message += f" (ratio={fragmentation_ratio:.4f})"
        super().__init__(full_message, error_code=2)


class DefragmentationError(MemOptimizerError):
    """
    碎片整理错误

    当碎片整理失败时抛出此异常。
    """

    def __init__(self, message: str, blocks_moved: Optional[int] = None):
        """
        初始化碎片整理错误

        Args:
            message: 错误消息
            blocks_moved: 已移动的块数（可选）
        """
        self.blocks_moved = blocks_moved
        full_message = f"Defragmentation failed: {message}"
        if blocks_moved is not None:
            full_message += f" (blocks_moved={blocks_moved})"
        super().__init__(full_message, error_code=3)


class NUMAError(MemOptimizerError):
    """
    NUMA操作错误

    当NUMA相关操作失败时抛出此异常。
    """

    def __init__(self, message: str, node: Optional[int] = None):
        """
        初始化NUMA错误

        Args:
            message: 错误消息
            node: 相关的NUMA节点（可选）
        """
        self.node = node
        full_message = f"NUMA error: {message}"
        if node is not None:
            full_message += f" (node={node})"
        super().__init__(full_message, error_code=4)


class NUMANotAvailableError(NUMAError):
    """
    NUMA不可用错误

    当系统不支持NUMA时抛出此异常。
    """

    def __init__(self, message: str = "NUMA is not available on this system"):
        super().__init__(message)


class NUMAMigrationError(NUMAError):
    """
    NUMA迁移错误

    当内存迁移失败时抛出此异常。
    """

    def __init__(self, source_node: int, target_node: int, size: int):
        """
        初始化NUMA迁移错误

        Args:
            source_node: 源节点
            target_node: 目标节点
            size: 迁移大小
        """
        self.source_node = source_node
        self.target_node = target_node
        self.size = size
        super().__init__(
            f"Migration failed from node {source_node} to {target_node}",
            node=target_node
        )


class ConfigurationError(MemOptimizerError):
    """
    配置错误

    当配置无效时抛出此异常。
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
        super().__init__(full_message, error_code=5)


class AllocatorError(MemOptimizerError):
    """
    分配器错误

    当分配器操作失败时抛出此异常。
    """

    def __init__(self, message: str, allocator_type: Optional[str] = None):
        """
        初始化分配器错误

        Args:
            message: 错误消息
            allocator_type: 分配器类型（可选）
        """
        self.allocator_type = allocator_type
        full_message = f"Allocator error: {message}"
        if allocator_type is not None:
            full_message += f" (allocator={allocator_type})"
        super().__init__(full_message, error_code=6)


class AllocatorNotInitializedError(AllocatorError):
    """
    分配器未初始化错误

    当分配器未正确初始化时抛出此异常。
    """

    def __init__(self, allocator_type: str):
        super().__init__(f"Allocator not initialized", allocator_type=allocator_type)


class InvalidBlockError(MemOptimizerError):
    """
    无效块错误

    当操作无效的内存块时抛出此异常。
    """

    def __init__(self, message: str, address: Optional[int] = None, size: Optional[int] = None):
        """
        初始化无效块错误

        Args:
            message: 错误消息
            address: 内存地址（可选）
            size: 块大小（可选）
        """
        self.address = address
        self.size = size
        full_message = f"Invalid block: {message}"
        if address is not None:
            full_message += f" (address=0x{address:x})"
        if size is not None:
            full_message += f" (size={size})"
        super().__init__(full_message, error_code=7)


class BlockNotFoundError(InvalidBlockError):
    """
    块未找到错误

    当指定的内存块不存在时抛出此异常。
    """

    def __init__(self, address: int):
        super().__init__("Block not found", address=address)


class DoubleFreeError(InvalidBlockError):
    """
    双重释放错误

    当尝试重复释放同一块内存时抛出此异常。
    """

    def __init__(self, address: int):
        super().__init__("Double free detected", address=address)


class CorruptionError(MemOptimizerError):
    """
    内存损坏错误

    当检测到内存损坏时抛出此异常。
    """

    def __init__(self, message: str, address: Optional[int] = None):
        """
        初始化内存损坏错误

        Args:
            message: 错误消息
            address: 损坏地址（可选）
        """
        self.address = address
        full_message = f"Memory corruption: {message}"
        if address is not None:
            full_message += f" (address=0x{address:x})"
        super().__init__(full_message, error_code=8)


class IntegrationError(MemOptimizerError):
    """
    集成错误

    当与其他模块集成失败时抛出此异常。
    """

    def __init__(self, message: str, module: Optional[str] = None):
        """
        初始化集成错误

        Args:
            message: 错误消息
            module: 相关模块名（可选）
        """
        self.module = module
        full_message = f"Integration error: {message}"
        if module is not None:
            full_message += f" (module={module})"
        super().__init__(full_message, error_code=9)


class MonitorError(MemOptimizerError):
    """
    监控错误

    当监控操作失败时抛出此异常。
    """

    def __init__(self, message: str, metric: Optional[str] = None):
        """
        初始化监控错误

        Args:
            message: 错误消息
            metric: 相关指标（可选）
        """
        self.metric = metric
        full_message = f"Monitor error: {message}"
        if metric is not None:
            full_message += f" (metric={metric})"
        super().__init__(full_message, error_code=10)


class PSIError(MonitorError):
    """
    PSI指标错误

    当PSI指标读取失败时抛出此异常。
    """

    def __init__(self, message: str = "Failed to read PSI metrics"):
        super().__init__(message, metric="psi")


class StrategyError(MemOptimizerError):
    """
    策略错误

    当策略选择失败时抛出此异常。
    """

    def __init__(self, message: str, strategy: Optional[str] = None):
        """
        初始化策略错误

        Args:
            message: 错误消息
            strategy: 相关策略（可选）
        """
        self.strategy = strategy
        full_message = f"Strategy error: {message}"
        if strategy is not None:
            full_message += f" (strategy={strategy})"
        super().__init__(full_message, error_code=11)


class RLTrainingError(StrategyError):
    """
    RL训练错误

    当强化学习训练失败时抛出此异常。
    """

    def __init__(self, message: str):
        super().__init__(message, strategy="rl")


class BanditError(StrategyError):
    """
    多臂老虎机错误

    当多臂老虎机算法失败时抛出此异常。
    """

    def __init__(self, message: str):
        super().__init__(message, strategy="bandit")
