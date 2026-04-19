# memopt 文档

## 目录

1. [API参考](api.md) - 完整的API文档
2. [使用示例](examples.md) - 详细的使用示例
3. [贡献指南](contributing.md) - 如何贡献代码

## 模块概览

### mem_mapper - 内存映射
高性能内存映射工具，支持NUMA感知和大页优化。

### data_compressor - 数据压缩
多算法压缩工具，支持ZSTD、LZ4、Brotli，自适应算法选择。

### stream_processor - 流式处理
低延迟流式处理框架，支持窗口操作和背压控制。

### mem_optimizer - 内存优化
智能内存分配器，支持Buddy、Slab、TLSF分配策略。

### lazy_evaluator - 惰性计算
惰性求值框架，支持记忆化和依赖图增量计算。

### sparse_array - 稀疏数组
高效稀疏数组实现，支持多种存储格式和GPU加速。

### mem_monitor - 内存监控
实时内存监控工具，支持生命周期分析和泄漏检测。

### buffer_manager - 缓冲区管理
高性能缓冲区管理器，支持无锁队列和ARC替换策略。
