# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-04-19

### Added
- 为各模块添加详细的实际场景示例
- 新增 3 个模块间集成示例，覆盖日常使用和科研场景
- 优化使用说明文档结构

### Documentation
- 更新 `使用说明.md`，添加 8 个模块的实际应用场景
- 新增 5 个模块间集成示例
- 完善文档结构和示例代码

## [1.0.0] - 2026-04-17

### Added

#### mem_mapper - 内存映射工具
- NUMA感知的并行内存映射
- 大页(HugeTLB)优化支持
- 跨平台兼容性 (Windows/Linux)
- 预取和访问模式建议 (madvise)
- 安全路径验证和符号链接检测

#### data_compressor - 数据压缩工具
- 多算法支持: ZSTD, LZ4, Brotli
- 自适应算法选择 (UCB Bandit)
- 数据类型自动检测
- 流式压缩支持
- KV Cache专用压缩 (ZSMerge, Lexico)
- 模型权重压缩 (BF16, FP32)

#### stream_processor - 流式处理器
- 惰性执行流处理
- 多种窗口类型: Tumbling, Sliding, Session, Count
- Chandy-Lamport检查点机制
- 水印机制处理乱序数据
- 令牌桶背压控制
- 与data_compressor集成

#### mem_optimizer - 内存分配优化器
- 多分配器: Buddy, Slab, TLSF
- 强化学习策略选择 (UCB Bandit)
- 碎片整理器
- NUMA协调器
- 内存监控集成

#### lazy_evaluator - 惰性计算工具
- Lazy[T]泛型类
- Memothunk实现
- 多级缓存 (LRU, MultiLevelCache)
- memoize装饰器
- 依赖图增量计算
- Stream Fusion优化

#### sparse_array - 稀疏数组
- 多存储格式: CSR, CSC, COO, BCSR, Bitmap
- 自动格式选择
- 算术运算和线性代数
- 块低秩压缩
- HSS矩阵支持
- NumPy/SciPy适配器

#### mem_monitor - 内存监控工具
- 实时内存监控
- 生命周期分析
- 热点分析
- 内存泄漏检测
- NUMA感知分层管理
- Prometheus指标导出
- psutil/tracemalloc适配器

#### buffer_manager - 缓冲区管理器
- BufferPool管理
- 无锁队列: SPSC, MPSC, MPMC
- RingBuffer环形缓冲区
- DoubleBuffer双缓冲
- ARC/LRU替换策略
- 预取策略
- 水位线流量控制

### Infrastructure
- setup.py 安装脚本
- requirements.txt 依赖管理
- pytest.ini 测试配置
- .gitignore 版本控制配置
- LICENSE (MIT)
- README.md 项目说明
- docs/ 文档目录
  - index.md
  - api.md
  - examples.md
  - contributing.md
- tests/ 集成测试
- benchmarks/ 基准测试

### Tests
- 547个单元测试全部通过
- 5个集成测试全部通过
- 代码质量评分 ≥ 8.5/10

## [0.1.0] - 2026-04-15

### Added
- 项目初始化
- 基础架构设计
- mem_mapper模块开发
- data_compressor模块开发
