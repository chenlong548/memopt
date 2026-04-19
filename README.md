# memopt - 高性能内存优化工具包

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

memopt 是一个高性能内存优化工具包，提供内存映射、数据压缩、流式处理、内存优化、惰性计算、稀疏数组、内存监控和缓冲区管理等功能。

## 核心模块

| 模块 | 功能 | 状态 |
|------|------|------|
| **mem_mapper** | 内存映射、NUMA感知、大页支持、预取优化 | ✅ |
| **data_compressor** | 多算法压缩(ZSTD/LZ4/Brotli)、自适应选择、KV Cache优化 | ✅ |
| **stream_processor** | 流式处理、窗口管理、检查点机制、背压控制 | ✅ |
| **mem_optimizer** | 内存分配优化(Buddy/Slab/TLSF)、RL策略、碎片整理 | ✅ |
| **lazy_evaluator** | 惰性计算、Memothunk、依赖图、流融合 | ✅ |
| **sparse_array** | 稀疏数组(CSR/CSC/COO/BCSR/Bitmap)、GPU加速 | ✅ |
| **mem_monitor** | 内存监控、生命周期分析、泄漏检测、NUMA感知 | ✅ |
| **buffer_manager** | 缓冲区管理、无锁队列(SPSC/MPSC/MPMC)、ARC替换 | ✅ |

## 安装

```bash
# 基础安装
pip install memopt

# GPU支持
pip install memopt[gpu]

# RL策略支持
pip install memopt[rl]

# 开发依赖
pip install memopt[dev]
```

## 快速开始

### 内存映射

```python
from mem_mapper import MemoryMapper

mapper = MemoryMapper()
region = mapper.map_file("large_file.bin", mode="readonly")
data = region.read(0, 1024)  # 读取前1024字节
mapper.unmap(region)
```

### 数据压缩

```python
from data_compressor import DataCompressor

compressor = DataCompressor(algorithm="zstd", level=3)
compressed = compressor.compress(data)
decompressed = compressor.decompress(compressed)
```

### 流式处理

```python
from stream_processor import Stream, TumblingWindow

stream = Stream.from_source(data_source)
result = (stream
    .map(lambda x: x * 2)
    .filter(lambda x: x > 10)
    .window(TumblingWindow(size=100))
    .reduce(lambda acc, x: acc + x)
    .collect())
```

### 内存优化

```python
from mem_optimizer import MemoryPool, BuddyAllocator

pool = MemoryPool(allocator=BuddyAllocator(block_size=4096))
block = pool.allocate(1024)
# 使用内存...
pool.deallocate(block)
```

### 惰性计算

```python
from lazy_evaluator import Lazy, memoize

@memoize(max_size=1000)
def expensive_computation(x):
    return x ** 2

lazy_value = Lazy(lambda: expensive_computation(100))
result = lazy_value.force()  # 只在需要时计算
```

### 稀疏数组

```python
from sparse_array import SparseArray
import numpy as np

dense = np.random.rand(1000, 1000)
dense[dense < 0.9] = 0
sparse = SparseArray.from_dense(dense)
result = sparse @ np.random.rand(1000)  # SpMV
```

### 内存监控

```python
from mem_monitor import MemoryMonitor, MonitorConfig

config = MonitorConfig(sample_interval=0.1)
monitor = MemoryMonitor(config)
monitor.start()

# ... 运行代码 ...

snapshot = monitor.get_snapshot()
print(f"Memory usage: {snapshot.get_usage_ratio():.2%}")
report = monitor.stop()
```

### 缓冲区管理

```python
from buffer_manager import BufferPool, SPSCQueue

# 缓冲池
pool = BufferPool(buffer_size=4096, num_buffers=16)
buffer = pool.acquire()
buffer.write(data)
pool.release(buffer)

# 无锁队列
queue = SPSCQueue(capacity=1000)
queue.enqueue(item)
item = queue.dequeue()
```

## 架构设计

每个模块采用分层架构设计：

```
模块/
├── core/           # 核心层 - 基础类和接口
├── [功能层]/       # 功能层 - 具体实现
├── integration/    # 集成层 - 模块间集成
└── tests/          # 测试层 - 单元测试
```

## 性能特性

- **内存映射**: 支持大页(2MB/1GB)、NUMA感知、预取优化
- **数据压缩**: 自适应算法选择、流式压缩、KV Cache专用压缩
- **流式处理**: 低延迟、背压控制、检查点恢复
- **内存优化**: RL驱动的分配策略、碎片整理、NUMA协调
- **惰性计算**: O(1)记忆化、依赖图增量计算
- **稀疏数组**: 多格式支持、GPU加速、自动格式选择
- **内存监控**: 低开销采样、生命周期分析、泄漏检测
- **缓冲区管理**: 无锁队列、ARC替换策略、水位线管理

## 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest mem_mapper/tests/
pytest data_compressor/tests/

# 带覆盖率报告
pytest --cov=memopt --cov-report=html
```

## 文档

详细文档请参阅 [docs/](docs/) 目录：

- [API参考](docs/api.md)
- [使用示例](docs/examples.md)
- [贡献指南](docs/contributing.md)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请参阅 [贡献指南](docs/contributing.md)。
