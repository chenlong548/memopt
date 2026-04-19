# 使用示例

## 1. 内存映射大文件

```python
from mem_mapper import MemoryMapper, Advice

# 创建映射器
mapper = MemoryMapper()

# 映射大文件
region = mapper.map_file("large_data.bin", mode="readonly")

# 顺序访问优化
mapper.advise(region, Advice.SEQUENTIAL)

# 读取数据
header = region.read(0, 1024)
data_chunk = region.read(1024, 4096)

# 清理
mapper.unmap(region)
```

## 2. 自适应数据压缩

```python
from data_compressor import DataCompressor, CompressionType

# 创建自适应压缩器
compressor = DataCompressor(algorithm="auto")

# 压缩数据
data = b"Hello, World!" * 10000
compressed = compressor.compress(data)

print(f"原始大小: {len(data)}")
print(f"压缩后大小: {len(compressed)}")
print(f"压缩率: {compressor.compression_ratio:.2%}")

# 解压
decompressed = compressor.decompress(compressed)
assert data == decompressed
```

## 3. 流式数据处理

```python
from stream_processor import Stream, TumblingWindow, SlidingWindow

# 创建数据流
data = range(1000)
stream = Stream.from_source(data)

# 链式处理
result = (stream
    .map(lambda x: x * 2)
    .filter(lambda x: x > 500)
    .window(TumblingWindow(size=100))
    .reduce(lambda acc, x: acc + x, initial=0)
    .collect())

print(f"处理结果: {result}")
```

## 4. 内存池管理

```python
from mem_optimizer import MemoryPool, BuddyAllocator

# 创建内存池
pool = MemoryPool(
    allocator=BuddyAllocator(block_size=4096),
    total_size=100 * 1024 * 1024  # 100MB
)

# 分配内存
block1 = pool.allocate(1024)
block2 = pool.allocate(2048)

# 使用内存
block1.write(b"Hello")
block2.write(b"World" * 100)

# 释放内存
pool.deallocate(block1)
pool.deallocate(block2)

# 碎片整理
pool.defragment()
```

## 5. 惰性计算与记忆化

```python
from lazy_evaluator import Lazy, memoize, DependencyGraph

# 记忆化函数
@memoize(max_size=1000)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 惰性值
lazy_result = Lazy(lambda: fibonacci(100))
print(lazy_result.force())  # 只在需要时计算

# 依赖图
graph = DependencyGraph()
graph.add_node("a", lambda: 1)
graph.add_node("b", lambda: graph.get_result("a") * 2)
graph.add_node("c", lambda: graph.get_result("b") + 3)

print(graph.evaluate("c"))  # 5
```

## 6. 稀疏矩阵运算

```python
from sparse_array import SparseArray, recommend_format
import numpy as np

# 创建稀疏矩阵
dense = np.random.rand(1000, 1000)
dense[dense < 0.95] = 0  # 95%稀疏度

sparse = SparseArray.from_dense(dense)

# 推荐最优格式
recommendation = recommend_format(sparse)
print(f"推荐格式: {recommendation['recommended_format']}")

# 矩阵运算
x = np.random.rand(1000)
y = sparse @ x  # SpMV

# 格式转换
csr = sparse.to_csr()
csc = sparse.to_csc()
```

## 7. 实时内存监控

```python
from mem_monitor import MemoryMonitor, MonitorConfig, WatermarkLevel

# 配置监控
config = MonitorConfig(
    sample_interval=0.1,
    history_size=1000
)

monitor = MemoryMonitor(config)

# 设置水位线回调
def on_high_usage(level, value):
    print(f"警告: 内存使用率达到 {value:.2%}")

monitor.set_threshold('memory_usage', 0.8, callback=on_high_usage)

# 开始监控
monitor.start()

# ... 运行你的代码 ...

# 获取快照
snapshot = monitor.get_snapshot()
print(f"当前内存使用: {snapshot.get_usage_ratio():.2%}")
print(f"RSS: {snapshot.rss / 1024 / 1024:.2f} MB")

# 停止监控
report = monitor.stop()
print(f"平均内存使用: {report.avg_memory_usage:.2%}")
```

## 8. 高性能缓冲区

```python
from buffer_manager import BufferPool, SPSCQueue, MPMCQueue, ARC

# 缓冲池
pool = BufferPool(buffer_size=4096, num_buffers=16)

# 获取缓冲区
buffer = pool.acquire()
buffer.write(b"Hello, Buffer!")
data = buffer.read(13)
pool.release(buffer)

# SPSC队列 (单生产者单消费者)
queue = SPSCQueue(capacity=1000)
queue.enqueue("item1")
item = queue.dequeue()

# ARC缓存
cache = ARC(capacity=100)
cache.put("key1", "value1")
value = cache.get("key1")
```

## 9. 模块集成示例

```python
from mem_mapper import MemoryMapper
from data_compressor import DataCompressor
from stream_processor import Stream
from buffer_manager import BufferPool

# 创建组件
mapper = MemoryMapper()
compressor = DataCompressor(algorithm="zstd")
pool = BufferPool(buffer_size=65536, num_buffers=8)

# 映射文件
region = mapper.map_file("large_file.bin")

# 使用缓冲区处理
buffer = pool.acquire()
chunk = region.read(0, 65536)
buffer.write(chunk)

# 压缩数据
compressed = compressor.compress(buffer.data)

# 流式处理
stream = Stream.from_source([compressed])
result = stream.map(lambda x: len(x)).collect()

# 清理
pool.release(buffer)
mapper.unmap(region)
```
