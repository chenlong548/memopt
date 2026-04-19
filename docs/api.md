# API 参考

## mem_mapper

### MemoryMapper

```python
class MemoryMapper:
    def map_file(self, path: str, mode: str = 'readonly',
                 numa_node: int = -1) -> MappedRegion:
        """映射文件到内存"""
    
    def unmap(self, region: MappedRegion) -> None:
        """解除映射"""
    
    def advise(self, region: MappedRegion, advice: str) -> None:
        """提供访问模式建议"""
```

### MappedRegion

```python
class MappedRegion:
    def read(self, offset: int, size: int) -> bytes:
        """读取数据"""
    
    def write(self, offset: int, data: bytes) -> None:
        """写入数据"""
    
    @property
    def size(self) -> int:
        """区域大小"""
```

---

## data_compressor

### DataCompressor

```python
class DataCompressor:
    def __init__(self, algorithm: str = "zstd", level: int = 3):
        """初始化压缩器"""
    
    def compress(self, data: bytes) -> bytes:
        """压缩数据"""
    
    def decompress(self, data: bytes) -> bytes:
        """解压数据"""
    
    @property
    def compression_ratio(self) -> float:
        """压缩率"""
```

---

## stream_processor

### Stream

```python
class Stream:
    @staticmethod
    def from_source(source: Iterable) -> 'Stream':
        """从数据源创建流"""
    
    def map(self, func: Callable) -> 'Stream':
        """映射转换"""
    
    def filter(self, pred: Callable) -> 'Stream':
        """过滤"""
    
    def window(self, window: Window) -> 'Stream':
        """窗口操作"""
    
    def reduce(self, func: Callable) -> 'Stream':
        """归约"""
    
    def collect(self) -> List:
        """收集结果"""
```

---

## mem_optimizer

### MemoryPool

```python
class MemoryPool:
    def __init__(self, allocator: Allocator = None, 
                 total_size: int = 1024*1024*1024):
        """初始化内存池"""
    
    def allocate(self, size: int) -> MemoryBlock:
        """分配内存"""
    
    def deallocate(self, block: MemoryBlock) -> None:
        """释放内存"""
    
    def defragment(self) -> None:
        """碎片整理"""
```

---

## lazy_evaluator

### Lazy

```python
class Lazy(Generic[T]):
    def __init__(self, thunk: Callable[[], T]):
        """创建惰性值"""
    
    def force(self) -> T:
        """强制求值"""
    
    def map(self, func: Callable[[T], U]) -> 'Lazy[U]':
        """映射操作"""
```

### memoize

```python
def memoize(max_size: int = 1000, ttl: float = None):
    """记忆化装饰器"""
```

---

## sparse_array

### SparseArray

```python
class SparseArray:
    @staticmethod
    def from_dense(arr: np.ndarray, threshold: float = 0.05) -> 'SparseArray':
        """从密集数组创建"""
    
    def to_dense(self) -> np.ndarray:
        """转换为密集数组"""
    
    def dot(self, other) -> 'SparseArray':
        """矩阵乘法"""
    
    @property
    def nnz(self) -> int:
        """非零元素数量"""
```

---

## mem_monitor

### MemoryMonitor

```python
class MemoryMonitor:
    def __init__(self, config: MonitorConfig = None):
        """初始化监控器"""
    
    def start(self) -> None:
        """开始监控"""
    
    def stop(self) -> MonitorReport:
        """停止监控并返回报告"""
    
    def get_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
    
    def set_threshold(self, metric: str, value: float, 
                      action: str = "alert") -> None:
        """设置阈值"""
```

---

## buffer_manager

### BufferPool

```python
class BufferPool:
    def __init__(self, buffer_size: int, num_buffers: int):
        """初始化缓冲池"""
    
    def acquire(self, timeout: float = None) -> Buffer:
        """获取缓冲区"""
    
    def release(self, buffer: Buffer) -> None:
        """释放缓冲区"""
```

### SPSCQueue

```python
class SPSCQueue:
    def __init__(self, capacity: int):
        """初始化SPSC队列"""
    
    def enqueue(self, item: Any) -> bool:
        """入队"""
    
    def dequeue(self) -> Optional[Any]:
        """出队"""
```
