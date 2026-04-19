"""
stream_processor 单元测试

测试stream_processor模块的核心功能。
"""

import pytest
import time
from typing import List

from stream_processor import (
    Record,
    Watermark,
    WatermarkGenerator,
    WatermarkStrategy,
    WatermarkTracker,
    Stream,
    StreamRecord,
    StreamType,
    KeyedStream,
    ExecutionContext,
    ExecutionConfig,
    ExecutionState,
    TaskMetrics,
    DAG,
    DAGNode,
    DAGEdge,
    CyclicDependencyError,
    InvalidOperatorError,
    CollectionSource,
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    KeyByOperator,
    ReduceOperator,
    PrintSink,
    CollectionSink,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    CountWindowAssigner,
    EventTimeTrigger,
    CountTrigger,
    Window,
    TokenBucket,
    LeakyBucket,
    SlidingWindowCounter,
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    FlowController,
    FlowControlConfig,
    BackpressureController,
    BackpressureConfig,
    BackpressureLevel,
    MemoryStateBackend,
    KeyValueState,
    ListState,
    SnapshotBuilder,
    SnapshotStore,
    CheckpointManager,
    CheckpointConfig,
    JsonSerializer,
    PickleSerializer,
    Counter,
    Gauge,
    Histogram,
    Meter,
    MetricsRegistry,
    generate_id,
    hash_value,
    timestamp_ms,
    format_bytes,
    chunk_list,
)


class TestRecord:
    """测试Record类"""

    def test_create_record(self):
        """测试创建记录"""
        record = Record(value="test_data")
        assert record.value == "test_data"
        assert record.key is None
        assert record.timestamp is not None

    def test_record_with_key(self):
        """测试带键的记录"""
        record = Record(value="test_data", key="test_key")
        assert record.key == "test_key"

    def test_get_key(self):
        """测试获取键"""
        record = Record(value="test_data")
        key = record.get_key()
        assert key is not None
        assert isinstance(key, str)

    def test_with_value(self):
        """测试创建新值的记录"""
        record1 = Record(value="data1", key="key1")
        record2 = record1.with_value("data2")
        assert record2.value == "data2"
        assert record2.key == "key1"

    def test_to_dict(self):
        """测试转换为字典"""
        record = Record(value="test_data", key="test_key")
        data = record.to_dict()
        assert data['value'] == "test_data"
        assert data['key'] == "test_key"


class TestWatermark:
    """测试Watermark类"""

    def test_create_watermark(self):
        """测试创建watermark"""
        watermark = Watermark(timestamp=100.0)
        assert watermark.timestamp == 100.0

    def test_watermark_comparison(self):
        """测试watermark比较"""
        wm1 = Watermark(timestamp=100.0)
        wm2 = Watermark(timestamp=200.0)
        assert wm1 < wm2
        assert wm2 > wm1
        assert wm1 <= wm2
        assert wm2 >= wm1


class TestWatermarkGenerator:
    """测试WatermarkGenerator类"""

    def test_create_generator(self):
        """测试创建生成器"""
        generator = WatermarkGenerator(out_of_orderness=5.0)
        assert generator.out_of_orderness == 5.0

    def test_update_watermark(self):
        """测试更新watermark"""
        generator = WatermarkGenerator(out_of_orderness=5.0)

        watermark = generator.update(100.0)
        assert watermark is not None
        assert watermark.timestamp == 95.0

        watermark = generator.update(110.0)
        assert watermark is not None
        assert watermark.timestamp == 105.0

    def test_watermark_strategy(self):
        """测试watermark策略"""
        generator = WatermarkStrategy.bounded_out_of_orderness(10.0)
        assert generator.out_of_orderness == 10.0


class TestWatermarkTracker:
    """测试WatermarkTracker类"""

    def test_create_tracker(self):
        """测试创建跟踪器"""
        tracker = WatermarkTracker(num_partitions=3)
        assert tracker.num_partitions == 3

    def test_update_partition_watermark(self):
        """测试更新分区watermark"""
        tracker = WatermarkTracker(num_partitions=3)

        tracker.update_partition_watermark(0, Watermark(timestamp=100.0))
        tracker.update_partition_watermark(1, Watermark(timestamp=150.0))
        tracker.update_partition_watermark(2, Watermark(timestamp=200.0))

        global_wm = tracker.get_global_watermark()
        assert global_wm.timestamp == 100.0


class TestStream:
    """测试Stream类"""

    def test_create_stream(self):
        """测试创建流"""
        stream = Stream(name="test_stream")
        assert stream.name == "test_stream"
        assert stream.parallelism == 1

    def test_emit_record(self):
        """测试发射记录"""
        stream = Stream(name="test_stream")
        record = Record(value="test_data")

        stream.emit(record)
        assert len(stream) == 1

    def test_stream_map(self):
        """测试流映射"""
        stream = Stream(name="test_stream")
        stream.emit(Record(value=1))
        stream.emit(Record(value=2))

        mapped = stream.map(lambda r: r.with_value(r.value * 2))
        assert len(mapped) == 2

    def test_stream_filter(self):
        """测试流过滤"""
        stream = Stream(name="test_stream")
        stream.emit(Record(value=1))
        stream.emit(Record(value=2))
        stream.emit(Record(value=3))

        filtered = stream.filter(lambda r: r.value % 2 == 0)
        assert len(filtered) == 1

    def test_key_by(self):
        """测试按键分组"""
        stream = Stream(name="test_stream")
        stream.emit(Record(value="a1", key="a"))
        stream.emit(Record(value="a2", key="a"))
        stream.emit(Record(value="b1", key="b"))

        keyed = stream.key_by(lambda r: r.key)
        assert "a" in keyed.get_keys()
        assert "b" in keyed.get_keys()


class TestDAG:
    """测试DAG类"""

    def test_create_dag(self):
        """测试创建DAG"""
        dag = DAG(name="test_dag")
        assert dag.name == "test_dag"
        assert len(dag) == 0

    def test_add_node(self):
        """测试添加节点"""
        dag = DAG(name="test_dag")
        node_id = dag.add_node(
            name="source",
            operator_type="source",
            parallelism=1
        )
        assert node_id is not None
        assert len(dag) == 1

    def test_add_edge(self):
        """测试添加边"""
        dag = DAG(name="test_dag")
        source_id = dag.add_node("source", "source")
        sink_id = dag.add_node("sink", "sink")

        dag.add_edge(source_id, sink_id)

        edges = dag.get_edges()
        assert len(edges) == 1

    def test_topological_sort(self):
        """测试拓扑排序"""
        dag = DAG(name="test_dag")
        source_id = dag.add_node("source", "source")
        transform_id = dag.add_node("transform", "transform")
        sink_id = dag.add_node("sink", "sink")

        dag.add_edge(source_id, transform_id)
        dag.add_edge(transform_id, sink_id)

        sorted_nodes = dag.topological_sort()
        assert sorted_nodes.index(source_id) < sorted_nodes.index(transform_id)
        assert sorted_nodes.index(transform_id) < sorted_nodes.index(sink_id)

    def test_cyclic_detection(self):
        """测试循环检测"""
        dag = DAG(name="test_dag")
        node1 = dag.add_node("node1", "transform")
        node2 = dag.add_node("node2", "transform")

        dag.add_edge(node1, node2)

        with pytest.raises(CyclicDependencyError):
            dag.add_edge(node2, node1)

    def test_validate_dag(self):
        """测试验证DAG"""
        dag = DAG(name="test_dag")
        source_id = dag.add_node("source", "source")
        sink_id = dag.add_node("sink", "sink")
        dag.add_edge(source_id, sink_id)

        assert dag.validate() is True


class TestExecutionContext:
    """测试ExecutionContext类"""

    def test_create_context(self):
        """测试创建上下文"""
        context = ExecutionContext(job_name="test_job")
        assert context.job_name == "test_job"
        assert context.get_state() == ExecutionState.CREATED

    def test_state_transitions(self):
        """测试状态转换"""
        context = ExecutionContext(job_name="test_job")

        context.start()
        assert context.is_running()

        context.complete()
        assert context.is_completed()


class TestOperators:
    """测试操作符"""

    def test_map_operator(self):
        """测试映射操作符"""
        operator = MapOperator(
            name="map",
            map_func=lambda x: x * 2
        )

        context = ExecutionContext(job_name="test")
        operator.open(context)

        record = Record(value=5)
        results = operator.process(record)

        assert len(results) == 1
        assert results[0].value == 10

    def test_filter_operator(self):
        """测试过滤操作符"""
        operator = FilterOperator(
            name="filter",
            predicate=lambda x: x > 5
        )

        context = ExecutionContext(job_name="test")
        operator.open(context)

        record1 = Record(value=3)
        record2 = Record(value=7)

        results1 = operator.process(record1)
        results2 = operator.process(record2)

        assert len(results1) == 0
        assert len(results2) == 1

    def test_flat_map_operator(self):
        """测试扁平映射操作符"""
        operator = FlatMapOperator(
            name="flat_map",
            flat_map_func=lambda x: [x, x * 2]
        )

        context = ExecutionContext(job_name="test")
        operator.open(context)

        record = Record(value=3)
        results = operator.process(record)

        assert len(results) == 2
        assert results[0].value == 3
        assert results[1].value == 6

    def test_reduce_operator(self):
        """测试归约操作符"""
        operator = ReduceOperator(
            name="reduce",
            reduce_func=lambda a, b: a + b
        )

        context = ExecutionContext(job_name="test")
        operator.open(context)

        records = [
            Record(value=1, key="a"),
            Record(value=2, key="a"),
            Record(value=3, key="a")
        ]

        for record in records:
            results = operator.process(record)

        assert len(results) == 1
        assert results[0].value == 6


class TestWindows:
    """测试窗口"""

    def test_tumbling_window(self):
        """测试滚动窗口"""
        assigner = TumblingWindowAssigner(size=10.0)

        record = Record(value="test", timestamp=15.0)
        windows = assigner.assign_windows(record)

        assert len(windows) == 1
        assert windows[0].start == 10.0
        assert windows[0].end == 20.0

    def test_sliding_window(self):
        """测试滑动窗口"""
        assigner = SlidingWindowAssigner(size=10.0, slide=5.0)

        record = Record(value="test", timestamp=12.0)
        windows = assigner.assign_windows(record)

        assert len(windows) > 0

    def test_count_window(self):
        """测试计数窗口"""
        assigner = CountWindowAssigner(count=5)

        for i in range(5):
            record = Record(value=i)
            windows = assigner.assign_windows(record)

        assert assigner.get_count() == 5

    def test_event_time_trigger(self):
        """测试事件时间触发器"""
        trigger = EventTimeTrigger()

        records = [Record(value="test", timestamp=100.0)]
        watermark = Watermark(timestamp=105.0)

        should_fire = trigger.should_fire(records, records[0], watermark)
        assert should_fire is True


class TestBackpressure:
    """测试背压控制"""

    def test_token_bucket(self):
        """测试令牌桶"""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        assert bucket.try_consume(5) is True
        assert bucket.try_consume(6) is False
        assert bucket.try_consume(5) is True

    def test_rate_limiter(self):
        """测试限流器"""
        config = RateLimitConfig(
            rate=10.0,
            capacity=10,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        limiter = RateLimiter(config)

        for _ in range(10):
            result = limiter.try_acquire()
            assert result.allowed is True

        result = limiter.try_acquire()
        assert result.allowed is False

    def test_flow_controller(self):
        """测试流量控制器"""
        config = FlowControlConfig(max_buffer_size=10)
        controller = FlowController(config)

        for i in range(10):
            assert controller.offer(i) is True

        assert controller.offer(11) is False

    def test_backpressure_controller(self):
        """测试背压控制器"""
        config = BackpressureConfig()
        controller = BackpressureController(config)

        controller.update_utilization(0.3)
        assert controller.get_level() == BackpressureLevel.NONE

        controller.update_utilization(0.8)
        assert controller.get_level() in [
            BackpressureLevel.MEDIUM,
            BackpressureLevel.HIGH
        ]


class TestCheckpoint:
    """测试检查点"""

    def test_memory_state_backend(self):
        """测试内存状态后端"""
        backend = MemoryStateBackend()

        backend.put("key1", "value1")
        assert backend.get("key1").value == "value1"

        backend.delete("key1")
        assert backend.get("key1") is None

    def test_key_value_state(self):
        """测试键值状态"""
        backend = MemoryStateBackend()
        state = KeyValueState(backend, namespace="test")

        state.put("key1", "value1")
        assert state.get("key1") == "value1"

        assert state.contains("key1") is True
        state.delete("key1")
        assert state.contains("key1") is False

    def test_snapshot(self):
        """测试快照"""
        builder = SnapshotBuilder(job_id="test_job", checkpoint_id=1)
        builder.add_state_data("key1", "value1")

        snapshot = builder.build()
        assert snapshot.metadata.job_id == "test_job"
        assert snapshot.metadata.checkpoint_id == 1

    def test_checkpoint_manager(self):
        """测试检查点管理器"""
        config = CheckpointConfig(checkpoint_interval=1.0)
        manager = CheckpointManager(
            job_id="test_job",
            config=config
        )

        result = manager.trigger_checkpoint()
        assert result.success is True


class TestSerialization:
    """测试序列化"""

    def test_json_serializer(self):
        """测试JSON序列化器"""
        serializer = JsonSerializer()

        data = {"key": "value", "number": 123}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data

    def test_pickle_serializer(self):
        """测试Pickle序列化器"""
        serializer = PickleSerializer()

        data = {"key": "value", "list": [1, 2, 3]}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data


class TestMetrics:
    """测试指标"""

    def test_counter(self):
        """测试计数器"""
        counter = Counter("test_counter")

        counter.increment()
        assert counter.get_count() == 1

        counter.increment(5)
        assert counter.get_count() == 6

    def test_gauge(self):
        """测试仪表"""
        gauge = Gauge("test_gauge")

        gauge.set(100.0)
        assert gauge.get_value() == 100.0

        gauge.increment(50.0)
        assert gauge.get_value() == 150.0

    def test_histogram(self):
        """测试直方图"""
        histogram = Histogram("test_histogram")

        for i in range(100):
            histogram.update(i)

        assert histogram.get_count() == 100
        assert histogram.get_min() == 0
        assert histogram.get_max() == 99

    def test_meter(self):
        """测试速率计"""
        meter = Meter("test_meter")

        for _ in range(10):
            meter.mark()

        assert meter.get_count() == 10

    def test_metrics_registry(self):
        """测试指标注册表"""
        registry = MetricsRegistry()

        counter = registry.counter("test_counter")
        counter.increment()

        gauge = registry.gauge("test_gauge")
        gauge.set(100.0)

        metrics = registry.get_all_metrics()
        assert "test_counter" in metrics
        assert "test_gauge" in metrics


class TestHelpers:
    """测试辅助函数"""

    def test_generate_id(self):
        """测试生成ID"""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2

    def test_hash_value(self):
        """测试哈希值"""
        hash1 = hash_value("test")
        hash2 = hash_value("test")
        assert hash1 == hash2

    def test_timestamp(self):
        """测试时间戳"""
        ts = timestamp_ms()
        assert ts > 0
        assert isinstance(ts, int)

    def test_format_bytes(self):
        """测试格式化字节"""
        assert format_bytes(500) == "500.00B"
        assert format_bytes(1024) == "1.00KB"
        assert format_bytes(1024 * 1024) == "1.00MB"

    def test_chunk_list(self):
        """测试分块列表"""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chunks = chunk_list(lst, 3)

        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert len(chunks[-1]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
