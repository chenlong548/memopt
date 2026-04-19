# 贡献指南

感谢您对 memopt 项目的关注！本文档将帮助您了解如何为项目做出贡献。

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/memopt/memopt.git
cd memopt
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

### 3. 安装开发依赖

```bash
pip install -e ".[dev]"
```

## 代码规范

### Python版本

- 支持 Python 3.8+
- 使用类型注解
- 遵循 PEP 8 规范

### 代码格式化

```bash
# 使用 Black 格式化代码
black .

# 使用 Flake8 检查代码
flake8 .

# 使用 Mypy 类型检查
mypy .
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def function(arg1: int, arg2: str) -> bool:
    """函数简短描述。

    详细描述（可选）。

    Args:
        arg1: 第一个参数的描述。
        arg2: 第二个参数的描述。

    Returns:
        返回值的描述。

    Raises:
        ValueError: 参数无效时抛出。

    Example:
        >>> function(1, "test")
        True
    """
    return True
```

## 项目结构

```
memopt/
├── mem_mapper/          # 内存映射模块
├── data_compressor/     # 数据压缩模块
├── stream_processor/    # 流式处理模块
├── mem_optimizer/       # 内存优化模块
├── lazy_evaluator/      # 惰性计算模块
├── sparse_array/        # 稀疏数组模块
├── mem_monitor/         # 内存监控模块
├── buffer_manager/      # 缓冲区管理模块
├── tests/               # 集成测试
├── benchmarks/          # 基准测试
├── docs/                # 文档
└── setup.py             # 安装脚本
```

## 模块开发规范

每个模块应遵循以下结构：

```
模块/
├── __init__.py          # 模块入口，导出公共API
├── core/                # 核心层
│   ├── __init__.py
│   ├── base.py          # 基础类和接口
│   └── exceptions.py    # 异常定义
├── [功能层]/            # 具体功能实现
├── integration/         # 与其他模块的集成
└── tests/               # 单元测试
    ├── __init__.py
    └── test_unit.py
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest mem_mapper/tests/

# 运行带覆盖率的测试
pytest --cov=memopt --cov-report=html

# 运行基准测试
pytest benchmarks/
```

### 测试规范

- 每个新功能必须有对应的测试
- 测试覆盖率应达到 80% 以上
- 使用 pytest 框架
- 测试文件命名为 `test_*.py`

```python
import unittest

class TestFeature(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        pass
    
    def test_basic_functionality(self):
        """测试基本功能"""
        self.assertEqual(function(1, 2), 3)
    
    def test_edge_cases(self):
        """测试边界情况"""
        with self.assertRaises(ValueError):
            function(-1, 0)
    
    def tearDown(self):
        """测试后清理"""
        pass
```

## 提交代码

### Git 工作流

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 提交信息规范

使用约定式提交：

- `feat:` 新功能
- `fix:` 修复bug
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具相关

示例：
```
feat(mem_mapper): 添加大页支持
fix(data_compressor): 修复ZSTD压缩内存泄漏
docs(readme): 更新安装说明
```

## 代码审查

所有 Pull Request 都需要经过代码审查：

1. 代码质量检查（Black, Flake8, Mypy）
2. 测试覆盖率检查
3. 功能正确性验证
4. 文档完整性检查

## 发布流程

1. 更新版本号
2. 更新 CHANGELOG
3. 创建 Git 标签
4. 构建发布包
5. 发布到 PyPI

```bash
# 构建
python setup.py sdist bdist_wheel

# 发布
twine upload dist/*
```

## 获取帮助

- 提交 Issue: https://github.com/memopt/memopt/issues
- 邮件: memopt@example.com

感谢您的贡献！
