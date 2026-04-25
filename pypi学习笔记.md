# PyPI 发布学习笔记

## 1. PyPI 账号准备

### 1.1 注册 PyPI 账号

1. 访问 https://pypi.org/account/register/
2. 填写用户名、邮箱和密码
3. 验证邮箱地址

### 1.2 创建 API Token

1. 登录 PyPI 账号
2. 进入 **Account settings** → **API tokens**
3. 创建 API Token，选择 "Entire account (all projects)"
4. 复制生成的 API Token（只显示一次，请妥善保存）

### 1.3 配置 .pypirc 文件

创建 `~/.pypirc` 文件（Windows 上是 `%USERPROFILE%\.pypirc`）：

```ini
[pypi]
username = __token__
password = your-pypi-api-token-here
```

## 2. 自动化发布（GitHub Actions）

### 2.1 配置 PyPI API Token

1. 登录 PyPI 账号
2. 进入 **Account settings** → **API tokens**
3. 创建 API Token，选择 "Entire account (all projects)"
4. 复制生成的 API Token

### 2.2 配置 GitHub Secrets

1. 进入 GitHub 仓库
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: 粘贴刚才复制的 API Token
6. 点击 **Add secret**

### 2.3 创建 GitHub Actions Workflow

创建 `.github/workflows/publish.yml` 文件：

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## 3. 手动发布流程

### 3.1 安装发布工具

```bash
pip install build twine
```

### 3.2 构建包

```bash
python -m build
```

这会生成：
- `dist/` 目录下的 `.tar.gz` 源码包
- `dist/` 目录下的 `.whl` wheel 包

### 3.3 检查包

```bash
twine check dist/*
```

### 3.4 上传到 PyPI

```bash
twine upload dist/*
```

## 4. 版本管理

### 4.1 更新版本号

在 `setup.py` 或 `pyproject.toml` 中更新版本号：

```python
# setup.py
setup(
    version="1.0.1",
    ...
)
```

或

```toml
# pyproject.toml
[project]
version = "1.0.1"
```

### 4.2 创建 Git Tag

```bash
git tag v1.0.1
git push origin v1.0.1
```

### 4.3 创建 GitHub Release

1. 进入 GitHub 仓库
2. 点击 **Releases** → **Draft a new release**
3. 选择刚才创建的 tag
4. 填写 Release 标题和说明
5. 点击 **Publish release**

## 5. 常见问题

### 5.1 包名已被占用

如果包名已被占用，需要：
1. 更改包名
2. 或联系 PyPI 管理员请求转移包名

### 5.2 上传失败

常见原因：
- API Token 无效或过期
- 版本号已存在
- 包结构不正确

### 5.3 测试发布

可以使用 TestPyPI 进行测试：

```bash
twine upload --repository testpypi dist/*
```

## 6. 最佳实践

### 6.1 版本号规范

使用语义化版本号：`MAJOR.MINOR.PATCH`
- MAJOR: 不兼容的 API 更改
- MINOR: 向后兼容的功能新增
- PATCH: 向后兼容的问题修复

### 6.2 README 和文档

确保：
- README.md 包含详细的使用说明
- 包含安装、使用示例
- 包含许可证信息

### 6.3 依赖管理

明确声明依赖：
- `install_requires`: 运行时依赖
- `extras_require`: 可选依赖
- `tests_require`: 测试依赖

## 7. 安全注意事项

### 7.1 API Token 安全

- 不要将 API Token 提交到代码库
- 使用 GitHub Secrets 存储 Token
- 定期更换 API Token
- 为不同项目创建不同的 Token

### 7.2 .pypirc 文件安全

- 不要将 .pypirc 文件提交到代码库
- 添加到 .gitignore
- 设置文件权限为 600

## 8. 参考资源

- [PyPI 官方文档](https://packaging.python.org/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
