"""
memopt - 高性能内存优化工具包

安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memopt",
    version="1.0.1",
    author="memopt Team",
    author_email="1741548594@qq.com",
    description="高性能内存优化工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenlong548/memopt",
    packages=find_packages(exclude=["tests*", "benchmarks*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Memory",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "zstandard>=0.18.0",
        "lz4>=4.0.0",
        "brotli>=1.0.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "gpu": [
            "cupy>=10.0.0",
            "numba>=0.55.0",
        ],
        "rl": [
            "torch>=1.10.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.4.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "sphinx>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memopt-monitor=mem_monitor.cli:main",
        ],
    },
)
