"""
data_compressor 测试运行器

运行所有测试并生成测试报告。
"""

import unittest
import os
import sys
import time
import json
from io import StringIO
from typing import Dict, Any, List

# 添加项目路径 - 确保data_compressor模块可以被导入
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class TestReport:
    """测试报告"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.failures: List[str] = []

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def end(self):
        """结束计时"""
        self.end_time = time.time()

    def add_result(self, test_class: str, result: unittest.TestResult):
        """添加测试结果"""
        self.results[test_class] = {
            'tests_run': result.testsRun,
            'errors': len(result.errors),
            'failures': len(result.failures),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
        }

        for test, traceback in result.errors:
            self.errors.append(f"{test_class}: {test}\n{traceback}")

        for test, traceback in result.failures:
            self.failures.append(f"{test_class}: {test}\n{traceback}")

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())

        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0

        return {
            'total_tests': total_tests,
            'total_errors': total_errors,
            'total_failures': total_failures,
            'total_skipped': total_skipped,
            'success_rate': (total_tests - total_errors - total_failures) / total_tests * 100 if total_tests > 0 else 0,
            'duration_seconds': duration,
            'passed': total_tests - total_errors - total_failures - total_skipped,
        }

    def generate_report(self) -> str:
        """生成报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("data_compressor 测试报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        summary = self.get_summary()
        lines.append("测试摘要:")
        lines.append(f"  总测试数: {summary['total_tests']}")
        lines.append(f"  通过: {summary['passed']}")
        lines.append(f"  失败: {summary['total_failures']}")
        lines.append(f"  错误: {summary['total_errors']}")
        lines.append(f"  跳过: {summary['total_skipped']}")
        lines.append(f"  成功率: {summary['success_rate']:.2f}%")
        lines.append(f"  耗时: {summary['duration_seconds']:.2f}秒")
        lines.append("")

        lines.append("-" * 50)
        lines.append("各测试类结果:")
        lines.append("-" * 50)

        for test_class, result in self.results.items():
            status = "通过" if result['success'] else "失败"
            lines.append(f"  {test_class}: {status}")
            lines.append(f"    测试数: {result['tests_run']}")
            lines.append(f"    错误: {result['errors']}")
            lines.append(f"    失败: {result['failures']}")
            lines.append(f"    跳过: {result['skipped']}")

        if self.errors:
            lines.append("")
            lines.append("-" * 50)
            lines.append("错误详情:")
            lines.append("-" * 50)
            for error in self.errors[:10]:  # 只显示前10个
                lines.append(error)
                lines.append("")

        if self.failures:
            lines.append("")
            lines.append("-" * 50)
            lines.append("失败详情:")
            lines.append("-" * 50)
            for failure in self.failures[:10]:  # 只显示前10个
                lines.append(failure)
                lines.append("")

        return "\n".join(lines)

    def save_report(self, filepath: str):
        """保存报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())

    def save_json(self, filepath: str):
        """保存JSON格式报告"""
        report_data = {
            'summary': self.get_summary(),
            'results': self.results,
            'errors': self.errors,
            'failures': self.failures,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)


def run_all_tests(verbose: int = 2) -> TestReport:
    """运行所有测试"""
    report = TestReport()
    report.start()

    # 发现并加载所有测试
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # 测试文件列表
    test_modules = [
        'test_unit',
        'test_security',
        'test_performance',
        'test_integration',
    ]

    for module_name in test_modules:
        print(f"\n{'=' * 50}")
        print(f"运行 {module_name}...")
        print(f"{'=' * 50}")

        try:
            # 导入模块
            module = __import__(module_name, globals(), locals(), ['*'])

            # 加载测试
            suite = loader.loadTestsFromModule(module)

            # 运行测试
            runner = unittest.TextTestRunner(verbosity=verbose)
            result = runner.run(suite)

            # 记录结果
            report.add_result(module_name, result)

        except Exception as e:
            print(f"加载测试模块 {module_name} 失败: {e}")
            report.errors.append(f"{module_name}: {str(e)}")

    report.end()
    return report


def run_quick_tests(verbose: int = 2) -> TestReport:
    """运行快速测试（跳过性能测试）"""
    report = TestReport()
    report.start()

    loader = unittest.TestLoader()

    test_modules = [
        'test_unit',
        'test_security',
        'test_integration',
    ]

    for module_name in test_modules:
        print(f"\n运行 {module_name}...")

        try:
            module = __import__(module_name, globals(), locals(), ['*'])
            suite = loader.loadTestsFromModule(module)
            runner = unittest.TextTestRunner(verbosity=verbose)
            result = runner.run(suite)
            report.add_result(module_name, result)
        except Exception as e:
            print(f"加载测试模块 {module_name} 失败: {e}")
            report.errors.append(f"{module_name}: {str(e)}")

    report.end()
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='运行 data_compressor 测试')
    parser.add_argument('--quick', action='store_true', help='运行快速测试（跳过性能测试）')
    parser.add_argument('--report', type=str, default=None, help='保存报告到文件')
    parser.add_argument('--json', type=str, default=None, help='保存JSON报告到文件')
    parser.add_argument('--verbose', '-v', type=int, default=2, help='详细程度')

    args = parser.parse_args()

    print("=" * 70)
    print("data_compressor 测试套件")
    print("=" * 70)

    if args.quick:
        report = run_quick_tests(args.verbose)
    else:
        report = run_all_tests(args.verbose)

    # 打印报告
    print("\n" + report.generate_report())

    # 保存报告
    if args.report:
        report.save_report(args.report)
        print(f"\n报告已保存到: {args.report}")

    if args.json:
        report.save_json(args.json)
        print(f"JSON报告已保存到: {args.json}")

    # 返回退出码
    summary = report.get_summary()
    if summary['total_errors'] > 0 or summary['total_failures'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
