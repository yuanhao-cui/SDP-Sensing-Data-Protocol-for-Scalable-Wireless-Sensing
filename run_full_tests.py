#!/usr/bin/env python3
"""
WSDP 全量测试规划与执行脚本

测试范围:
- 5个数据集: elderAL, gait, widar, xrf55, zte
- 12个模型: csimodel, mlpmodel, cnn1dmodel, cnn2dmodel, lstmmodel, 
            resnet1d, resnet2d, bilstmattention, efficientnetcsi,
            visiontransformercsi, mambacsi, graphneuralcsi
- 20个算法: denoise(3), calibrate(4), normalize(2), interpolate(3),
            detect(2), extract_features(4), outliers(2)
- 60个pipeline组合: 5数据集 × 12模型

执行策略:
- L1: 单元测试 (现有279个) - 快速
- L2: 模型×合成数据 (12模型×5测试=60个) - 中等
- L3: 算法全量 (20算法×多场景≈100个) - 中等
- L4: Pipeline集成 (60组合) - 慢速，需下载数据
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# 测试配置
TEST_CONFIG = {
    "datasets": ["elderAL", "gait", "widar", "xrf55", "zte"],
    "models": [
        "csimodel", "mlpmodel", "cnn1dmodel", "cnn2dmodel", "lstmmodel",
        "resnet1d", "resnet2d", "bilstmattention", "efficientnetcsi",
        "visiontransformercsi", "mambacsi", "graphneuralcsi"
    ],
    "algorithms": {
        "denoise": ["wavelet", "butterworth", "savgol"],
        "calibrate": ["linear", "polynomial", "stc", "robust"],
        "normalize": ["z-score", "min-max"],
        "interpolate": ["linear", "cubic", "nearest"],
        "detect": ["activity", "change_point"],
        "extract_features": ["doppler", "entropy", "ratio", "decomposition"],
        "outliers": ["iqr", "z-score"]
    }
}


def run_test_group(test_files, group_name, parallel=True):
    """运行一组测试文件"""
    print(f"\n{'='*60}")
    print(f"开始执行: {group_name}")
    print(f"{'='*60}")
    
    results = []
    start_time = time.time()
    
    if parallel and len(test_files) > 1:
        # 并行执行
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_single_test, f): f for f in test_files}
            for future in as_completed(futures):
                test_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": test_file,
                        "status": "ERROR",
                        "error": str(e)
                    })
    else:
        # 串行执行
        for test_file in test_files:
            result = run_single_test(test_file)
            results.append(result)
    
    elapsed = time.time() - start_time
    
    # 汇总结果
    passed = sum(1 for r in results if r.get("status") == "PASSED")
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    errors = sum(1 for r in results if r.get("status") == "ERROR")
    
    print(f"\n{group_name} 结果汇总:")
    print(f"  通过: {passed} | 失败: {failed} | 错误: {errors}")
    print(f"  耗时: {elapsed:.2f}s")
    
    return {
        "group": group_name,
        "results": results,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "elapsed": elapsed
    }


def run_single_test(test_file):
    """运行单个测试文件"""
    cmd = [
        sys.executable, "-m", "pytest", test_file,
        "-v", "--tb=short", "--no-header"
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        # 解析结果
        stdout = result.stdout
        if "passed" in stdout and "failed" not in stdout.lower():
            status = "PASSED"
        elif "failed" in stdout.lower():
            status = "FAILED"
        else:
            status = "UNKNOWN"
        
        return {
            "file": test_file,
            "status": status,
            "returncode": result.returncode,
            "summary": extract_summary(stdout)
        }
    except subprocess.TimeoutExpired:
        return {
            "file": test_file,
            "status": "TIMEOUT",
            "error": "Test timed out after 300s"
        }
    except Exception as e:
        return {
            "file": test_file,
            "status": "ERROR",
            "error": str(e)
        }


def extract_summary(stdout):
    """从pytest输出中提取摘要"""
    for line in stdout.split('\n'):
        if 'passed' in line or 'failed' in line:
            return line.strip()
    return "N/A"


def run_level1_unit_tests():
    """L1: 现有单元测试 (279个)"""
    return run_test_group(
        ["tests/"],  # 运行所有现有测试
        "L1: 单元测试 (279个)",
        parallel=False  # pytest内部已并行
    )


def run_level2_model_tests():
    """L2: 模型全量测试 (12模型 × 5场景 = 60个)"""
    return run_test_group(
        ["tests/test_all_models_full.py"],
        "L2: 模型全量测试 (12模型×5场景)",
        parallel=False
    )


def run_level3_algorithm_tests():
    """L3: 算法全量测试 (20算法 × 多场景 ≈ 100个)"""
    return run_test_group(
        ["tests/test_all_algorithms_full.py"],
        "L3: 算法全量测试 (20算法)",
        parallel=False
    )


def run_level4_pipeline_tests(data_dir, output_dir):
    """L4: Pipeline集成测试 (5数据集 × 12模型 = 60组合)"""
    print(f"\n{'='*60}")
    print(f"开始执行: L4 Pipeline集成测试 (60组合)")
    print(f"{'='*60}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    results = []
    start_time = time.time()
    
    # 为每个数据集创建一个测试进程
    dataset_tests = []
    for dataset in TEST_CONFIG["datasets"]:
        dataset_path = Path(data_dir) / dataset
        if dataset_path.exists():
            dataset_tests.append((dataset, dataset_path))
        else:
            print(f"  ⚠️  跳过 {dataset}: 数据目录不存在")
    
    print(f"\n可用数据集: {[d[0] for d in dataset_tests]}")
    
    # 并行执行每个数据集的测试
    with ProcessPoolExecutor(max_workers=min(4, len(dataset_tests))) as executor:
        futures = {
            executor.submit(
                run_dataset_pipelines,
                dataset, path, output_dir
            ): dataset for dataset, path in dataset_tests
        }
        
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "dataset": dataset,
                    "status": "ERROR",
                    "error": str(e)
                })
    
    elapsed = time.time() - start_time
    
    # 汇总
    total_combinations = len(results) * len(TEST_CONFIG["models"])
    passed = sum(r.get("passed", 0) for r in results)
    failed = sum(r.get("failed", 0) for r in results)
    
    print(f"\nL4 Pipeline结果汇总:")
    print(f"  数据集: {len(results)} 个")
    print(f"  模型: {len(TEST_CONFIG['models'])} 个")
    print(f"  总组合: {total_combinations}")
    print(f"  通过: {passed} | 失败: {failed}")
    print(f"  耗时: {elapsed:.2f}s ({elapsed/60:.1f}min)")
    
    return {
        "group": "L4: Pipeline集成测试",
        "results": results,
        "passed": passed,
        "failed": failed,
        "elapsed": elapsed
    }


def run_dataset_pipelines(dataset, dataset_path, output_dir):
    """测试单个数据集的所有模型组合"""
    from wsdp import pipeline
    
    results = []
    dataset_output = Path(output_dir) / dataset
    
    for model_name in TEST_CONFIG["models"]:
        try:
            # 运行简化版pipeline (1个epoch, 1个seed)
            pipeline(
                input_path=str(dataset_path),
                output_folder=str(dataset_output / model_name),
                dataset=dataset,
                num_epochs=1,
                num_seeds=1
            )
            results.append({
                "dataset": dataset,
                "model": model_name,
                "status": "PASSED"
            })
        except Exception as e:
            results.append({
                "dataset": dataset,
                "model": model_name,
                "status": "FAILED",
                "error": str(e)
            })
    
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    
    return {
        "dataset": dataset,
        "passed": passed,
        "failed": failed,
        "results": results
    }


def generate_report(all_results, output_path):
    """生成测试报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_groups": len(all_results),
            "total_passed": sum(r.get("passed", 0) for r in all_results),
            "total_failed": sum(r.get("failed", 0) for r in all_results),
            "total_errors": sum(r.get("errors", 0) for r in all_results),
            "total_elapsed": sum(r.get("elapsed", 0) for r in all_results)
        },
        "details": all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"测试报告已保存: {output_path}")
    print(f"{'='*60}")
    print(f"总通过: {report['summary']['total_passed']}")
    print(f"总失败: {report['summary']['total_failed']}")
    print(f"总错误: {report['summary']['total_errors']}")
    print(f"总耗时: {report['summary']['total_elapsed']:.2f}s")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='WSDP全量测试')
    parser.add_argument('--level', choices=['1', '2', '3', '4', 'all'], 
                        default='all', help='测试层级')
    parser.add_argument('--data-dir', default='./data', 
                        help='数据集根目录')
    parser.add_argument('--output-dir', default='./test_output',
                        help='测试输出目录')
    parser.add_argument('--report', default='test_report.json',
                        help='测试报告路径')
    args = parser.parse_args()
    
    all_results = []
    
    # L1: 单元测试
    if args.level in ['1', 'all']:
        all_results.append(run_level1_unit_tests())
    
    # L2: 模型测试
    if args.level in ['2', 'all']:
        all_results.append(run_level2_model_tests())
    
    # L3: 算法测试
    if args.level in ['3', 'all']:
        all_results.append(run_level3_algorithm_tests())
    
    # L4: Pipeline测试 (需要数据)
    if args.level in ['4', 'all']:
        all_results.append(run_level4_pipeline_tests(
            args.data_dir, args.output_dir
        ))
    
    # 生成报告
    generate_report(all_results, args.report)


if __name__ == '__main__':
    main()
