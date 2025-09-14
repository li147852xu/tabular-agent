#!/usr/bin/env python3
"""
Tabular Agent v1.0.0 完整演示脚本
展示从数据生成到模型卡报告的完整流程
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print('='*60)

def print_step(step, description):
    """打印步骤"""
    print(f"\n📋 步骤 {step}: {description}")
    print("-" * 40)

def run_command(cmd, description, check=True):
    """运行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {e}")
        if e.stderr:
            print("错误信息:", e.stderr)
        return False

def create_sample_data():
    """创建示例数据"""
    print_step(1, "创建示例数据")
    
    # 设置随机种子
    np.random.seed(42)
    n_samples = 200
    
    # 生成特征
    X = np.random.randn(n_samples, 5)
    
    # 生成目标变量（二分类）
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(5)])
    df['target'] = y
    
    # 保存数据
    os.makedirs('demo_data', exist_ok=True)
    df[:160].to_csv('demo_data/train.csv', index=False)  # 训练集
    df[160:].to_csv('demo_data/test.csv', index=False)   # 测试集
    
    print(f"✅ 数据已创建:")
    print(f"   - 训练集: {df[:160].shape}")
    print(f"   - 测试集: {df[160:].shape}")
    print(f"   - 目标变量分布: {df['target'].value_counts().to_dict()}")
    
    return True

def run_basic_pipeline():
    """运行基础管道"""
    print_step(2, "运行基础ML管道")
    
    cmd = """
    tabular-agent run \
        --train demo_data/train.csv \
        --test demo_data/test.csv \
        --target target \
        --out demo_results/basic \
        --verbose
    """
    
    success = run_command(cmd, "基础管道")
    if success:
        print("✅ 基础管道运行成功")
        return True
    return False

def run_advanced_pipeline():
    """运行高级管道"""
    print_step(3, "运行高级ML管道（包含稳定性评估）")
    
    cmd = """
    tabular-agent run \
        --train demo_data/train.csv \
        --test demo_data/test.csv \
        --target target \
        --out demo_results/advanced \
        --stability-runs 5 \
        --calibration isotonic \
        --verbose
    """
    
    success = run_command(cmd, "高级管道")
    if success:
        print("✅ 高级管道运行成功")
        return True
    return False

def run_audit_pipeline():
    """运行数据审计"""
    print_step(4, "运行数据泄漏审计")
    
    cmd = """
    tabular-agent audit \
        --data demo_data/train.csv \
        --target target \
        --out demo_results/audit \
        --verbose
    """
    
    success = run_command(cmd, "数据审计")
    if success:
        print("✅ 数据审计完成")
        return True
    return False

def show_results():
    """显示结果"""
    print_step(5, "查看结果")
    
    results_dir = Path("demo_results")
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return False
    
    print("📊 生成的文件:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(str(results_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size:,} bytes)")
    
    return True

def open_model_card():
    """打开模型卡"""
    print_step(6, "打开模型卡报告")
    
    # 查找最新的模型卡
    model_cards = list(Path("demo_results").rglob("model_card.html"))
    if not model_cards:
        print("❌ 未找到模型卡文件")
        return False
    
    latest_card = max(model_cards, key=os.path.getctime)
    print(f"📄 模型卡位置: {latest_card}")
    
    # 尝试打开模型卡
    if sys.platform == "darwin":  # macOS
        run_command(f"open {latest_card}", "打开模型卡", check=False)
    elif sys.platform == "win32":  # Windows
        run_command(f"start {latest_card}", "打开模型卡", check=False)
    else:  # Linux
        run_command(f"xdg-open {latest_card}", "打开模型卡", check=False)
    
    print("✅ 模型卡已打开")
    return True

def run_tests():
    """运行测试"""
    print_step(7, "运行单元测试")
    
    cmd = "pytest tests/ -v --tb=short"
    success = run_command(cmd, "单元测试")
    if success:
        print("✅ 所有测试通过")
        return True
    return False

def main():
    """主函数"""
    print_header("Tabular Agent v1.0.0 完整演示")
    
    print("""
🎯 本演示将展示以下功能:
   1. 创建示例数据
   2. 运行基础ML管道
   3. 运行高级ML管道（稳定性评估）
   4. 运行数据泄漏审计
   5. 查看生成的结果
   6. 打开模型卡报告
   7. 运行单元测试
    """)
    
    # 检查是否安装了tabular-agent
    try:
        import tabular_agent
        print(f"✅ Tabular Agent已安装: {tabular_agent.__version__}")
    except ImportError:
        print("❌ Tabular Agent未安装，请先运行: pip install -e .")
        return 1
    
    # 执行演示步骤
    steps = [
        create_sample_data,
        run_basic_pipeline,
        run_advanced_pipeline,
        run_audit_pipeline,
        show_results,
        open_model_card,
        run_tests,
    ]
    
    success_count = 0
    for i, step in enumerate(steps, 1):
        try:
            if step():
                success_count += 1
            else:
                print(f"⚠️  步骤 {i} 失败，继续执行...")
        except Exception as e:
            print(f"❌ 步骤 {i} 出现异常: {e}")
    
    # 总结
    print_header("演示完成")
    print(f"✅ 成功完成 {success_count}/{len(steps)} 个步骤")
    
    if success_count == len(steps):
        print("🎉 所有功能演示成功！")
        print("\n📚 接下来可以:")
        print("   - 查看生成的模型卡报告")
        print("   - 尝试使用自己的数据")
        print("   - 探索更多高级功能")
        print("   - 查看文档: https://github.com/li147852xu/tabular-agent")
        return 0
    else:
        print("⚠️  部分功能演示失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
