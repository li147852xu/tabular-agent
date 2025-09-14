# Tabular Agent v0.1

## 概述 / Overview

Tabular Agent v0.1 是一个端到端的表格机器学习管道，从CSV数据到模型卡报告，无需LLM/Agent功能。

Tabular Agent v0.1 is an end-to-end tabular machine learning pipeline from CSV data to model card reports without LLM/Agent functionality.

## 功能特性 / Features

### 核心功能 / Core Features

- **数据画像 / Data Profiling**: 识别缺失值、异常值、高基数特征、目标分布和时间列
- **泄漏审计 / Leakage Audit**: 集成leakage-buster工具，检测数据泄漏
- **特征工程 / Feature Engineering**: 时间感知的特征工程，包括目标编码、WOE、滚动特征等
- **模型训练 / Model Training**: 支持LightGBM、XGBoost、CatBoost、线性模型和HistGBDT
- **超参数调优 / Hyperparameter Tuning**: 使用Optuna进行并行超参数优化
- **模型融合 / Model Blending**: 基础融合策略（均值、排序均值、逻辑均值）
- **模型评估 / Model Evaluation**: 全面的评估指标（AUC、KS、PR-AUC等）
- **报告生成 / Report Generation**: 生成专业的HTML模型卡

### 技术特点 / Technical Features

- **时间感知 / Time-aware**: 支持时间序列数据的特征工程
- **防泄漏 / Leakage-proof**: 内置防泄漏保护机制
- **可重现 / Reproducible**: 固定种子确保结果可重现
- **可扩展 / Scalable**: 支持并行处理和资源管理

## 安装 / Installation

```bash
pip install -e .
```

## 使用方法 / Usage

### 基本用法 / Basic Usage

```bash
tabular-agent run --train data/train.csv --test data/test.csv --target target --out runs/demo
```

### 时间序列数据 / Time Series Data

```bash
tabular-agent run --train data/train.csv --test data/test.csv --target target --time-col date --out runs/demo
```

### 高级配置 / Advanced Configuration

```bash
tabular-agent run \
  --train data/train.csv \
  --test data/test.csv \
  --target target \
  --time-col date \
  --n-jobs 32 \
  --time-budget 180 \
  --out runs/demo
```

## 参数说明 / Parameters

| 参数 / Parameter | 说明 / Description | 默认值 / Default |
|------------------|-------------------|------------------|
| `--train` | 训练数据路径 / Training data path | 必需 / Required |
| `--test` | 测试数据路径 / Test data path | 必需 / Required |
| `--target` | 目标列名 / Target column name | 必需 / Required |
| `--time-col` | 时间列名 / Time column name | 可选 / Optional |
| `--n-jobs` | 并行作业数 / Number of parallel jobs | 1 |
| `--time-budget` | 时间预算（秒）/ Time budget (seconds) | 300 |
| `--out` | 输出目录 / Output directory | 必需 / Required |
| `--audit-cli` | 泄漏审计CLI路径 / Leakage audit CLI path | 可选 / Optional |
| `--config` | 配置文件路径 / Configuration file path | 可选 / Optional |
| `--seed` | 随机种子 / Random seed | 42 |
| `--verbose` | 详细输出 / Verbose output | False |

## 输出结果 / Output Results

### 文件结构 / File Structure

```
runs/
└── demo/
    └── 20240101_120000/
        ├── meta.json          # 元数据 / Metadata
        ├── results.json       # 结果数据 / Results data
        └── model_card.html    # 模型卡报告 / Model card report
```

### 模型卡内容 / Model Card Contents

- **数据概览 / Data Overview**: 数据统计和特征分析
- **数据质量 / Data Quality**: 缺失值、异常值分析
- **泄漏审计 / Leakage Audit**: 数据泄漏检测结果
- **特征工程 / Feature Engineering**: 特征工程过程和结果
- **模型性能 / Model Performance**: 详细的性能指标
- **特征重要性 / Feature Importance**: 特征重要性分析
- **校准分析 / Calibration Analysis**: 模型校准曲线
- **稳定性分析 / Stability Analysis**: 模型稳定性评估

## 示例 / Examples

### 二进制分类 / Binary Classification

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/binary_demo
```

### 时间序列预测 / Time Series Forecasting

```bash
tabular-agent run \
  --train examples/train_timeseries.csv \
  --test examples/test_timeseries.csv \
  --target target \
  --time-col date \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/timeseries_demo
```

## 技术架构 / Technical Architecture

### 核心模块 / Core Modules

- **io.py**: 数据加载和保存
- **profile.py**: 数据画像和分析
- **audit.py**: 泄漏审计
- **fe/**: 特征工程模块
- **models/**: 模型训练和评估
- **tune/**: 超参数调优
- **blend/**: 模型融合
- **evaluate/**: 模型评估
- **report/**: 报告生成

### 依赖项 / Dependencies

- Python >= 3.8
- NumPy, Pandas, Scikit-learn
- LightGBM, XGBoost, CatBoost
- Optuna (超参数优化)
- Jinja2 (模板引擎)
- Matplotlib, Seaborn, Plotly (可视化)

## 许可证 / License

MIT License

## 贡献 / Contributing

欢迎提交Issue和Pull Request！

Welcome to submit issues and pull requests!

## 更新日志 / Changelog

### v0.1.0 (2024-01-01)
- 初始版本发布
- 实现基本的ML管道功能
- 支持多种模型和评估指标
- 生成专业的模型卡报告
