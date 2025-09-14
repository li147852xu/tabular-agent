# Tabular Agent

## 概述 / Overview

Tabular Agent 是一个端到端的表格机器学习管道，从CSV数据到模型卡报告，支持智能规划、RAG引证和风险分析。

Tabular Agent is an end-to-end tabular machine learning pipeline from CSV data to model card reports, supporting intelligent planning, RAG citations, and risk analysis.

## 版本历史 / Version History

### v0.2.0 (Latest) - 智能规划与RAG引证
- 智能规划器（LLM+规则混合）
- RAG引证系统
- 增强的模型卡报告
- 安全护栏机制

### v0.1.0 - 基础ML管道
- 数据画像和泄漏审计
- 特征工程和模型训练
- 超参数调优和模型融合
- 基础模型卡报告

## 快速开始 / Quick Start

### 安装 / Installation

```bash
pip install -e .
```

### 基本使用 / Basic Usage

```bash
# 自动模式（推荐）/ Auto mode (recommended)
tabular-agent run --train data/train.csv --test data/test.csv --target target --out runs/demo

# 时间序列数据 / Time series data
tabular-agent run --train data/train.csv --test data/test.csv --target target --time-col date --out runs/demo

# 高级配置 / Advanced configuration
tabular-agent run \
  --train data/train.csv \
  --test data/test.csv \
  --target target \
  --time-col date \
  --planner auto \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/demo
```

## 功能特性 / Features

### 核心功能 / Core Features

- **数据画像 / Data Profiling**: 自动分析数据质量和特征分布
- **泄漏审计 / Leakage Audit**: 检测数据泄漏和重复
- **特征工程 / Feature Engineering**: 时间感知的特征工程
- **模型训练 / Model Training**: 支持多种主流模型
- **超参数调优 / Hyperparameter Tuning**: 使用Optuna进行优化
- **模型融合 / Model Blending**: 多种融合策略
- **模型评估 / Model Evaluation**: 全面的评估指标
- **报告生成 / Report Generation**: 专业的HTML模型卡

### v0.2新增功能 / v0.2 New Features

- **智能规划器 / Intelligent Planner**: LLM+规则混合决策
- **RAG引证系统 / RAG Citation System**: 基于历史运行的知识引用
- **安全护栏 / Security Guard**: 严格的白名单验证
- **环境变量支持 / Environment Variable Support**: 灵活的配置管理

## 详细文档 / Detailed Documentation

- [v0.1 README](README_v0.1.md) - 基础功能文档
- [v0.2 README](README_v0.2.md) - 智能规划功能文档

## 示例 / Examples

### 二进制分类 / Binary Classification

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --planner auto \
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
  --planner auto \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/timeseries_demo
```

## 测试 / Testing

```bash
# 运行所有测试 / Run all tests
pytest

# 运行特定模块测试 / Run specific module tests
pytest -q -k "planner or kb"
pytest -q -k "reflector or stability"
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

### v0.2新增模块 / v0.2 New Modules

- **agent/planner.py**: 智能规划器
- **agent/kb/**: 知识库和RAG系统
- **core/orchestrator.py**: 增强的管道编排器

## 依赖项 / Dependencies

- Python >= 3.8
- NumPy, Pandas, Scikit-learn
- LightGBM, XGBoost, CatBoost
- Optuna (超参数优化)
- Jinja2 (模板引擎)
- Matplotlib, Seaborn, Plotly (可视化)
- Pydantic (数据验证)

## 许可证 / License

MIT License

## 贡献 / Contributing

欢迎提交Issue和Pull Request！

Welcome to submit issues and pull requests!

## 更新日志 / Changelog

### v0.2.0 (2024-01-14)
- 新增智能规划器（Planner）模块
- 实现RAG引证系统
- 增强模型卡报告
- 添加安全护栏机制
- 支持环境变量配置
- 完善测试覆盖

### v0.1.0 (2024-01-01)
- 初始版本发布
- 实现基本的ML管道功能
- 支持多种模型和评估指标
- 生成专业的模型卡报告