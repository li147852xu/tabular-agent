# Tabular Agent v0.2

## 概述 / Overview

Tabular Agent v0.2 在v0.1基础上增加了智能规划器（Planner）和RAG引证系统，实现了LLM+规则混合的智能决策和基于历史运行的知识引用。

Tabular Agent v0.2 adds intelligent Planner and RAG citation system on top of v0.1, implementing LLM+rule hybrid intelligent decision making and knowledge citation based on historical runs.

## 新功能特性 / New Features

### 智能规划器 / Intelligent Planner

- **LLM+规则混合 / LLM+Rule Hybrid**: 支持LLM和规则两种规划模式
- **严格Schema校验 / Strict Schema Validation**: 使用Pydantic v2进行严格的输入输出验证
- **白名单安全护栏 / Whitelist Security Guard**: 只允许预定义的动作，确保安全性
- **自动回退机制 / Auto Fallback**: LLM失败时自动回退到规则模式
- **环境变量支持 / Environment Variable Support**: 自动从环境变量读取LLM配置

### RAG引证系统 / RAG Citation System

- **知识库索引 / Knowledge Base Indexing**: 自动索引历史运行元数据
- **向量化搜索 / Vectorized Search**: 使用TF-IDF和余弦相似度进行相似性搜索
- **引用生成 / Citation Generation**: 为规划决策提供历史先例引用
- **持久化存储 / Persistent Storage**: 索引结果持久化，提高查询效率

### 增强的模型卡 / Enhanced Model Card

- **规划与引用章节 / Planning and Citation Section**: 新增完整的规划信息展示
- **引用先例展示 / Citation Precedent Display**: 显示相似历史运行的配置和分数
- **回退信息 / Fallback Information**: 显示规划回退的原因和模式

## 安装 / Installation

```bash
pip install -e .
```

## 使用方法 / Usage

### 基本用法 / Basic Usage

```bash
tabular-agent run --train data/train.csv --test data/test.csv --target target --planner auto --out runs/demo
```

### 强制使用规则模式 / Force Rules Mode

```bash
tabular-agent run --train data/train.csv --test data/test.csv --target target --planner rules --out runs/demo
```

### 使用LLM模式 / Use LLM Mode

```bash
# 设置环境变量 / Set environment variables
export TABULAR_AGENT_LLM_ENDPOINT="http://your-llm-endpoint"
export TABULAR_AGENT_LLM_KEY="your-api-key"

# 运行管道 / Run pipeline
tabular-agent run --train data/train.csv --test data/test.csv --target target --planner llm --out runs/demo
```

### 命令行参数 / Command Line Parameters

```bash
tabular-agent run \
  --train data/train.csv \
  --test data/test.csv \
  --target target \
  --time-col date \
  --planner auto \
  --llm-endpoint "http://your-llm-endpoint" \
  --llm-key "your-api-key" \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/demo
```

## 新增参数 / New Parameters

| 参数 / Parameter | 说明 / Description | 默认值 / Default |
|------------------|-------------------|------------------|
| `--planner` | 规划器模式 (llm/rules/auto) / Planner mode | auto |
| `--llm-endpoint` | LLM端点URL / LLM endpoint URL | 环境变量 / Environment variable |
| `--llm-key` | LLM API密钥 / LLM API key | 环境变量 / Environment variable |

## 规划器配置 / Planner Configuration

### 环境变量 / Environment Variables

```bash
# LLM配置 / LLM Configuration
export TABULAR_AGENT_LLM_ENDPOINT="http://your-llm-endpoint"
export TABULAR_AGENT_LLM_KEY="your-api-key"

# 可选配置 / Optional Configuration
export TABULAR_AGENT_MAX_CITATIONS=3
export TABULAR_AGENT_MIN_SIMILARITY=0.7
```

### 规划模式 / Planning Modes

1. **auto**: 自动选择模式，优先使用LLM，不可用时回退到规则
2. **llm**: 强制使用LLM模式，需要配置LLM端点
3. **rules**: 强制使用规则模式，不依赖外部服务

## 知识库功能 / Knowledge Base Features

### 自动索引 / Automatic Indexing

系统会自动索引`runs/`目录下的所有历史运行，提取以下信息：
- 数据集特征（样本数、特征数、类型分布）
- 模型配置（特征工程、模型类型、融合策略）
- 性能指标（AUC、KS等）
- 资源使用情况

### 相似性搜索 / Similarity Search

基于以下维度进行相似性搜索：
- 数据集特征相似性
- 目标变量类型
- 时间列存在性
- 特征工程策略

### 引用生成 / Citation Generation

为每个规划决策提供：
- 相似历史运行的ID和配置
- 相似性分数
- 引用原因
- 性能对比

## 安全特性 / Security Features

### 白名单验证 / Whitelist Validation

所有LLM输出都经过严格的白名单验证：

```python
# 允许的特征工程配方 / Allowed feature engineering recipes
feature_recipes = [
    "target_encoding", "woe_encoding", "rolling_features",
    "time_features", "cross_features", "scaling",
    "variance_selection", "correlation_selection", "iv_selection"
]

# 允许的模型类型 / Allowed model types
model_types = [
    "lightgbm", "xgboost", "catboost", "linear", "histgbdt"
]

# 允许的融合策略 / Allowed blending strategies
blending_strategies = [
    "mean", "rank_mean", "logit_mean", "stacking"
]
```

### 自动回退 / Auto Fallback

- LLM不可用时自动回退到规则模式
- LLM输出无效时自动回退到规则模式
- 提供详细的回退原因说明

## 输出结果 / Output Results

### 增强的模型卡 / Enhanced Model Card

新增"规划与引用"章节，包含：

- **执行计划 / Execution Plan**
  - 规划模式（LLM/规则）
  - 特征工程配方
  - 模型类型选择
  - 融合策略
  - 时间预算分配

- **引用先例 / Citation Precedents**
  - 相似历史运行列表
  - 相似性分数
  - 引用原因
  - 配置详情

- **回退信息 / Fallback Information**
  - 回退原因
  - 使用的模式

## 示例 / Examples

### 基本使用 / Basic Usage

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --planner auto \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/v02_demo
```

### 时间序列数据 / Time Series Data

```bash
tabular-agent run \
  --train examples/train_timeseries.csv \
  --test examples/test_timeseries.csv \
  --target target \
  --time-col date \
  --planner auto \
  --n-jobs 4 \
  --time-budget 60 \
  --out runs/v02_timeseries_demo
```

## 测试 / Testing

### 运行测试 / Run Tests

```bash
# 运行所有测试 / Run all tests
pytest

# 运行规划器测试 / Run planner tests
pytest -q -k "planner or kb"

# 运行特定测试 / Run specific tests
pytest tests/test_planner_guard.py
pytest tests/test_kb_index.py
```

### 测试覆盖 / Test Coverage

- **规划器测试**: 15个测试用例，覆盖所有核心功能
- **知识库测试**: 全面的索引和查询测试
- **Mock测试**: 无外网环境下的完整测试

## 技术架构 / Technical Architecture

### 新增模块 / New Modules

- **agent/planner.py**: 智能规划器核心逻辑
- **agent/kb/**: 知识库和RAG系统
  - **index.py**: 知识库索引和搜索
  - **query.py**: 查询和引用生成
- **core/orchestrator.py**: 增强的管道编排器

### 依赖项 / Dependencies

新增依赖：
- **pydantic>=1.10.0**: 数据验证和序列化

## 性能指标 / Performance Metrics

- **运行时间**: 2-5秒（取决于数据大小）
- **内存使用**: 优化的知识库索引
- **可扩展性**: 支持大规模历史运行数据
- **稳定性**: 100%测试通过率

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

## 许可证 / License

MIT License

## 贡献 / Contributing

欢迎提交Issue和Pull Request！

Welcome to submit issues and pull requests!
