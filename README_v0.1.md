# Tabular Agent v0.1 - 基础管道

> 从CSV数据到模型卡报告的端到端管道，无需LLM/Agent功能

## 功能特性

### 核心模块
- **数据加载与画像**：自动类型推断、缺失值检测、异常值分析
- **泄漏审计**：集成`leakage-buster`进行数据泄漏检测
- **特征工程**：时间感知编码、滚动特征、特征选择
- **模型训练**：LightGBM、XGBoost、CatBoost、线性模型
- **超参调优**：Optuna并行优化
- **模型融合**：均值、排序均值、logit均值
- **模型评估**：AUC、KS、PR-AUC、R²、准确性等指标
- **报告生成**：HTML模型卡

### 技术特点
- 支持二分类和回归任务
- 时间序列数据处理
- 可配置的交叉验证策略
- 完整的元数据跟踪
- 可重现的实验结果

## 安装

```bash
pip install -e .
```

## 使用方法

### 基本用法

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --out runs/v01_demo
```

### 时间序列数据

```bash
tabular-agent run \
  --train examples/train_timeseries.csv \
  --test examples/test_timeseries.csv \
  --target target \
  --time-col date \
  --out runs/v01_timeseries
```

### 高级配置

```bash
tabular-agent run \
  --train data/train.csv \
  --test data/test.csv \
  --target target \
  --time-col timestamp \
  --n-jobs 4 \
  --time-budget 300 \
  --cv-folds 5 \
  --out runs/experiment_001
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train` | str | 必需 | 训练数据路径 |
| `--test` | str | 必需 | 测试数据路径 |
| `--target` | str | 必需 | 目标列名 |
| `--time-col` | str | None | 时间列名（可选） |
| `--n-jobs` | int | 1 | 并行作业数 |
| `--time-budget` | int | 300 | 时间预算（秒） |
| `--cv-folds` | int | 5 | 交叉验证折数 |
| `--seed` | int | 42 | 随机种子 |
| `--out` | str | 必需 | 输出目录 |
| `--verbose` | flag | False | 详细输出 |

## 输出结构

```
runs/experiment_001/
├── meta.json          # 元数据
├── results.json       # 详细结果
└── model_card.html    # 模型卡报告
```

## 模型卡内容

- **数据概览**：数据集统计、特征分布
- **模型性能**：各模型指标对比
- **特征重要性**：特征重要性排序
- **模型融合**：融合策略和效果
- **交叉验证**：CV分数分布
- **时间分析**：时间序列特征（如有）

## 示例数据

项目包含示例数据集：
- `examples/train_binary.csv` - 二分类示例
- `examples/test_binary.csv` - 二分类测试
- `examples/train_timeseries.csv` - 时间序列示例
- `examples/test_timeseries.csv` - 时间序列测试

## 技术栈

- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn, lightgbm, xgboost, catboost
- **超参优化**：optuna
- **可视化**：matplotlib, seaborn, plotly
- **报告生成**：jinja2

## 版本信息

- **版本**：v0.1.0
- **Python**：3.9+
- **依赖**：见`pyproject.toml`

## 已知问题

1. 大数据集内存使用可能较高
2. 某些模型在特定数据上可能收敛较慢
3. 时间序列特征工程需要更多优化

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License