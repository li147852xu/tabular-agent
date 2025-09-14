# Tabular Agent v0.3 - 风险分析与稳定性评估

> 可靠的模型风险分析，稳定性评估，校准优化，提供专业的模型卡报告

## 功能特性

### 新增功能
- **Reflector模块**：后训练风险分析和重试建议
- **稳定性评估**：多次重复OOF评估，方差分析
- **风险分级**：High/Med/Low风险等级
- **校准优化**：Isotonic/Platt校准方法
- **增强模型卡**：风险矩阵、稳定性仪表板、阈值建议

### 核心模块
- **Reflector**：风险分析引擎
- **StabilityEvaluator**：稳定性评估器
- **Risk Analysis**：过拟合、泄漏、不稳定性、校准分析
- **Retry Suggestions**：可执行的重试建议

## 安装

```bash
pip install -e .
```

## 使用方法

### 基本用法（带稳定性评估）

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --stability-runs 5 \
  --calibration isotonic \
  --out runs/v03_demo
```

### 高级配置（带风险策略）

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --stability-runs 10 \
  --calibration platt \
  --risk-policy conf/risk_policy.yaml \
  --n-jobs 8 \
  --time-budget 180 \
  --out runs/v03_advanced
```

### 快速测试

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --stability-runs 3 \
  --calibration none \
  --out runs/v03_quick
```

## 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stability-runs` | int | 5 | 稳定性评估运行次数 |
| `--calibration` | str | none | 校准方法：isotonic/platt/none |
| `--risk-policy` | str | None | 风险策略配置文件路径 |

## 风险分析

### 过拟合检测
- 训练集vs OOF分数差异
- 时间窗口漂移分析
- 风险等级：High/Med/Low

### 泄漏检测
- 特征泄漏回溯分析
- 目标泄漏检测
- 时间泄漏检查

### 不稳定性检测
- 扰动敏感性分析
- 种子波动检测
- 特征重要性稳定性

### 校准分析
- 概率校准质量
- 校准曲线分析
- 阈值建议

## 稳定性评估

### 多次运行
- 不同随机种子
- 数据扰动
- 并行/串行执行

### 指标计算
- AUC方差和标准差
- 置信区间
- 稳定性等级（A-F）

### 特征稳定性
- 特征重要性波动
- 稳定性排序
- 关键特征识别

## 校准优化

### Isotonic校准
- 单调性保持
- 适合大多数情况
- 计算效率高

### Platt校准
- 逻辑回归校准
- 适合小数据集
- 概率分布优化

### 无校准
- 保持原始概率
- 适合已校准模型
- 减少计算开销

## 风险策略配置

### 配置文件示例
```yaml
# conf/risk_policy.yaml
overfitting:
  train_oof_diff_threshold: 0.05
  time_drift_threshold: 0.1

leakage:
  feature_correlation_threshold: 0.9
  target_leakage_threshold: 0.95

instability:
  auc_cv_threshold: 0.1
  feature_importance_cv_threshold: 0.2

calibration:
  brier_score_threshold: 0.25
  reliability_diagram_threshold: 0.1

retry_suggestions:
  overfitting: "考虑增加正则化或减少模型复杂度"
  leakage: "检查特征工程，移除可疑特征"
  instability: "增加数据量或使用集成方法"
  calibration: "调整概率阈值或使用校准方法"
```

## 模型卡增强

### 风险分析章节
- 风险摘要和等级
- 详细风险指标
- 重试建议

### 稳定性仪表板
- 稳定性指标可视化
- 多次运行结果对比
- 特征稳定性分析

### 阈值建议
- 不同成本权重下的阈值
- Youden's J统计量
- KS最大化阈值

## 输出结构

```
runs/experiment_003/
├── meta.json          # 元数据（含风险分析）
├── results.json       # 详细结果（含稳定性数据）
├── model_card.html    # 模型卡报告（含风险章节）
└── risk_analysis.json # 风险分析详情
```

## 技术架构

### Reflector模块
- `RiskLevel`：风险等级枚举
- `RiskType`：风险类型枚举
- `RiskIndicator`：风险指标类
- `RetrySuggestion`：重试建议类

### StabilityEvaluator
- 多次运行管理
- 并行/串行执行
- 指标计算和汇总
- 特征稳定性分析

### 风险分析引擎
- 规则基础分析
- 阈值比较
- 建议生成
- 配置驱动

## 性能优化

- 并行稳定性评估
- 内存使用优化
- 缓存机制
- 增量计算

## 版本信息

- **版本**：v0.3.0
- **Python**：3.9+
- **新增依赖**：pydantic>=1.10.0

## 已知问题

1. 稳定性评估可能耗时较长
2. 某些风险阈值需要调优
3. 校准方法选择需要经验

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License
