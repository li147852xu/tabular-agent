# Tabular Agent v0.2 - 智能规划与RAG引证

> 基于LLM+规则的混合规划器，支持RAG引证历史运行，提供安全护栏和规则回退

## 功能特性

### 新增功能
- **智能规划器**：LLM+规则混合，Pydantic严格校验
- **RAG引证系统**：索引历史运行元数据，提供参考先例
- **安全护栏**：LLM输出白名单验证，自动回退到规则
- **知识库**：向量+BM25混合检索
- **规划报告**：展示规划过程、回退原因、引用先例

### 核心模块
- **Planner**：智能规划器，支持LLM和规则两种模式
- **Knowledge Base**：历史运行元数据索引和检索
- **Citation System**：引用先例生成和展示
- **Schema Validation**：严格的Pydantic数据校验

## 安装

```bash
pip install -e .
```

## 使用方法

### 基本用法（自动模式）

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --planner auto \
  --out runs/v02_demo
```

### LLM模式（需要API密钥）

```bash
export LLM_ENDPOINT="https://api.openai.com/v1"
export LLM_KEY="your-api-key"

tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --planner llm \
  --llm-endpoint $LLM_ENDPOINT \
  --llm-key $LLM_KEY \
  --out runs/v02_llm
```

### 规则模式

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --planner rules \
  --out runs/v02_rules
```

## 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--planner` | str | auto | 规划器模式：llm/rules/auto |
| `--llm-endpoint` | str | None | LLM API端点 |
| `--llm-key` | str | None | LLM API密钥 |

## 规划器模式

### 自动模式（推荐）
- 优先尝试LLM规划
- LLM失败时自动回退到规则
- 无需配置API密钥

### LLM模式
- 使用LLM进行智能规划
- 需要提供API密钥
- 支持OpenAI兼容接口

### 规则模式
- 使用预定义规则进行规划
- 无需外部依赖
- 稳定可靠

## RAG引证系统

### 知识库索引
- 自动索引`runs/`目录下的历史运行
- 提取数据集名、列、参数、分数、资源开销
- 支持向量和BM25混合检索

### 引用先例
- 规划器决策时引用相似历史运行
- 展示run_id、分数、配置信息
- 在模型卡中显示引用块

## 安全护栏

### 白名单验证
- LLM只能选择预定义的动作
- 特征工程、模型、调参空间都有白名单
- 超出白名单的输出会被拒绝

### 自动回退
- LLM输出验证失败时自动回退
- 记录回退原因
- 确保系统稳定性

## 模型卡增强

### 规划与引用章节
- 显示使用的规划模式
- 展示回退原因（如有）
- 列出引用的历史先例
- 显示规划详情

## 配置示例

### 环境变量
```bash
export LLM_ENDPOINT="https://api.openai.com/v1"
export LLM_KEY="sk-..."
```

### 规划器配置
```python
# 在代码中配置
planner_config = PlanningConfig(
    mode=PlannerMode.AUTO,
    llm_endpoint="https://api.openai.com/v1",
    llm_key="sk-..."
)
```

## 输出结构

```
runs/experiment_002/
├── meta.json          # 元数据（包含规划信息）
├── results.json       # 详细结果
├── model_card.html    # 模型卡报告（含规划章节）
└── kb/               # 知识库文件
    ├── vector_index.pkl
    └── bm25_index.pkl
```

## 技术架构

### Planner模块
- `PlannerMode`：规划模式枚举
- `PlanningConfig`：配置类
- `PlanningResult`：结果类
- `Citation`：引用类

### Knowledge Base
- 向量索引：使用scikit-learn的TfidfVectorizer
- BM25索引：使用rank_bm25
- 元数据存储：JSON格式

### 安全机制
- Pydantic严格校验
- 白名单验证
- 异常处理和回退

## 性能优化

- 知识库索引缓存
- 并行检索
- 增量更新
- 内存优化

## 版本信息

- **版本**：v0.2.0
- **Python**：3.9+
- **新增依赖**：pydantic>=1.10.0

## 已知问题

1. LLM API调用可能失败，会自动回退
2. 知识库索引大文件时可能较慢
3. 某些LLM输出格式可能不标准

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License