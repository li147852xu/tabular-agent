# Tabular Agent

Automated tabular ML pipeline from CSV to model card reports.

## Features

- **Data Profiling**: Comprehensive data analysis including missing values, outliers, cardinality, and data quality scoring
- **Leakage Audit**: Integration with leakage-buster for data leakage detection
- **Time-Aware Feature Engineering**: Advanced feature engineering with time-aware cross-validation
- **Model Training**: Support for LightGBM, XGBoost, CatBoost, Linear models, and HistGBDT
- **Hyperparameter Tuning**: Optuna-based parallel hyperparameter optimization
- **Model Blending**: Multiple blending strategies including mean, rank-based, and logit-space blending
- **Comprehensive Evaluation**: AUC, KS, calibration analysis, threshold optimization, and stability analysis
- **Model Cards**: Beautiful HTML model cards with interactive visualizations

## Installation

```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --n-jobs 8 \
  --time-budget 120 \
  --out runs/demo
```

### Time-Aware Features

```bash
tabular-agent run \
  --train examples/train_timeseries.csv \
  --test examples/test_timeseries.csv \
  --target target \
  --time-col date \
  --n-jobs 8 \
  --time-budget 120 \
  --out runs/timeseries_demo
```

### With Leakage Audit

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --audit-cli /path/to/leakage-buster \
  --n-jobs 8 \
  --time-budget 120 \
  --out runs/audited_demo
```

## Configuration

The pipeline can be configured using YAML files. See `conf/defaults.yaml` for available options.

### Example Configuration

```yaml
# conf/custom.yaml
target: "target"
time_col: "date"
n_jobs: 8
time_budget: 300
model_names: ["lightgbm", "xgboost", "catboost"]
cv_folds: 5

feature_config:
  encoders: ["target", "woe"]
  rolling_features: true
  window_sizes: [7, 14, 30]
  feature_selection: true
  min_variance: 0.01
  max_correlation: 0.95
  min_iv: 0.02
```

## Pipeline Components

### 1. Data Profiling
- Missing value analysis
- Outlier detection
- Cardinality analysis
- Target analysis
- Time series analysis
- Data quality scoring

### 2. Leakage Audit
- Duplicate row detection
- Target leakage detection
- Time leakage detection
- Correlation analysis

### 3. Feature Engineering
- Target encoding with time-aware CV
- Weight of Evidence (WOE) encoding
- Rolling window features
- Time-based features
- Feature selection
- Scaling and normalization

### 4. Model Training
- LightGBM
- XGBoost
- CatBoost
- Linear models
- Histogram-based Gradient Boosting

### 5. Hyperparameter Tuning
- Optuna-based optimization
- Parallel tuning
- Time budget allocation
- Multi-model tuning

### 6. Model Blending
- Mean blending
- Rank-based blending
- Logit-space blending
- Stacking with meta-learners

### 7. Model Evaluation
- Classification metrics (AUC, Precision, Recall, F1)
- Regression metrics (RMSE, MAE, R²)
- Calibration analysis
- Threshold optimization
- Stability analysis

### 8. Model Card Generation
- Interactive HTML reports
- Performance visualizations
- Data quality insights
- Leakage audit results
- Feature importance plots
- Calibration diagrams

## Output Structure

```
runs/
└── 20240101_120000/
    ├── model_card.html          # Main model card
    ├── results.json            # Complete results
    └── meta.json              # Metadata and configuration
```

## Development

### Running Tests

```bash
pytest -q
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Examples

### Generate Example Data

```bash
python src/tabular_agent/examples/generate_example_data.py
```

### Run Pipeline

```bash
tabular-agent run \
  --train examples/train_binary.csv \
  --test examples/test_binary.csv \
  --target target \
  --n-jobs 8 \
  --time-budget 120 \
  --out runs/demo
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- Jinja2
- Matplotlib
- Seaborn
- Plotly

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Changelog

### v0.1.0
- Initial release
- Complete pipeline from CSV to model card
- Support for multiple models and blending strategies
- Comprehensive evaluation and reporting
