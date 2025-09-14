"""Generate example datasets for testing tabular-agent."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings


def generate_binary_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate binary classification dataset."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for i, feat_idx in enumerate(informative_features):
        X[:, feat_idx] += np.random.randn(n_samples) * 0.5
    
    # Create target with some noise
    target = np.zeros(n_samples)
    for feat_idx in informative_features:
        target += X[:, feat_idx] * np.random.randn()
    
    # Add noise
    target += np.random.randn(n_samples) * noise
    
    # Convert to binary
    target = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    return df


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate regression dataset."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for i, feat_idx in enumerate(informative_features):
        X[:, feat_idx] += np.random.randn(n_samples) * 0.5
    
    # Create target
    target = np.zeros(n_samples)
    for feat_idx in informative_features:
        target += X[:, feat_idx] * np.random.randn()
    
    # Add noise
    target += np.random.randn(n_samples) * noise
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    return df


def generate_time_series_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate time series dataset with temporal features."""
    np.random.seed(random_state)
    
    # Generate time index
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Add temporal patterns
    for i in range(n_features):
        # Add trend
        X[:, i] += np.linspace(0, 1, n_samples) * np.random.randn()
        
        # Add seasonality
        X[:, i] += np.sin(2 * np.pi * np.arange(n_samples) / 365) * np.random.randn()
    
    # Create informative features
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for i, feat_idx in enumerate(informative_features):
        X[:, feat_idx] += np.random.randn(n_samples) * 0.5
    
    # Create target with temporal dependency
    target = np.zeros(n_samples)
    for feat_idx in informative_features:
        target += X[:, feat_idx] * np.random.randn()
    
    # Add temporal trend to target
    target += np.linspace(0, 0.5, n_samples)
    
    # Add noise
    target += np.random.randn(n_samples) * noise
    
    # Convert to binary for classification
    target = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    df['date'] = dates
    
    return df


def generate_categorical_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_categorical: int = 5,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate dataset with categorical features."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for i, feat_idx in enumerate(informative_features):
        X[:, feat_idx] += np.random.randn(n_samples) * 0.5
    
    # Convert some features to categorical
    categorical_features = np.random.choice(n_features, n_categorical, replace=False)
    for feat_idx in categorical_features:
        n_categories = np.random.randint(3, 8)
        X[:, feat_idx] = np.random.randint(0, n_categories, n_samples)
    
    # Create target
    target = np.zeros(n_samples)
    for feat_idx in informative_features:
        if feat_idx in categorical_features:
            # For categorical features, use different weights per category
            categories = np.unique(X[:, feat_idx])
            for cat in categories:
                mask = X[:, feat_idx] == cat
                target[mask] += np.random.randn() * mask.sum()
        else:
            target += X[:, feat_idx] * np.random.randn()
    
    # Add noise
    target += np.random.randn(n_samples) * noise
    
    # Convert to binary
    target = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    # Convert categorical features to string type
    for feat_idx in categorical_features:
        df[f'feature_{feat_idx}'] = df[f'feature_{feat_idx}'].astype(str)
    
    return df


def generate_missing_data(
    df: pd.DataFrame,
    missing_ratio: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Add missing values to dataset."""
    np.random.seed(random_state)
    
    df_missing = df.copy()
    
    # Add missing values to random positions
    n_missing = int(len(df) * missing_ratio)
    missing_indices = np.random.choice(len(df), n_missing, replace=False)
    missing_cols = np.random.choice(df.columns, n_missing, replace=True)
    
    for idx, col in zip(missing_indices, missing_cols):
        df_missing.loc[idx, col] = np.nan
    
    return df_missing


def main():
    """Generate example datasets."""
    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating example datasets...")
    
    # Binary classification
    print("Generating binary classification dataset...")
    df_binary = generate_binary_classification_data(n_samples=1000, n_features=20)
    df_binary.to_csv(output_dir / "binary_classification.csv", index=False)
    
    # Regression
    print("Generating regression dataset...")
    df_regression = generate_regression_data(n_samples=1000, n_features=20)
    df_regression.to_csv(output_dir / "regression.csv", index=False)
    
    # Time series
    print("Generating time series dataset...")
    df_timeseries = generate_time_series_data(n_samples=1000, n_features=20)
    df_timeseries.to_csv(output_dir / "timeseries.csv", index=False)
    
    # Categorical
    print("Generating categorical dataset...")
    df_categorical = generate_categorical_data(n_samples=1000, n_features=20)
    df_categorical.to_csv(output_dir / "categorical.csv", index=False)
    
    # Missing data
    print("Generating missing data dataset...")
    df_missing = generate_missing_data(df_binary, missing_ratio=0.1)
    df_missing.to_csv(output_dir / "missing_data.csv", index=False)
    
    # Split datasets for train/test
    print("Splitting datasets for train/test...")
    
    # Binary classification train/test
    train_binary = df_binary.iloc[:800]
    test_binary = df_binary.iloc[800:]
    train_binary.to_csv(output_dir / "train_binary.csv", index=False)
    test_binary.to_csv(output_dir / "test_binary.csv", index=False)
    
    # Time series train/test
    train_timeseries = df_timeseries.iloc[:800]
    test_timeseries = df_timeseries.iloc[800:]
    train_timeseries.to_csv(output_dir / "train_timeseries.csv", index=False)
    test_timeseries.to_csv(output_dir / "test_timeseries.csv", index=False)
    
    print("Example datasets generated successfully!")
    print(f"Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
