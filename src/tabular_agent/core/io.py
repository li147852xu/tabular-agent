"""Data I/O and type inference utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings


def read_csv(
    file_path: Union[str, Path],
    target_col: str,
    time_col: Optional[str] = None,
    sparse_threshold: float = 0.5,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read CSV file with automatic type inference and sparse column handling.
    
    Args:
        file_path: Path to CSV file
        target_col: Name of target column
        time_col: Name of time column (optional)
        sparse_threshold: Threshold for considering a column sparse
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Tuple of (dataframe, metadata)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read CSV with basic settings
    df = pd.read_csv(file_path, **kwargs)
    
    # Basic validation
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    if time_col and time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    
    # Infer data types
    df, type_info = _infer_types(df, target_col, time_col)
    
    # Handle sparse columns
    df, sparse_info = _handle_sparse_columns(df, sparse_threshold)
    
    # Basic data quality checks
    quality_info = _check_data_quality(df, target_col, time_col)
    
    metadata = {
        "file_path": str(file_path),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "type_info": type_info,
        "sparse_info": sparse_info,
        "quality_info": quality_info,
    }
    
    return df, metadata


def _infer_types(
    df: pd.DataFrame, 
    target_col: str, 
    time_col: Optional[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Infer and convert data types."""
    type_info = {}
    
    for col in df.columns:
        if col == target_col:
            # Target column type inference
            if df[col].dtype == 'object':
                # Check if it's categorical or numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                    type_info[col] = "numeric"
                except (ValueError, TypeError):
                    # Treat as categorical
                    df[col] = df[col].astype('category')
                    type_info[col] = "categorical"
            else:
                type_info[col] = "numeric"
                
        elif col == time_col:
            # Time column conversion
            try:
                df[col] = pd.to_datetime(df[col])
                type_info[col] = "datetime"
            except (ValueError, TypeError):
                warnings.warn(f"Could not convert time column '{col}' to datetime")
                type_info[col] = "object"
                
        else:
            # Regular column type inference
            if df[col].dtype == 'object':
                # Try to convert to numeric first
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                    type_info[col] = "numeric"
                except (ValueError, TypeError):
                    # Check if it's actually categorical
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1:  # Less than 10% unique values
                        df[col] = df[col].astype('category')
                        type_info[col] = "categorical"
                    else:
                        type_info[col] = "text"
            else:
                type_info[col] = "numeric"
    
    return df, type_info


def _handle_sparse_columns(
    df: pd.DataFrame, 
    sparse_threshold: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle sparse columns by converting to sparse format or dropping."""
    sparse_info = {}
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            continue
            
        # Calculate sparsity
        null_ratio = df[col].isnull().sum() / len(df)
        zero_ratio = (df[col] == 0).sum() / len(df)
        sparsity = max(null_ratio, zero_ratio)
        
        sparse_info[col] = {
            "sparsity": sparsity,
            "null_ratio": null_ratio,
            "zero_ratio": zero_ratio,
        }
        
        if sparsity > sparse_threshold:
            # Skip sparse conversion for now due to compatibility issues
            # TODO: Implement proper sparse array handling
            sparse_info[col]["converted_to_sparse"] = False
        else:
            sparse_info[col]["converted_to_sparse"] = False
    
    return df, sparse_info


def _check_data_quality(
    df: pd.DataFrame, 
    target_col: str, 
    time_col: Optional[str]
) -> Dict[str, Any]:
    """Perform basic data quality checks."""
    quality_info = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }
    
    # Target column specific checks
    if target_col in df.columns:
        target_info = {
            "unique_values": df[target_col].nunique(),
            "value_counts": df[target_col].value_counts().to_dict(),
        }
        
        # Check if binary classification
        if df[target_col].nunique() == 2:
            target_info["is_binary"] = True
            target_info["class_balance"] = df[target_col].value_counts(normalize=True).to_dict()
        else:
            target_info["is_binary"] = False
            
        quality_info["target"] = target_info
    
    # Time column specific checks
    if time_col and time_col in df.columns:
        time_info = {
            "min_date": df[time_col].min(),
            "max_date": df[time_col].max(),
            "date_range_days": (df[time_col].max() - df[time_col].min()).days,
            "missing_dates": df[time_col].isnull().sum(),
        }
        quality_info["time"] = time_info
    
    return quality_info


def save_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    format: str = "csv",
    **kwargs
) -> None:
    """Save dataframe to file in specified format."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "csv":
        df.to_csv(file_path, index=False, **kwargs)
    elif format.lower() == "parquet":
        df.to_parquet(file_path, index=False, **kwargs)
    elif format.lower() == "pickle":
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_data(
    file_path: Union[str, Path],
    format: str = "csv",
    **kwargs
) -> pd.DataFrame:
    """Load dataframe from file in specified format."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format.lower() == "csv":
        return pd.read_csv(file_path, **kwargs)
    elif format.lower() == "parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif format.lower() == "pickle":
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
