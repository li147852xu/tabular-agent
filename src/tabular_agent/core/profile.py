"""Data profiling and analysis utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder


class DataProfiler:
    """Comprehensive data profiler for tabular data."""
    
    def __init__(self, target_col: str, time_col: Optional[str] = None):
        """
        Initialize data profiler.
        
        Args:
            target_col: Name of target column
            time_col: Name of time column (optional)
        """
        self.target_col = target_col
        self.time_col = time_col
        self.profile_results = {}
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data profiling.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing profiling results
        """
        self.profile_results = {
            "basic_info": self._basic_info(df),
            "missing_analysis": self._missing_analysis(df),
            "outlier_analysis": self._outlier_analysis(df),
            "cardinality_analysis": self._cardinality_analysis(df),
            "target_analysis": self._target_analysis(df),
            "time_analysis": self._time_analysis(df) if self.time_col else None,
            "correlation_analysis": self._correlation_analysis(df),
            "data_quality_score": self._data_quality_score(df),
        }
        
        return self.profile_results
    
    def _basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic dataset information."""
        return {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['category', 'object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        }
    
    def _missing_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Missing value analysis."""
        missing_stats = df.isnull().sum()
        missing_ratio = missing_stats / len(df)
        
        return {
            "missing_counts": missing_stats.to_dict(),
            "missing_ratios": missing_ratio.to_dict(),
            "columns_with_missing": missing_stats[missing_stats > 0].to_dict(),
            "high_missing_columns": missing_ratio[missing_ratio > 0.5].to_dict(),
            "missing_patterns": self._analyze_missing_patterns(df),
        }
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_matrix = df.isnull()
        
        # Check for completely missing rows
        complete_missing_rows = missing_matrix.all(axis=1).sum()
        
        # Check for columns that are missing together
        missing_correlations = missing_matrix.corr()
        
        return {
            "complete_missing_rows": complete_missing_rows,
            "missing_correlations": missing_correlations.to_dict(),
        }
    
    def _outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Outlier detection and analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if col == self.target_col:
                continue
                
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_outliers = values[z_scores > 3]
            
            outlier_info[col] = {
                "iqr_outliers": len(outliers),
                "iqr_outlier_ratio": len(outliers) / len(values),
                "z_score_outliers": len(z_outliers),
                "z_score_outlier_ratio": len(z_outliers) / len(values),
                "outlier_values": outliers.tolist()[:10],  # First 10 outliers
            }
        
        return outlier_info
    
    def _cardinality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cardinality of categorical columns."""
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns
        cardinality_info = {}
        
        for col in categorical_cols:
            if col == self.target_col:
                continue
                
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Check for high cardinality
            is_high_cardinality = cardinality_ratio > 0.5
            
            # Check for potential ID columns
            is_potential_id = (
                cardinality_ratio > 0.95 and 
                unique_count > 1000 and
                df[col].dtype == 'object'
            )
            
            cardinality_info[col] = {
                "unique_count": unique_count,
                "cardinality_ratio": cardinality_ratio,
                "is_high_cardinality": is_high_cardinality,
                "is_potential_id": is_potential_id,
                "most_common": df[col].value_counts().head(5).to_dict(),
            }
        
        return cardinality_info
    
    def _target_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Target variable analysis."""
        if self.target_col not in df.columns:
            return {}
        
        target_series = df[self.target_col].dropna()
        
        # Basic statistics
        target_info = {
            "count": len(target_series),
            "missing_count": df[self.target_col].isnull().sum(),
            "unique_count": target_series.nunique(),
            "dtype": str(target_series.dtype),
        }
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(target_series):
            try:
                target_info.update({
                    "mean": target_series.mean(),
                    "std": target_series.std(),
                    "min": target_series.min(),
                    "max": target_series.max(),
                    "skewness": target_series.skew(),
                    "kurtosis": target_series.kurtosis(),
                })
            except (TypeError, AttributeError):
                # Handle sparse arrays or other special types
                target_info.update({
                    "mean": float(target_series.mean()),
                    "std": float(target_series.std()),
                    "min": float(target_series.min()),
                    "max": float(target_series.max()),
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                })
            
            # Check if binary
            if target_series.nunique() == 2:
                target_info["is_binary"] = True
                target_info["class_balance"] = target_series.value_counts(normalize=True).to_dict()
                target_info["positive_class_ratio"] = target_series.mean()
            else:
                target_info["is_binary"] = False
        else:
            # Categorical target
            target_info["is_binary"] = False
            target_info["class_distribution"] = target_series.value_counts(normalize=True).to_dict()
        
        return target_info
    
    def _time_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Time column analysis."""
        if not self.time_col or self.time_col not in df.columns:
            return {}
        
        time_series = df[self.time_col].dropna()
        
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            return {"error": "Time column is not datetime type"}
        
        return {
            "count": len(time_series),
            "missing_count": df[self.time_col].isnull().sum(),
            "min_date": time_series.min(),
            "max_date": time_series.max(),
            "date_range_days": (time_series.max() - time_series.min()).days,
            "is_sorted": time_series.is_monotonic_increasing,
            "duplicate_dates": time_series.duplicated().sum(),
            "date_frequency": self._analyze_date_frequency(time_series),
        }
    
    def _analyze_date_frequency(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze date frequency patterns."""
        try:
            # Try to infer frequency
            freq = pd.infer_freq(time_series)
            
            # Check for gaps
            time_diffs = time_series.diff().dropna()
            if len(time_diffs) > 0:
                most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
            else:
                most_common_diff = None
            
            return {
                "inferred_frequency": freq,
                "most_common_interval": str(most_common_diff) if most_common_diff else None,
                "interval_consistency": len(time_diffs.unique()) == 1 if len(time_diffs) > 0 else False,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Correlation analysis between features and target."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.target_col not in numeric_cols:
            return {"error": "Target column is not numeric"}
        
        correlations = {}
        target_correlations = {}
        
        for col in numeric_cols:
            if col == self.target_col:
                continue
            
            # Calculate correlation with target
            corr_with_target = df[col].corr(df[self.target_col])
            if not np.isnan(corr_with_target):
                target_correlations[col] = corr_with_target
        
        # Sort by absolute correlation
        sorted_correlations = sorted(
            target_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            "target_correlations": dict(sorted_correlations),
            "high_correlation_features": {
                k: v for k, v in target_correlations.items() 
                if abs(v) > 0.5
            },
            "low_correlation_features": {
                k: v for k, v in target_correlations.items() 
                if abs(v) < 0.1
            },
        }
    
    def _data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality score."""
        scores = {}
        
        # Missing data score (0-100, higher is better)
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores["missing_data"] = max(0, 100 - missing_ratio * 100)
        
        # Duplicate rows score
        duplicate_ratio = df.duplicated().sum() / len(df)
        scores["duplicates"] = max(0, 100 - duplicate_ratio * 100)
        
        # Data type consistency score
        type_consistency = 100  # Assume good if we got here
        scores["type_consistency"] = type_consistency
        
        # Overall score
        overall_score = np.mean(list(scores.values()))
        scores["overall"] = overall_score
        
        return scores
    
    def get_recommendations(self) -> List[str]:
        """Get data quality recommendations based on profiling results."""
        recommendations = []
        
        if not self.profile_results:
            return recommendations
        
        # Missing data recommendations
        missing_analysis = self.profile_results.get("missing_analysis", {})
        high_missing = missing_analysis.get("high_missing_columns", {})
        if high_missing:
            recommendations.append(
                f"Consider dropping columns with >50% missing values: {list(high_missing.keys())}"
            )
        
        # High cardinality recommendations
        cardinality_analysis = self.profile_results.get("cardinality_analysis", {})
        for col, info in cardinality_analysis.items():
            if info.get("is_high_cardinality", False):
                recommendations.append(
                    f"High cardinality column '{col}' may need encoding or grouping"
                )
            if info.get("is_potential_id", False):
                recommendations.append(
                    f"Column '{col}' appears to be an ID column and should be excluded from modeling"
                )
        
        # Outlier recommendations
        outlier_analysis = self.profile_results.get("outlier_analysis", {})
        for col, info in outlier_analysis.items():
            if info.get("iqr_outlier_ratio", 0) > 0.1:
                recommendations.append(
                    f"Column '{col}' has many outliers ({info['iqr_outlier_ratio']:.1%}) - consider outlier treatment"
                )
        
        # Target recommendations
        target_analysis = self.profile_results.get("target_analysis", {})
        if target_analysis.get("is_binary", False):
            class_balance = target_analysis.get("class_balance", {})
            if class_balance:
                min_ratio = min(class_balance.values())
                if min_ratio < 0.1:
                    recommendations.append(
                        "Severe class imbalance detected - consider resampling or class weights"
                    )
        
        return recommendations
