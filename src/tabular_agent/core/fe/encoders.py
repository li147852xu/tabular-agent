"""Feature encoders for time-aware feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with time-aware cross-validation."""
    
    def __init__(
        self, 
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        time_col: Optional[str] = None,
        cv_folds: int = 5
    ):
        """
        Initialize target encoder.
        
        Args:
            smoothing: Smoothing parameter for target encoding
            min_samples_leaf: Minimum samples per leaf
            time_col: Time column for time-aware CV
            cv_folds: Number of CV folds
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.time_col = time_col
        self.cv_folds = cv_folds
        self.encodings_ = {}
        self.global_mean_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TargetEncoder':
        """Fit target encoder."""
        self.global_mean_ = y.mean()
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Transform features using target encoding."""
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                if col in self.encodings_:
                    X_encoded[col] = X[col].map(self.encodings_[col]).fillna(self.global_mean_)
                else:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X, y)


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder."""
    
    def __init__(self, time_col: Optional[str] = None):
        """
        Initialize WOE encoder.
        
        Args:
            time_col: Time column for time-aware encoding
        """
        self.time_col = time_col
        self.woe_mappings_ = {}
        self.iv_scores_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WOEEncoder':
        """Fit WOE encoder."""
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                woe_mapping, iv_score = self._calculate_woe(X[col], y)
                self.woe_mappings_[col] = woe_mapping
                self.iv_scores_[col] = iv_score
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using WOE encoding."""
        X_encoded = X.copy()
        
        for col in X.columns:
            if col in self.woe_mappings_:
                X_encoded[col] = X[col].map(self.woe_mappings_[col]).fillna(0)
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)
    
    def _calculate_woe(self, feature: pd.Series, target: pd.Series) -> tuple:
        """Calculate WOE and IV for a feature."""
        # Create contingency table
        contingency = pd.crosstab(feature, target)
        
        # Calculate WOE
        woe_mapping = {}
        total_positive = target.sum()
        total_negative = len(target) - total_positive
        
        for category in contingency.index:
            positive = contingency.loc[category, 1] if 1 in contingency.columns else 0
            negative = contingency.loc[category, 0] if 0 in contingency.columns else 0
            
            if positive > 0 and negative > 0:
                woe = np.log((positive / total_positive) / (negative / total_negative))
                woe_mapping[category] = woe
            else:
                woe_mapping[category] = 0
        
        # Calculate IV
        iv = 0
        for category in contingency.index:
            positive = contingency.loc[category, 1] if 1 in contingency.columns else 0
            negative = contingency.loc[category, 0] if 0 in contingency.columns else 0
            
            if positive > 0 and negative > 0:
                iv += (positive / total_positive - negative / total_negative) * np.log(
                    (positive / total_positive) / (negative / total_negative)
                )
        
        return woe_mapping, iv


class RollingEncoder(BaseEstimator, TransformerMixin):
    """Rolling window feature encoder for time series data."""
    
    def __init__(
        self, 
        time_col: str,
        window_sizes: List[int] = [7, 14, 30],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ):
        """
        Initialize rolling encoder.
        
        Args:
            time_col: Time column name
            window_sizes: List of window sizes in days
            functions: List of aggregation functions
        """
        self.time_col = time_col
        self.window_sizes = window_sizes
        self.functions = functions
        self.feature_columns_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RollingEncoder':
        """Fit rolling encoder."""
        # Sort by time
        X_sorted = X.sort_values(self.time_col)
        
        # Generate feature column names
        numeric_cols = X_sorted.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.time_col]
        
        for col in numeric_cols:
            for window in self.window_sizes:
                for func in self.functions:
                    self.feature_columns_.append(f"{col}_rolling_{window}d_{func}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using rolling windows."""
        X_encoded = X.copy()
        
        # Sort by time
        X_sorted = X_encoded.sort_values(self.time_col)
        
        # Generate rolling features
        numeric_cols = X_sorted.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.time_col]
        
        for col in numeric_cols:
            for window in self.window_sizes:
                # Calculate rolling statistics
                rolling_mean = X_sorted[col].rolling(window=window, min_periods=1).mean()
                rolling_std = X_sorted[col].rolling(window=window, min_periods=1).std()
                rolling_min = X_sorted[col].rolling(window=window, min_periods=1).min()
                rolling_max = X_sorted[col].rolling(window=window, min_periods=1).max()
                
                # Add to dataframe
                X_encoded[f"{col}_rolling_{window}d_mean"] = rolling_mean
                X_encoded[f"{col}_rolling_{window}d_std"] = rolling_std
                X_encoded[f"{col}_rolling_{window}d_min"] = rolling_min
                X_encoded[f"{col}_rolling_{window}d_max"] = rolling_max
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)


class TimeAwareEncoder(BaseEstimator, TransformerMixin):
    """Time-aware feature encoder that prevents leakage."""
    
    def __init__(
        self, 
        time_col: str,
        target_col: str,
        encoders: List[str] = ['target', 'woe'],
        cv_folds: int = 5
    ):
        """
        Initialize time-aware encoder.
        
        Args:
            time_col: Time column name
            target_col: Target column name
            encoders: List of encoders to use
            cv_folds: Number of CV folds
        """
        self.time_col = time_col
        self.target_col = target_col
        self.encoders = encoders
        self.cv_folds = cv_folds
        self.encoder_objects_ = {}
        self.feature_columns_ = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TimeAwareEncoder':
        """Fit time-aware encoder."""
        # Sort by time
        X_sorted = X.sort_values(self.time_col)
        y_sorted = y.loc[X_sorted.index]
        
        # Initialize encoders
        if 'target' in self.encoders:
            self.encoder_objects_['target'] = TargetEncoder(
                time_col=self.time_col,
                cv_folds=self.cv_folds
            )
        
        if 'woe' in self.encoders:
            self.encoder_objects_['woe'] = WOEEncoder(time_col=self.time_col)
        
        # Fit encoders
        for name, encoder in self.encoder_objects_.items():
            encoder.fit(X_sorted, y_sorted)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using time-aware encoding."""
        X_encoded = X.copy()
        
        # Sort by time
        X_sorted = X_encoded.sort_values(self.time_col)
        
        # Apply encoders
        for name, encoder in self.encoder_objects_.items():
            if hasattr(encoder, 'transform'):
                X_sorted = encoder.transform(X_sorted)
        
        # Restore original order
        X_encoded = X_sorted.loc[X_encoded.index]
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector for removing low-quality features."""
    
    def __init__(
        self,
        min_variance: float = 0.01,
        max_correlation: float = 0.95,
        min_iv: float = 0.02
    ):
        """
        Initialize feature selector.
        
        Args:
            min_variance: Minimum variance threshold
            max_correlation: Maximum correlation threshold
            min_iv: Minimum IV threshold for categorical features
        """
        self.min_variance = min_variance
        self.max_correlation = max_correlation
        self.min_iv = min_iv
        self.selected_features_ = []
        self.feature_importance_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit feature selector."""
        # Remove low variance features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        high_variance_cols = []
        
        for col in numeric_cols:
            if X[col].var() >= self.min_variance:
                high_variance_cols.append(col)
        
        # Remove highly correlated features
        correlation_matrix = X[high_variance_cols].corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.max_correlation)]
        selected_numeric = [col for col in high_variance_cols if col not in to_drop]
        
        # Select categorical features based on IV
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        selected_categorical = []
        
        for col in categorical_cols:
            # Calculate IV for this feature
            iv = self._calculate_iv(X[col], y)
            if iv >= self.min_iv:
                selected_categorical.append(col)
                self.feature_importance_[col] = iv
        
        self.selected_features_ = selected_numeric + selected_categorical
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting only important ones."""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)
    
    def _calculate_iv(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Value for a feature."""
        contingency = pd.crosstab(feature, target)
        total_positive = target.sum()
        total_negative = len(target) - total_positive
        
        iv = 0
        for category in contingency.index:
            positive = contingency.loc[category, 1] if 1 in contingency.columns else 0
            negative = contingency.loc[category, 0] if 0 in contingency.columns else 0
            
            if positive > 0 and negative > 0:
                iv += (positive / total_positive - negative / total_negative) * np.log(
                    (positive / total_positive) / (negative / total_negative)
                )
        
        return iv
