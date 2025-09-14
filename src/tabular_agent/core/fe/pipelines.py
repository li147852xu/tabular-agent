"""Feature engineering pipelines for time-aware processing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings

from .encoders import (
    TargetEncoder, WOEEncoder, RollingEncoder, 
    TimeAwareEncoder, FeatureSelector
)


class TimeAwarePipeline(BaseEstimator, TransformerMixin):
    """Time-aware feature engineering pipeline."""
    
    def __init__(
        self,
        target_col: str,
        time_col: Optional[str] = None,
        feature_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize time-aware pipeline.
        
        Args:
            target_col: Target column name
            time_col: Time column name (optional)
            feature_config: Feature engineering configuration
        """
        self.target_col = target_col
        self.time_col = time_col
        self.feature_config = feature_config or {}
        self.pipeline_ = None
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TimeAwarePipeline':
        """Fit the pipeline."""
        # Create pipeline based on configuration
        self.pipeline_ = self._create_pipeline()
        
        # Fit pipeline
        self.pipeline_.fit(X, y)
        
        # Get feature names
        self.feature_names_ = self._get_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if self.pipeline_ is None:
            raise ValueError("Pipeline not fitted yet")
        
        return self.pipeline_.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)
    
    def _create_pipeline(self) -> Pipeline:
        """Create feature engineering pipeline."""
        steps = []
        
        # Time-aware encoding
        if self.time_col:
            steps.append(('time_aware_encoder', TimeAwareEncoder(
                time_col=self.time_col,
                target_col=self.target_col,
                encoders=self.feature_config.get('encoders', ['target', 'woe']),
                cv_folds=self.feature_config.get('cv_folds', 5)
            )))
        
        # Rolling features
        if self.time_col and self.feature_config.get('rolling_features', True):
            steps.append(('rolling_encoder', RollingEncoder(
                time_col=self.time_col,
                window_sizes=self.feature_config.get('window_sizes', [7, 14, 30]),
                functions=self.feature_config.get('rolling_functions', ['mean', 'std', 'min', 'max'])
            )))
        
        # Feature selection
        if self.feature_config.get('feature_selection', True):
            steps.append(('feature_selector', FeatureSelector(
                min_variance=self.feature_config.get('min_variance', 0.01),
                max_correlation=self.feature_config.get('max_correlation', 0.95),
                min_iv=self.feature_config.get('min_iv', 0.02)
            )))
        
        # Scaling (disabled for now to avoid DataFrame conversion issues)
        # if self.feature_config.get('scaling', True):
        #     steps.append(('scaler', StandardScaler()))
        
        return Pipeline(steps)
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation."""
        if self.pipeline_ is None:
            return []
        
        # Get feature names from the last step
        last_step = self.pipeline_.steps[-1][1]
        if hasattr(last_step, 'feature_names_in_'):
            return last_step.feature_names_in_.tolist()
        else:
            return [f"feature_{i}" for i in range(X.shape[1])]


class CrossValidationEncoder(BaseEstimator, TransformerMixin):
    """Cross-validation encoder to prevent leakage."""
    
    def __init__(
        self,
        encoder: BaseEstimator,
        cv_folds: int = 5,
        time_col: Optional[str] = None
    ):
        """
        Initialize CV encoder.
        
        Args:
            encoder: Base encoder to use
            cv_folds: Number of CV folds
            time_col: Time column for time-aware CV
        """
        self.encoder = encoder
        self.cv_folds = cv_folds
        self.time_col = time_col
        self.encoders_ = []
        self.cv_splits_ = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CrossValidationEncoder':
        """Fit CV encoder."""
        # Create CV splits
        if self.time_col:
            # Time-aware CV
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            self.cv_splits_ = list(tscv.split(X))
        else:
            # Regular CV
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            self.cv_splits_ = list(kf.split(X))
        
        # Fit encoders for each fold
        self.encoders_ = []
        for train_idx, _ in self.cv_splits_:
            encoder = self.encoder.__class__(**self.encoder.get_params())
            encoder.fit(X.iloc[train_idx], y.iloc[train_idx])
            self.encoders_.append(encoder)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using CV encoders."""
        X_encoded = X.copy()
        
        # Apply each encoder to its corresponding fold
        for i, (_, val_idx) in enumerate(self.cv_splits_):
            if i < len(self.encoders_):
                X_encoded.iloc[val_idx] = self.encoders_[i].transform(X.iloc[val_idx])
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)


class FeatureEngineeringPipeline:
    """Main feature engineering pipeline orchestrator."""
    
    def __init__(
        self,
        target_col: str,
        time_col: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            target_col: Target column name
            time_col: Time column name (optional)
            config: Configuration dictionary
        """
        self.target_col = target_col
        self.time_col = time_col
        self.config = config or {}
        self.pipeline_ = None
        self.feature_importance_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEngineeringPipeline':
        """Fit the feature engineering pipeline."""
        # Create pipeline
        self.pipeline_ = TimeAwarePipeline(
            target_col=self.target_col,
            time_col=self.time_col,
            feature_config=self.config
        )
        
        # Fit pipeline
        self.pipeline_.fit(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if self.pipeline_ is None:
            raise ValueError("Pipeline not fitted yet")
        
        return self.pipeline_.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        if self.pipeline_ is None:
            return []
        return self.pipeline_.feature_names_
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance_
    
    def create_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if not self.time_col or self.time_col not in X.columns:
            return X
        
        X_time = X.copy()
        time_series = pd.to_datetime(X_time[self.time_col])
        
        # Basic time features
        X_time[f'{self.time_col}_year'] = time_series.dt.year
        X_time[f'{self.time_col}_month'] = time_series.dt.month
        X_time[f'{self.time_col}_day'] = time_series.dt.day
        X_time[f'{self.time_col}_weekday'] = time_series.dt.weekday
        X_time[f'{self.time_col}_hour'] = time_series.dt.hour
        X_time[f'{self.time_col}_quarter'] = time_series.dt.quarter
        
        # Cyclical encoding
        X_time[f'{self.time_col}_month_sin'] = np.sin(2 * np.pi * time_series.dt.month / 12)
        X_time[f'{self.time_col}_month_cos'] = np.cos(2 * np.pi * time_series.dt.month / 12)
        X_time[f'{self.time_col}_day_sin'] = np.sin(2 * np.pi * time_series.dt.day / 31)
        X_time[f'{self.time_col}_day_cos'] = np.cos(2 * np.pi * time_series.dt.day / 31)
        X_time[f'{self.time_col}_weekday_sin'] = np.sin(2 * np.pi * time_series.dt.weekday / 7)
        X_time[f'{self.time_col}_weekday_cos'] = np.cos(2 * np.pi * time_series.dt.weekday / 7)
        
        return X_time
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        X_interact = X.copy()
        
        # Get numeric columns
        numeric_cols = X_interact.select_dtypes(include=[np.number]).columns
        
        # Create pairwise interactions for top correlated features
        if len(numeric_cols) > 1:
            corr_matrix = X_interact[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Get top correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.3:  # Threshold for interaction
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Create interaction features
            for col1, col2 in high_corr_pairs[:10]:  # Limit to 10 interactions
                X_interact[f'{col1}_x_{col2}'] = X_interact[col1] * X_interact[col2]
                X_interact[f'{col1}_div_{col2}'] = X_interact[col1] / (X_interact[col2] + 1e-8)
        
        return X_interact
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numeric columns."""
        X_poly = X.copy()
        
        # Get numeric columns
        numeric_cols = X_poly.select_dtypes(include=[np.number]).columns
        
        # Create polynomial features for top features
        for col in numeric_cols[:5]:  # Limit to top 5 features
            for d in range(2, degree + 1):
                X_poly[f'{col}_poly_{d}'] = X_poly[col] ** d
        
        return X_poly
