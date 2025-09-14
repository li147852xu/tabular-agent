"""Model registry and configuration."""

from typing import Dict, Any, List, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, **kwargs):
        """Initialize model with parameters."""
        self.params = kwargs
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Make probability predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """Get feature importance."""
        pass


class ModelRegistry:
    """Registry for available models."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models = {}
        self._register_default_models()
    
    def register(self, name: str, model_class: type, default_params: Dict[str, Any]):
        """Register a new model."""
        self.models[name] = {
            'class': model_class,
            'default_params': default_params
        }
    
    def get_model(self, name: str, **params) -> BaseModel:
        """Get a model instance."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        model_info = self.models[name]
        model_class = model_info['class']
        default_params = model_info['default_params'].copy()
        default_params.update(params)
        
        return model_class(**default_params)
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.models.keys())
    
    def get_default_params(self, name: str) -> Dict[str, Any]:
        """Get default parameters for a model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        return self.models[name]['default_params'].copy()
    
    def _register_default_models(self):
        """Register default models."""
        # LightGBM
        self.register('lightgbm', LightGBMModel, {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        })
        
        # XGBoost
        self.register('xgboost', XGBoostModel, {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        })
        
        # CatBoost
        self.register('catboost', CatBoostModel, {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'random_seed': 42,
            'verbose': False
        })
        
        # Linear Regression/Logistic Regression
        self.register('linear', LinearModel, {
            'random_state': 42
        })
        
        # Histogram-based Gradient Boosting
        self.register('histgbdt', HistGBDTModel, {
            'max_iter': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        })


# Model implementations
class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize LightGBM model."""
        super().__init__(**kwargs)
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    def fit(self, X, y, **kwargs):
        """Fit LightGBM model."""
        # Determine objective
        if len(np.unique(y)) == 2:
            objective = 'binary'
            eval_metric = 'binary_logloss'
        else:
            objective = 'regression'
            eval_metric = 'rmse'
        
        # Create dataset
        train_data = self.lgb.Dataset(X, label=y)
        
        # Train model
        self.model = self.lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data],
            callbacks=[self.lgb.early_stopping(10), self.lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importance(importance_type='gain')


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize XGBoost model."""
        super().__init__(**kwargs)
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def fit(self, X, y, **kwargs):
        """Fit XGBoost model."""
        # Determine if classification or regression
        if len(np.unique(y)) == 2:
            # Binary classification
            self.model = self.xgb.XGBClassifier(**self.params)
        else:
            # Regression
            self.model = self.xgb.XGBRegressor(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Check if it's a classifier
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regressors, return predictions as probabilities
            pred = self.model.predict(X)
            # Convert to 2D array for consistency
            if pred.ndim == 1:
                return np.column_stack([1 - pred, pred])
            else:
                return pred
    
    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize CatBoost model."""
        super().__init__(**kwargs)
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
            self.CatBoostRegressor = CatBoostRegressor
            self.CatBoostClassifier = CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    def fit(self, X, y, **kwargs):
        """Fit CatBoost model."""
        # Determine if classification or regression
        if len(np.unique(y)) == 2:
            self.model = self.CatBoostClassifier(**self.params)
        else:
            self.model = self.CatBoostRegressor(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class LinearModel(BaseModel):
    """Linear model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize linear model."""
        super().__init__(**kwargs)
        from sklearn.linear_model import LogisticRegression, LinearRegression
        self.LogisticRegression = LogisticRegression
        self.LinearRegression = LinearRegression
    
    def fit(self, X, y, **kwargs):
        """Fit linear model."""
        # Determine if classification or regression
        if len(np.unique(y)) == 2:
            self.model = self.LogisticRegression(**self.params)
        else:
            self.model = self.LinearRegression(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return np.abs(self.model.coef_[0])


class HistGBDTModel(BaseModel):
    """Histogram-based Gradient Boosting model wrapper."""
    
    def __init__(self, **kwargs):
        """Initialize HistGBDT model."""
        super().__init__(**kwargs)
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        self.HistGradientBoostingRegressor = HistGradientBoostingRegressor
        self.HistGradientBoostingClassifier = HistGradientBoostingClassifier
    
    def fit(self, X, y, **kwargs):
        """Fit HistGBDT model."""
        # Determine if classification or regression
        if len(np.unique(y)) == 2:
            self.model = self.HistGradientBoostingClassifier(**self.params)
        else:
            self.model = self.HistGradientBoostingRegressor(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


# Global registry instance
model_registry = ModelRegistry()
