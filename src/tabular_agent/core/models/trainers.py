"""Model training utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import warnings
import time

from .registry import model_registry, BaseModel


class ModelTrainer:
    """Unified model trainer with cross-validation support."""
    
    def __init__(
        self,
        model_name: str,
        cv_folds: int = 5,
        time_col: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize model trainer.
        
        Args:
            model_name: Name of model to train
            cv_folds: Number of CV folds
            time_col: Time column for time-aware CV
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.time_col = time_col
        self.random_state = random_state
        self.model = None
        self.cv_scores = {}
        self.training_time = 0
        self.is_fitted = False
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> 'ModelTrainer':
        """
        Fit the model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Get model instance
        if params:
            self.model = model_registry.get_model(self.model_name, **params)
        else:
            self.model = model_registry.get_model(self.model_name)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) == 2
        
        # Perform cross-validation
        if self.time_col:
            # Time-aware CV
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_splits = list(tscv.split(X))
        else:
            # Regular CV
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_splits = list(kf.split(X))
        
        # Calculate CV scores
        cv_scores = []
        for train_idx, val_idx in cv_splits:
            # Handle both DataFrame and numpy array inputs
            if hasattr(X, 'iloc'):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if hasattr(y, 'iloc'):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and fit model for this fold
            fold_model = model_registry.get_model(self.model_name, **params) if params else model_registry.get_model(self.model_name)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if is_classification:
                y_proba = fold_model.predict_proba(X_val)
                # Handle both 1D and 2D probability arrays
                if y_proba.ndim == 1:
                    y_pred = y_proba
                else:
                    y_pred = y_proba[:, 1]
                # Calculate AUC for binary classification
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val, y_pred)
            else:
                y_pred = fold_model.predict(X_val)
                # Calculate R² for regression
                from sklearn.metrics import r2_score
                score = r2_score(y_val, y_pred)
            
            cv_scores.append(score)
        
        self.cv_scores = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'min': np.min(cv_scores),
            'max': np.max(cv_scores)
        }
        
        # Fit final model on all data
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def get_cv_scores(self) -> Dict[str, float]:
        """Get cross-validation scores."""
        return self.cv_scores
    
    def get_training_time(self) -> float:
        """Get training time in seconds."""
        return self.training_time


class EnsembleTrainer:
    """Ensemble trainer for multiple models."""
    
    def __init__(
        self,
        model_names: List[str],
        cv_folds: int = 5,
        time_col: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            model_names: List of model names to train
            cv_folds: Number of CV folds
            time_col: Time column for time-aware CV
            random_state: Random state for reproducibility
        """
        self.model_names = model_names
        self.cv_folds = cv_folds
        self.time_col = time_col
        self.random_state = random_state
        self.trainers = {}
        self.is_fitted = False
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> 'EnsembleTrainer':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters for each model
            
        Returns:
            Self
        """
        for model_name in self.model_names:
            model_params = params.get(model_name, {}) if params else {}
            
            trainer = ModelTrainer(
                model_name=model_name,
                cv_folds=self.cv_folds,
                time_col=self.time_col,
                random_state=self.random_state
            )
            
            trainer.fit(X, y, model_params)
            self.trainers[model_name] = trainer
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions (average)."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = []
        for trainer in self.trainers.values():
            predictions.append(trainer.predict(X))
        
        return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble probability predictions (average)."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = []
        for trainer in self.trainers.values():
            predictions.append(trainer.predict_proba(X))
        
        return np.mean(predictions, axis=0)
    
    def get_ensemble_scores(self) -> Dict[str, Dict[str, float]]:
        """Get CV scores for all models in ensemble."""
        return {name: trainer.get_cv_scores() for name, trainer in self.trainers.items()}
    
    def get_best_model(self) -> str:
        """Get the best performing model based on CV scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        best_score = -np.inf
        best_model = None
        
        for name, trainer in self.trainers.items():
            cv_scores = trainer.get_cv_scores()
            if cv_scores['mean'] > best_score:
                best_score = cv_scores['mean']
                best_model = name
        
        return best_model
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        return {name: trainer.get_feature_importance() for name, trainer in self.trainers.items()}


class ModelEvaluator:
    """Model evaluation utilities."""
    
    def __init__(self, is_classification: bool = True):
        """
        Initialize model evaluator.
        
        Args:
            is_classification: Whether this is a classification task
        """
        self.is_classification = is_classification
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for classification)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.is_classification:
            return self._evaluate_classification(y_true, y_pred, y_proba)
        else:
            return self._evaluate_regression(y_true, y_pred)
    
    def _evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate classification performance."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def _evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression performance."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error
        )
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        metrics['mape'] = mape
        
        return metrics
