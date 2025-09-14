"""Basic model blending strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import warnings


class ModelBlender:
    """Main model blender class that coordinates different blending strategies."""
    
    def __init__(self):
        """Initialize model blender."""
        self.blenders = {
            'mean': MeanBlender,
            'rank-mean': RankMeanBlender,
            'logit-mean': LogitMeanBlender,
            'stacking': StackingBlender
        }
    
    def blend_models(self, model_results: List[Dict[str, Any]], strategy: str = 'mean') -> Dict[str, Any]:
        """
        Blend multiple models using specified strategy.
        
        Args:
            model_results: List of model results dictionaries
            strategy: Blending strategy to use
            
        Returns:
            Blending results dictionary
        """
        if strategy not in self.blenders:
            raise ValueError(f"Unknown blending strategy: {strategy}")
        
        # Create blender instance
        BlenderClass = self.blenders[strategy]
        blender = BlenderClass()
        
        # Extract predictions and labels
        predictions = []
        labels = None
        
        for result in model_results:
            if 'predictions' in result:
                predictions.append(result['predictions'])
            if 'probabilities' in result and labels is None:
                # Use probabilities for classification
                labels = result.get('labels', None)
        
        if not predictions:
            raise ValueError("No predictions found in model results")
        
        # Convert to numpy arrays
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Perform blending
        if strategy in ['mean', 'rank-mean', 'logit-mean']:
            blended_pred = blender.fit_transform(predictions)
        else:
            # For stacking, we need more complex setup
            blended_pred = blender.fit_transform(predictions)
        
        # Calculate metrics
        if labels is not None:
            from sklearn.metrics import accuracy_score, roc_auc_score
            accuracy = accuracy_score(labels, blended_pred > 0.5)
            try:
                auc = roc_auc_score(labels, blended_pred)
            except:
                auc = 0.5
        else:
            accuracy = 0.0
            auc = 0.5
        
        return {
            'strategy': strategy,
            'blended_predictions': blended_pred,
            'accuracy': accuracy,
            'auc': auc,
            'n_models': len(model_results),
            'summary': {
                'strategy': strategy,
                'n_models': len(model_results),
                'accuracy': accuracy,
                'auc': auc
            }
        }


class MeanBlender(BaseEstimator, TransformerMixin):
    """Simple mean blending of model predictions."""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize mean blender.
        
        Args:
            weights: Weights for each model (if None, equal weights)
        """
        self.weights = weights
        self.n_models = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MeanBlender':
        """
        Fit the blender.
        
        Args:
            X: Array of shape (n_samples, n_models) with predictions
            y: Target values (not used for mean blending)
        """
        self.n_models = X.shape[1]
        
        if self.weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        elif len(self.weights) != self.n_models:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({self.n_models})")
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform predictions using mean blending."""
        return np.average(X, axis=1, weights=self.weights)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform predictions."""
        return self.fit(X, y).transform(X)


class RankMeanBlender(BaseEstimator, TransformerMixin):
    """Rank-based mean blending."""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize rank mean blender.
        
        Args:
            weights: Weights for each model (if None, equal weights)
        """
        self.weights = weights
        self.n_models = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RankMeanBlender':
        """
        Fit the blender.
        
        Args:
            X: Array of shape (n_samples, n_models) with predictions
            y: Target values (not used for rank blending)
        """
        self.n_models = X.shape[1]
        
        if self.weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        elif len(self.weights) != self.n_models:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({self.n_models})")
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform predictions using rank-based blending."""
        # Convert to ranks
        ranks = np.zeros_like(X)
        for i in range(X.shape[1]):
            ranks[:, i] = np.argsort(np.argsort(X[:, i]))
        
        # Weighted average of ranks
        blended_ranks = np.average(ranks, axis=1, weights=self.weights)
        
        # Convert back to probabilities (normalize to [0, 1])
        n_samples = X.shape[0]
        blended_probs = blended_ranks / (n_samples - 1)
        
        return blended_probs
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform predictions."""
        return self.fit(X, y).transform(X)


class LogitMeanBlender(BaseEstimator, TransformerMixin):
    """Logit-space mean blending."""
    
    def __init__(self, weights: Optional[List[float]] = None, epsilon: float = 1e-15):
        """
        Initialize logit mean blender.
        
        Args:
            weights: Weights for each model (if None, equal weights)
            epsilon: Small value to avoid log(0)
        """
        self.weights = weights
        self.epsilon = epsilon
        self.n_models = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LogitMeanBlender':
        """
        Fit the blender.
        
        Args:
            X: Array of shape (n_samples, n_models) with predictions
            y: Target values (not used for logit blending)
        """
        self.n_models = X.shape[1]
        
        if self.weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        elif len(self.weights) != self.n_models:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({self.n_models})")
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform predictions using logit-space blending."""
        # Clip probabilities to avoid log(0) and log(1)
        X_clipped = np.clip(X, self.epsilon, 1 - self.epsilon)
        
        # Convert to logits
        logits = np.log(X_clipped / (1 - X_clipped))
        
        # Weighted average of logits
        blended_logits = np.average(logits, axis=1, weights=self.weights)
        
        # Convert back to probabilities
        blended_probs = 1 / (1 + np.exp(-blended_logits))
        
        return blended_probs
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform predictions."""
        return self.fit(X, y).transform(X)


class StackingBlender(BaseEstimator, TransformerMixin):
    """Stacking blender using a meta-learner."""
    
    def __init__(
        self,
        meta_learner: Optional[BaseEstimator] = None,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize stacking blender.
        
        Args:
            meta_learner: Meta-learner for stacking (if None, auto-select)
            cv_folds: Number of CV folds for stacking
            random_state: Random state for reproducibility
        """
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.meta_model_ = None
        self.n_models = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingBlender':
        """
        Fit the stacking blender.
        
        Args:
            X: Array of shape (n_samples, n_models) with predictions
            y: Target values
        """
        self.n_models = X.shape[1]
        
        # Auto-select meta-learner if not provided
        if self.meta_learner is None:
            if len(np.unique(y)) == 2:
                self.meta_learner = LogisticRegression(random_state=self.random_state)
            else:
                self.meta_learner = LinearRegression()
        
        # Fit meta-learner
        self.meta_model_ = self.meta_learner
        self.meta_model_.fit(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform predictions using stacking."""
        if self.meta_model_ is None:
            raise ValueError("Stacking blender not fitted yet")
        
        return self.meta_model_.predict(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform predictions."""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from meta-learner."""
        if self.meta_model_ is None:
            raise ValueError("Stacking blender not fitted yet")
        
        if hasattr(self.meta_model_, 'coef_'):
            return np.abs(self.meta_model_.coef_[0])
        elif hasattr(self.meta_model_, 'feature_importances_'):
            return self.meta_model_.feature_importances_
        else:
            return np.ones(self.n_models) / self.n_models


class BlendingEnsemble:
    """Ensemble of different blending strategies."""
    
    def __init__(
        self,
        strategies: List[str] = ['mean', 'rank_mean', 'logit_mean'],
        weights: Optional[List[float]] = None,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize blending ensemble.
        
        Args:
            strategies: List of blending strategies to use
            weights: Weights for each strategy (if None, equal weights)
            cv_folds: Number of CV folds for evaluation
            random_state: Random state for reproducibility
        """
        self.strategies = strategies
        self.weights = weights
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.blenders = {}
        self.strategy_weights = {}
        self.is_fitted = False
    
    def fit(
        self, 
        predictions: Dict[str, np.ndarray], 
        y: np.ndarray
    ) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            y: Target values
        """
        # Convert predictions to array
        model_names = list(predictions.keys())
        pred_array = np.column_stack([predictions[name] for name in model_names])
        
        # Initialize blenders
        for strategy in self.strategies:
            if strategy == 'mean':
                self.blenders[strategy] = MeanBlender()
            elif strategy == 'rank_mean':
                self.blenders[strategy] = RankMeanBlender()
            elif strategy == 'logit_mean':
                self.blenders[strategy] = LogitMeanBlender()
            elif strategy == 'stacking':
                self.blenders[strategy] = StackingBlender(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unknown blending strategy: {strategy}")
        
        # Fit blenders and evaluate
        strategy_scores = {}
        for strategy, blender in self.blenders.items():
            try:
                # Fit blender
                blender.fit(pred_array, y)
                
                # Evaluate using cross-validation
                if strategy == 'stacking':
                    # For stacking, use the meta-learner's score
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(
                        blender.meta_model_, pred_array, y,
                        cv=self.cv_folds, scoring='roc_auc' if len(np.unique(y)) == 2 else 'r2'
                    )
                    strategy_scores[strategy] = np.mean(scores)
                else:
                    # For other strategies, use simple evaluation
                    pred = blender.transform(pred_array)
                    if len(np.unique(y)) == 2:
                        from sklearn.metrics import roc_auc_score
                        strategy_scores[strategy] = roc_auc_score(y, pred)
                    else:
                        from sklearn.metrics import r2_score
                        strategy_scores[strategy] = r2_score(y, pred)
                
            except Exception as e:
                warnings.warn(f"Failed to fit {strategy} blender: {e}")
                strategy_scores[strategy] = 0.0
        
        # Set strategy weights based on performance
        if self.weights is None:
            # Use performance-based weights
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                self.strategy_weights = {
                    strategy: score / total_score 
                    for strategy, score in strategy_scores.items()
                }
            else:
                # Equal weights if all strategies failed
                self.strategy_weights = {
                    strategy: 1.0 / len(self.strategies) 
                    for strategy in self.strategies
                }
        else:
            # Use provided weights
            if len(self.weights) != len(self.strategies):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of strategies ({len(self.strategies)})")
            self.strategy_weights = dict(zip(self.strategies, self.weights))
        
        self.is_fitted = True
        return self
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Blending ensemble not fitted yet")
        
        # Convert predictions to array
        model_names = list(predictions.keys())
        pred_array = np.column_stack([predictions[name] for name in model_names])
        
        # Get predictions from each strategy
        strategy_predictions = {}
        for strategy, blender in self.blenders.items():
            try:
                strategy_predictions[strategy] = blender.transform(pred_array)
            except Exception as e:
                warnings.warn(f"Failed to predict with {strategy} blender: {e}")
                # Use mean as fallback
                strategy_predictions[strategy] = np.mean(pred_array, axis=1)
        
        # Weighted average of strategy predictions
        final_pred = np.zeros(len(pred_array))
        for strategy, pred in strategy_predictions.items():
            weight = self.strategy_weights.get(strategy, 0.0)
            final_pred += weight * pred
        
        return final_pred
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get weights for each blending strategy."""
        return self.strategy_weights
    
    def get_strategy_scores(self) -> Dict[str, float]:
        """Get performance scores for each strategy."""
        return getattr(self, 'strategy_scores', {})


def create_blender(
    strategy: str,
    **kwargs
) -> BaseEstimator:
    """
    Create a blender with the specified strategy.
    
    Args:
        strategy: Blending strategy name
        **kwargs: Additional arguments for the blender
        
    Returns:
        Blender instance
    """
    if strategy == 'mean':
        return MeanBlender(**kwargs)
    elif strategy == 'rank_mean':
        return RankMeanBlender(**kwargs)
    elif strategy == 'logit_mean':
        return LogitMeanBlender(**kwargs)
    elif strategy == 'stacking':
        return StackingBlender(**kwargs)
    else:
        raise ValueError(f"Unknown blending strategy: {strategy}")


def blend_predictions(
    predictions: Dict[str, np.ndarray],
    y: np.ndarray,
    strategy: str = 'mean',
    **kwargs
) -> np.ndarray:
    """
    Blend predictions using the specified strategy.
    
    Args:
        predictions: Dictionary mapping model names to predictions
        y: Target values
        strategy: Blending strategy name
        **kwargs: Additional arguments for the blender
        
    Returns:
        Blended predictions
    """
    # Convert predictions to array
    model_names = list(predictions.keys())
    pred_array = np.column_stack([predictions[name] for name in model_names])
    
    # Create and fit blender
    blender = create_blender(strategy, **kwargs)
    blender.fit(pred_array, y)
    
    # Return blended predictions
    return blender.transform(pred_array)
