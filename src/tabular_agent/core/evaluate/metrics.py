"""Comprehensive metrics calculation for model evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import warnings


class MetricsCalculator:
    """Comprehensive metrics calculator for classification and regression."""
    
    def __init__(self, is_classification: bool = True):
        """
        Initialize metrics calculator.
        
        Args:
            is_classification: Whether this is a classification task
        """
        self.is_classification = is_classification
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate all relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for classification)
            threshold: Classification threshold (if None, use 0.5)
            
        Returns:
            Dictionary containing all metrics
        """
        if self.is_classification:
            return self._calculate_classification_metrics(y_true, y_pred, y_proba, threshold)
        else:
            return self._calculate_regression_metrics(y_true, y_pred)
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        if threshold is None:
            threshold = 0.5
        
        # Convert probabilities to predictions if needed
        if y_proba is not None:
            # Ensure y_proba is 1D for binary classification
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]  # Take positive class probability
            y_pred_from_proba = (y_proba >= threshold).astype(int)
        else:
            y_pred_from_proba = y_pred
            y_proba = y_pred  # Use predictions as probabilities
        
        # Ensure predictions are 1D and integer
        if y_pred_from_proba.ndim > 1:
            y_pred_from_proba = y_pred_from_proba.ravel()
        y_pred_from_proba = y_pred_from_proba.astype(int)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_from_proba),
            'precision': precision_score(y_true, y_pred_from_proba, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred_from_proba, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred_from_proba, average='binary', zero_division=0),
            'threshold': threshold,
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_from_proba)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            })
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                warnings.warn(f"Failed to calculate probability metrics: {e}")
                metrics.update({
                    'auc': 0.0,
                    'pr_auc': 0.0,
                    'log_loss': float('inf')
                })
        
        # Additional metrics
        metrics.update({
            'ks_statistic': self._calculate_ks_statistic(y_true, y_proba),
            'gini_coefficient': 2 * metrics.get('auc', 0) - 1,
            'precision_recall_curve': self._calculate_pr_curve(y_true, y_proba),
            'roc_curve': self._calculate_roc_curve(y_true, y_proba),
        })
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        metrics['mape'] = mape
        
        # Additional metrics
        metrics.update({
            'mape_robust': self._calculate_robust_mape(y_true, y_pred),
            'smape': self._calculate_smape(y_true, y_pred),
            'wmape': self._calculate_wmape(y_true, y_pred),
            'residuals': y_true - y_pred,
            'residual_stats': self._calculate_residual_stats(y_true, y_pred),
        })
        
        return metrics
    
    def _calculate_ks_statistic(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        if y_proba is None:
            return 0.0
        
        try:
            from scipy import stats
            
            # Get probabilities for each class
            proba_0 = y_proba[y_true == 0]
            proba_1 = y_proba[y_true == 1]
            
            if len(proba_0) == 0 or len(proba_1) == 0:
                return 0.0
            
            # Calculate KS statistic
            ks_stat, _ = stats.ks_2samp(proba_0, proba_1)
            return ks_stat
            
        except Exception:
            return 0.0
    
    def _calculate_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate precision-recall curve."""
        if y_proba is None:
            return {'precision': np.array([]), 'recall': np.array([]), 'thresholds': np.array([])}
        
        try:
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            return {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
        except Exception:
            return {'precision': np.array([]), 'recall': np.array([]), 'thresholds': np.array([])}
    
    def _calculate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate ROC curve."""
        if y_proba is None:
            return {'fpr': np.array([]), 'tpr': np.array([]), 'thresholds': np.array([])}
        
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }
        except Exception:
            return {'fpr': np.array([]), 'tpr': np.array([]), 'thresholds': np.array([])}
    
    def _calculate_robust_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate robust MAPE (median absolute percentage error)."""
        try:
            mape = np.median(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            return mape
        except Exception:
            return float('inf')
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate symmetric MAPE."""
        try:
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
            return smape
        except Exception:
            return float('inf')
    
    def _calculate_wmape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate weighted MAPE."""
        try:
            wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
            return wmape
        except Exception:
            return float('inf')
    
    def _calculate_residual_stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate residual statistics."""
        residuals = y_true - y_pred
        
        return {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'skewness': self._calculate_skewness(residuals),
            'kurtosis': self._calculate_kurtosis(residuals),
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        try:
            from scipy import stats
            return stats.skew(data)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        try:
            from scipy import stats
            return stats.kurtosis(data)
        except Exception:
            return 0.0


class CalibrationAnalyzer:
    """Analyze model calibration."""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins
    
    def analyze_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze model calibration.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary containing calibration analysis
        """
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=self.n_bins
            )
            
            # Calculate Brier score
            brier_score = np.mean((y_proba - y_true) ** 2)
            
            # Calculate reliability diagram
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Calculate MCE (Maximum Calibration Error)
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            return {
                'brier_score': brier_score,
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'bin_boundaries': bin_boundaries,
                'reliability_diagram': {
                    'bin_lowers': bin_lowers,
                    'bin_uppers': bin_uppers,
                    'accuracy_in_bins': [y_true[(y_proba > bl) & (y_proba <= bu)].mean() 
                                       for bl, bu in zip(bin_lowers, bin_uppers)],
                    'confidence_in_bins': [y_proba[(y_proba > bl) & (y_proba <= bu)].mean() 
                                        for bl, bu in zip(bin_lowers, bin_uppers)]
                }
            }
            
        except Exception as e:
            warnings.warn(f"Failed to analyze calibration: {e}")
            return {
                'brier_score': float('inf'),
                'expected_calibration_error': float('inf'),
                'maximum_calibration_error': float('inf'),
                'fraction_of_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'bin_boundaries': np.array([]),
                'reliability_diagram': {}
            }


class ThresholdOptimizer:
    """Optimize classification threshold."""
    
    def __init__(self, metric: str = 'f1'):
        """
        Initialize threshold optimizer.
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'youden')
        """
        self.metric = metric
        self.best_threshold = 0.5
        self.best_score = 0.0
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: Thresholds to test (if None, use 100 evenly spaced values)
            
        Returns:
            Dictionary containing optimization results
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)
        
        scores = []
        for threshold in thresholds:
            # Ensure y_proba is 1D for binary classification
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba_1d = y_proba[:, 1]  # Take positive class probability
            else:
                y_proba_1d = y_proba.ravel() if y_proba.ndim > 1 else y_proba
            
            y_pred = (y_proba_1d >= threshold).astype(int)
            score = self._calculate_metric(y_true, y_pred, y_proba_1d, threshold)
            scores.append(score)
        
        scores = np.array(scores)
        best_idx = np.argmax(scores)
        
        self.best_threshold = thresholds[best_idx]
        self.best_score = scores[best_idx]
        
        return {
            'best_threshold': self.best_threshold,
            'best_score': self.best_score,
            'thresholds': thresholds,
            'scores': scores,
            'metric': self.metric
        }
    
    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        threshold: float
    ) -> float:
        """Calculate the specified metric."""
        # Ensure predictions are 1D and integer
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        y_pred = y_pred.astype(int)
        
        if self.metric == 'f1':
            return f1_score(y_true, y_pred, average='binary', zero_division=0)
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred, average='binary', zero_division=0)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred, average='binary', zero_division=0)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'youden':
            # Youden's J statistic (sensitivity + specificity - 1)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                return sensitivity + specificity - 1
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


class StabilityAnalyzer:
    """Analyze model stability across different groups."""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize stability analyzer.
        
        Args:
            n_bins: Number of bins for stability analysis
        """
        self.n_bins = n_bins
    
    def analyze_stability(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze model stability across groups.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            groups: Group labels for stability analysis
            
        Returns:
            Dictionary containing stability analysis
        """
        if groups is None:
            # Use probability bins as groups
            # Ensure y_proba is 1D
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba_1d = y_proba[:, 1]  # Take positive class probability
            else:
                y_proba_1d = y_proba.ravel() if y_proba.ndim > 1 else y_proba
            groups = pd.cut(y_proba_1d, bins=self.n_bins, labels=False)
        
        stability_metrics = {}
        
        # Calculate metrics for each group
        group_metrics = {}
        for group_id in np.unique(groups):
            if pd.isna(group_id):
                continue
                
            group_mask = groups == group_id
            group_y_true = y_true[group_mask]
            group_y_proba = y_proba[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            # Ensure group_y_proba is 1D for AUC calculation
            if group_y_proba.ndim > 1 and group_y_proba.shape[1] > 1:
                group_y_proba_1d = group_y_proba[:, 1]  # Take positive class probability
            else:
                group_y_proba_1d = group_y_proba.ravel() if group_y_proba.ndim > 1 else group_y_proba
            
            # Calculate basic metrics for this group
            group_metrics[group_id] = {
                'size': len(group_y_true),
                'positive_rate': group_y_true.mean(),
                'mean_probability': group_y_proba_1d.mean(),
                'std_probability': group_y_proba_1d.std(),
                'auc': roc_auc_score(group_y_true, group_y_proba_1d) if len(np.unique(group_y_true)) > 1 else 0.5
            }
        
        # Calculate stability metrics
        if len(group_metrics) > 1:
            # Calculate PSI (Population Stability Index)
            psi = self._calculate_psi(y_proba, groups)
            
            # Calculate variance in metrics across groups
            aucs = [metrics['auc'] for metrics in group_metrics.values()]
            positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
            
            stability_metrics = {
                'population_stability_index': psi,
                'auc_variance': np.var(aucs),
                'auc_std': np.std(aucs),
                'positive_rate_variance': np.var(positive_rates),
                'positive_rate_std': np.std(positive_rates),
                'group_metrics': group_metrics,
                'n_groups': len(group_metrics)
            }
        else:
            stability_metrics = {
                'population_stability_index': 0.0,
                'auc_variance': 0.0,
                'auc_std': 0.0,
                'positive_rate_variance': 0.0,
                'positive_rate_std': 0.0,
                'group_metrics': group_metrics,
                'n_groups': len(group_metrics)
            }
        
        return stability_metrics
    
    def _calculate_psi(self, y_proba: np.ndarray, groups: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        try:
            # Calculate probability distributions for each group
            group_distributions = {}
            for group_id in np.unique(groups):
                if pd.isna(group_id):
                    continue
                group_mask = groups == group_id
                group_proba = y_proba[group_mask]
                
                # Create histogram
                hist, bin_edges = np.histogram(group_proba, bins=self.n_bins, range=(0, 1))
                group_distributions[group_id] = hist / len(group_proba)
            
            # Calculate PSI between all pairs of groups
            psi_values = []
            group_ids = list(group_distributions.keys())
            
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    dist1 = group_distributions[group_ids[i]]
                    dist2 = group_distributions[group_ids[j]]
                    
                    # Calculate PSI
                    psi = 0
                    for k in range(len(dist1)):
                        if dist1[k] > 0 and dist2[k] > 0:
                            psi += (dist1[k] - dist2[k]) * np.log(dist1[k] / dist2[k])
                    
                    psi_values.append(psi)
            
            return np.mean(psi_values) if psi_values else 0.0
            
        except Exception:
            return 0.0
