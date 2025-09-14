"""Stability evaluation module for tabular-agent v0.3."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib

from ..models.trainers import ModelTrainer
from ..tune.optuna import OptunaTuner
from ..evaluate.metrics import MetricsCalculator
from ...agent.reflector import StabilityMetrics


class StabilityEvaluator:
    """Evaluator for model stability across multiple runs."""
    
    def __init__(self, n_runs: int = 5, random_seeds: Optional[List[int]] = None):
        """
        Initialize stability evaluator.
        
        Args:
            n_runs: Number of stability evaluation runs
            random_seeds: List of random seeds for reproducibility
        """
        self.n_runs = n_runs
        self.random_seeds = random_seeds or list(range(42, 42 + n_runs))
        
        if len(self.random_seeds) != n_runs:
            self.random_seeds = list(range(42, 42 + n_runs))
    
    def evaluate_stability(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_config: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate model stability across multiple runs.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_config: Model configuration
            feature_importance: Feature importance for analysis
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing stability results
        """
        print(f"Running stability evaluation with {self.n_runs} runs...")
        
        # Run stability evaluation
        if n_jobs > 1:
            results = self._evaluate_parallel(
                X_train, y_train, X_test, y_test, model_config, n_jobs
            )
        else:
            results = self._evaluate_sequential(
                X_train, y_train, X_test, y_test, model_config
            )
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(results)
        
        # Analyze feature importance stability
        feature_stability = self._analyze_feature_stability(results, feature_importance)
        
        # Calculate additional metrics for template compatibility
        auc_scores = [r['metrics'].get('auc', 0.0) for r in results if r.get('success', False)]
        positive_rates = [r['metrics'].get('positive_rate', 0.0) for r in results if r.get('success', False)]
        
        return {
            "n_runs": self.n_runs,
            "seeds": self.random_seeds,
            "results": results,
            "stability_metrics": stability_metrics,
            "feature_stability": feature_stability,
            "summary": self._generate_summary(stability_metrics, feature_stability),
            "population_stability_index": stability_metrics.get('coefficient_of_variation', 0.0),
            "auc_std": np.std(auc_scores) if auc_scores else 0.0,
            "positive_rate_std": np.std(positive_rates) if positive_rates else 0.0
        }
    
    def _evaluate_sequential(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run stability evaluation sequentially."""
        results = []
        
        for i, seed in enumerate(self.random_seeds):
            print(f"  Run {i+1}/{self.n_runs} (seed={seed})...")
            
            # Set random seed
            np.random.seed(seed)
            
            # Train model
            start_time = time.time()
            model_result = self._train_single_model(
                X_train, y_train, X_test, y_test, model_config, seed
            )
            training_time = time.time() - start_time
            
            model_result['seed'] = seed
            model_result['training_time'] = training_time
            results.append(model_result)
        
        return results
    
    def _evaluate_parallel(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_config: Dict[str, Any],
        n_jobs: int
    ) -> List[Dict[str, Any]]:
        """Run stability evaluation in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_seed = {
                executor.submit(
                    self._train_single_model,
                    X_train, y_train, X_test, y_test, model_config, seed
                ): seed for seed in self.random_seeds
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_seed)):
                seed = future_to_seed[future]
                print(f"  Run {i+1}/{self.n_runs} (seed={seed}) completed...")
                
                try:
                    model_result = future.result()
                    model_result['seed'] = seed
                    results.append(model_result)
                except Exception as e:
                    print(f"  Warning: Run with seed {seed} failed: {e}")
                    # Add failed result for consistency
                    results.append({
                        'seed': seed,
                        'success': False,
                        'error': str(e),
                        'metrics': {},
                        'feature_importance': {}
                    })
        
        return results
    
    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_config: Dict[str, Any],
        seed: int
    ) -> Dict[str, Any]:
        """Train a single model for stability evaluation."""
        try:
            # Set random seed
            np.random.seed(seed)
            
            # Create model trainer
            trainer = ModelTrainer(model_config)
            
            # Train model
            model = trainer.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(X_train.columns, np.abs(model.coef_[0])))
            
            return {
                'success': True,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'model_type': model_config.get('type', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metrics': {},
                'feature_importance': {}
            }
    
    def _calculate_stability_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate stability metrics from results."""
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'cv_scores': [],
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'coefficient_of_variation': 0.0,
                'confidence_interval': (0.0, 0.0),
                'stability_grade': 'F'
            }
        
        # Extract AUC scores
        cv_scores = [r['metrics'].get('auc', 0.0) for r in successful_results]
        cv_scores = np.array(cv_scores)
        
        # Calculate basic statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        min_score = np.min(cv_scores)
        max_score = np.max(cv_scores)
        
        # Calculate coefficient of variation
        cv_coefficient = std_score / mean_score if mean_score > 0 else 0
        
        # Calculate confidence interval (95%)
        n = len(cv_scores)
        if n > 1:
            t_value = 1.96  # Approximate for 95% confidence
            margin_error = t_value * (std_score / np.sqrt(n))
            confidence_interval = (mean_score - margin_error, mean_score + margin_error)
        else:
            confidence_interval = (mean_score, mean_score)
        
        # Calculate stability grade
        stability_grade = self._calculate_stability_grade(cv_coefficient)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'coefficient_of_variation': cv_coefficient,
            'confidence_interval': confidence_interval,
            'stability_grade': stability_grade,
            'n_successful_runs': len(successful_results),
            'n_failed_runs': len(results) - len(successful_results)
        }
    
    def _calculate_stability_grade(self, cv_coefficient: float) -> str:
        """Calculate stability grade based on coefficient of variation."""
        if cv_coefficient < 0.05:
            return 'A'  # Excellent
        elif cv_coefficient < 0.10:
            return 'B'  # Good
        elif cv_coefficient < 0.15:
            return 'C'  # Fair
        elif cv_coefficient < 0.20:
            return 'D'  # Poor
        else:
            return 'F'  # Very Poor
    
    def _analyze_feature_stability(
        self, 
        results: List[Dict[str, Any]], 
        reference_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze feature importance stability across runs."""
        # Filter successful results with feature importance
        successful_results = [
            r for r in results 
            if r.get('success', False) and r.get('feature_importance')
        ]
        
        if not successful_results:
            return {
                'feature_importance_stability': {},
                'top_features': [],
                'stability_summary': 'No feature importance data available'
            }
        
        # Collect feature importance across runs
        feature_importance_runs = [r['feature_importance'] for r in successful_results]
        
        # Calculate stability for each feature
        feature_stability = {}
        all_features = set()
        for run_importance in feature_importance_runs:
            all_features.update(run_importance.keys())
        
        for feature in all_features:
            importance_values = []
            for run_importance in feature_importance_runs:
                if feature in run_importance:
                    importance_values.append(run_importance[feature])
            
            if importance_values:
                importance_values = np.array(importance_values)
                mean_importance = np.mean(importance_values)
                std_importance = np.std(importance_values)
                cv_importance = std_importance / mean_importance if mean_importance > 0 else 0
                
                feature_stability[feature] = {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'cv_importance': cv_importance,
                    'stability_grade': self._calculate_stability_grade(cv_importance)
                }
        
        # Sort features by mean importance
        sorted_features = sorted(
            feature_stability.items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        )
        
        # Get top features
        top_features = [f[0] for f in sorted_features[:10]]
        
        return {
            'feature_importance_stability': feature_stability,
            'top_features': top_features,
            'stability_summary': f"Analyzed {len(feature_stability)} features across {len(successful_results)} runs"
        }
    
    def _generate_summary(
        self, 
        stability_metrics: Dict[str, Any], 
        feature_stability: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of stability evaluation."""
        cv_coefficient = stability_metrics['coefficient_of_variation']
        stability_grade = stability_metrics['stability_grade']
        
        # Determine overall stability assessment
        if stability_grade in ['A', 'B']:
            overall_assessment = "Stable"
            recommendation = "Model shows good stability across runs"
        elif stability_grade == 'C':
            overall_assessment = "Moderately Stable"
            recommendation = "Consider additional regularization or more data"
        elif stability_grade == 'D':
            overall_assessment = "Unstable"
            recommendation = "Model needs improvement - consider different approach"
        else:
            overall_assessment = "Very Unstable"
            recommendation = "Model is unreliable - significant changes needed"
        
        return {
            'overall_assessment': overall_assessment,
            'stability_grade': stability_grade,
            'cv_coefficient': cv_coefficient,
            'recommendation': recommendation,
            'confidence_interval': stability_metrics['confidence_interval'],
            'n_successful_runs': stability_metrics.get('n_successful_runs', 0),
            'n_failed_runs': stability_metrics.get('n_failed_runs', 0)
        }
