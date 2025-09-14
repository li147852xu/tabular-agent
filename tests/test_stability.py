"""Tests for stability evaluation functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from tabular_agent.core.evaluate.stability import StabilityEvaluator


class TestStabilityEvaluator:
    """Test stability evaluation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_test = pd.Series(np.random.randint(0, 2, 20))
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.model_config = {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
        }
    
    def test_stability_evaluator_initialization(self):
        """Test StabilityEvaluator initialization."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        assert evaluator.n_runs == 5
        assert len(evaluator.random_seeds) == 5
        assert evaluator.random_seeds[0] == 42
    
    def test_stability_evaluator_custom_seeds(self):
        """Test StabilityEvaluator with custom seeds."""
        custom_seeds = [1, 2, 3, 4, 5]
        evaluator = StabilityEvaluator(n_runs=5, random_seeds=custom_seeds)
        
        assert evaluator.random_seeds == custom_seeds
    
    def test_calculate_stability_metrics(self):
        """Test stability metrics calculation."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        scores = [0.8, 0.82, 0.79, 0.81, 0.80]
        metrics = evaluator._calculate_stability_metrics([
            {'success': True, 'metrics': {'auc': score}} for score in scores
        ])
        
        assert metrics['mean_score'] == pytest.approx(0.804, abs=0.001)
        assert metrics['std_score'] > 0
        assert metrics['min_score'] == 0.79
        assert metrics['max_score'] == 0.82
        assert metrics['coefficient_of_variation'] > 0
        assert metrics['stability_grade'] in ['A', 'B', 'C', 'D', 'F']
        assert metrics['n_successful_runs'] == 5
        assert metrics['n_failed_runs'] == 0
    
    def test_calculate_stability_metrics_with_failures(self):
        """Test stability metrics calculation with failed runs."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        results = [
            {'success': True, 'metrics': {'auc': 0.8}},
            {'success': False, 'error': 'Training failed'},
            {'success': True, 'metrics': {'auc': 0.82}},
            {'success': True, 'metrics': {'auc': 0.79}},
            {'success': False, 'error': 'Training failed'}
        ]
        
        metrics = evaluator._calculate_stability_metrics(results)
        
        assert metrics['n_successful_runs'] == 3
        assert metrics['n_failed_runs'] == 2
        assert len(metrics['cv_scores']) == 3
    
    def test_calculate_stability_grade(self):
        """Test stability grade calculation."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        # Test different CV coefficients
        assert evaluator._calculate_stability_grade(0.03) == 'A'  # Excellent
        assert evaluator._calculate_stability_grade(0.08) == 'B'  # Good
        assert evaluator._calculate_stability_grade(0.12) == 'C'  # Fair
        assert evaluator._calculate_stability_grade(0.18) == 'D'  # Poor
        assert evaluator._calculate_stability_grade(0.25) == 'F'  # Very Poor
    
    def test_analyze_feature_stability(self):
        """Test feature importance stability analysis."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        # Mock results with feature importance
        results = [
            {
                'success': True,
                'feature_importance': {
                    'feature_0': 0.3,
                    'feature_1': 0.2,
                    'feature_2': 0.1
                }
            },
            {
                'success': True,
                'feature_importance': {
                    'feature_0': 0.32,
                    'feature_1': 0.18,
                    'feature_2': 0.12
                }
            },
            {
                'success': True,
                'feature_importance': {
                    'feature_0': 0.28,
                    'feature_1': 0.22,
                    'feature_2': 0.08
                }
            }
        ]
        
        feature_stability = evaluator._analyze_feature_stability(results)
        
        assert 'feature_importance_stability' in feature_stability
        assert 'top_features' in feature_stability
        assert 'stability_summary' in feature_stability
        
        # Check feature stability data
        stability_data = feature_stability['feature_importance_stability']
        assert 'feature_0' in stability_data
        assert 'mean_importance' in stability_data['feature_0']
        assert 'cv_importance' in stability_data['feature_0']
        assert 'stability_grade' in stability_data['feature_0']
    
    def test_generate_summary(self):
        """Test summary generation."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        stability_metrics = {
            'coefficient_of_variation': 0.08,
            'stability_grade': 'B',
            'confidence_interval': (0.75, 0.85),
            'n_successful_runs': 5,
            'n_failed_runs': 0
        }
        
        feature_stability = {
            'stability_summary': 'Analyzed 3 features across 5 runs'
        }
        
        summary = evaluator._generate_summary(stability_metrics, feature_stability)
        
        assert 'overall_assessment' in summary
        assert 'stability_grade' in summary
        assert 'cv_coefficient' in summary
        assert 'recommendation' in summary
        assert 'confidence_interval' in summary
        assert 'n_successful_runs' in summary
        assert 'n_failed_runs' in summary
    
    @patch('tabular_agent.core.evaluate.stability.ModelTrainer')
    @patch('tabular_agent.core.evaluate.stability.MetricsCalculator')
    def test_train_single_model_success(self, mock_metrics_class, mock_trainer_class):
        """Test successful single model training."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_model
        mock_trainer_class.return_value = mock_trainer
        
        # Mock metrics calculator
        mock_metrics_calc = Mock()
        mock_metrics_calc.calculate_metrics.return_value = {'auc': 0.8, 'accuracy': 0.75}
        mock_metrics_class.return_value = mock_metrics_calc
        
        result = evaluator._train_single_model(
            self.X_train, self.y_train, self.X_test, self.y_test,
            self.model_config, 42
        )
        
        assert result['success'] is True
        assert 'metrics' in result
        assert 'feature_importance' in result
        assert result['model_type'] == 'lightgbm'
    
    @patch('tabular_agent.core.evaluate.stability.ModelTrainer')
    def test_train_single_model_failure(self, mock_trainer_class):
        """Test failed single model training."""
        evaluator = StabilityEvaluator(n_runs=5)
        
        # Mock trainer to raise exception
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer
        
        result = evaluator._train_single_model(
            self.X_train, self.y_train, self.X_test, self.y_test,
            self.model_config, 42
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error'] == "Training failed"
        assert 'metrics' in result
        assert 'feature_importance' in result
    
    def test_evaluate_stability_sequential(self):
        """Test sequential stability evaluation."""
        evaluator = StabilityEvaluator(n_runs=3)
        
        with patch.object(evaluator, '_train_single_model') as mock_train:
            # Mock successful training
            mock_train.return_value = {
                'success': True,
                'metrics': {'auc': 0.8},
                'feature_importance': {'feature_0': 0.3, 'feature_1': 0.2}
            }
            
            results = evaluator._evaluate_sequential(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.model_config
            )
            
            assert len(results) == 3
            assert all(r['success'] for r in results)
            assert mock_train.call_count == 3
    
    def test_evaluate_stability_parallel(self):
        """Test parallel stability evaluation."""
        evaluator = StabilityEvaluator(n_runs=3)
        
        with patch.object(evaluator, '_train_single_model') as mock_train:
            # Mock successful training
            mock_train.return_value = {
                'success': True,
                'metrics': {'auc': 0.8},
                'feature_importance': {'feature_0': 0.3, 'feature_1': 0.2}
            }
            
            results = evaluator._evaluate_parallel(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.model_config, n_jobs=2
            )
            
            assert len(results) == 3
            assert all(r['success'] for r in results)
            assert mock_train.call_count == 3
    
    @patch('tabular_agent.core.evaluate.stability.ModelTrainer')
    def test_evaluate_stability_integration(self, mock_trainer_class):
        """Test full stability evaluation integration."""
        evaluator = StabilityEvaluator(n_runs=3)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_model
        mock_trainer_class.return_value = mock_trainer
        
        results = evaluator.evaluate_stability(
            self.X_train, self.y_train, self.X_test, self.y_test,
            self.model_config, n_jobs=1
        )
        
        assert 'n_runs' in results
        assert 'seeds' in results
        assert 'results' in results
        assert 'stability_metrics' in results
        assert 'feature_stability' in results
        assert 'summary' in results
        
        assert results['n_runs'] == 3
        assert len(results['seeds']) == 3
        assert len(results['results']) == 3
