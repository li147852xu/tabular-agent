"""Tests for Reflector risk analysis functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from tabular_agent.agent.reflector import (
    Reflector, ReflectorConfig, RiskLevel, RiskType, RetryAction
)


class TestReflector:
    """Test Reflector functionality."""
    
    def test_reflector_initialization(self):
        """Test Reflector initialization."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        assert reflector.config == config
        assert reflector.risk_policy is not None
        assert 'risk_thresholds' in reflector.risk_policy
    
    def test_analyze_overfitting_high_risk(self):
        """Test overfitting analysis with high risk."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        model_results = {
            'train_metrics': {'auc': 0.95},
            'val_metrics': {'auc': 0.75}  # Larger gap to trigger high risk
        }
        
        risks = reflector.analyze_risks(model_results)
        
        assert len(risks) == 1
        assert risks[0].risk_type == RiskType.OVERFITTING
        assert risks[0].level == RiskLevel.HIGH
        assert risks[0].score > 0.1  # High gap
    
    def test_analyze_overfitting_no_risk(self):
        """Test overfitting analysis with no risk."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        model_results = {
            'train_metrics': {'auc': 0.85},
            'val_metrics': {'auc': 0.84}
        }
        
        risks = reflector.analyze_risks(model_results)
        
        # Should not detect overfitting
        overfitting_risks = [r for r in risks if r.risk_type == RiskType.OVERFITTING]
        assert len(overfitting_risks) == 0
    
    def test_analyze_leakage_high_risk(self):
        """Test leakage analysis with high risk."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        audit_results = {
            'status': 'success',
            'leakage_indicators': {
                'duplicate_rows': {
                    'is_leakage': True,
                    'ratio': 0.08  # Higher ratio to trigger high risk
                }
            }
        }
        
        risks = reflector.analyze_risks({}, audit_results=audit_results)
        
        assert len(risks) == 1
        assert risks[0].risk_type == RiskType.LEAKAGE
        assert risks[0].level == RiskLevel.HIGH
        assert risks[0].score > 0.05
    
    def test_analyze_instability_high_risk(self):
        """Test instability analysis with high risk."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        stability_results = {
            'cv_scores': [0.8, 0.4, 0.9, 0.3, 0.5]  # Very high variance
        }
        
        risks = reflector.analyze_risks({}, stability_results=stability_results)
        
        assert len(risks) == 1
        assert risks[0].risk_type == RiskType.INSTABILITY
        assert risks[0].level == RiskLevel.HIGH
    
    def test_analyze_calibration_high_risk(self):
        """Test calibration analysis with high risk."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        calibration_results = {
            'calibration_error': 0.16  # Higher error to trigger high risk
        }
        
        risks = reflector.analyze_risks({}, calibration_results=calibration_results)
        
        assert len(risks) == 1
        assert risks[0].risk_type == RiskType.CALIBRATION
        assert risks[0].level == RiskLevel.HIGH
    
    def test_generate_retry_suggestions(self):
        """Test retry suggestions generation."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        # Create mock risks
        from tabular_agent.agent.reflector import RiskIndicator
        risks = [
            RiskIndicator(
                risk_type=RiskType.OVERFITTING,
                level=RiskLevel.HIGH,
                score=0.15,
                description="High overfitting",
                evidence={'gap': 0.15},
                suggestions=[RetryAction.SHRINK_SEARCH_SPACE, RetryAction.ADD_REGULARIZATION]
            )
        ]
        
        suggestions = reflector.generate_retry_suggestions(risks)
        
        assert suggestions['priority'] == 'high'
        assert len(suggestions['suggestions']) > 0
        assert suggestions['total_risks'] == 1
        assert suggestions['high_risks'] == 1
    
    def test_calculate_stability_metrics(self):
        """Test stability metrics calculation."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        scores = [0.8, 0.82, 0.79, 0.81, 0.80]
        metrics = reflector.calculate_stability_metrics(scores)
        
        assert metrics.mean_score == pytest.approx(0.804, abs=0.001)
        assert metrics.std_score > 0
        assert metrics.min_score == 0.79
        assert metrics.max_score == 0.82
        assert len(metrics.confidence_interval) == 2
        assert metrics.coefficient_of_variation > 0
    
    def test_risk_policy_loading(self):
        """Test risk policy loading from file."""
        config = ReflectorConfig(risk_policy_path="conf/risk_policy.yaml")
        reflector = Reflector(config)
        
        assert reflector.risk_policy is not None
        assert 'risk_thresholds' in reflector.risk_policy
        assert 'retry_suggestions' in reflector.risk_policy
    
    def test_action_descriptions(self):
        """Test action description generation."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        description = reflector._get_action_description("shrink_search_space")
        assert "Reduce hyperparameter" in description
        
        implementation = reflector._get_action_implementation("shrink_search_space")
        assert "max_depth" in implementation
    
    def test_multiple_risks_analysis(self):
        """Test analysis with multiple risk types."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        model_results = {
            'train_metrics': {'auc': 0.95},
            'val_metrics': {'auc': 0.80}
        }
        
        audit_results = {
            'status': 'success',
            'leakage_indicators': {
                'duplicate_rows': {
                    'is_leakage': True,
                    'ratio': 0.1
                }
            }
        }
        
        stability_results = {
            'cv_scores': [0.8, 0.6, 0.9, 0.7, 0.5]
        }
        
        risks = reflector.analyze_risks(
            model_results, stability_results, None, audit_results
        )
        
        # Should detect multiple risks
        assert len(risks) >= 2
        
        risk_types = [r.risk_type for r in risks]
        assert RiskType.OVERFITTING in risk_types
        assert RiskType.LEAKAGE in risk_types
        assert RiskType.INSTABILITY in risk_types
    
    def test_no_risks_detected(self):
        """Test when no risks are detected."""
        config = ReflectorConfig()
        reflector = Reflector(config)
        
        model_results = {
            'train_metrics': {'auc': 0.85},
            'val_metrics': {'auc': 0.84}
        }
        
        audit_results = {
            'status': 'success',
            'leakage_indicators': {
                'duplicate_rows': {'is_leakage': False}
            }
        }
        
        stability_results = {
            'cv_scores': [0.84, 0.85, 0.83, 0.86, 0.84]
        }
        
        risks = reflector.analyze_risks(
            model_results, stability_results, None, audit_results
        )
        
        # Should detect no risks
        assert len(risks) == 0
