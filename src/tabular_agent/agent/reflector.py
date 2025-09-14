"""Reflector module for risk analysis and retry suggestions in tabular-agent v0.3."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pathlib import Path
import json
import yaml
from dataclasses import dataclass

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk levels for model analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskType(str, Enum):
    """Types of risks to analyze."""
    OVERFITTING = "overfitting"
    LEAKAGE = "leakage"
    INSTABILITY = "instability"
    CALIBRATION = "calibration"
    DATA_DRIFT = "data_drift"


class RetryAction(str, Enum):
    """Actions to take for retry suggestions."""
    SHRINK_SEARCH_SPACE = "shrink_search_space"
    ADD_REGULARIZATION = "add_regularization"
    CHANGE_CV_STRATEGY = "change_cv_strategy"
    REMOVE_FEATURES = "remove_features"
    RECALCULATE_FEATURES = "recalculate_features"
    INCREASE_DATA = "increase_data"
    CHANGE_MODEL = "change_model"


@dataclass
class RiskIndicator:
    """Risk indicator with score and details."""
    risk_type: RiskType
    level: RiskLevel
    score: float
    description: str
    evidence: Dict[str, Any]
    suggestions: List[RetryAction]


@dataclass
class StabilityMetrics:
    """Stability metrics from repeated evaluation."""
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    confidence_interval: Tuple[float, float]
    coefficient_of_variation: float


class ReflectorConfig(BaseModel):
    """Configuration for the Reflector."""
    risk_thresholds: Dict[str, float] = Field(
        default={
            "overfitting": 0.1,  # Train vs OOF difference
            "leakage": 0.05,     # Leakage score threshold
            "instability": 0.15, # CV coefficient of variation
            "calibration": 0.1,  # Calibration error threshold
            "data_drift": 0.2    # PSI threshold
        },
        description="Risk thresholds for different risk types"
    )
    stability_runs: int = Field(default=5, description="Number of stability evaluation runs")
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")
    enable_retry_suggestions: bool = Field(default=True, description="Enable retry suggestions")
    risk_policy_path: Optional[str] = Field(default=None, description="Path to risk policy YAML file")


class Reflector:
    """Reflector for risk analysis and retry suggestions."""
    
    def __init__(self, config: ReflectorConfig):
        """Initialize Reflector."""
        self.config = config
        self.risk_policy = self._load_risk_policy()
    
    def _load_risk_policy(self) -> Dict[str, Any]:
        """Load risk policy from YAML file."""
        if self.config.risk_policy_path and Path(self.config.risk_policy_path).exists():
            with open(self.config.risk_policy_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_risk_policy()
    
    def _get_default_risk_policy(self) -> Dict[str, Any]:
        """Get default risk policy."""
        return {
            "risk_thresholds": self.config.risk_thresholds,
            "retry_suggestions": {
                "overfitting": [
                    "shrink_search_space",
                    "add_regularization",
                    "change_cv_strategy"
                ],
                "leakage": [
                    "remove_features",
                    "recalculate_features"
                ],
                "instability": [
                    "change_cv_strategy",
                    "add_regularization",
                    "increase_data"
                ],
                "calibration": [
                    "change_model",
                    "add_regularization"
                ],
                "data_drift": [
                    "recalculate_features",
                    "increase_data"
                ]
            },
            "risk_descriptions": {
                "overfitting": "Model shows signs of overfitting with significant performance gap between training and validation",
                "leakage": "Potential data leakage detected in features or data preparation",
                "instability": "Model performance varies significantly across different random seeds",
                "calibration": "Model predictions are poorly calibrated and unreliable",
                "data_drift": "Significant distribution shift detected between train and test data"
            }
        }
    
    def analyze_risks(
        self,
        model_results: Dict[str, Any],
        stability_results: Optional[Dict[str, Any]] = None,
        calibration_results: Optional[Dict[str, Any]] = None,
        audit_results: Optional[Dict[str, Any]] = None
    ) -> List[RiskIndicator]:
        """Analyze risks and return risk indicators."""
        risks = []
        
        # Analyze overfitting
        overfitting_risk = self._analyze_overfitting(model_results)
        if overfitting_risk:
            risks.append(overfitting_risk)
        
        # Analyze leakage
        if audit_results:
            leakage_risk = self._analyze_leakage(audit_results)
            if leakage_risk:
                risks.append(leakage_risk)
        
        # Analyze instability
        if stability_results:
            instability_risk = self._analyze_instability(stability_results)
            if instability_risk:
                risks.append(instability_risk)
        
        # Analyze calibration
        if calibration_results:
            calibration_risk = self._analyze_calibration(calibration_results)
            if calibration_risk:
                risks.append(calibration_risk)
        
        return risks
    
    def _analyze_overfitting(self, model_results: Dict[str, Any]) -> Optional[RiskIndicator]:
        """Analyze overfitting risk."""
        if 'train_metrics' not in model_results or 'val_metrics' not in model_results:
            return None
        
        train_auc = model_results['train_metrics'].get('auc', 0)
        val_auc = model_results['val_metrics'].get('auc', 0)
        
        if train_auc == 0 or val_auc == 0:
            return None
        
        gap = train_auc - val_auc
        threshold = self.config.risk_thresholds['overfitting']
        
        if gap > threshold:
            level = RiskLevel.HIGH if gap > threshold * 1.5 else RiskLevel.MEDIUM
            return RiskIndicator(
                risk_type=RiskType.OVERFITTING,
                level=level,
                score=gap,
                description=f"Training AUC ({train_auc:.3f}) significantly higher than validation AUC ({val_auc:.3f})",
                evidence={
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                    "gap": gap,
                    "threshold": threshold
                },
                suggestions=self.risk_policy['retry_suggestions']['overfitting']
            )
        
        return None
    
    def _analyze_leakage(self, audit_results: Dict[str, Any]) -> Optional[RiskIndicator]:
        """Analyze leakage risk."""
        if audit_results.get('status') != 'success':
            return None
        
        leakage_indicators = audit_results.get('leakage_indicators', {})
        max_leakage_score = 0
        leakage_evidence = {}
        
        # Check duplicate rows
        if leakage_indicators.get('duplicate_rows', {}).get('is_leakage', False):
            ratio = leakage_indicators['duplicate_rows'].get('ratio', 0)
            max_leakage_score = max(max_leakage_score, ratio)
            leakage_evidence['duplicate_ratio'] = ratio
        
        # Check target leakage
        if leakage_indicators.get('target_leakage', {}).get('is_leakage', False):
            suspicious_features = leakage_indicators['target_leakage'].get('suspicious_features', [])
            max_leakage_score = max(max_leakage_score, 0.5)  # Binary indicator
            leakage_evidence['suspicious_features'] = suspicious_features
        
        # Check time leakage
        if leakage_indicators.get('time_leakage', {}).get('is_leakage', False):
            max_leakage_score = max(max_leakage_score, 0.3)  # Binary indicator
            leakage_evidence['time_leakage'] = True
        
        threshold = self.config.risk_thresholds['leakage']
        
        if max_leakage_score > threshold:
            level = RiskLevel.HIGH if max_leakage_score > threshold * 1.5 else RiskLevel.MEDIUM
            return RiskIndicator(
                risk_type=RiskType.LEAKAGE,
                level=level,
                score=max_leakage_score,
                description=f"Data leakage detected with score {max_leakage_score:.3f}",
                evidence=leakage_evidence,
                suggestions=self.risk_policy['retry_suggestions']['leakage']
            )
        
        return None
    
    def _analyze_instability(self, stability_results: Dict[str, Any]) -> Optional[RiskIndicator]:
        """Analyze model instability risk."""
        if 'cv_scores' not in stability_results:
            return None
        
        cv_scores = stability_results['cv_scores']
        if len(cv_scores) < 2:
            return None
        
        cv_scores = np.array(cv_scores)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        cv_coefficient = std_score / mean_score if mean_score > 0 else 0
        
        threshold = self.config.risk_thresholds['instability']
        
        if cv_coefficient > threshold:
            level = RiskLevel.HIGH if cv_coefficient > threshold * 1.5 else RiskLevel.MEDIUM
            return RiskIndicator(
                risk_type=RiskType.INSTABILITY,
                level=level,
                score=cv_coefficient,
                description=f"Model instability detected with CV coefficient {cv_coefficient:.3f}",
                evidence={
                    "cv_scores": cv_scores.tolist(),
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "cv_coefficient": cv_coefficient
                },
                suggestions=self.risk_policy['retry_suggestions']['instability']
            )
        
        return None
    
    def _analyze_calibration(self, calibration_results: Dict[str, Any]) -> Optional[RiskIndicator]:
        """Analyze calibration risk."""
        if 'calibration_error' not in calibration_results:
            return None
        
        calibration_error = calibration_results['calibration_error']
        threshold = self.config.risk_thresholds['calibration']
        
        if calibration_error > threshold:
            level = RiskLevel.HIGH if calibration_error > threshold * 1.5 else RiskLevel.MEDIUM
            return RiskIndicator(
                risk_type=RiskType.CALIBRATION,
                level=level,
                score=calibration_error,
                description=f"Poor calibration detected with error {calibration_error:.3f}",
                evidence={
                    "calibration_error": calibration_error,
                    "threshold": threshold
                },
                suggestions=self.risk_policy['retry_suggestions']['calibration']
            )
        
        return None
    
    def generate_retry_suggestions(self, risks: List[RiskIndicator]) -> Dict[str, Any]:
        """Generate retry suggestions based on risk analysis."""
        if not self.config.enable_retry_suggestions:
            return {"suggestions": [], "priority": "none"}
        
        # Group suggestions by priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for risk in risks:
            if risk.level == RiskLevel.HIGH:
                high_priority.extend(risk.suggestions)
            elif risk.level == RiskLevel.MEDIUM:
                medium_priority.extend(risk.suggestions)
            else:
                low_priority.extend(risk.suggestions)
        
        # Remove duplicates and create detailed suggestions
        suggestions = []
        
        for action in set(high_priority):
            suggestions.append({
                "action": action,
                "priority": "high",
                "description": self._get_action_description(action),
                "implementation": self._get_action_implementation(action)
            })
        
        for action in set(medium_priority):
            if action not in [s["action"] for s in suggestions]:
                suggestions.append({
                    "action": action,
                    "priority": "medium",
                    "description": self._get_action_description(action),
                    "implementation": self._get_action_implementation(action)
                })
        
        for action in set(low_priority):
            if action not in [s["action"] for s in suggestions]:
                suggestions.append({
                    "action": action,
                    "priority": "low",
                    "description": self._get_action_description(action),
                    "implementation": self._get_action_implementation(action)
                })
        
        return {
            "suggestions": suggestions,
            "priority": "high" if high_priority else "medium" if medium_priority else "low",
            "total_risks": len(risks),
            "high_risks": len([r for r in risks if r.level == RiskLevel.HIGH])
        }
    
    def _get_action_description(self, action: str) -> str:
        """Get human-readable description for action."""
        descriptions = {
            "shrink_search_space": "Reduce hyperparameter search space to prevent overfitting",
            "add_regularization": "Add regularization (L1/L2) to reduce overfitting",
            "change_cv_strategy": "Use time-based or stratified CV to improve stability",
            "remove_features": "Remove suspicious features that may cause leakage",
            "recalculate_features": "Recalculate features with proper time windows",
            "increase_data": "Collect more training data to improve stability",
            "change_model": "Try a different model type (e.g., linear vs tree-based)"
        }
        return descriptions.get(action, f"Implement {action}")
    
    def _get_action_implementation(self, action: str) -> str:
        """Get implementation guidance for action."""
        implementations = {
            "shrink_search_space": "Reduce max_depth, min_samples_split, learning_rate ranges",
            "add_regularization": "Add alpha parameter for L1/L2 regularization",
            "change_cv_strategy": "Use TimeSeriesSplit or GroupKFold instead of KFold",
            "remove_features": "Remove features with high correlation to target or suspicious patterns",
            "recalculate_features": "Ensure feature calculation respects time boundaries",
            "increase_data": "Collect more historical data or use data augmentation",
            "change_model": "Switch between linear models, tree-based models, or neural networks"
        }
        return implementations.get(action, f"Implement {action} according to best practices")
    
    def calculate_stability_metrics(self, scores: List[float]) -> StabilityMetrics:
        """Calculate stability metrics from repeated evaluation scores."""
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Calculate confidence interval
        n = len(scores)
        if n > 1:
            t_value = 1.96  # Approximate for 95% confidence
            margin_error = t_value * (std_score / np.sqrt(n))
            confidence_interval = (mean_score - margin_error, mean_score + margin_error)
        else:
            confidence_interval = (mean_score, mean_score)
        
        coefficient_of_variation = std_score / mean_score if mean_score > 0 else 0
        
        return StabilityMetrics(
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            confidence_interval=confidence_interval,
            coefficient_of_variation=coefficient_of_variation
        )
