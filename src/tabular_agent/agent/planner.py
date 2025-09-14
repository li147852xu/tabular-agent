"""Planner module for tabular-agent v0.2 with LLM+rule hybrid approach."""

import json
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pydantic
from pydantic import BaseModel, Field, field_validator

# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .kb.index import KnowledgeBase


class PlannerMode(str, Enum):
    """Planner execution modes."""
    LLM = "llm"
    RULES = "rules"
    AUTO = "auto"


class FeatureRecipe(str, Enum):
    """Allowed feature engineering recipes."""
    TARGET_ENCODING = "target_encoding"
    WOE_ENCODING = "woe_encoding"
    ROLLING_FEATURES = "rolling_features"
    TIME_FEATURES = "time_features"
    CROSS_FEATURES = "cross_features"
    SCALING = "scaling"
    VARIANCE_SELECTION = "variance_selection"
    CORRELATION_SELECTION = "correlation_selection"
    IV_SELECTION = "iv_selection"


class ModelType(str, Enum):
    """Allowed model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LINEAR = "linear"
    HISTGBDT = "histgbdt"


class BlendingStrategy(str, Enum):
    """Allowed blending strategies."""
    MEAN = "mean"
    RANK_MEAN = "rank_mean"
    LOGIT_MEAN = "logit_mean"
    STACKING = "stacking"


class OptunaSearchSpace(str, Enum):
    """Allowed Optuna search spaces."""
    LIGHTGBM_DEFAULT = "lightgbm_default"
    XGBOOST_DEFAULT = "xgboost_default"
    CATBOOST_DEFAULT = "catboost_default"
    LINEAR_DEFAULT = "linear_default"
    HISTGBDT_DEFAULT = "histgbdt_default"


class Citation(BaseModel):
    """Citation reference to prior runs."""
    run_id: str = Field(..., description="Run ID from runs metadata")
    score: float = Field(..., description="Performance score")
    config: Dict[str, Any] = Field(..., description="Configuration used")
    dataset_similarity: float = Field(..., description="Dataset similarity score")
    reason: str = Field(..., description="Reason for citation")


class PlanningConfig(BaseModel):
    """Configuration for the planner."""
    mode: PlannerMode = Field(default=PlannerMode.AUTO, description="Planner mode")
    llm_endpoint: Optional[str] = Field(default=None, description="LLM endpoint URL")
    llm_key: Optional[str] = Field(default=None, description="LLM API key")
    max_citations: int = Field(default=3, description="Maximum number of citations")
    min_similarity_threshold: float = Field(default=0.7, description="Minimum similarity for citations")
    fallback_on_error: bool = Field(default=True, description="Fallback to rules on LLM error")
    
    @field_validator('llm_key', mode='before')
    @classmethod
    def get_llm_key_from_env(cls, v):
        """Get LLM key from environment if not provided."""
        return v or os.getenv('TABULAR_AGENT_LLM_KEY')
    
    @field_validator('llm_endpoint', mode='before')
    @classmethod
    def get_llm_endpoint_from_env(cls, v):
        """Get LLM endpoint from environment if not provided."""
        return v or os.getenv('TABULAR_AGENT_LLM_ENDPOINT')


class PlanningResult(BaseModel):
    """Result of planning process."""
    success: bool = Field(..., description="Whether planning succeeded")
    mode_used: PlannerMode = Field(..., description="Mode actually used")
    plan: Dict[str, Any] = Field(..., description="Generated plan")
    citations: List[Citation] = Field(default=[], description="Citations to prior runs")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class Planner:
    """Hybrid planner with LLM+rule fallback and strict schema validation."""
    
    def __init__(self, config: PlanningConfig, knowledge_base: Optional['KnowledgeBase'] = None):
        """Initialize planner."""
        self.config = config
        self.kb = knowledge_base
        self.whitelist_actions = {
            'feature_recipes': [recipe.value for recipe in FeatureRecipe],
            'model_types': [model.value for model in ModelType],
            'blending_strategies': [strategy.value for strategy in BlendingStrategy],
            'optuna_spaces': [space.value for space in OptunaSearchSpace]
        }
    
    def plan(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> PlanningResult:
        """Generate execution plan based on data characteristics."""
        
        # Determine execution mode
        mode = self._determine_mode()
        
        if mode == PlannerMode.LLM:
            return self._plan_with_llm(data_schema, profile_summary, constraints)
        else:
            return self._plan_with_rules(data_schema, profile_summary, constraints)
    
    def _determine_mode(self) -> PlannerMode:
        """Determine which mode to use based on config and availability."""
        if self.config.mode == PlannerMode.RULES:
            return PlannerMode.RULES
        elif self.config.mode == PlannerMode.LLM:
            if self._is_llm_available():
                return PlannerMode.LLM
            else:
                return PlannerMode.RULES
        else:  # AUTO
            if self._is_llm_available():
                return PlannerMode.LLM
            else:
                return PlannerMode.RULES
    
    def _is_llm_available(self) -> bool:
        """Check if LLM is available."""
        return bool(self.config.llm_endpoint and self.config.llm_key)
    
    def _plan_with_llm(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> PlanningResult:
        """Generate plan using LLM with strict validation."""
        try:
            # Get citations from knowledge base
            citations = []
            if self.kb:
                citations = self.kb.query_similar_runs(
                    data_schema, profile_summary, 
                    max_results=self.config.max_citations,
                    min_similarity=self.config.min_similarity_threshold
                )
            
            # Prepare context for LLM
            context = self._prepare_llm_context(data_schema, profile_summary, constraints, citations)
            
            # Call LLM (mock implementation for now)
            llm_response = self._call_llm(context)
            
            # Parse and validate LLM response
            plan = self._parse_and_validate_llm_response(llm_response)
            
            return PlanningResult(
                success=True,
                mode_used=PlannerMode.LLM,
                plan=plan,
                citations=citations
            )
            
        except Exception as e:
            if self.config.fallback_on_error:
                return self._plan_with_rules(
                    data_schema, profile_summary, constraints,
                    fallback_reason=f"LLM error: {str(e)}"
                )
            else:
                return PlanningResult(
                    success=False,
                    mode_used=PlannerMode.LLM,
                    plan={},
                    citations=[],
                    error_message=str(e)
                )
    
    def _plan_with_rules(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        constraints: Dict[str, Any],
        fallback_reason: Optional[str] = None
    ) -> PlanningResult:
        """Generate plan using rule-based approach."""
        
        # Get citations from knowledge base
        citations = []
        if self.kb:
            citations = self.kb.query_similar_runs(
                data_schema, profile_summary,
                max_results=self.config.max_citations,
                min_similarity=self.config.min_similarity_threshold
            )
        
        # Rule-based planning logic
        plan = self._generate_rule_based_plan(data_schema, profile_summary, constraints)
        
        return PlanningResult(
            success=True,
            mode_used=PlannerMode.RULES,
            plan=plan,
            citations=citations,
            fallback_reason=fallback_reason
        )
    
    def _prepare_llm_context(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        constraints: Dict[str, Any],
        citations: List[Citation]
    ) -> str:
        """Prepare context for LLM."""
        context = {
            "data_schema": data_schema,
            "profile_summary": profile_summary,
            "constraints": constraints,
            "citations": [citation.dict() for citation in citations],
            "whitelist_actions": self.whitelist_actions
        }
        return json.dumps(context, indent=2)
    
    def _call_llm(self, context: str) -> str:
        """Call LLM with context (mock implementation)."""
        # Mock LLM response for testing
        # In real implementation, this would call the actual LLM API
        mock_response = {
            "feature_recipes": ["target_encoding", "woe_encoding", "rolling_features"],
            "model_types": ["lightgbm", "xgboost"],
            "blending_strategy": "mean",
            "optuna_search_space": "lightgbm_default",
            "time_budget_allocation": {
                "feature_engineering": 0.3,
                "model_training": 0.6,
                "hyperparameter_tuning": 0.1
            }
        }
        return json.dumps(mock_response)
    
    def _parse_and_validate_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response against whitelist."""
        try:
            plan = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        
        # Validate against whitelist
        validated_plan = {}
        
        # Validate feature recipes
        if 'feature_recipes' in plan:
            recipes = plan['feature_recipes']
            if not isinstance(recipes, list):
                raise ValueError("feature_recipes must be a list")
            validated_recipes = []
            for recipe in recipes:
                if recipe not in self.whitelist_actions['feature_recipes']:
                    raise ValueError(f"Invalid feature recipe: {recipe}")
                validated_recipes.append(recipe)
            validated_plan['feature_recipes'] = validated_recipes
        
        # Validate model types
        if 'model_types' in plan:
            models = plan['model_types']
            if not isinstance(models, list):
                raise ValueError("model_types must be a list")
            validated_models = []
            for model in models:
                if model not in self.whitelist_actions['model_types']:
                    raise ValueError(f"Invalid model type: {model}")
                validated_models.append(model)
            validated_plan['model_types'] = validated_models
        
        # Validate blending strategy
        if 'blending_strategy' in plan:
            strategy = plan['blending_strategy']
            if strategy not in self.whitelist_actions['blending_strategies']:
                raise ValueError(f"Invalid blending strategy: {strategy}")
            validated_plan['blending_strategy'] = strategy
        
        # Validate optuna search space
        if 'optuna_search_space' in plan:
            space = plan['optuna_search_space']
            if space not in self.whitelist_actions['optuna_spaces']:
                raise ValueError(f"Invalid optuna search space: {space}")
            validated_plan['optuna_search_space'] = space
        
        # Add other validated fields
        for key in ['time_budget_allocation', 'n_jobs', 'cv_folds']:
            if key in plan:
                validated_plan[key] = plan[key]
        
        return validated_plan
    
    def _generate_rule_based_plan(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate plan using rule-based logic."""
        
        plan = {
            "feature_recipes": [],
            "model_types": [],
            "blending_strategy": "mean",
            "optuna_search_space": "lightgbm_default",
            "time_budget_allocation": {
                "feature_engineering": 0.3,
                "model_training": 0.6,
                "hyperparameter_tuning": 0.1
            }
        }
        
        # Rule-based feature recipe selection
        if profile_summary.get('has_time_column', False):
            plan["feature_recipes"].extend(["time_features", "rolling_features"])
        
        if profile_summary.get('categorical_columns', 0) > 0:
            plan["feature_recipes"].extend(["target_encoding", "woe_encoding"])
        
        if profile_summary.get('numerical_columns', 0) > 5:
            plan["feature_recipes"].extend(["variance_selection", "correlation_selection"])
        
        plan["feature_recipes"].extend(["scaling", "cross_features"])
        
        # Rule-based model selection
        n_samples = profile_summary.get('n_samples', 0)
        if n_samples < 1000:
            plan["model_types"] = ["lightgbm", "linear"]
        elif n_samples < 10000:
            plan["model_types"] = ["lightgbm", "xgboost", "linear"]
        else:
            plan["model_types"] = ["lightgbm", "xgboost", "catboost", "histgbdt"]
        
        # Rule-based blending strategy
        if len(plan["model_types"]) > 2:
            plan["blending_strategy"] = "rank_mean"
        
        return plan
