"""Tests for planner guard functionality."""

import pytest
import json
from unittest.mock import Mock, patch
from tabular_agent.agent.planner import Planner, PlanningConfig, PlannerMode


class TestPlannerGuard:
    """Test planner guard functionality."""
    
    def test_whitelist_validation(self):
        """Test that planner validates against whitelist."""
        config = PlanningConfig(mode=PlannerMode.RULES)
        planner = Planner(config)
        
        # Test valid feature recipes
        valid_recipes = ["target_encoding", "woe_encoding", "rolling_features"]
        for recipe in valid_recipes:
            assert recipe in planner.whitelist_actions['feature_recipes']
        
        # Test invalid feature recipes
        invalid_recipes = ["invalid_recipe", "custom_encoding"]
        for recipe in invalid_recipes:
            assert recipe not in planner.whitelist_actions['feature_recipes']
    
    def test_llm_response_validation(self):
        """Test LLM response validation against whitelist."""
        config = PlanningConfig(mode=PlannerMode.LLM)
        planner = Planner(config)
        
        # Valid response
        valid_response = {
            "feature_recipes": ["target_encoding", "woe_encoding"],
            "model_types": ["lightgbm", "xgboost"],
            "blending_strategy": "mean",
            "optuna_search_space": "lightgbm_default"
        }
        
        result = planner._parse_and_validate_llm_response(json.dumps(valid_response))
        assert result["feature_recipes"] == ["target_encoding", "woe_encoding"]
        assert result["model_types"] == ["lightgbm", "xgboost"]
        assert result["blending_strategy"] == "mean"
        assert result["optuna_search_space"] == "lightgbm_default"
        
        # Invalid response - should raise error
        invalid_response = {
            "feature_recipes": ["invalid_recipe"],
            "model_types": ["lightgbm"],
            "blending_strategy": "mean",
            "optuna_search_space": "lightgbm_default"
        }
        
        with pytest.raises(ValueError, match="Invalid feature recipe"):
            planner._parse_and_validate_llm_response(json.dumps(invalid_response))
    
    def test_fallback_on_llm_error(self):
        """Test fallback to rules when LLM fails."""
        config = PlanningConfig(mode=PlannerMode.LLM, fallback_on_error=True, llm_endpoint="http://test", llm_key="test")
        planner = Planner(config)
        
        data_schema = {"target": "y", "columns": ["x1", "x2"]}
        profile_summary = {"n_samples": 1000, "n_features": 2}
        constraints = {"time_budget": 300}
        
        # Mock LLM to raise error
        with patch.object(planner, '_call_llm', side_effect=Exception("LLM error")):
            result = planner.plan(data_schema, profile_summary, constraints)
            
            assert result.success
            assert result.mode_used == PlannerMode.RULES
            assert result.fallback_reason is not None
            assert "LLM error" in result.fallback_reason
    
    def test_rules_planning(self):
        """Test rule-based planning."""
        config = PlanningConfig(mode=PlannerMode.RULES)
        planner = Planner(config)
        
        data_schema = {"target": "y", "columns": ["x1", "x2"]}
        profile_summary = {
            "n_samples": 1000, 
            "n_features": 2,
            "categorical_columns": 1,
            "numerical_columns": 1,
            "has_time_column": False
        }
        constraints = {"time_budget": 300}
        
        result = planner.plan(data_schema, profile_summary, constraints)
        
        assert result.success
        assert result.mode_used == PlannerMode.RULES
        assert "feature_recipes" in result.plan
        assert "model_types" in result.plan
        assert "blending_strategy" in result.plan
        assert all(recipe in planner.whitelist_actions['feature_recipes'] 
                  for recipe in result.plan["feature_recipes"])
        assert all(model in planner.whitelist_actions['model_types'] 
                  for model in result.plan["model_types"])
    
    def test_auto_mode_selection(self):
        """Test automatic mode selection."""
        # Test with LLM available
        config = PlanningConfig(mode=PlannerMode.AUTO, llm_endpoint="http://test", llm_key="test")
        planner = Planner(config)
        
        data_schema = {"target": "y", "columns": ["x1", "x2"]}
        profile_summary = {"n_samples": 1000, "n_features": 2}
        constraints = {"time_budget": 300}
        
        with patch.object(planner, '_call_llm', return_value='{"feature_recipes": ["target_encoding"]}'):
            result = planner.plan(data_schema, profile_summary, constraints)
            assert result.mode_used == PlannerMode.LLM
        
        # Test without LLM available
        config_no_llm = PlanningConfig(mode=PlannerMode.AUTO)
        planner_no_llm = Planner(config_no_llm)
        
        result = planner_no_llm.plan(data_schema, profile_summary, constraints)
        assert result.mode_used == PlannerMode.RULES
    
    def test_planning_with_citations(self):
        """Test planning with knowledge base citations."""
        from tabular_agent.agent.planner import Citation
        
        config = PlanningConfig(mode=PlannerMode.RULES)
        mock_kb = Mock()
        mock_citation = Citation(
            run_id="run1", 
            score=0.8, 
            config={"test": "config"}, 
            dataset_similarity=0.8, 
            reason="Similar dataset"
        )
        mock_kb.query_similar_runs.return_value = [mock_citation]
        
        planner = Planner(config, mock_kb)
        
        data_schema = {"target": "y", "columns": ["x1", "x2"]}
        profile_summary = {"n_samples": 1000, "n_features": 2}
        constraints = {"time_budget": 300}
        
        result = planner.plan(data_schema, profile_summary, constraints)
        
        assert result.success
        assert len(result.citations) == 1
        assert result.citations[0].run_id == "run1"
        assert result.citations[0].score == 0.8
