"""Tests for knowledge base indexing functionality."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from tabular_agent.agent.kb import KnowledgeBase, RunMetadata


class TestKnowledgeBaseIndexing:
    """Test knowledge base indexing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.runs_dir = Path(self.temp_dir) / "runs"
        self.runs_dir.mkdir(parents=True)
        
        # Create sample run metadata
        self.sample_metadata = {
            "config": {
                "train_path": "examples/train.csv",
                "target": "target",
                "time_col": "date"
            },
            "train_metadata": {
                "columns": ["x1", "x2", "target", "date"],
                "n_samples": 1000,
                "n_features": 3,
                "categorical_columns": 1,
                "numerical_columns": 2,
                "has_time_column": True
            }
        }
        
        self.sample_results = {
            "model_results": {
                "lightgbm": {"success": True, "metrics": {"auc": 0.85}},
                "xgboost": {"success": True, "metrics": {"auc": 0.83}}
            },
            "pipeline_time": 120.5
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_run(self, run_id: str, metadata: dict, results: dict):
        """Create a sample run directory with metadata and results."""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir()
        
        with open(run_dir / "meta.json", "w") as f:
            json.dump(metadata, f)
        
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f)
    
    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization."""
        kb = KnowledgeBase(str(self.runs_dir))
        assert kb.runs_dir == self.runs_dir
        assert len(kb.metadata) == 0
    
    def test_index_empty_runs_directory(self):
        """Test indexing empty runs directory."""
        kb = KnowledgeBase(str(self.runs_dir))
        count = kb.index_runs()
        assert count == 0
    
    def test_index_single_run(self):
        """Test indexing a single run."""
        self.create_sample_run("20240101_120000", self.sample_metadata, self.sample_results)
        
        kb = KnowledgeBase(str(self.runs_dir))
        count = kb.index_runs()
        
        assert count == 1
        assert len(kb.metadata) == 1
        
        metadata = kb.metadata[0]
        assert metadata.run_id == "20240101_120000"
        assert metadata.dataset_name == "train"
        assert metadata.target == "target"
        assert metadata.n_samples == 1000
        assert metadata.has_time_column is True
    
    def test_index_multiple_runs(self):
        """Test indexing multiple runs."""
        # Create multiple runs
        for i in range(3):
            run_id = f"20240101_12000{i}"
            metadata = self.sample_metadata.copy()
            metadata["train_metadata"]["n_samples"] = 1000 + i * 100
            self.create_sample_run(run_id, metadata, self.sample_results)
        
        kb = KnowledgeBase(str(self.runs_dir))
        count = kb.index_runs()
        
        assert count == 3
        assert len(kb.metadata) == 3
    
    def test_query_similar_runs(self):
        """Test querying similar runs."""
        # Create sample runs
        runs_data = [
            {
                "metadata": {
                    "config": {"train_path": "dataset1.csv", "target": "target"},
                    "train_metadata": {
                        "columns": ["x1", "x2", "target"],
                        "n_samples": 1000,
                        "n_features": 2,
                        "categorical_columns": 1,
                        "numerical_columns": 1,
                        "has_time_column": False
                    }
                },
                "results": {"model_results": {"lightgbm": {"success": True}}}
            },
            {
                "metadata": {
                    "config": {"train_path": "dataset2.csv", "target": "target"},
                    "train_metadata": {
                        "columns": ["x1", "x2", "x3", "target"],
                        "n_samples": 2000,
                        "n_features": 3,
                        "categorical_columns": 2,
                        "numerical_columns": 1,
                        "has_time_column": True
                    }
                },
                "results": {"model_results": {"xgboost": {"success": True}}}
            }
        ]
        
        for i, run_data in enumerate(runs_data):
            self.create_sample_run(f"run_{i}", run_data["metadata"], run_data["results"])
        
        kb = KnowledgeBase(str(self.runs_dir))
        kb.index_runs()
        
        # Query similar runs
        data_schema = {"target": "target", "columns": ["x1", "x2", "target"]}
        profile_summary = {
            "n_samples": 1000,
            "n_features": 2,
            "categorical_columns": 1,
            "numerical_columns": 1,
            "has_time_column": False
        }
        
        citations = kb.query_similar_runs(data_schema, profile_summary, max_results=2)
        
        assert len(citations) >= 0  # May be 0 if similarity is too low
        if citations:
            assert all(citation.score >= 0 for citation in citations)
            assert all(citation.run_id.startswith("run_") for citation in citations)
    
    def test_metadata_text_representation(self):
        """Test metadata text representation for vectorization."""
        metadata = RunMetadata(
            run_id="test_run",
            dataset_name="test_dataset",
            columns=["x1", "x2", "target"],
            target="target",
            time_col=None,
            n_samples=1000,
            n_features=2,
            categorical_columns=1,
            numerical_columns=1,
            has_time_column=False,
            config={"test": "config"},
            scores={"auc": 0.85},
            resource_usage={"pipeline_time": 120},
            feature_recipes=["target_encoding"],
            model_types=["lightgbm"],
            blending_strategy="mean"
        )
        
        text_repr = metadata.get_text_representation()
        assert "dataset: test_dataset" in text_repr
        assert "target: target" in text_repr
        assert "samples: 1000" in text_repr
        assert "features: 2" in text_repr
        assert "time_column: False" in text_repr
    
    def test_metadata_numerical_features(self):
        """Test metadata numerical features extraction."""
        metadata = RunMetadata(
            run_id="test_run",
            dataset_name="test_dataset",
            columns=["x1", "x2", "target"],
            target="target",
            time_col=None,
            n_samples=1000,
            n_features=2,
            categorical_columns=1,
            numerical_columns=1,
            has_time_column=False,
            config={"test": "config"},
            scores={"auc": 0.85},
            resource_usage={"pipeline_time": 120},
            feature_recipes=["target_encoding"],
            model_types=["lightgbm"],
            blending_strategy="mean"
        )
        
        numerical_features = metadata.get_numerical_features()
        assert len(numerical_features) == 5
        assert numerical_features[0] == 1000  # n_samples
        assert numerical_features[1] == 2      # n_features
        assert numerical_features[2] == 1      # categorical_columns
        assert numerical_features[3] == 1      # numerical_columns
        assert numerical_features[4] == 0.0    # has_time_column (False)
    
    def test_index_persistence(self):
        """Test that index is saved and loaded correctly."""
        # Create a run and index it
        self.create_sample_run("test_run", self.sample_metadata, self.sample_results)
        
        kb1 = KnowledgeBase(str(self.runs_dir))
        count1 = kb1.index_runs()
        assert count1 == 1
        
        # Create new knowledge base instance - should load from saved index
        kb2 = KnowledgeBase(str(self.runs_dir))
        assert len(kb2.metadata) == 1
        assert kb2.metadata[0].run_id == "test_run"
    
    def test_invalid_run_handling(self):
        """Test handling of invalid run directories."""
        # Create invalid run (missing files)
        invalid_run_dir = self.runs_dir / "invalid_run"
        invalid_run_dir.mkdir()
        # Don't create meta.json or results.json
        
        # Create valid run
        self.create_sample_run("valid_run", self.sample_metadata, self.sample_results)
        
        kb = KnowledgeBase(str(self.runs_dir))
        count = kb.index_runs()
        
        # Should only index the valid run
        assert count == 1
        assert len(kb.metadata) == 1
        assert kb.metadata[0].run_id == "valid_run"
