"""Knowledge base indexing for runs metadata."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from .query import CitationQuery, SimilarityResult


class RunMetadata:
    """Metadata for a single run."""
    
    def __init__(
        self,
        run_id: str,
        dataset_name: str,
        columns: List[str],
        target: str,
        time_col: Optional[str],
        n_samples: int,
        n_features: int,
        categorical_columns: int,
        numerical_columns: int,
        has_time_column: bool,
        config: Dict[str, Any],
        scores: Dict[str, float],
        resource_usage: Dict[str, Any],
        feature_recipes: List[str],
        model_types: List[str],
        blending_strategy: str
    ):
        """Initialize run metadata."""
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.columns = columns
        self.target = target
        self.time_col = time_col
        self.n_samples = n_samples
        self.n_features = n_features
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.has_time_column = has_time_column
        self.config = config
        self.scores = scores
        self.resource_usage = resource_usage
        self.feature_recipes = feature_recipes
        self.model_types = model_types
        self.blending_strategy = blending_strategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'dataset_name': self.dataset_name,
            'columns': self.columns,
            'target': self.target,
            'time_col': self.time_col,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'has_time_column': self.has_time_column,
            'config': self.config,
            'scores': self.scores,
            'resource_usage': self.resource_usage,
            'feature_recipes': self.feature_recipes,
            'model_types': self.model_types,
            'blending_strategy': self.blending_strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def get_text_representation(self) -> str:
        """Get text representation for vectorization."""
        text_parts = [
            f"dataset: {self.dataset_name}",
            f"target: {self.target}",
            f"samples: {self.n_samples}",
            f"features: {self.n_features}",
            f"categorical: {self.categorical_columns}",
            f"numerical: {self.numerical_columns}",
            f"time_column: {self.has_time_column}",
            f"recipes: {' '.join(self.feature_recipes)}",
            f"models: {' '.join(self.model_types)}",
            f"blending: {self.blending_strategy}"
        ]
        return " ".join(text_parts)
    
    def get_numerical_features(self) -> np.ndarray:
        """Get numerical features for similarity calculation."""
        return np.array([
            self.n_samples,
            self.n_features,
            self.categorical_columns,
            self.numerical_columns,
            float(self.has_time_column)
        ])


class KnowledgeBase:
    """Knowledge base for indexing and querying runs metadata."""
    
    def __init__(self, runs_dir: str = "runs"):
        """Initialize knowledge base."""
        self.runs_dir = Path(runs_dir)
        self.metadata: List[RunMetadata] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.text_vectors: Optional[np.ndarray] = None
        self.numerical_features: Optional[np.ndarray] = None
        self.index_file = self.runs_dir / "kb_index.pkl"
        
        # Load existing index if available
        self._load_index()
    
    def index_runs(self, force_rebuild: bool = False) -> int:
        """Index all runs in the runs directory."""
        if not force_rebuild and self.index_file.exists():
            return len(self.metadata)
        
        self.metadata = []
        
        # Find all run directories
        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        
        for run_dir in run_dirs:
            try:
                metadata = self._extract_run_metadata(run_dir)
                if metadata:
                    self.metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to index run {run_dir}: {e}")
                continue
        
        # Build vector representations
        self._build_vectors()
        
        # Save index
        self._save_index()
        
        return len(self.metadata)
    
    def _extract_run_metadata(self, run_dir: Path) -> Optional[RunMetadata]:
        """Extract metadata from a run directory."""
        meta_file = run_dir / "meta.json"
        results_file = run_dir / "results.json"
        
        if not meta_file.exists() or not results_file.exists():
            return None
        
        # Load metadata
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract information
        config = meta_data.get('config', {})
        train_metadata = meta_data.get('train_metadata', {})
        
        # Get dataset name from path or config
        dataset_name = config.get('train_path', 'unknown').split('/')[-1].replace('.csv', '')
        
        # Extract column information
        columns = train_metadata.get('columns', [])
        target = config.get('target', 'unknown')
        time_col = config.get('time_col')
        
        # Extract data characteristics
        n_samples = train_metadata.get('n_samples', 0)
        n_features = train_metadata.get('n_features', 0)
        categorical_columns = train_metadata.get('categorical_columns', 0)
        numerical_columns = train_metadata.get('numerical_columns', 0)
        has_time_column = bool(time_col)
        
        # Extract scores
        scores = {}
        if 'evaluation_results' in results_data:
            eval_results = results_data['evaluation_results']
            if 'best_model' in eval_results:
                best_model = eval_results['best_model']
                scores = best_model.get('metrics', {})
        
        # Extract resource usage
        resource_usage = {
            'pipeline_time': results_data.get('pipeline_time', 0),
            'memory_usage': results_data.get('memory_usage', 0)
        }
        
        # Extract feature recipes and model types from results
        feature_recipes = []
        model_types = []
        blending_strategy = "mean"
        
        if 'model_results' in results_data:
            model_results = results_data['model_results']
            for model_name, model_data in model_results.items():
                if model_data.get('success', False):
                    model_types.append(model_name)
        
        # Try to extract feature recipes from configuration
        if 'feature_engineering' in results_data:
            fe_config = results_data['feature_engineering']
            feature_recipes = fe_config.get('recipes_used', [])
        
        return RunMetadata(
            run_id=run_dir.name,
            dataset_name=dataset_name,
            columns=columns,
            target=target,
            time_col=time_col,
            n_samples=n_samples,
            n_features=n_features,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            has_time_column=has_time_column,
            config=config,
            scores=scores,
            resource_usage=resource_usage,
            feature_recipes=feature_recipes,
            model_types=model_types,
            blending_strategy=blending_strategy
        )
    
    def _build_vectors(self):
        """Build vector representations for similarity search."""
        if not self.metadata:
            return
        
        # Build text vectors
        texts = [meta.get_text_representation() for meta in self.metadata]
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.text_vectors = self.vectorizer.fit_transform(texts).toarray()
        
        # Build numerical features
        self.numerical_features = np.array([meta.get_numerical_features() for meta in self.metadata])
    
    def query_similar_runs(
        self,
        data_schema: Dict[str, Any],
        profile_summary: Dict[str, Any],
        max_results: int = 3,
        min_similarity: float = 0.7
    ) -> List[CitationQuery]:
        """Query similar runs based on data characteristics."""
        if not self.metadata or self.text_vectors is None:
            return []
        
        # Create query representation
        query_text = self._create_query_text(data_schema, profile_summary)
        query_vector = self.vectorizer.transform([query_text]).toarray()
        
        # Calculate text similarity
        text_similarities = cosine_similarity(query_vector, self.text_vectors)[0]
        
        # Calculate numerical similarity
        query_numerical = self._create_query_numerical(data_schema, profile_summary)
        numerical_similarities = self._calculate_numerical_similarity(query_numerical)
        
        # Combine similarities (weighted average)
        combined_similarities = 0.7 * text_similarities + 0.3 * numerical_similarities
        
        # Get top results
        top_indices = np.argsort(combined_similarities)[::-1]
        
        citations = []
        for idx in top_indices:
            if combined_similarities[idx] >= min_similarity and len(citations) < max_results:
                metadata = self.metadata[idx]
                citation = CitationQuery(
                    run_id=metadata.run_id,
                    score=combined_similarities[idx],
                    config=metadata.config,
                    dataset_similarity=combined_similarities[idx],
                    reason=f"Similar dataset characteristics (similarity: {combined_similarities[idx]:.3f})"
                )
                citations.append(citation)
        
        return citations
    
    def _create_query_text(self, data_schema: Dict[str, Any], profile_summary: Dict[str, Any]) -> str:
        """Create text representation for query."""
        text_parts = [
            f"target: {data_schema.get('target', 'unknown')}",
            f"samples: {profile_summary.get('n_samples', 0)}",
            f"features: {profile_summary.get('n_features', 0)}",
            f"categorical: {profile_summary.get('categorical_columns', 0)}",
            f"numerical: {profile_summary.get('numerical_columns', 0)}",
            f"time_column: {profile_summary.get('has_time_column', False)}"
        ]
        return " ".join(text_parts)
    
    def _create_query_numerical(self, data_schema: Dict[str, Any], profile_summary: Dict[str, Any]) -> np.ndarray:
        """Create numerical representation for query."""
        return np.array([
            profile_summary.get('n_samples', 0),
            profile_summary.get('n_features', 0),
            profile_summary.get('categorical_columns', 0),
            profile_summary.get('numerical_columns', 0),
            float(profile_summary.get('has_time_column', False))
        ])
    
    def _calculate_numerical_similarity(self, query_numerical: np.ndarray) -> np.ndarray:
        """Calculate numerical similarity."""
        if self.numerical_features is None:
            return np.zeros(len(self.metadata))
        
        # Normalize features
        query_norm = query_numerical / (np.linalg.norm(query_numerical) + 1e-8)
        features_norm = self.numerical_features / (np.linalg.norm(self.numerical_features, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        similarities = np.dot(features_norm, query_norm)
        return similarities
    
    def _save_index(self):
        """Save index to disk."""
        if not self.runs_dir.exists():
            self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'metadata': [meta.to_dict() for meta in self.metadata],
            'text_vectors': self.text_vectors.tolist() if self.text_vectors is not None else None,
            'numerical_features': self.numerical_features.tolist() if self.numerical_features is not None else None
        }
        
        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)
    
    def _load_index(self):
        """Load index from disk."""
        if not self.index_file.exists():
            return
        
        try:
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.metadata = [RunMetadata.from_dict(meta) for meta in index_data['metadata']]
            self.text_vectors = np.array(index_data['text_vectors']) if index_data['text_vectors'] else None
            self.numerical_features = np.array(index_data['numerical_features']) if index_data['numerical_features'] else None
            
            # Rebuild vectorizer
            if self.metadata:
                texts = [meta.get_text_representation() for meta in self.metadata]
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.vectorizer.fit(texts)
                
        except Exception as e:
            print(f"Warning: Failed to load knowledge base index: {e}")
            self.metadata = []
            self.text_vectors = None
            self.numerical_features = None
