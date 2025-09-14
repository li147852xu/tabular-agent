"""Query module for knowledge base citations."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationQuery(BaseModel):
    """Citation query result."""
    run_id: str = Field(..., description="Run ID")
    score: float = Field(..., description="Similarity score")
    config: Dict[str, Any] = Field(..., description="Configuration used")
    dataset_similarity: float = Field(..., description="Dataset similarity score")
    reason: str = Field(..., description="Reason for citation")


class SimilarityResult(BaseModel):
    """Similarity search result."""
    run_id: str = Field(..., description="Run ID")
    similarity_score: float = Field(..., description="Overall similarity score")
    text_similarity: float = Field(..., description="Text similarity score")
    numerical_similarity: float = Field(..., description="Numerical similarity score")
    metadata: Dict[str, Any] = Field(..., description="Run metadata")


class CitationReason(str, Enum):
    """Reasons for citation."""
    SIMILAR_DATASET = "similar_dataset"
    SIMILAR_FEATURES = "similar_features"
    SIMILAR_PERFORMANCE = "similar_performance"
    SIMILAR_CONFIG = "similar_config"
    BEST_PERFORMANCE = "best_performance"
