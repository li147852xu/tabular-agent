"""Main pipeline orchestrator for tabular-agent."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
import warnings

from .io import read_csv, save_data
from .profile import DataProfiler
from .audit import LeakageAuditor
from .fe.pipelines import FeatureEngineeringPipeline
from .models.trainers import ModelTrainer, EnsembleTrainer
from .tune.optuna import OptunaTuner, MultiModelTuner
from .blend.basic import BlendingEnsemble
from .evaluate.metrics import MetricsCalculator, CalibrationAnalyzer, ThresholdOptimizer, StabilityAnalyzer
from .report.card import ModelCardGenerator
from ..agent import Planner, PlanningConfig, KnowledgeBase


class PipelineOrchestrator:
    """Main orchestrator for the tabular-agent pipeline."""
    
    def __init__(self, config: Dict[str, Any], planner_mode: str = "auto", llm_endpoint: Optional[str] = None, llm_key: Optional[str] = None):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Configuration dictionary
            planner_mode: Planner mode (llm, rules, auto)
            llm_endpoint: LLM endpoint URL
            llm_key: LLM API key
        """
        self.config = config
        self.results = {}
        
        # Set random seed
        if 'seed' in config:
            np.random.seed(config['seed'])
        
        # Initialize planner
        self.planner_config = PlanningConfig(
            mode=planner_mode,
            llm_endpoint=llm_endpoint,
            llm_key=llm_key
        )
        self.knowledge_base = KnowledgeBase()
        self.planner = Planner(self.planner_config, self.knowledge_base)
        self.planning_result = None
        
        # Initialize components
        self.data_profiler = None
        self.leakage_auditor = None
        self.feature_pipeline = None
        self.model_trainer = None
        self.tuner = None
        self.blender = None
        self.metrics_calculator = None
        self.calibration_analyzer = None
        self.threshold_optimizer = None
        self.stability_analyzer = None
        self.model_card_generator = None
    
    def run(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing all results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Step 0: Index knowledge base and plan
            print("Step 0: Indexing knowledge base and planning...")
            self.knowledge_base.index_runs()
            
            # Step 1: Load and profile data
            print("Step 1: Loading and profiling data...")
            train_df, train_metadata = self._load_and_profile_data(
                self.config['train_path'], 
                self.config['target'],
                self.config.get('time_col')
            )
            
            test_df, test_metadata = self._load_and_profile_data(
                self.config['test_path'],
                self.config['target'],
                self.config.get('time_col')
            )
            
            # Generate plan based on data characteristics
            print("Step 1.5: Generating execution plan...")
            data_schema = {
                'target': self.config['target'],
                'time_col': self.config.get('time_col'),
                'columns': train_metadata.get('columns', [])
            }
            
            constraints = {
                'time_budget': self.config.get('time_budget', 300),
                'n_jobs': self.config.get('n_jobs', 1),
                'memory_limit': self.config.get('memory_limit', None)
            }
            
            self.planning_result = self.planner.plan(
                data_schema, train_metadata, constraints
            )
            
            if self.planning_result.success:
                print(f"Planning successful using {self.planning_result.mode_used} mode")
                if self.planning_result.citations:
                    print(f"Found {len(self.planning_result.citations)} similar runs for reference")
            else:
                print(f"Planning failed: {self.planning_result.error_message}")
                print("Falling back to default configuration")
            
            # Step 2: Leakage audit
            print("Step 2: Performing leakage audit...")
            audit_results = self._perform_leakage_audit(train_df, test_df)
            
            # Step 3: Feature engineering
            print("Step 3: Feature engineering...")
            X_train, y_train, X_test = self._perform_feature_engineering(
                train_df, test_df, self.config['target']
            )
            
            # Step 4: Model training and tuning
            print("Step 4: Model training and tuning...")
            model_results = self._train_and_tune_models(X_train, y_train, X_test)
            
            # Step 5: Model evaluation
            print("Step 5: Model evaluation...")
            evaluation_results = self._evaluate_models(
                model_results, y_train, X_test, test_df[self.config['target']]
            )
            
            # Step 6: Generate model card
            print("Step 6: Generating model card...")
            model_card_path = self._generate_model_card(
                evaluation_results, train_metadata, audit_results, output_dir, self.planning_result
            )
            
            # Compile results
            self.results = {
                'train_metadata': train_metadata,
                'test_metadata': test_metadata,
                'audit_results': audit_results,
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'model_card_path': model_card_path,
                'pipeline_time': time.time() - start_time
            }
            
            # Save results
            def convert_numpy_types(obj):
                """Convert numpy types to Python types for JSON serialization."""
                if isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'dtype'):  # Handle pandas types
                    return str(obj)
                else:
                    return obj
            
            # Convert numpy types before saving
            results_serializable = convert_numpy_types(self.results)
            
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results_serializable, f, indent=2, default=str)
            
            print(f"Pipeline completed successfully in {self.results['pipeline_time']:.2f} seconds")
            print(f"Model card saved to: {model_card_path}")
            
            return self.results
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            if self.config.get('verbose', False):
                import traceback
                traceback.print_exc()
            raise
    
    def _load_and_profile_data(
        self, 
        file_path: str, 
        target_col: str, 
        time_col: Optional[str]
    ) -> tuple:
        """Load and profile data."""
        # Load data
        df, metadata = read_csv(file_path, target_col, time_col)
        
        # Profile data
        profiler = DataProfiler(target_col, time_col)
        profile_results = profiler.profile(df)
        
        # Add profiling results to metadata
        metadata['profile'] = profile_results
        
        return df, metadata
    
    def _perform_leakage_audit(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform leakage audit."""
        auditor = LeakageAuditor(self.config.get('audit_cli'))
        return auditor.audit(
            train_df, 
            test_df, 
            self.config['target'],
            self.config.get('time_col')
        )
    
    def _perform_feature_engineering(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str
    ) -> tuple:
        """Perform feature engineering."""
        # Initialize feature pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(
            target_col=target_col,
            time_col=self.config.get('time_col'),
            config=self.config.get('feature_config', {})
        )
        
        # Fit on training data
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        self.feature_pipeline.fit(X_train, y_train)
        
        # Transform both train and test
        X_train_transformed = self.feature_pipeline.transform(X_train)
        X_test_transformed = self.feature_pipeline.transform(test_df.drop(columns=[target_col]))
        
        return X_train_transformed, y_train, X_test_transformed
    
    def _train_and_tune_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train and tune models."""
        # Determine if classification or regression
        is_classification = len(np.unique(y_train)) == 2
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(is_classification)
        
        # Get model names from config
        model_names = self.config.get('model_names', ['lightgbm', 'xgboost', 'catboost'])
        
        # Train individual models
        model_results = {}
        for model_name in model_names:
            print(f"Training {model_name}...")
            
            # Create trainer
            trainer = ModelTrainer(
                model_name=model_name,
                cv_folds=self.config.get('cv_folds', 5),
                time_col=self.config.get('time_col'),
                random_state=self.config.get('seed', 42)
            )
            
            # Train model
            trainer.fit(X_train, y_train)
            
            # Make predictions
            y_pred = trainer.predict(X_test)
            y_proba = trainer.predict_proba(X_test) if is_classification else None
            
            # Store results
            model_results[model_name] = {
                'trainer': trainer,
                'predictions': y_pred,
                'probabilities': y_proba,
                'cv_scores': trainer.get_cv_scores(),
                'feature_importance': trainer.get_feature_importance()
            }
        
        return model_results
    
    def _evaluate_models(
        self, 
        model_results: Dict[str, Any], 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate models comprehensively."""
        evaluation_results = {}
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_train)) == 2
        
        for model_name, results in model_results.items():
            print(f"Evaluating {model_name}...")
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_test, 
                results['predictions'], 
                results['probabilities']
            )
            
            # Calibration analysis
            if is_classification and results['probabilities'] is not None:
                calibration_analyzer = CalibrationAnalyzer()
                calibration_results = calibration_analyzer.analyze_calibration(
                    y_test, results['probabilities']
                )
            else:
                calibration_results = {}
            
            # Threshold optimization
            if is_classification and results['probabilities'] is not None:
                threshold_optimizer = ThresholdOptimizer()
                threshold_results = threshold_optimizer.optimize_threshold(
                    y_test, results['probabilities']
                )
            else:
                threshold_results = {}
            
            # Stability analysis
            stability_analyzer = StabilityAnalyzer()
            stability_results = stability_analyzer.analyze_stability(
                y_test, results['probabilities'] if results['probabilities'] is not None else results['predictions']
            )
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'calibration': calibration_results,
                'threshold_optimization': threshold_results,
                'stability': stability_results
            }
        
        return evaluation_results
    
    def _generate_model_card(
        self, 
        evaluation_results: Dict[str, Any], 
        train_metadata: Dict[str, Any], 
        audit_results: Dict[str, Any], 
        output_dir: Path,
        planning_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate model card."""
        # Get best model
        best_model = self._get_best_model(evaluation_results)
        best_results = evaluation_results[best_model]
        
        # Get feature importance
        feature_importance = {}
        for model_name, results in evaluation_results.items():
            if 'feature_importance' in results:
                feature_importance.update(results['feature_importance'])
        
        # Generate model card
        generator = ModelCardGenerator(output_dir)
        model_card_path = generator.generate_model_card(
            model_results=best_results['metrics'],
            data_profile=train_metadata['profile'],
            audit_results=audit_results,
            feature_importance=feature_importance,
            calibration_results=best_results['calibration'],
            stability_results=best_results['stability'],
            metadata={
                'best_model': best_model,
                'target': self.config['target'],
                'time_col': self.config.get('time_col'),
                'n_jobs': self.config.get('n_jobs', 1),
                'time_budget': self.config.get('time_budget', 300),
                'seed': self.config.get('seed', 42)
            },
            planning_result=planning_result.dict() if planning_result else None
        )
        
        return model_card_path
    
    def _get_best_model(self, evaluation_results: Dict[str, Any]) -> str:
        """Get the best performing model."""
        best_score = -np.inf
        best_model = None
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            # Use AUC for classification, R² for regression
            if 'auc' in metrics:
                score = metrics['auc']
            elif 'r2' in metrics:
                score = metrics['r2']
            else:
                score = metrics.get('accuracy', 0)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model or list(evaluation_results.keys())[0]
