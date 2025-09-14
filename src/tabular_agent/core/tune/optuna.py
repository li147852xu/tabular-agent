"""Optuna-based hyperparameter tuning."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..models.registry import model_registry
from ..models.trainers import ModelTrainer


class OptunaTuner:
    """Optuna-based hyperparameter tuner with parallel optimization."""
    
    def __init__(
        self,
        model_name: str,
        time_budget: int = 300,
        n_trials: int = 100,
        cv_folds: int = 5,
        time_col: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        """
        Initialize Optuna tuner.
        
        Args:
            model_name: Name of model to tune
            time_budget: Time budget in seconds
            n_trials: Maximum number of trials
            cv_folds: Number of CV folds
            time_col: Time column for time-aware CV
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.model_name = model_name
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.time_col = time_col
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        self.best_params = None
        self.best_score = None
        self.tuning_results = {}
        self.lock = threading.Lock()
    
    def tune(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        objective_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            objective_func: Custom objective function
            
        Returns:
            Dictionary containing tuning results
        """
        start_time = time.time()
        
        # Create objective function
        if objective_func is None:
            objective_func = self._create_objective(X, y)
        
        # Optimize
        if self.n_jobs > 1:
            self._optimize_parallel(objective_func)
        else:
            self.study.optimize(
                objective_func,
                n_trials=self.n_trials,
                timeout=self.time_budget
            )
        
        # Get results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        tuning_time = time.time() - start_time
        
        self.tuning_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'tuning_time': tuning_time,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in self.study.trials
            ]
        }
        
        return self.tuning_results
    
    def _create_objective(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create objective function for Optuna."""
        def objective(trial):
            # Get parameter suggestions
            params = self._suggest_params(trial)
            
            # Create trainer
            trainer = ModelTrainer(
                model_name=self.model_name,
                cv_folds=self.cv_folds,
                time_col=self.time_col,
                random_state=self.random_state
            )
            
            try:
                # Fit model
                trainer.fit(X, y, params)
                
                # Get CV score
                cv_scores = trainer.get_cv_scores()
                score = cv_scores['mean']
                
                # Report intermediate value for pruning
                trial.report(score, step=0)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                # Return a very low score for failed trials
                return -np.inf
        
        return objective
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        # Get default parameters
        default_params = model_registry.get_default_params(self.model_name)
        
        # Define parameter search spaces
        param_spaces = self._get_param_spaces()
        
        # Suggest parameters
        params = {}
        for param_name, param_space in param_spaces.items():
            if param_name in default_params:
                if param_space['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_space['low'], 
                        param_space['high'],
                        step=param_space.get('step', 1)
                    )
                elif param_space['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_space['low'], 
                        param_space['high'],
                        log=param_space.get('log', False)
                    )
                elif param_space['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_space['choices']
                    )
        
        return params
    
    def _get_param_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter search spaces for different models."""
        spaces = {
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 1000, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 1000, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
            },
            'catboost': {
                'iterations': {'type': 'int', 'low': 50, 'high': 1000, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'depth': {'type': 'int', 'low': 3, 'high': 12},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bylevel': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0},
            },
            'linear': {
                'C': {'type': 'float', 'low': 0.001, 'high': 100.0, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'lbfgs', 'saga']},
            },
            'histgbdt': {
                'max_iter': {'type': 'int', 'low': 50, 'high': 1000, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'l2_regularization': {'type': 'float', 'low': 0.0, 'high': 1.0},
            }
        }
        
        return spaces.get(self.model_name, {})
    
    def _optimize_parallel(self, objective_func: Callable):
        """Optimize with parallel trials."""
        def run_trial(trial_number):
            """Run a single trial."""
            try:
                trial = self.study.ask()
                value = objective_func(trial)
                self.study.tell(trial, value)
                return trial_number, value
            except Exception as e:
                warnings.warn(f"Trial {trial_number} failed: {e}")
                return trial_number, None
        
        # Run trials in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(self.n_trials):
                future = executor.submit(run_trial, i)
                futures.append(future)
            
            # Wait for completion or timeout
            start_time = time.time()
            for future in as_completed(futures):
                if time.time() - start_time > self.time_budget:
                    break
                
                try:
                    trial_number, value = future.result()
                    if value is not None:
                        with self.lock:
                            if len(self.study.trials) >= self.n_trials:
                                break
                except Exception as e:
                    warnings.warn(f"Future failed: {e}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        return self.best_params or {}
    
    def get_best_score(self) -> float:
        """Get best score achieved."""
        return self.best_score or -np.inf
    
    def get_tuning_results(self) -> Dict[str, Any]:
        """Get complete tuning results."""
        return self.tuning_results
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # Plot parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
        except Exception as e:
            warnings.warn(f"Failed to create plots: {e}")


class MultiModelTuner:
    """Tuner for multiple models with time budget allocation."""
    
    def __init__(
        self,
        model_names: List[str],
        time_budget: int = 300,
        cv_folds: int = 5,
        time_col: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        """
        Initialize multi-model tuner.
        
        Args:
            model_names: List of model names to tune
            time_budget: Total time budget in seconds
            cv_folds: Number of CV folds
            time_col: Time column for time-aware CV
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.model_names = model_names
        self.time_budget = time_budget
        self.cv_folds = cv_folds
        self.time_col = time_col
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.tuning_results = {}
        self.best_model = None
        self.best_score = -np.inf
    
    def tune_all(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Tune all models with time budget allocation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing tuning results for all models
        """
        # Allocate time budget equally among models
        time_per_model = self.time_budget // len(self.model_names)
        
        for model_name in self.model_names:
            print(f"Tuning {model_name}...")
            
            tuner = OptunaTuner(
                model_name=model_name,
                time_budget=time_per_model,
                n_trials=50,  # Reduced trials per model
                cv_folds=self.cv_folds,
                time_col=self.time_col,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            results = tuner.tune(X, y)
            self.tuning_results[model_name] = results
            
            # Track best model
            if results['best_score'] > self.best_score:
                self.best_score = results['best_score']
                self.best_model = model_name
        
        return self.tuning_results
    
    def get_best_model(self) -> str:
        """Get the best performing model name."""
        return self.best_model
    
    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """Get best parameters for a specific model."""
        if model_name not in self.tuning_results:
            return {}
        return self.tuning_results[model_name].get('best_params', {})
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all tuning results."""
        return self.tuning_results
