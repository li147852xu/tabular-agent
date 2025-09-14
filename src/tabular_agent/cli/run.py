"""Main CLI entry point for tabular-agent."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
import numpy as np
import yaml
from tabular_agent.core.orchestrator import PipelineOrchestrator


def main() -> None:
    """Main CLI entry point."""
    cli()


@click.group()
def cli():
    """Tabular Agent - Automated ML pipeline from CSV to model card reports."""
    pass


@cli.command()
@click.option("--train", required=True, help="Path to training CSV file")
@click.option("--test", required=True, help="Path to test CSV file")
@click.option("--target", required=True, help="Target column name")
@click.option("--time-col", help="Time column name for time-aware features")
@click.option("--n-jobs", default=1, type=int, help="Number of parallel jobs")
@click.option("--time-budget", default=300, type=int, help="Time budget in seconds")
@click.option("--out", required=True, help="Output directory for results")
@click.option("--audit-cli", help="Path to leakage-buster CLI (optional)")
@click.option("--config", help="Path to configuration YAML file")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--planner", default="auto", type=click.Choice(["llm", "rules", "auto"]), help="Planner mode")
@click.option("--llm-endpoint", help="LLM endpoint URL")
@click.option("--llm-key", help="LLM API key")
@click.option("--stability-runs", default=5, type=int, help="Number of stability evaluation runs")
@click.option("--calibration", default="isotonic", type=click.Choice(["isotonic", "platt", "none"]), help="Calibration method")
@click.option("--risk-policy", help="Path to risk policy YAML file")
def run(
    train: str,
    test: str,
    target: str,
    time_col: Optional[str],
    n_jobs: int,
    time_budget: int,
    out: str,
    audit_cli: Optional[str],
    config: Optional[str],
    seed: int,
    verbose: bool,
    planner: str,
    llm_endpoint: Optional[str],
    llm_key: Optional[str],
    stability_runs: int,
    calibration: str,
    risk_policy: Optional[str],
) -> None:
    """Run tabular-agent pipeline from CSV to model card report."""
    try:
        # Create output directory
        output_dir = Path(out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        config_dict = _load_config(config) if config else {}
        
        # Override with CLI arguments
        config_dict.update({
            "train_path": train,
            "test_path": test,
            "target": target,
            "time_col": time_col,
            "n_jobs": n_jobs,
            "time_budget": time_budget,
            "audit_cli": audit_cli,
            "seed": seed,
            "verbose": verbose,
            "stability_runs": stability_runs,
            "calibration": calibration,
            "risk_policy": risk_policy,
        })
        
        # Initialize and run orchestrator
        orchestrator = PipelineOrchestrator(
            config_dict, 
            planner_mode=planner, 
            llm_endpoint=llm_endpoint, 
            llm_key=llm_key
        )
        
        if verbose:
            click.echo(f"Starting tabular-agent pipeline...")
            click.echo(f"Run directory: {run_dir}")
            click.echo(f"Configuration: {json.dumps(config_dict, indent=2)}")
        
        # Run the pipeline
        results = orchestrator.run(run_dir)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "git_hash": _get_git_hash(),
            "seed": seed,
            "args": {
                "train": train,
                "test": test,
                "target": target,
                "time_col": time_col,
                "n_jobs": n_jobs,
                "time_budget": time_budget,
                "audit_cli": audit_cli,
            },
            "version": "0.1.0",
            "results": results,
        }
        
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
        metadata_serializable = convert_numpy_types(metadata)
        
        with open(run_dir / "meta.json", "w") as f:
            json.dump(metadata_serializable, f, indent=2, default=str)
        
        if verbose:
            click.echo(f"Pipeline completed successfully!")
            click.echo(f"Results saved to: {run_dir}")
            click.echo(f"Model card: {run_dir / 'model_card.html'}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_git_hash() -> str:
    """Get current git hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    main()
