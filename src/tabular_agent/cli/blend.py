"""Blend subcommand for tabular-agent v1.0."""

import click
import json
import pandas as pd
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any, Optional, List

from ..core.blend.basic import ModelBlender


def blend(
    models: str,
    out: str = "blend_results",
    method: str = "auto",
    strategy: str = "mean",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Blend multiple models using ensemble methods.
    
    Args:
        models: Path to model results directory
        out: Output directory
        method: Blending method (auto/crediblend/builtin)
        strategy: Blending strategy (mean/rank-mean/logit-mean)
        verbose: Verbose output
    
    Returns:
        Blending results dictionary
    """
    if verbose:
        click.echo(f"Starting model blending...")
        click.echo(f"Models directory: {models}")
        click.echo(f"Method: {method}")
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Output: {out}")
    
    # Create output directory
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if models directory exists
    models_dir = Path(models)
    if not models_dir.exists():
        click.echo(f"Models directory not found: {models}", err=True)
        sys.exit(1)
    
    # Determine blending method
    if method == "auto":
        # Try crediblend first, fallback to builtin
        try:
            result = subprocess.run(['crediblend', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                method = "crediblend"
            else:
                method = "builtin"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            method = "builtin"
    
    # Perform blending
    if method == "crediblend":
        blend_results = _blend_with_crediblend(models_dir, output_dir, strategy, verbose)
    else:
        blend_results = _blend_with_builtin(models_dir, output_dir, strategy, verbose)
    
    # Save results
    results_file = output_dir / "blend_results.json"
    with open(results_file, 'w') as f:
        json.dump(blend_results, f, indent=2, default=str)
    
    # Generate blend summary for model card
    blend_summary_file = output_dir / "blend_summary.json"
    with open(blend_summary_file, 'w') as f:
        json.dump(blend_results.get('summary', {}), f, indent=2, default=str)
    
    if verbose:
        click.echo(f"Blend results saved to: {results_file}")
        click.echo(f"Blend summary saved to: {blend_summary_file}")
    
    return blend_results


def _blend_with_crediblend(
    models_dir: Path, 
    output_dir: Path, 
    strategy: str, 
    verbose: bool
) -> Dict[str, Any]:
    """Blend using crediblend CLI tool."""
    if verbose:
        click.echo("Using crediblend for model blending...")
    
    try:
        # Look for blend_summary.json in models directory
        blend_summary_file = models_dir / "blend_summary.json"
        if not blend_summary_file.exists():
            # Create a basic blend summary from model results
            blend_summary = _create_basic_blend_summary(models_dir)
        else:
            with open(blend_summary_file, 'r') as f:
                blend_summary = json.load(f)
        
        # Run crediblend
        cmd = ['crediblend', '--input', str(blend_summary_file), '--output', str(output_dir)]
        if strategy != "mean":
            cmd.extend(['--strategy', strategy])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse crediblend output
            blend_results = {
                'method': 'crediblend',
                'status': 'success',
                'strategy': strategy,
                'summary': blend_summary,
                'raw_output': result.stdout
            }
        else:
            raise Exception(f"crediblend failed: {result.stderr}")
            
    except Exception as e:
        if verbose:
            click.echo(f"crediblend failed: {e}")
            click.echo("Falling back to builtin blending...")
        return _blend_with_builtin(models_dir, output_dir, strategy, verbose)
    
    return blend_results


def _blend_with_builtin(
    models_dir: Path, 
    output_dir: Path, 
    strategy: str, 
    verbose: bool
) -> Dict[str, Any]:
    """Blend using builtin ensemble methods."""
    if verbose:
        click.echo("Using builtin model blending...")
    
    # Load model results
    model_results = _load_model_results(models_dir)
    
    # Create blender
    blender = ModelBlender()
    
    # Perform blending
    blend_results = blender.blend_models(model_results, strategy=strategy)
    
    blend_results['method'] = 'builtin'
    blend_results['status'] = 'success'
    blend_results['strategy'] = strategy
    
    return blend_results


def _load_model_results(models_dir: Path) -> List[Dict[str, Any]]:
    """Load model results from directory."""
    model_results = []
    
    # Look for results.json files
    for results_file in models_dir.glob("**/results.json"):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                model_results.append(results)
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")
    
    return model_results


def _create_basic_blend_summary(models_dir: Path) -> Dict[str, Any]:
    """Create a basic blend summary from model results."""
    model_results = _load_model_results(models_dir)
    
    summary = {
        'models': [],
        'strategy': 'mean',
        'weights': {},
        'performance': {}
    }
    
    for i, results in enumerate(model_results):
        model_name = results.get('best_model', f'model_{i}')
        summary['models'].append(model_name)
        summary['weights'][model_name] = 1.0 / len(model_results)
        summary['performance'][model_name] = results.get('metrics', {})
    
    return summary
