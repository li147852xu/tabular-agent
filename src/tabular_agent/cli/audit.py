"""Audit subcommand for tabular-agent v1.0."""

import click
import pandas as pd
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any, Optional

from ..core.profile import DataProfiler
from ..core.audit import LeakageAuditor


def audit(
    data: str,
    target: str,
    time_col: Optional[str] = None,
    out: str = "audit_results",
    method: str = "auto",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform data leakage audit.
    
    Args:
        data: Path to data file
        target: Target column name
        time_col: Time column name (optional)
        out: Output directory
        method: Audit method (auto/leakage-buster/builtin)
        verbose: Verbose output
    
    Returns:
        Audit results dictionary
    """
    if verbose:
        click.echo(f"Starting data leakage audit...")
        click.echo(f"Data: {data}")
        click.echo(f"Target: {target}")
        click.echo(f"Time column: {time_col}")
        click.echo(f"Method: {method}")
        click.echo(f"Output: {out}")
    
    # Create output directory
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(data)
        if verbose:
            click.echo(f"Loaded data: {df.shape}")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    # Determine audit method
    if method == "auto":
        # Try leakage-buster first, fallback to builtin
        try:
            result = subprocess.run(['leakage-buster', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                method = "leakage-buster"
            else:
                method = "builtin"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            method = "builtin"
    
    # Perform audit
    if method == "leakage-buster":
        audit_results = _audit_with_leakage_buster(df, target, time_col, output_dir, verbose)
    else:
        audit_results = _audit_with_builtin(df, target, time_col, output_dir, verbose)
    
    # Save results
    results_file = output_dir / "audit_results.json"
    with open(results_file, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    
    if verbose:
        click.echo(f"Audit results saved to: {results_file}")
    
    return audit_results


def _audit_with_leakage_buster(
    df: pd.DataFrame, 
    target: str, 
    time_col: Optional[str], 
    output_dir: Path, 
    verbose: bool
) -> Dict[str, Any]:
    """Audit using leakage-buster CLI tool."""
    if verbose:
        click.echo("Using leakage-buster for audit...")
    
    # Save data temporarily
    temp_data = output_dir / "temp_data.csv"
    df.to_csv(temp_data, index=False)
    
    try:
        # Run leakage-buster
        cmd = ['leakage-buster', '--data', str(temp_data), '--target', target]
        if time_col:
            cmd.extend(['--time-col', time_col])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse leakage-buster output
            audit_results = {
                'method': 'leakage-buster',
                'status': 'success',
                'leakage_score': 0.0,  # Parse from output
                'suspicious_features': [],
                'recommendations': [],
                'raw_output': result.stdout
            }
        else:
            raise Exception(f"leakage-buster failed: {result.stderr}")
            
    except Exception as e:
        if verbose:
            click.echo(f"leakage-buster failed: {e}")
            click.echo("Falling back to builtin audit...")
        return _audit_with_builtin(df, target, time_col, output_dir, verbose)
    
    finally:
        # Clean up temp file
        if temp_data.exists():
            temp_data.unlink()
    
    return audit_results


def _audit_with_builtin(
    df: pd.DataFrame, 
    target: str, 
    time_col: Optional[str], 
    output_dir: Path, 
    verbose: bool
) -> Dict[str, Any]:
    """Audit using builtin leakage detection."""
    if verbose:
        click.echo("Using builtin leakage detection...")
    
    # Profile data
    profiler = DataProfiler()
    profile = profiler.profile_data(df, target, time_col)
    
    # Perform leakage audit
    auditor = LeakageAuditor()
    audit_results = auditor.audit(df, target, time_col, profile)
    
    audit_results['method'] = 'builtin'
    audit_results['status'] = 'success'
    
    return audit_results
