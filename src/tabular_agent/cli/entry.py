"""Main CLI entry point for tabular-agent v1.0."""

import click
import sys
from pathlib import Path

from .run import run
from .audit import audit
from .blend import blend


@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    Tabular Agent v1.0 - Automated ML Pipeline
    
    A comprehensive machine learning pipeline for tabular data that includes:
    - Data profiling and leakage auditing
    - Intelligent feature engineering
    - Model training and hyperparameter optimization
    - Model blending and ensemble methods
    - Risk analysis and stability evaluation
    - Professional model card generation
    
    For more information, visit: https://github.com/li147852xu/tabular-agent
    """
    pass


@main.command()
@click.option('--train', required=True, help='Path to training data (CSV)')
@click.option('--test', required=True, help='Path to test data (CSV)')
@click.option('--target', required=True, help='Target column name')
@click.option('--time-col', help='Time column name (optional)')
@click.option('--n-jobs', default=1, help='Number of parallel jobs')
@click.option('--time-budget', default=300, help='Time budget in seconds')
@click.option('--cv-folds', default=5, help='Number of CV folds')
@click.option('--seed', default=42, help='Random seed')
@click.option('--out', required=True, help='Output directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--planner', default='auto', type=click.Choice(['llm', 'rules', 'auto']), help='Planner mode')
@click.option('--llm-endpoint', help='LLM API endpoint')
@click.option('--llm-key', help='LLM API key')
@click.option('--stability-runs', default=5, help='Number of stability evaluation runs')
@click.option('--calibration', default='none', type=click.Choice(['isotonic', 'platt', 'none']), help='Calibration method')
@click.option('--risk-policy', help='Risk policy configuration file')
def run_pipeline(**kwargs):
    """Run the complete ML pipeline from data to model card."""
    # Import here to avoid circular imports
    from .run import run_pipeline_impl
    run_pipeline_impl(**kwargs)


@main.command()
@click.option('--data', required=True, help='Path to data file (CSV)')
@click.option('--target', required=True, help='Target column name')
@click.option('--time-col', help='Time column name (optional)')
@click.option('--out', required=True, help='Output directory')
@click.option('--method', default='auto', type=click.Choice(['auto', 'leakage-buster', 'builtin']), help='Audit method')
@click.option('--verbose', is_flag=True, help='Verbose output')
def audit_data(**kwargs):
    """Perform data leakage audit."""
    from .audit import audit
    audit(**kwargs)


@main.command()
@click.option('--models', required=True, help='Path to model results directory')
@click.option('--out', required=True, help='Output directory')
@click.option('--method', default='auto', type=click.Choice(['auto', 'crediblend', 'builtin']), help='Blending method')
@click.option('--strategy', default='mean', type=click.Choice(['mean', 'rank-mean', 'logit-mean']), help='Blending strategy')
@click.option('--verbose', is_flag=True, help='Verbose output')
def blend_models(**kwargs):
    """Blend multiple models using ensemble methods."""
    from .blend import blend
    blend(**kwargs)


# Add subcommands to main group
main.add_command(run_pipeline, name='run')
main.add_command(audit_data, name='audit')
main.add_command(blend_models, name='blend')


if __name__ == '__main__':
    main()
