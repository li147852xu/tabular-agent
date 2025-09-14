"""Model card generation for tabular-agent."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import base64
from datetime import datetime
import warnings

from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


class ModelCardGenerator:
    """Generate comprehensive model cards in HTML format."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize model card generator.
        
        Args:
            output_dir: Output directory for the model card
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_model_card(
        self,
        model_results: Dict[str, Any],
        data_profile: Dict[str, Any],
        audit_results: Dict[str, Any],
        feature_importance: Dict[str, float],
        calibration_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        metadata: Dict[str, Any],
        planning_result: Optional[Dict[str, Any]] = None,
        risk_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive model card.
        
        Args:
            model_results: Model performance results
            data_profile: Data profiling results
            audit_results: Leakage audit results
            feature_importance: Feature importance scores
            calibration_results: Model calibration results
            stability_results: Model stability results
            metadata: Additional metadata
            
        Returns:
            Path to generated HTML file
        """
        # Generate plots
        plots = self._generate_plots(
            model_results, data_profile, calibration_results, stability_results, feature_importance
        )
        
        # Prepare data for template
        template_data = self._prepare_template_data(
            model_results, data_profile, audit_results, feature_importance,
            calibration_results, stability_results, metadata, plots, planning_result, risk_analysis
        )
        
        # Load and render template
        template = self._load_template()
        html_content = template.render(**template_data)
        
        # Save HTML file
        html_path = self.output_dir / "model_card.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_plots(
        self,
        model_results: Dict[str, Any],
        data_profile: Dict[str, Any],
        calibration_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """Generate plots for the model card."""
        plots = {}
        
        try:
            # ROC Curve
            if 'roc_curve' in model_results:
                plots['roc_curve'] = self._plot_roc_curve(model_results['roc_curve'])
            
            # Precision-Recall Curve
            if 'precision_recall_curve' in model_results:
                plots['pr_curve'] = self._plot_pr_curve(model_results['precision_recall_curve'])
            
            # Calibration Plot
            if calibration_results:
                plots['calibration'] = self._plot_calibration(calibration_results)
            
            # Feature Importance
            if feature_importance and len(feature_importance) > 0:
                plots['feature_importance'] = self._plot_feature_importance(feature_importance)
            
            # Data Distribution
            if data_profile:
                plots['data_distribution'] = self._plot_data_distribution(data_profile)
            
            # Stability Analysis
            if stability_results:
                plots['stability'] = self._plot_stability(stability_results)
            
        except Exception as e:
            warnings.warn(f"Failed to generate some plots: {e}")
        
        return plots
    
    def _plot_roc_curve(self, roc_data: Dict[str, np.ndarray]) -> str:
        """Generate ROC curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name='ROC Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _plot_pr_curve(self, pr_data: Dict[str, np.ndarray]) -> str:
        """Generate Precision-Recall curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pr_data['recall'],
            y=pr_data['precision'],
            mode='lines',
            name='PR Curve',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=500,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _plot_calibration(self, calibration_data: Dict[str, Any]) -> str:
        """Generate calibration plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Reliability Diagram', 'Calibration Curve'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Reliability diagram
        bin_lowers = calibration_data['reliability_diagram']['bin_lowers']
        bin_uppers = calibration_data['reliability_diagram']['bin_uppers']
        accuracy_in_bins = calibration_data['reliability_diagram']['accuracy_in_bins']
        confidence_in_bins = calibration_data['reliability_diagram']['confidence_in_bins']
        
        fig.add_trace(
            go.Bar(
                x=[(l + u) / 2 for l, u in zip(bin_lowers, bin_uppers)],
                y=accuracy_in_bins,
                name='Accuracy',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Calibration curve
        fig.add_trace(
            go.Scatter(
                x=calibration_data['mean_predicted_value'],
                y=calibration_data['fraction_of_positives'],
                mode='lines+markers',
                name='Calibration Curve',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Calibration Analysis',
            width=1000,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _plot_feature_importance(self, feature_importance: Dict[str, float]) -> str:
        """Generate feature importance plot."""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:20]  # Top 20 features
        
        features, importances = zip(*top_features)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            name='Feature Importance'
        ))
        
        fig.update_layout(
            title='Top 20 Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            width=600,
            height=500
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _plot_data_distribution(self, data_profile: Dict[str, Any]) -> str:
        """Generate data distribution plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Values', 'Data Types', 'Target Distribution', 'Data Quality Score'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Missing values
        missing_data = data_profile.get('missing_analysis', {}).get('missing_ratios', {})
        if missing_data:
            fig.add_trace(
                go.Bar(
                    x=list(missing_data.keys()),
                    y=list(missing_data.values()),
                    name='Missing Ratio'
                ),
                row=1, col=1
            )
        
        # Data types
        dtypes = data_profile.get('basic_info', {}).get('dtypes', {})
        if dtypes:
            fig.add_trace(
                go.Pie(
                    labels=list(dtypes.keys()),
                    values=list(dtypes.values()),
                    name='Data Types'
                ),
                row=1, col=2
            )
        
        # Target distribution
        target_info = data_profile.get('target_analysis', {})
        if target_info.get('is_binary', False):
            class_balance = target_info.get('class_balance', {})
            if class_balance:
                fig.add_trace(
                    go.Bar(
                        x=list(class_balance.keys()),
                        y=list(class_balance.values()),
                        name='Class Balance'
                    ),
                    row=2, col=1
                )
        
        # Data quality score
        quality_score = data_profile.get('data_quality_score', {}).get('overall', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Data Distribution Analysis',
            width=1000,
            height=600
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _plot_stability(self, stability_results: Dict[str, Any]) -> str:
        """Generate stability analysis plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('AUC by Group', 'Positive Rate by Group'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        group_metrics = stability_results.get('group_metrics', {})
        if group_metrics:
            groups = list(group_metrics.keys())
            aucs = [group_metrics[g]['auc'] for g in groups]
            positive_rates = [group_metrics[g]['positive_rate'] for g in groups]
            
            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=aucs,
                    name='AUC by Group'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=positive_rates,
                    name='Positive Rate by Group'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Model Stability Analysis',
            width=1000,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _prepare_template_data(
        self,
        model_results: Dict[str, Any],
        data_profile: Dict[str, Any],
        audit_results: Dict[str, Any],
        feature_importance: Dict[str, float],
        calibration_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        metadata: Dict[str, Any],
        plots: Dict[str, str],
        planning_result: Optional[Dict[str, Any]] = None,
        risk_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare data for template rendering."""
        return {
            'model_results': model_results,
            'data_profile': data_profile,
            'audit_results': audit_results,
            'feature_importance': feature_importance,
            'calibration_results': calibration_results,
            'stability_results': stability_results,
            'metadata': metadata,
            'plots': plots,
            'planning_result': planning_result,
            'risk_analysis': risk_analysis,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': '0.3.0'
        }
    
    def _load_template(self) -> Template:
        """Load the HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card - Tabular Agent</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .feature-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }
        .feature-item {
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .feature-name {
            font-weight: bold;
            color: #333;
        }
        .feature-importance {
            color: #007bff;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }
        
        /* Risk Analysis Styles */
        .risk-matrix {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            justify-content: center;
        }
        .risk-item {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            min-width: 120px;
        }
        .risk-item.high-risk {
            background-color: #ffebee;
            border: 2px solid #f44336;
        }
        .risk-item.medium-risk {
            background-color: #fff3e0;
            border: 2px solid #ff9800;
        }
        .risk-item.low-risk {
            background-color: #e8f5e8;
            border: 2px solid #4caf50;
        }
        .risk-count {
            display: block;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .risk-label {
            font-size: 0.9em;
            text-transform: uppercase;
            font-weight: bold;
        }
        .risk-card {
            margin: 15px 0;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid;
        }
        .risk-card.high {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        .risk-card.medium {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        .risk-card.low {
            background-color: #e8f5e8;
            border-left-color: #4caf50;
        }
        .risk-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .risk-level {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .risk-level.high {
            background-color: #f44336;
            color: white;
        }
        .risk-level.medium {
            background-color: #ff9800;
            color: white;
        }
        .risk-level.low {
            background-color: #4caf50;
            color: white;
        }
        .risk-evidence, .risk-suggestions {
            margin: 15px 0;
        }
        .risk-evidence ul, .risk-suggestions ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .suggestion-card {
            margin: 15px 0;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        .suggestion-card.high {
            background-color: #ffebee;
            border-color: #f44336;
        }
        .suggestion-card.medium {
            background-color: #fff3e0;
            border-color: #ff9800;
        }
        .suggestion-card.low {
            background-color: #e8f5e8;
            border-color: #4caf50;
        }
        .suggestion-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .suggestion-priority {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .suggestion-priority.high {
            background-color: #f44336;
            color: white;
        }
        .suggestion-priority.medium {
            background-color: #ff9800;
            color: white;
        }
        .suggestion-priority.low {
            background-color: #4caf50;
            color: white;
        }
        .stability-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-value.grade-A { color: #4caf50; }
        .metric-value.grade-B { color: #8bc34a; }
        .metric-value.grade-C { color: #ff9800; }
        .metric-value.grade-D { color: #ff5722; }
        .metric-value.grade-F { color: #f44336; }
        .metric-label {
            font-size: 0.9em;
            color: #666;
        }
        .stability-summary {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Model Card</h1>
            <p>Generated by Tabular Agent v{{ version }} on {{ timestamp }}</p>
        </div>

        <!-- Model Performance -->
        <div class="section">
            <h2>Model Performance</h2>
            <div class="metrics-grid">
                {% if model_results.auc %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(model_results.auc) }}</div>
                    <div class="metric-label">AUC</div>
                </div>
                {% endif %}
                {% if model_results.accuracy %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(model_results.accuracy) }}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                {% endif %}
                {% if model_results.precision %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(model_results.precision) }}</div>
                    <div class="metric-label">Precision</div>
                </div>
                {% endif %}
                {% if model_results.recall %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(model_results.recall) }}</div>
                    <div class="metric-label">Recall</div>
                </div>
                {% endif %}
                {% if model_results.f1 %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(model_results.f1) }}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                {% endif %}
            </div>
            
            {% if plots.roc_curve %}
            <div class="plot-container">
                {{ plots.roc_curve|safe }}
            </div>
            {% endif %}
            
            {% if plots.pr_curve %}
            <div class="plot-container">
                {{ plots.pr_curve|safe }}
            </div>
            {% endif %}
        </div>

        <!-- Data Profile -->
        <div class="section">
            <h2>Data Profile</h2>
            {% if data_profile.basic_info %}
            <div class="info">
                <strong>Dataset Shape:</strong> {{ data_profile.basic_info.shape[0] }} rows × {{ data_profile.basic_info.shape[1] }} columns
            </div>
            {% endif %}
            
            {% if plots.data_distribution %}
            <div class="plot-container">
                {{ plots.data_distribution|safe }}
            </div>
            {% endif %}
            
            {% if data_profile.data_quality_score %}
            <div class="info">
                <strong>Data Quality Score:</strong> {{ "%.1f"|format(data_profile.data_quality_score.overall) }}/100
            </div>
            {% endif %}
        </div>

        <!-- Planning and Citations -->
        {% if planning_result %}
        <div class="section">
            <h2>Planning and Citations</h2>
            <div class="planning-info">
                <h3>Execution Plan</h3>
                <div class="plan-details">
                    <p><strong>Planning Mode:</strong> {{ planning_result.mode_used|title }}</p>
                    {% if planning_result.fallback_reason %}
                    <div class="warning">
                        <strong>⚠️ Fallback Applied:</strong> {{ planning_result.fallback_reason }}
                    </div>
                    {% endif %}
                    
                    <h4>Feature Engineering Recipes</h4>
                    <ul>
                        {% for recipe in planning_result.plan.feature_recipes %}
                        <li>{{ recipe|replace('_', ' ')|title }}</li>
                        {% endfor %}
                    </ul>
                    
                    <h4>Model Types</h4>
                    <ul>
                        {% for model in planning_result.plan.model_types %}
                        <li>{{ model|title }}</li>
                        {% endfor %}
                    </ul>
                    
                    <h4>Blending Strategy</h4>
                    <p>{{ planning_result.plan.blending_strategy|replace('_', ' ')|title }}</p>
                    
                    {% if planning_result.plan.time_budget_allocation %}
                    <h4>Time Budget Allocation</h4>
                    <ul>
                        {% for component, allocation in planning_result.plan.time_budget_allocation.items() %}
                        <li>{{ component|replace('_', ' ')|title }}: {{ "%.1f"|format(allocation * 100) }}%</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                
                {% if planning_result.citations %}
                <h3>Reference Citations</h3>
                <div class="citations">
                    <p>This plan was informed by {{ planning_result.citations|length }} similar prior runs:</p>
                    {% for citation in planning_result.citations %}
                    <div class="citation">
                        <h4>Run {{ citation.run_id }}</h4>
                        <p><strong>Similarity Score:</strong> {{ "%.3f"|format(citation.score) }}</p>
                        <p><strong>Reason:</strong> {{ citation.reason }}</p>
                        <details>
                            <summary>Configuration Details</summary>
                            <pre>{{ citation.config|tojson(indent=2) }}</pre>
                        </details>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        <!-- Risk Analysis and Stability -->
        {% if risk_analysis %}
        <div class="section">
            <h2>Risk Analysis and Stability</h2>
            
            <!-- Risk Summary -->
            <div class="risk-summary">
                <h3>Risk Summary</h3>
                <div class="risk-matrix">
                    <div class="risk-item high-risk">
                        <span class="risk-count">{{ risk_analysis.risk_summary.high_risks }}</span>
                        <span class="risk-label">High Risk</span>
                    </div>
                    <div class="risk-item medium-risk">
                        <span class="risk-count">{{ risk_analysis.risk_summary.medium_risks }}</span>
                        <span class="risk-label">Medium Risk</span>
                    </div>
                    <div class="risk-item low-risk">
                        <span class="risk-count">{{ risk_analysis.risk_summary.low_risks }}</span>
                        <span class="risk-label">Low Risk</span>
                    </div>
                </div>
            </div>
            
            <!-- Risk Details -->
            {% if risk_analysis.risks %}
            <div class="risk-details">
                <h3>Risk Details</h3>
                {% for risk in risk_analysis.risks %}
                <div class="risk-card {{ risk.level }}">
                    <div class="risk-header">
                        <h4>{{ risk.risk_type|replace('_', ' ')|title }}</h4>
                        <span class="risk-level {{ risk.level }}">{{ risk.level|title }}</span>
                    </div>
                    <p class="risk-description">{{ risk.description }}</p>
                    <div class="risk-evidence">
                        <strong>Evidence:</strong>
                        <ul>
                            {% for key, value in risk.evidence.items() %}
                            <li><strong>{{ key|replace('_', ' ')|title }}:</strong> {{ value }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    <div class="risk-suggestions">
                        <strong>Suggested Actions:</strong>
                        <ul>
                            {% for suggestion in risk.suggestions %}
                            <li>{{ suggestion|replace('_', ' ')|title }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <!-- Retry Suggestions -->
            {% if risk_analysis.retry_suggestions.suggestions %}
            <div class="retry-suggestions">
                <h3>Retry Suggestions</h3>
                <div class="suggestions-priority {{ risk_analysis.retry_suggestions.priority }}">
                    <strong>Priority Level:</strong> {{ risk_analysis.retry_suggestions.priority|title }}
                </div>
                {% for suggestion in risk_analysis.retry_suggestions.suggestions %}
                <div class="suggestion-card {{ suggestion.priority }}">
                    <div class="suggestion-header">
                        <h4>{{ suggestion.action|replace('_', ' ')|title }}</h4>
                        <span class="suggestion-priority {{ suggestion.priority }}">{{ suggestion.priority|title }}</span>
                    </div>
                    <p class="suggestion-description">{{ suggestion.description }}</p>
                    <div class="suggestion-implementation">
                        <strong>Implementation:</strong> {{ suggestion.implementation }}
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <!-- Stability Dashboard -->
            {% if stability_results %}
            <div class="stability-dashboard">
                <h3>Model Stability Dashboard</h3>
                <div class="stability-metrics">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.3f"|format(stability_results.stability_metrics.mean_score) }}</div>
                        <div class="metric-label">Mean Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.3f"|format(stability_results.stability_metrics.std_score) }}</div>
                        <div class="metric-label">Std Deviation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.3f"|format(stability_results.stability_metrics.coefficient_of_variation) }}</div>
                        <div class="metric-label">CV Coefficient</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value grade-{{ stability_results.stability_metrics.stability_grade }}">{{ stability_results.stability_metrics.stability_grade }}</div>
                        <div class="metric-label">Stability Grade</div>
                    </div>
                </div>
                <div class="stability-summary">
                    <p><strong>Overall Assessment:</strong> {{ stability_results.summary.overall_assessment }}</p>
                    <p><strong>Recommendation:</strong> {{ stability_results.summary.recommendation }}</p>
                    <p><strong>Confidence Interval:</strong> ({{ "%.3f"|format(stability_results.stability_metrics.confidence_interval[0]) }}, {{ "%.3f"|format(stability_results.stability_metrics.confidence_interval[1]) }})</p>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Leakage Audit -->
        <div class="section">
            <h2>Leakage Audit</h2>
            {% if audit_results.status == 'success' %}
                {% if audit_results.leakage_indicators.duplicate_rows.is_leakage %}
                <div class="warning">
                    <strong>⚠️ Duplicate Rows Detected:</strong> {{ "%.1f"|format(audit_results.leakage_indicators.duplicate_rows.ratio * 100) }}% overlap between train and test sets
                </div>
                {% endif %}
                {% if audit_results.leakage_indicators.target_leakage.is_leakage %}
                <div class="warning">
                    <strong>⚠️ Target Leakage Detected:</strong> {{ audit_results.leakage_indicators.target_leakage.suspicious_features|join(', ') }}
                </div>
                {% endif %}
                {% if audit_results.leakage_indicators.get('time_leakage', {}).get('is_leakage', False) %}
                <div class="warning">
                    <strong>⚠️ Time Leakage Detected:</strong> Test set contains data from before training period
                </div>
                {% endif %}
                {% if not audit_results.leakage_indicators.get('duplicate_rows', {}).get('is_leakage', False) and not audit_results.leakage_indicators.get('target_leakage', {}).get('is_leakage', False) and not audit_results.leakage_indicators.get('time_leakage', {}).get('is_leakage', False) %}
                <div class="success">
                    <strong>✅ No Obvious Leakage Detected</strong>
                </div>
                {% endif %}
            {% else %}
            <div class="warning">
                <strong>⚠️ Audit Failed:</strong> {{ audit_results.error }}
            </div>
            {% endif %}
        </div>

        <!-- Feature Importance -->
        <div class="section">
            <h2>Feature Importance</h2>
            {% if plots.feature_importance %}
            <div class="plot-container">
                {{ plots.feature_importance|safe }}
            </div>
            {% endif %}
            
            {% if feature_importance %}
            <div class="feature-list">
                {% for feature, importance in feature_importance.items()[:10] %}
                <div class="feature-item">
                    <div class="feature-name">{{ feature }}</div>
                    <div class="feature-importance">Importance: {{ "%.4f"|format(importance) }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <!-- Model Calibration -->
        <div class="section">
            <h2>Model Calibration</h2>
            {% if calibration_results %}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(calibration_results.brier_score) }}</div>
                    <div class="metric-label">Brier Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(calibration_results.expected_calibration_error) }}</div>
                    <div class="metric-label">Expected Calibration Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(calibration_results.maximum_calibration_error) }}</div>
                    <div class="metric-label">Maximum Calibration Error</div>
                </div>
            </div>
            
            {% if plots.calibration %}
            <div class="plot-container">
                {{ plots.calibration|safe }}
            </div>
            {% endif %}
            {% endif %}
        </div>

        <!-- Model Stability -->
        <div class="section">
            <h2>Model Stability</h2>
            {% if stability_results %}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(stability_results.population_stability_index) }}</div>
                    <div class="metric-label">Population Stability Index</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(stability_results.auc_std) }}</div>
                    <div class="metric-label">AUC Standard Deviation</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.4f"|format(stability_results.positive_rate_std) }}</div>
                    <div class="metric-label">Positive Rate Std Dev</div>
                </div>
            </div>
            
            {% if plots.stability %}
            <div class="plot-container">
                {{ plots.stability|safe }}
            </div>
            {% endif %}
            {% endif %}
        </div>

        <!-- Metadata -->
        <div class="section">
            <h2>Model Metadata</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                {% for key, value in metadata.items() %}
                <tr>
                    <td>{{ key|title }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="footer">
            <p>Generated by Tabular Agent v{{ version }} | {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_str)
