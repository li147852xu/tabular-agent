"""Data leakage audit using leakage-buster integration."""

import pandas as pd
import numpy as np
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings


class LeakageAuditor:
    """Data leakage auditor with leakage-buster integration."""
    
    def __init__(self, audit_cli: Optional[str] = None):
        """
        Initialize leakage auditor.
        
        Args:
            audit_cli: Path to leakage-buster CLI (optional)
        """
        self.audit_cli = audit_cli
        self.audit_results = {}
    
    def audit(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform leakage audit on train/test data.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            target_col: Target column name
            time_col: Time column name (optional)
            
        Returns:
            Dictionary containing audit results
        """
        # Try leakage-buster integration first
        if self.audit_cli:
            try:
                self.audit_results = self._audit_with_leakage_buster(
                    train_df, test_df, target_col, time_col
                )
            except Exception as e:
                warnings.warn(f"Leakage-buster audit failed: {e}")
                self.audit_results = self._basic_leakage_audit(
                    train_df, test_df, target_col, time_col
                )
        else:
            # Fallback to basic leakage detection
            self.audit_results = self._basic_leakage_audit(
                train_df, test_df, target_col, time_col
            )
        
        return self.audit_results
    
    def _audit_with_leakage_buster(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Audit using leakage-buster CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save data to temporary files
            train_path = temp_path / "train.csv"
            test_path = temp_path / "test.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Prepare leakage-buster command
            cmd = [
                self.audit_cli,
                "--train", str(train_path),
                "--test", str(test_path),
                "--target", target_col,
                "--output", str(temp_path / "audit_results.json")
            ]
            
            if time_col:
                cmd.extend(["--time-col", time_col])
            
            # Run leakage-buster
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                
                # Load results
                results_path = temp_path / "audit_results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        audit_results = json.load(f)
                else:
                    # Parse stdout if no JSON output
                    audit_results = self._parse_leakage_buster_output(result.stdout)
                
                return {
                    "method": "leakage-buster",
                    "status": "success",
                    "results": audit_results,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                
            except subprocess.CalledProcessError as e:
                return {
                    "method": "leakage-buster",
                    "status": "error",
                    "error": str(e),
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                }
            except subprocess.TimeoutExpired:
                return {
                    "method": "leakage-buster",
                    "status": "timeout",
                    "error": "Leakage-buster timed out after 5 minutes",
                }
    
    def _parse_leakage_buster_output(self, stdout: str) -> Dict[str, Any]:
        """Parse leakage-buster stdout output."""
        # This is a basic parser - would need to be customized based on actual output format
        lines = stdout.strip().split('\n')
        results = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        results[key] = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        results[key] = float(value)
                    else:
                        results[key] = value
                except ValueError:
                    results[key] = value
        
        return results
    
    def _basic_leakage_audit(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Basic leakage detection without leakage-buster."""
        audit_results = {
            "method": "basic",
            "status": "success",
            "leakage_indicators": {},
            "recommendations": [],
        }
        
        # Check for exact duplicates between train and test
        train_features = train_df.drop(columns=[target_col])
        test_features = test_df.drop(columns=[target_col])
        
        # Find common rows
        common_rows = pd.merge(train_features, test_features, how='inner')
        duplicate_ratio = len(common_rows) / len(test_features)
        
        audit_results["leakage_indicators"]["duplicate_rows"] = {
            "count": len(common_rows),
            "ratio": duplicate_ratio,
            "is_leakage": duplicate_ratio > 0.01,  # More than 1% overlap
        }
        
        if duplicate_ratio > 0.01:
            audit_results["recommendations"].append(
                f"High overlap between train and test sets ({duplicate_ratio:.1%}) - potential data leakage"
            )
        
        # Check for target leakage in features
        target_leakage = self._check_target_leakage(train_df, test_df, target_col)
        audit_results["leakage_indicators"]["target_leakage"] = target_leakage
        
        if target_leakage["suspicious_features"]:
            audit_results["recommendations"].append(
                f"Potential target leakage in features: {target_leakage['suspicious_features']}"
            )
        
        # Check for time leakage
        if time_col:
            time_leakage = self._check_time_leakage(train_df, test_df, time_col)
            audit_results["leakage_indicators"]["time_leakage"] = time_leakage
            
            if time_leakage["is_leakage"]:
                audit_results["recommendations"].append(
                    "Time leakage detected - test set contains data from before training period"
                )
        
        # Check for high correlation features
        correlation_leakage = self._check_correlation_leakage(train_df, test_df, target_col)
        audit_results["leakage_indicators"]["correlation_leakage"] = correlation_leakage
        
        if correlation_leakage["high_correlation_features"]:
            audit_results["recommendations"].append(
                f"Features with suspiciously high correlation: {correlation_leakage['high_correlation_features']}"
            )
        
        return audit_results
    
    def _check_target_leakage(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str
    ) -> Dict[str, Any]:
        """Check for target leakage in features."""
        suspicious_features = []
        
        # Check if any feature perfectly predicts target
        for col in train_df.columns:
            if col == target_col:
                continue
            
            if train_df[col].dtype in ['object', 'category']:
                # For categorical features, check if any category perfectly predicts target
                for category in train_df[col].unique():
                    if pd.isna(category):
                        continue
                    
                    category_mask = train_df[col] == category
                    if category_mask.sum() > 1:  # At least 2 samples
                        target_values = train_df.loc[category_mask, target_col]
                        if target_values.nunique() == 1:  # All same target value
                            suspicious_features.append(f"{col}={category}")
            else:
                # For numeric features, check for perfect correlation
                if train_df[col].corr(train_df[target_col]) > 0.99:
                    suspicious_features.append(col)
        
        return {
            "suspicious_features": suspicious_features,
            "is_leakage": len(suspicious_features) > 0,
        }
    
    def _check_time_leakage(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        time_col: str
    ) -> Dict[str, Any]:
        """Check for time leakage."""
        train_times = pd.to_datetime(train_df[time_col])
        test_times = pd.to_datetime(test_df[time_col])
        
        train_max = train_times.max()
        test_min = test_times.min()
        
        # Check if test set has data from before training period
        is_leakage = test_min < train_max
        
        return {
            "train_max_time": train_max,
            "test_min_time": test_min,
            "is_leakage": is_leakage,
            "time_gap_days": (test_min - train_max).days if not is_leakage else 0,
        }
    
    def _check_correlation_leakage(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target_col: str
    ) -> Dict[str, Any]:
        """Check for suspiciously high correlations."""
        high_correlation_features = []
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        for col in numeric_cols:
            if col in train_df.columns and col in test_df.columns:
                train_corr = train_df[col].corr(train_df[target_col])
                test_corr = test_df[col].corr(test_df[target_col])
                
                # Check for suspiciously high correlation
                if abs(train_corr) > 0.9 or abs(test_corr) > 0.9:
                    high_correlation_features.append({
                        "feature": col,
                        "train_correlation": train_corr,
                        "test_correlation": test_corr,
                    })
        
        return {
            "high_correlation_features": high_correlation_features,
            "is_leakage": len(high_correlation_features) > 0,
        }
    
    def get_summary(self) -> str:
        """Get a summary of audit results."""
        if not self.audit_results:
            return "No audit results available"
        
        method = self.audit_results.get("method", "unknown")
        status = self.audit_results.get("status", "unknown")
        
        summary_parts = [f"Audit Method: {method}", f"Status: {status}"]
        
        if status == "success":
            leakage_indicators = self.audit_results.get("leakage_indicators", {})
            
            # Check for various leakage types
            if leakage_indicators.get("duplicate_rows", {}).get("is_leakage", False):
                summary_parts.append("⚠️  Duplicate rows detected between train/test")
            
            if leakage_indicators.get("target_leakage", {}).get("is_leakage", False):
                summary_parts.append("⚠️  Potential target leakage detected")
            
            if leakage_indicators.get("time_leakage", {}).get("is_leakage", False):
                summary_parts.append("⚠️  Time leakage detected")
            
            if leakage_indicators.get("correlation_leakage", {}).get("is_leakage", False):
                summary_parts.append("⚠️  Suspiciously high correlations detected")
            
            if not any([
                leakage_indicators.get("duplicate_rows", {}).get("is_leakage", False),
                leakage_indicators.get("target_leakage", {}).get("is_leakage", False),
                leakage_indicators.get("time_leakage", {}).get("is_leakage", False),
                leakage_indicators.get("correlation_leakage", {}).get("is_leakage", False),
            ]):
                summary_parts.append("✅ No obvious leakage detected")
        
        return "\n".join(summary_parts)
