#!/usr/bin/env python3
"""Test script for tabular-agent pipeline."""

import subprocess
import sys
from pathlib import Path


def test_basic_pipeline():
    """Test basic binary classification pipeline."""
    print("Testing basic binary classification pipeline...")
    
    cmd = [
        "tabular-agent", "run",
        "--train", "examples/train_binary.csv",
        "--test", "examples/test_binary.csv", 
        "--target", "target",
        "--n-jobs", "2",
        "--time-budget", "30",
        "--out", "runs/test_basic"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Basic pipeline test passed!")
        return True
    else:
        print(f"❌ Basic pipeline test failed: {result.stderr}")
        return False


def test_timeseries_pipeline():
    """Test time series pipeline."""
    print("Testing time series pipeline...")
    
    cmd = [
        "tabular-agent", "run",
        "--train", "examples/train_timeseries.csv",
        "--test", "examples/test_timeseries.csv",
        "--target", "target", 
        "--time-col", "date",
        "--n-jobs", "2",
        "--time-budget", "30",
        "--out", "runs/test_timeseries"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Time series pipeline test passed!")
        return True
    else:
        print(f"❌ Time series pipeline test failed: {result.stderr}")
        return False


def check_output_files(run_dir):
    """Check if required output files exist."""
    run_path = Path(run_dir)
    
    # Find the latest run directory
    run_dirs = [d for d in run_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return False
    
    latest_run = max(run_dirs, key=lambda x: x.name)
    
    required_files = ["model_card.html", "results.json", "meta.json"]
    
    for file_name in required_files:
        file_path = latest_run / file_name
        if not file_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
    
    print(f"✅ All required files found in {latest_run}")
    return True


def main():
    """Run all tests."""
    print("Running tabular-agent pipeline tests...\n")
    
    # Generate example data if not exists
    if not Path("examples/train_binary.csv").exists():
        print("Generating example data...")
        subprocess.run([sys.executable, "src/tabular_agent/examples/generate_example_data.py"])
    
    # Run tests
    tests = [
        test_basic_pipeline,
        test_timeseries_pipeline,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Check output files
    print("Checking output files...")
    if check_output_files("runs/test_basic") and check_output_files("runs/test_timeseries"):
        print("✅ All output files generated correctly!")
    else:
        print("❌ Some output files are missing!")
        return 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
