#!/usr/bin/env python3
"""Local test script for tabular-agent v1.0"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run local tests"""
    print("🧪 Running tabular-agent v1.0 local tests...")
    
    # Test 1: Check installation
    success1 = run_command("pip show tabular-agent", "Check package installation")
    
    # Test 2: Check CLI help
    success2 = run_command("tabular-agent --help", "Check CLI help")
    
    # Test 3: Check subcommands
    success3 = run_command("tabular-agent run --help", "Check run subcommand")
    success4 = run_command("tabular-agent audit --help", "Check audit subcommand")
    success5 = run_command("tabular-agent blend --help", "Check blend subcommand")
    
    # Test 4: Run basic pipeline
    success6 = run_command(
        "tabular-agent run --train examples/train_binary.csv --test examples/test_binary.csv --target target --out runs/local_test --verbose",
        "Run basic pipeline"
    )
    
    # Test 5: Run pytest
    success7 = run_command("pytest tests/ -v", "Run unit tests")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    tests = [
        ("Package installation", success1),
        ("CLI help", success2),
        ("Run subcommand", success3),
        ("Audit subcommand", success4),
        ("Blend subcommand", success5),
        ("Basic pipeline", success6),
        ("Unit tests", success7),
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
