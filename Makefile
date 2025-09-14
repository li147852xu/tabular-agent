# Tabular Agent v1.0 Makefile

.PHONY: help venv install test build clean selfcheck demo

# Default target
help:
	@echo "Tabular Agent v1.0 - Available targets:"
	@echo ""
	@echo "  venv       - Create virtual environment"
	@echo "  install    - Install package in development mode"
	@echo "  test       - Run tests with coverage"
	@echo "  build      - Build package and Docker image"
	@echo "  clean      - Clean build artifacts"
	@echo "  selfcheck  - Complete self-check (venv + install + test + demo)"
	@echo "  demo       - Run demo on example datasets"
	@echo ""

# Virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created in ./venv"
	@echo "Activate with: source venv/bin/activate"

# Install package
install: venv
	@echo "Installing package in development mode..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[dev,audit,blend]
	@echo "Package installed successfully"

# Run tests
test: install
	@echo "Running tests with coverage..."
	./venv/bin/pytest tests/ --cov=src/ --cov-report=html --cov-report=term-missing -v
	@echo "Tests completed. Coverage report: htmlcov/index.html"

# Build package and Docker image
build: install
	@echo "Building package..."
	./venv/bin/python -m build
	@echo "Package built successfully"
	@echo "Building Docker image..."
	docker build -t tabular-agent:latest .
	@echo "Docker image built successfully"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf runs/
	@echo "Clean completed"

# Complete self-check
selfcheck: clean venv install test demo
	@echo ""
	@echo "=========================================="
	@echo "Self-check completed successfully!"
	@echo "=========================================="
	@echo ""
	@echo "Summary:"
	@echo "  ✓ Virtual environment created"
	@echo "  ✓ Package installed with all dependencies"
	@echo "  ✓ All tests passed"
	@echo "  ✓ Demo runs completed"
	@echo ""
	@echo "Ready for production deployment!"

# Run demo on example datasets
demo: install
	@echo "Running demo on example datasets..."
	@echo ""
	@echo "1. Binary classification demo..."
	./venv/bin/tabular-agent run \
		--train examples/train_binary.csv \
		--test examples/test_binary.csv \
		--target target \
		--n-jobs 2 \
		--time-budget 60 \
		--out runs/demo_binary \
		--verbose
	@echo ""
	@echo "2. Time series demo..."
	./venv/bin/tabular-agent run \
		--train examples/train_timeseries.csv \
		--test examples/test_timeseries.csv \
		--target target \
		--time-col date \
		--n-jobs 2 \
		--time-budget 60 \
		--out runs/demo_timeseries \
		--verbose
	@echo ""
	@echo "3. Audit demo..."
	./venv/bin/tabular-agent audit \
		--data examples/train_binary.csv \
		--target target \
		--out runs/audit_demo \
		--verbose
	@echo ""
	@echo "4. Blend demo..."
	./venv/bin/tabular-agent blend \
		--models runs/demo_binary \
		--out runs/blend_demo \
		--verbose
	@echo ""
	@echo "Demo completed successfully!"
	@echo "Results saved in runs/ directory"

# Development helpers
lint: install
	@echo "Running linters..."
	./venv/bin/flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	./venv/bin/flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	./venv/bin/mypy src/ --ignore-missing-imports
	@echo "Linting completed"

format: install
	@echo "Formatting code..."
	./venv/bin/black src/ tests/
	./venv/bin/isort src/ tests/
	@echo "Code formatted"

# Docker helpers
docker-run: build
	@echo "Running tabular-agent in Docker..."
	docker run --rm -v $(PWD)/examples:/app/examples -v $(PWD)/runs:/app/runs \
		tabular-agent:latest run \
		--train examples/train_binary.csv \
		--test examples/test_binary.csv \
		--target target \
		--out runs/docker_demo

# Release helpers
release-check: install
	@echo "Checking release readiness..."
	./venv/bin/python -m build
	./venv/bin/twine check dist/*
	@echo "Release check passed"

# Performance test
perf-test: install
	@echo "Running performance test..."
	time ./venv/bin/tabular-agent run \
		--train examples/train_binary.csv \
		--test examples/test_binary.csv \
		--target target \
		--n-jobs 4 \
		--time-budget 120 \
		--out runs/perf_test \
		--verbose
