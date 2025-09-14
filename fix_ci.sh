#!/bin/bash
# Quick fix script for tabular-agent CI/CD issues

echo "🔧 Fixing tabular-agent CI/CD issues..."

# 1. Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Not in tabular-agent directory"
    exit 1
fi

# 2. Install dependencies
echo "📦 Installing dependencies..."
pip install -e .[dev] || {
    echo "❌ Failed to install dependencies"
    exit 1
}

# 3. Run tests
echo "🧪 Running tests..."
pytest tests/ -v || {
    echo "❌ Tests failed"
    exit 1
}

# 4. Test CLI
echo "🔍 Testing CLI..."
tabular-agent --help > /dev/null || {
    echo "❌ CLI test failed"
    exit 1
}

# 5. Test pipeline
echo "🚀 Testing pipeline..."
tabular-agent run \
    --train examples/train_binary.csv \
    --test examples/test_binary.csv \
    --target target \
    --out runs/fix_test \
    --verbose || {
    echo "❌ Pipeline test failed"
    exit 1
}

# 6. Check GitHub Actions files
echo "📋 Checking GitHub Actions configuration..."
if [ ! -f ".github/workflows/ci.yml" ]; then
    echo "❌ Missing .github/workflows/ci.yml"
    exit 1
fi

if [ ! -f ".github/workflows/release.yml" ]; then
    echo "❌ Missing .github/workflows/release.yml"
    exit 1
fi

# 7. Check for common issues
echo "🔍 Checking for common issues..."

# Check Python versions in CI
if grep -q "python-version.*3\.1[^0-9]" .github/workflows/*.yml; then
    echo "⚠️  Warning: Found Python 3.1 reference (should be 3.11)"
fi

# Check for missing secrets
if ! grep -q "secrets\." .github/workflows/release.yml; then
    echo "⚠️  Warning: No secrets configured in release workflow"
fi

echo "✅ All checks passed!"
echo ""
echo "📋 Next steps:"
echo "1. Configure GitHub Secrets (see docs/setup-secrets.md)"
echo "2. Push changes to trigger CI"
echo "3. Create v1.0.0 tag to trigger release"
echo ""
echo "🎉 tabular-agent v1.0.0 is ready for production!"
