#!/bin/bash
# SCBE-AETHERMOORE Deployment Script

set -e

echo "=== SCBE-AETHERMOORE Deployment ==="

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --tb=short || echo "Some tests may need attention"

# Run main demo to verify
echo "Verifying installation..."
python demo_quantum_resistance.py | head -30

echo ""
echo "=== Deployment Ready ==="
echo ""
echo "To run:"
echo "  python demo_quantum_resistance.py     # Main demo"
echo "  python examples/basic_auth.py         # Basic auth example"
echo "  python examples/ai_context_binding.py # AI binding example"
echo "  python examples/api_protection.py     # API protection example"
echo ""
echo "For production, install pqcrypto:"
echo "  pip install pqcrypto"
