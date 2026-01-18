# Contributing to SCBE-AETHERMOORE Python SDK

Thank you for your interest in contributing to the SCBE-AETHERMOORE Python SDK! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize the project's best interests

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- pip >= 23.0
- Docker (optional, for containerized testing)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/aws-lambda-simple-web-app.git
cd aws-lambda-simple-web-app
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to verify setup
pytest -v

# Check imports
python -c "from scbe_production import __version__; print(f'SCBE v{__version__}')"
```

---

## Making Changes

### Branch Naming Convention

Use descriptive branch names:

```
feature/add-new-cipher-mode
fix/geoseal-threshold-calculation
docs/update-api-reference
refactor/simplify-governance-engine
```

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=symphonic_cipher --cov=scbe_production --cov-report=term-missing

# Run specific test file
pytest tests/test_pqc.py -v

# Run tests matching pattern
pytest -k "test_governance" -v

# Run with parallel execution
pytest -n auto
```

### Test Structure

```
tests/
├── __init__.py
├── test_pqc.py              # Post-quantum crypto tests
├── test_governance.py       # Governance engine tests
├── test_geoseal.py          # GeoSeal manifold tests
└── conftest.py              # Pytest fixtures
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from scbe_production.service import SCBEProductionService

class TestSCBEService:
    """Tests for SCBEProductionService."""

    @pytest.fixture
    def service(self):
        """Create a test service instance."""
        return SCBEProductionService()

    def test_health_check(self, service):
        """Test health check returns healthy status."""
        result = service.health_check()
        assert result["status"] == "healthy"
        assert "components" in result

    def test_seal_memory(self, service):
        """Test memory sealing operation."""
        shard = service.seal_memory(
            plaintext=b"test data",
            agent_id="test-agent",
            topic="testing",
            position=(1, 2, 3, 5, 8, 13),
        )
        assert shard.sealed_data is not None
        assert shard.topic == "testing"
```

---

## Code Style

### Python Style Guide

We follow PEP 8 with these tools:

```bash
# Format code with Black
black symphonic_cipher/ scbe_production/ tests/

# Check with flake8
flake8 symphonic_cipher/ scbe_production/ tests/

# Type checking with mypy
mypy symphonic_cipher/ scbe_production/
```

### Configuration

See `pyproject.toml` for tool configurations:

```toml
[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
```

### Docstring Convention

Use Google-style docstrings:

```python
def seal_memory(
    self,
    plaintext: bytes,
    agent_id: str,
    topic: str,
    position: Tuple[int, ...],
) -> MemoryShard:
    """Seal a memory shard with full SCBE protection.

    Applies the 14-layer security pipeline including PQC encryption,
    Sacred Tongue encoding, and GeoSeal manifold binding.

    Args:
        plaintext: The raw data to seal.
        agent_id: Unique identifier for the owning agent.
        topic: Classification topic for the memory.
        position: Fibonacci spiral position tuple.

    Returns:
        A sealed MemoryShard with cryptographic protection.

    Raises:
        ValidationError: If inputs fail validation.
        PQCError: If cryptographic operations fail.

    Example:
        >>> service = SCBEProductionService()
        >>> shard = service.seal_memory(
        ...     plaintext=b"secret",
        ...     agent_id="agent-001",
        ...     topic="classified",
        ...     position=(1, 2, 3, 5, 8, 13),
        ... )
    """
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Examples

```
feat(governance): add SNAP protocol failsafe

Implements the SNAP (Secure Noise Abort Protocol) that triggers
when risk thresholds are exceeded. This provides a cryptographic
failsafe that destroys the secret rather than allowing breach.

Closes #42
```

```
fix(geoseal): correct interior threshold calculation

The threshold was too restrictive at 0.3, causing false exterior
classifications for benign requests. Changed to 0.5 based on
mathematical analysis of the dual-space projection.
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] No linting errors (`flake8`)
- [ ] Type hints are correct (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

### 2. PR Description Template

```markdown
## Summary
Brief description of changes.

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
Describe how you tested the changes.

## Screenshots (if applicable)

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### 3. Review Process

1. Create PR against `main` branch
2. Automated CI checks run
3. Code review by maintainers
4. Address feedback
5. Squash and merge

---

## Architecture Overview

### Package Structure

```
scbe-aethermoore/
├── symphonic_cipher/        # Core cipher implementation
│   ├── scbe_aethermoore/   # Main SCBE modules
│   │   ├── pqc/            # Post-quantum cryptography
│   │   ├── layers/         # 14-layer pipeline
│   │   ├── spiral_seal/    # SpiralSeal SS1
│   │   └── qc_lattice/     # Quasicrystal lattice
│   └── tests/              # Unit tests
│
├── scbe_production/         # Production-ready service
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration management
│   ├── exceptions.py       # Error hierarchy
│   ├── logging.py          # Audit logging
│   └── service.py          # Unified API
│
└── docs/                    # Documentation
```

### Key Components

1. **PQC Layer**: ML-KEM-768 + ML-DSA-65 post-quantum cryptography
2. **Sacred Tongues**: 6-tongue × 256-token steganographic encoding
3. **GeoSeal Manifold**: Dual-space (sphere + hypercube) trust verification
4. **Governance Engine**: Risk scoring with ALLOW/QUARANTINE/DENY/SNAP
5. **Harmonic Wall**: Exponential cost growth for adversarial drift

---

## Questions?

- Open an [issue](https://github.com/issdandavis/aws-lambda-simple-web-app/issues)
- Check existing documentation in `/docs`

Thank you for contributing!
