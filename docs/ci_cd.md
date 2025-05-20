# CI/CD Pipeline for CDAS

This document describes the continuous integration and continuous deployment (CI/CD) pipeline for the Construction Document Analysis System (CDAS).

## Overview

The CI/CD pipeline automates the following processes:

1. **Testing**: Runs unit tests, integration tests, and code quality checks
2. **Building**: Packages the application for distribution
3. **Deploying**: Publishes releases to PyPI (when applicable)

## CI Pipeline Components

### GitHub Actions Workflows

The CI pipeline uses GitHub Actions with the following workflows:

1. **Test Workflow** (`.github/workflows/test.yml`):
   - Runs on multiple Python versions (3.8, 3.9, 3.10)
   - Performs code formatting checks
   - Runs linters and type checkers
   - Executes unit and integration tests
   - Generates and uploads coverage reports

2. **Deploy Workflow** (`.github/workflows/deploy.yml`):
   - Triggers on release creation
   - Builds distribution packages
   - Publishes to TestPyPI for verification
   - Publishes to PyPI for tagged releases

### Local Development Tools

For local development and testing, the following tools are available:

1. **Makefile**:
   - `make test`: Run all tests
   - `make unit`: Run unit tests only
   - `make integration`: Run integration tests only
   - `make coverage`: Run tests with coverage reporting
   - `make lint`: Run linters
   - `make type`: Run type checkers
   - `make format`: Format code with Black
   - `make clean`: Remove temporary files
   - `make all`: Run all checks and tests

2. **Tox**:
   - Tests across multiple Python versions
   - Isolates testing environments
   - Run with `tox` command

3. **Setup Script**:
   - `scripts/setup_testing.py`: Sets up a testing environment

## Test Configuration

### pytest Configuration

The `pytest.ini` file configures test discovery and execution:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --verbose
markers =
    unit: mark a test as a unit test
    integration: mark a test as an integration test
    slow: mark a test as slow
```

### Coverage Configuration

The `.codecov.yml` file configures coverage reporting:

- Sets coverage targets (80%)
- Configures reporting precision and thresholds
- Customizes PR comments

## Workflow

### Development Workflow

1. Create a feature branch from `develop`
2. Run tests locally with `make test`
3. Submit a PR to merge into `develop`
4. CI automatically runs tests on the PR
5. After review and approval, merge into `develop`

### Release Workflow

1. Create a release branch from `develop`
2. Version bump in `setup.py` or `pyproject.toml`
3. Submit a PR to merge into `main`
4. After approval and merge, create a GitHub release (with tag)
5. CI automatically publishes to PyPI

## Setting Up the CI Pipeline

### GitHub Setup

1. Go to GitHub repository settings
2. Configure secrets for PyPI deployment:
   - `PYPI_API_TOKEN`: API token for PyPI
   - `TEST_PYPI_API_TOKEN`: API token for TestPyPI

### Local Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run the setup script:
   ```bash
   python scripts/setup_testing.py
   ```

## Running Tests Locally

### Using Makefile

```bash
# Run all tests
make test

# Run with coverage report
make coverage

# Run all checks (format, lint, type, test)
make all
```

### Using Tox

```bash
# Run on all Python versions
tox

# Run on specific Python version
tox -e py39

# Run linting only
tox -e lint

# Run type checking only
tox -e type
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_db/test_models.py

# Run with coverage
pytest --cov=cdas

# Run integration tests only
pytest tests/test_integration/
```