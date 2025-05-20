.PHONY: test lint type format coverage clean all help integration unit

# Variables
PYTHON = python
PYTEST = pytest
PYTEST_ARGS = --verbose
PYTEST_COV_ARGS = --cov=cdas --cov-report=term-missing
FLAKE8 = flake8
BLACK = black
MYPY = mypy

help:
	@echo "Available commands:"
	@echo "  make test        - Run all tests"
	@echo "  make unit        - Run unit tests"
	@echo "  make integration - Run integration tests"
	@echo "  make coverage    - Run tests with coverage report"
	@echo "  make lint        - Run linter"
	@echo "  make type        - Run type checker"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make all         - Run all checks and tests"

test:
	$(PYTEST) $(PYTEST_ARGS)

unit:
	$(PYTEST) $(PYTEST_ARGS) tests/test_db/ tests/test_analysis/ tests/test_financial_analysis/ tests/test_document_processor/ tests/test_reporting/ tests/test_ai/

integration:
	$(PYTEST) $(PYTEST_ARGS) tests/test_integration/

coverage:
	$(PYTEST) $(PYTEST_COV_ARGS)

lint:
	$(FLAKE8) cdas/ tests/

type:
	$(MYPY) cdas/

format:
	$(BLACK) cdas/ tests/

clean:
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type test