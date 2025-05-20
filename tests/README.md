# CDAS Testing Documentation

This directory contains tests for the Construction Document Analysis System (CDAS). The tests are organized into the following categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the interaction between components
- **Functional Tests**: Test the system from the user's perspective

## Test Structure

```
tests/
├─ test_ai/                 # AI component tests
├─ test_analysis/           # Analysis component tests
├─ test_db/                 # Database component tests
├─ test_document_processor/ # Document processor tests
├─ test_financial_analysis/ # Financial analysis tests
├─ test_integration/        # Integration tests
└─ test_reporting/          # Reporting component tests
```

## Running Tests

### Running All Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=cdas --cov-report=term-missing
```

### Running Integration Tests

Integration tests verify that the system components work together correctly. These tests simulate real-world usage scenarios and validate end-to-end workflows.

```bash
# Run all integration tests
pytest tests/test_integration/

# Using the dedicated script
./tests/run_integration_tests.py

# With verbose output and coverage
./tests/run_integration_tests.py --verbose --coverage
```

### Running Specific Test Categories

```bash
# Run database tests
pytest tests/test_db/

# Run document processor tests
pytest tests/test_document_processor/

# Run financial analysis tests
pytest tests/test_financial_analysis/
```

## Integration Test Scenarios

The integration tests cover the following key scenarios:

1. **Document Processing**
   - Document ingestion from various file formats
   - Extraction of structured data
   - Storage in the database
   - Handling of related documents

2. **Financial Analysis**
   - Pattern detection across documents
   - Anomaly detection in financial data
   - Relationship tracking between documents
   - Chronological analysis of financial transactions

3. **CLI Commands**
   - Document management commands
   - Analysis commands
   - Reporting commands

## Test Fixtures

The integration tests use the following fixtures:

- `test_db_path`: Creates a temporary database file
- `test_docs_dir`: Creates a temporary directory for test documents
- `test_session`: Creates a database session for testing
- `document_processor`: Creates a document processor instance
- `analysis_engine`: Creates a financial analysis engine instance
- `sample_payment_app`: Creates a sample payment application document
- `sample_change_order`: Creates a sample change order document

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Test Organization**: 
   - Place unit tests in the appropriate module directory
   - Place integration tests in the `test_integration` directory

2. **Test Naming**:
   - Name test files with the `test_` prefix
   - Name test functions with the `test_` prefix
   - Use descriptive names that indicate what is being tested

3. **Test Documentation**:
   - Add a docstring to each test module explaining what it tests
   - Add a docstring to each test function explaining the test scenario

4. **Test Coverage**:
   - Aim for high test coverage (>80%)
   - Ensure critical paths are well-tested
   - Use parameterized tests for testing multiple scenarios