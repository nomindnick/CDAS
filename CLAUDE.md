# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Construction Document Analysis System (CDAS) is a specialized tool designed for attorneys representing public agencies (particularly school districts) in construction disputes. Its primary purpose is to process, analyze, and reconcile financial information across various document types to:

1. Track financial claims and counterclaims between parties (district vs. contractor)
2. Identify discrepancies in amounts across different document types
3. Detect suspicious financial patterns (e.g., rejected change orders reappearing in payment applications)
4. Generate comprehensive reports for dispute resolution conferences
5. Provide evidence-backed analysis with direct citations to source documents

## Project Architecture

The system follows a modular architecture with these core components:

- **Document Processor**: Extracts structured data from construction documents (PDFs, Excel, images)
- **Database Schema**: Stores document metadata, line items, relationships, and analysis flags
- **Financial Analysis Engine**: Detects patterns, anomalies, and relationships in financial data
- **AI Integration**: Uses LLMs for document understanding and analysis
- **Reporting System**: Generates well-structured reports with evidence citations
- **Command Line Interface**: Main user interface for all system functions
- **Network Analysis**: Builds and analyzes relationship networks between documents

## Project Structure

```
cdas/                   # Main package directory
├── __init__.py         # Package initialization
├── ai/                 # AI integration module (LLM integration)
├── analysis/           # General analysis tools
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── db/                 # Database management
├── document_processor/ # Document processing system
├── financial_analysis/ # Financial analysis engine
└── reporting/          # Report generation system

docs/                   # Documentation
logs/                   # Log files (not in version control)
migrations/             # Database migration scripts
scripts/                # Utility scripts
tests/                  # Test suite
├── logs/               # Test log files (not in version control)
└── synthetic_data/     # Test data for integration tests
```

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Setup testing environment
python scripts/setup_testing.py

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration values
```

### Database Commands

```bash
# Initialize database schema
python -m cdas.db.init

# Run database migrations
alembic upgrade head

# Reset database (development only)
python -m cdas.db.reset

# Create a new migration
alembic revision --autogenerate -m "Description of changes"
```

### Running the Application

```bash
# Run the CLI application
python -m cdas.cli

# Run with debug logging
python -m cdas.cli --verbose

# Get help on commands
python -m cdas.cli --help
python -m cdas.cli doc --help
python -m cdas.cli analyze --help
```

### Testing

Using Makefile:

```bash
# Run all tests
make test

# Run unit tests only
make unit

# Run integration tests only
make integration

# Run tests with coverage report
make coverage

# Run all checks (format, lint, type, test)
make all
```

Using pytest directly:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_document_processor/

# Run specific test
pytest tests/test_document_processor/test_pdf.py::test_extract_amounts

# Run tests with coverage report
pytest --cov=cdas --cov-report=term-missing

# Run integration tests
pytest tests/test_integration/
```

Using Tox for multi-environment testing:

```bash
# Run tests on all Python versions
tox

# Run on specific Python version
tox -e py39

# Run linting only
tox -e lint

# Run type checking only
tox -e type
```

### Code Quality

```bash
# Run code formatter
make format
# or directly:
black cdas/ tests/

# Run linter
make lint
# or directly:
flake8 cdas/ tests/

# Run type checker
make type
# or directly:
mypy cdas/

# Clean temporary files
make clean
```

## Core Components

### Document Processor

The Document Processor extracts structured data from various document types:

- PDF files: Uses pdfplumber for text and table extraction
- Excel spreadsheets: Uses pandas for data extraction
- Scanned documents: Uses OCR for text extraction and handwriting recognition

Example:
```python
from cdas.document_processor.processor import DocumentProcessor
from cdas.db.session import get_session

# Create session
session = get_session()

# Initialize document processor
processor = DocumentProcessor(session)

# Process a document
result = processor.process_document(
    "path/to/document.pdf",
    doc_type="payment_app",
    party="contractor"
)
```

Using the factory:
```python
from cdas.db.session import session_scope
from cdas.document_processor.factory import DocumentProcessorFactory

with session_scope() as session:
    factory = DocumentProcessorFactory()
    
    # Process a single document in one call
    result = factory.process_single_document(
        session,
        "path/to/document.pdf",
        "payment_app",
        "contractor"
    )
```

### Financial Analysis Engine

The Financial Analysis Engine identifies patterns and anomalies in financial data:

- Pattern Detection: Finds recurring amounts, reappearing amounts, inconsistent markups
- Anomaly Detection: Identifies statistical outliers and rule-based anomalies
- Amount Matching: Traces amounts across documents, with support for fuzzy matching
- Chronological Analysis: Tracks financial changes over time

Example:
```python
from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.db.session import get_session

# Create session
session = get_session()

# Initialize analysis engine
engine = FinancialAnalysisEngine(session)

# Find suspicious patterns
patterns = engine.find_suspicious_patterns()

# Analyze a specific amount
amount_analysis = engine.analyze_amount(12345.67)
```

### Network Analysis

The Network Analysis component builds and analyzes relationship networks:

```python
from cdas.db.session import session_scope
from cdas.analysis.network import NetworkAnalyzer

with session_scope() as session:
    # Create network analyzer
    analyzer = NetworkAnalyzer(session)
    
    # Build document network
    graph = analyzer.build_document_network()
    
    # Find circular references
    cycles = analyzer.find_circular_references()
    
    # Find isolated documents
    isolated = analyzer.find_isolated_documents()
    
    # Visualize the network
    output_path = analyzer.visualize_network(output_path="network.png")
```

### AI Integration

The AI Integration component enhances the system with language model capabilities:

- Document Understanding: Extracts meaning from unstructured text
- Pattern Recognition: Identifies complex financial patterns
- Report Generation: Creates natural language summaries of findings
- Interactive Querying: Allows natural language querying of documents

Example:
```python
from cdas.ai.llm import LLMManager
from cdas.ai.agents.investigator import InvestigatorAgent
from cdas.db.session import get_session

# Create session
session = get_session()

# Initialize LLM manager
llm_manager = LLMManager()

# Initialize investigator agent
agent = InvestigatorAgent(session, llm_manager)

# Investigate a question
results = agent.investigate("What evidence suggests the contractor double-billed for HVAC equipment?")
```

### Reporting System

The Reporting System generates well-structured, evidence-backed reports:

- Multiple report types: Summary, detailed, evidence chains, custom
- Multiple output formats: Markdown, HTML, PDF, Excel
- Evidence citation: Links findings to source documents
- Visual elements: Charts, tables, and timelines

Example:
```python
from cdas.reporting.generator import ReportGenerator
from cdas.db.session import get_session

# Create session
session = get_session()

# Initialize report generator
report_generator = ReportGenerator(session)

# Generate summary report
report = report_generator.generate_summary_report(report_data, format='pdf', output_path='report.pdf')
```

## Database Schema

The database schema is designed to track financial information across documents:

- **Documents**: Stores metadata about each document
- **Pages**: Stores individual pages within documents
- **Line Items**: Stores financial line items extracted from documents
- **Document Relationships**: Tracks relationships between documents
- **Financial Transactions**: Tracks financial transactions across documents
- **Analysis Flags**: Stores flags for suspicious patterns or anomalies

## Database Operations

The system provides a rich set of database operations:

```python
from cdas.db.session import session_scope
from cdas.db.operations import (
    register_document, register_page, store_line_items,
    create_document_relationship, find_matching_amounts,
    create_analysis_flag, search_documents, search_line_items
)

with session_scope() as session:
    # Find documents with matching amounts
    matches = find_matching_amounts(session, 12345.67, tolerance=0.01)
    
    # Search for documents
    change_orders = search_documents(
        session,
        doc_type="change_order",
        party="contractor",
        status="rejected"
    )
    
    # Search for line items
    hvac_items = search_line_items(
        session,
        description_keyword="HVAC",
        min_amount=5000,
        max_amount=10000,
        doc_type="payment_app"
    )
```

## CLI Usage

The CLI is the primary interface for interacting with the system:

```bash
# Document Management
python -m cdas.cli doc ingest contract.pdf --type change_order --party contractor
python -m cdas.cli doc list --type payment_app
python -m cdas.cli doc show doc_123abc --items

# Financial Analysis
python -m cdas.cli analyze patterns --min-confidence 0.8
python -m cdas.cli analyze amount 12345.67
python -m cdas.cli analyze document doc_123abc

# Querying
python -m cdas.cli query search "HVAC installation" --type payment_app
python -m cdas.cli query find --min 5000 --max 10000 --desc "electrical"
python -m cdas.cli query ask "When was the first time the contractor billed for elevator maintenance?"

# Reporting
python -m cdas.cli report summary financial_summary.pdf
python -m cdas.cli report detailed detailed_analysis.pdf --include-evidence
python -m cdas.cli report evidence 23456.78 evidence_chain.pdf
```

## Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines and use Black for formatting
2. **Testing**: Write unit tests for all new functionality
3. **Documentation**: Document all classes, methods, and functions
4. **Error Handling**: Implement robust error handling for all user-facing functions
5. **Type Annotations**: Use type annotations for all function parameters and return values

## Logging

The system uses Python's built-in logging module with a standardized configuration:

```python
import logging
from cdas.config import get_logging_config

# Get logging configuration
logging_config = get_logging_config()

# Configure logging
logging.config.dictConfig(logging_config)

# Get logger for current module
logger = logging.getLogger(__name__)

# Use logger
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

Log files are stored in the following locations:
- Application logs: `logs/cdas.log`
- Test logs: `tests/logs/test.log`

The log level can be configured in the `.env` file:
```
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/cdas.log
```

## CI/CD Pipeline

The project uses a CI/CD pipeline with GitHub Actions that:

1. Runs tests on multiple Python versions (3.8, 3.9, 3.10)
2. Performs code formatting checks
3. Runs linters and type checkers
4. Executes unit and integration tests
5. Generates and uploads coverage reports
6. Publishes to PyPI for tagged releases

For local development, use:
- `make all` to run all checks and tests
- `tox` to test across multiple Python versions