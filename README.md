# Construction Document Analysis System (CDAS)

A specialized tool for analyzing construction documents to track financial claims, identify discrepancies, and generate evidence-backed reports for dispute resolution.

## Project Overview

The Construction Document Analysis System (CDAS) is designed for attorneys representing public agencies (particularly school districts) in construction disputes. Its primary purpose is to process, analyze, and reconcile financial information across various document types to:

1. Track financial claims and counterclaims between parties (district vs. contractor)
2. Identify discrepancies in amounts across different document types
3. Detect suspicious financial patterns (e.g., rejected change orders reappearing in payment applications)
4. Generate comprehensive reports for dispute resolution conferences
5. Provide evidence-backed analysis with direct citations to source documents

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

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"

# Initialize database
python -m cdas.db.init
```

## Getting Started

### Database Setup

The system uses a database to store document information, extracted data, and analysis results:

```python
from cdas.db.session import session_scope
from cdas.db.operations import register_document

# Use the session context manager for transactions
with session_scope() as session:
    # Register a document
    doc = register_document(
        session,
        file_path="path/to/document.pdf",
        doc_type="change_order",
        party="contractor",
        metadata={"project": "School Renovation"}
    )

    print(f"Document registered with ID: {doc.doc_id}")
```

### Document Processing

The system can extract structured data from various construction document types:

- PDF files (contracts, change orders, payment applications)
- Excel spreadsheets (schedules, invoices)
- Scanned documents (via OCR)
- Images with handwritten content

Example:

```python
from cdas.db.session import session_scope
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType

# Use session context manager
with session_scope() as session:
    # Create document processor
    factory = DocumentProcessorFactory()
    processor = factory.create_processor(session)
    
    # Process a document
    result = processor.process_document(
        file_path="path/to/document.pdf",
        doc_type=DocumentType.PAYMENT_APP,
        party=PartyType.CONTRACTOR,
        save_to_db=True
    )
    
    # Access extracted data
    if result.success:
        print(f"Document ID: {result.document_id}")
        print(f"Document metadata: {result.metadata}")
        print(f"Extracted {len(result.line_items)} line items")
        for item in result.line_items:
            print(f"{item['description']}: ${item.get('amount')}")
    else:
        print(f"Error: {result.error}")
```

You can also use the factory's convenience method:

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
    
    if result.success:
        print(f"Successfully processed document: {result.document_id}")
```

## Features

### Database Schema

- Comprehensive schema for storing construction document data
- Support for document metadata, pages, and line items
- Relationship tracking between documents
- Financial transaction tracking
- Specialized tables for change orders and payment applications
- Analysis flags for suspicious patterns
- Report generation and evidence linking

### Document Processor

- Specialized extractors for each document type
- OCR for scanned documents
- Handwriting recognition capabilities
- Extracts tabular data from structured documents
- Parses and normalizes financial information
- Preserves document metadata and source information

### Financial Analysis

- Pattern detection for recurring amounts
- Anomaly detection (statistical and rule-based)
- Amount matching across documents with fuzzy matching
- Chronological analysis of financial changes
- Network analysis of document and financial relationships
- Suspicious pattern detection with configurable confidence thresholds

### Network Analysis

- Build and visualize relationship graphs between documents
- Detect circular references and suspicious financial patterns
- Identify isolated documents and missing relationships
- Calculate centrality measures to find key documents
- Generate community detection for related document groups

### AI Integration (Coming Soon)

- Document understanding using LLMs
- Pattern recognition for complex financial patterns
- Report generation with natural language summaries
- Interactive querying of documents

### Reporting System (Coming Soon)

- Multiple report types (summary, detailed, evidence chains)
- Multiple output formats (Markdown, HTML, PDF, Excel)
- Evidence citation linking findings to source documents
- Visual elements (charts, tables, timelines)

## CLI Usage

The system provides a comprehensive command-line interface:

```bash
# Database Management
python -m cdas.db.init
python -m cdas.db.reset

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

You can get help on any command by adding the `--help` flag:

```bash
python -m cdas.cli --help
python -m cdas.cli doc --help
python -m cdas.cli analyze --help
```

## Financial Analysis

The system provides powerful financial analysis capabilities:

```python
from cdas.db.session import session_scope
from cdas.financial_analysis.engine import FinancialAnalysisEngine

with session_scope() as session:
    # Create analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Find suspicious patterns
    patterns = engine.find_suspicious_patterns(min_confidence=0.8)
    
    for pattern in patterns:
        print(f"Pattern: {pattern['type']} - {pattern['description']}")
        print(f"Confidence: {pattern['confidence']:.2f}")
    
    # Analyze a specific amount
    amount_analysis = engine.analyze_amount(12345.67)
    
    if amount_analysis['matches']:
        print(f"Found {len(amount_analysis['matches'])} matches for ${amount_analysis['amount']}")
        
        if amount_analysis['anomalies']:
            print(f"Detected {len(amount_analysis['anomalies'])} anomalies")
            for anomaly in amount_analysis['anomalies']:
                print(f"  {anomaly['explanation']}")
    
    # Analyze a document
    doc_analysis = engine.analyze_document("doc_123abc")
    
    if doc_analysis['patterns'] or doc_analysis['anomalies']:
        print("Detected issues in document:")
        for issue in doc_analysis['patterns'] + doc_analysis['anomalies']:
            print(f"  {issue['type']}: {issue['explanation']}")
```

## Network Analysis

The system can build and analyze relationship networks:

```python
from cdas.db.session import session_scope
from cdas.analysis.network import NetworkAnalyzer

with session_scope() as session:
    # Create network analyzer
    analyzer = NetworkAnalyzer(session)
    
    # Build document network
    graph = analyzer.build_document_network()
    print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Find circular references
    cycles = analyzer.find_circular_references()
    for cycle in cycles:
        print(f"Circular reference: {' -> '.join(cycle)} -> {cycle[0]}")
    
    # Find isolated documents
    isolated = analyzer.find_isolated_documents()
    print(f"Found {len(isolated)} isolated documents")
    
    # Visualize the network
    output_path = analyzer.visualize_network(output_path="network.png")
    print(f"Network visualization saved to {output_path}")
```

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
    
    for match in matches:
        print(f"Found matching amount in document: {match.document.file_name}")
        
    # Search for documents
    change_orders = search_documents(
        session,
        doc_type="change_order",
        party="contractor",
        status="rejected"
    )
    
    print(f"Found {len(change_orders)} rejected change orders")
    
    # Search for line items by amount range and description
    hvac_items = search_line_items(
        session,
        description_keyword="HVAC",
        min_amount=5000,
        max_amount=10000,
        doc_type="payment_app"
    )
    
    print(f"Found {len(hvac_items)} HVAC line items between $5,000 and $10,000")
```

## Testing

The project includes a comprehensive test suite with unit tests and integration tests:

```bash
# Run all tests
make test

# Run unit tests only
make unit

# Run integration tests only
make integration

# Run tests with coverage report
make coverage

# Run the full CI pipeline locally (format, lint, type check, test)
make all
```

For more detailed testing:

```bash
# Run tests with pytest directly
pytest

# Run specific test categories
pytest tests/test_db/
pytest tests/test_integration/

# Run with coverage
pytest --cov=cdas --cov-report=term-missing

# Run tests across multiple Python versions
tox
```

See the [Testing Documentation](tests/README.md) and [CI/CD Documentation](docs/ci_cd.md) for more details.

## Documentation

For detailed documentation, refer to the following:

- [Project Architecture](docs/architecture.md)
- [Database Schema](docs/database-schema.md)
- [Document Processing](docs/document-processor.md)
- [Financial Analysis](docs/financial-analysis.md)
- [AI Integration](docs/ai-integration.md)
- [Reporting System](docs/reporting-system.md)
- [CLI Reference](docs/cli-reference.md)
- [Testing Guide](tests/README.md)
- [CI/CD Pipeline](docs/ci_cd.md)

## Development

```bash
# Set up development environment
python scripts/setup_testing.py

# Format code
black cdas/ tests/

# Run linter
flake8 cdas/ tests/

# Run type checker
mypy cdas/

# Create database migration
alembic revision --autogenerate -m "Description of changes"

# Apply database migrations
alembic upgrade head
```

Using the Makefile:

```bash
# Run all checks (format, lint, type, test)
make all

# Format code
make format

# Run linter
make lint

# Run type checker
make type

# Run tests
make test

# Generate coverage report
make coverage

# Clean temporary files
make clean
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.