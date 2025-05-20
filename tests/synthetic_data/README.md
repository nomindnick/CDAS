# CDAS Synthetic Test Suite

This directory contains a comprehensive synthetic test suite for the Construction Document Analysis System (CDAS). The test suite consists of artificially created construction documents with intentionally embedded issues of varying difficulty to evaluate the system's detection capabilities.

## Overview

The synthetic test suite includes:

1. A fictional construction project (Oakridge Elementary School Renovation)
2. Multiple document types (contracts, payment applications, change orders, correspondence, invoices)
3. 20 embedded issues ranging from easy to difficult to detect
4. Scripts for processing documents and evaluating system performance

## Directory Structure

- `contracts/` - Contract documents
- `payment_apps/` - Payment applications 
- `change_orders/` - Change orders (approved and rejected)
- `correspondence/` - Letters between project parties
- `invoices/` - Subcontractor invoices
- `project_info.md` - Details about the fictional project
- `manifest.md` - List of all test documents and embedded issues
- `document_import.csv` - CSV file for bulk document import
- `import_documents.py` - Script to import documents from CSV
- `run_synthetic_tests.py` - Script to run tests and evaluate results

## Document Properties

The documents in this test suite were designed to:

1. Represent realistic construction documentation
2. Include common construction document formats
3. Contain a mix of normal, valid project information
4. Embed specific testing issues of varying complexity
5. Create relationships between documents that require analysis to detect

## Embedded Issues

The test suite includes 20 embedded issues:

- **Easy Issues**: Simple to detect with basic data processing
- **Medium Issues**: Require cross-document analysis and pattern recognition
- **Hard Issues**: Need sophisticated analysis techniques to identify

See `manifest.md` for the complete list of issues.

## Running the Tests

### 1. Import Documents

You can import all documents at once using the CSV file:

```bash
python import_documents.py document_import.csv --reset
```

The `--reset` flag will reset the database before importing.

### 2. Run the Tests

To run the synthetic tests and evaluate the system:

```bash
python run_synthetic_tests.py
```

Or to start with a fresh database:

```bash
python run_synthetic_tests.py --reset
```

### 3. Analyze Results

The test script will:

1. Process all documents
2. Run financial analysis
3. Perform network analysis
4. Compare detected issues with the expected issues
5. Calculate and report the detection rate

## Evaluation Metrics

The main evaluation metric is the **detection rate**, which is the percentage of embedded issues that the system successfully detects.

## Extending the Test Suite

To add more test cases:

1. Create new document files in the appropriate subdirectories
2. Update the `document_import.csv` file with the new documents
3. Add the expected issues to the `EXPECTED_ISSUES` dictionary in `run_synthetic_tests.py`

## Purpose

This synthetic test suite serves several purposes:

1. **Evaluation**: Measure how well the system detects known issues
2. **Regression Testing**: Ensure new features don't break existing functionality
3. **Development Guidance**: Highlight areas where detection capabilities could be improved
4. **Demonstration**: Show the types of issues the system is designed to detect
5. **Benchmarking**: Establish performance baselines for future enhancements