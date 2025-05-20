# Construction Document Analysis System (CDAS) - Usage Guide

This guide provides comprehensive instructions for using the Construction Document Analysis System (CDAS), a specialized tool designed for attorneys representing public agencies in construction disputes.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Document Management](#document-management)
4. [Financial Analysis](#financial-analysis)
5. [Network Analysis](#network-analysis)
6. [Reporting](#reporting)
7. [AI-Assisted Analysis](#ai-assisted-analysis)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)

## Introduction

CDAS helps you:
- Track financial claims between parties
- Identify discrepancies across document types
- Detect suspicious financial patterns
- Generate evidence-backed reports
- Query your document set using natural language

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database (recommended for production)
- API keys for AI services (if using AI features)

### Step 1: Install the Package

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the package
pip install -e .

# Install with development dependencies (if needed)
pip install -e ".[dev]"
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
# Mandatory settings:
# - Database connection
# - API keys (if using AI features)
# - Log file locations
```

### Step 3: Initialize Database

```bash
# Initialize database schema
python -m cdas.db.init

# Run database migrations
alembic upgrade head
```

### Step 4: Verify Installation

```bash
# Run the CLI application to verify installation
python -m cdas.cli --help
```

## Document Management

CDAS can process various document types including PDFs, Excel files, and scanned documents.

### Importing Documents

```bash
# Import a single document
python -m cdas.cli doc ingest path/to/document.pdf --type payment_app --party contractor

# Import multiple documents from a CSV manifest
python -m cdas.cli doc batch-import manifest.csv

# CSV format example:
# path,type,party,date,reference
# documents/contract.pdf,contract,district,2023-01-15,Contract-2023-001
# documents/invoice1.pdf,invoice,contractor,2023-02-20,INV-2023-001
```

### Listing Documents

```bash
# List all documents
python -m cdas.cli doc list

# List documents by type
python -m cdas.cli doc list --type change_order

# List documents by party
python -m cdas.cli doc list --party contractor

# List documents by date range
python -m cdas.cli doc list --from 2023-01-01 --to 2023-06-30
```

### Viewing Document Details

```bash
# View document details (use ID from 'doc list' command)
python -m cdas.cli doc show doc_123abc

# View document details with line items
python -m cdas.cli doc show doc_123abc --items

# Export document to text
python -m cdas.cli doc export doc_123abc output.txt
```

## Financial Analysis

CDAS provides tools to analyze financial data across documents.

### Running Pattern Analysis

```bash
# Run pattern analysis on all documents
python -m cdas.cli analyze patterns

# Run with minimum confidence level
python -m cdas.cli analyze patterns --min-confidence 0.8

# Run pattern analysis for specific document types
python -m cdas.cli analyze patterns --doc-types payment_app,change_order
```

### Analyzing Specific Amounts

```bash
# Trace an amount through all documents
python -m cdas.cli analyze amount 12345.67

# Use fuzzy matching for approximate amounts
python -m cdas.cli analyze amount 12345.67 --tolerance 0.01

# Analyze amount with specific description keywords
python -m cdas.cli analyze amount 12345.67 --keywords "HVAC,installation"
```

### Analyzing Documents

```bash
# Analyze a specific document
python -m cdas.cli analyze document doc_123abc

# Find documents with suspicious patterns
python -m cdas.cli analyze suspicious --min-confidence 0.7
```

## Network Analysis

Visualize and analyze relationships between documents.

```bash
# Generate a network visualization
python -m cdas.cli network visualize --output network.png

# Find circular references
python -m cdas.cli network circular

# Find isolated documents
python -m cdas.cli network isolated

# Analyze connections for a specific document
python -m cdas.cli network connections doc_123abc
```

## Reporting

Generate detailed reports based on your analysis.

### Types of Reports

```bash
# Generate a summary report
python -m cdas.cli report summary financial_summary.pdf

# Generate a detailed report
python -m cdas.cli report detailed detailed_analysis.pdf

# Generate an evidence chain for a specific amount
python -m cdas.cli report evidence 23456.78 evidence_chain.pdf

# Generate a custom report
python -m cdas.cli report custom --template custom_template.j2 custom_report.pdf
```

### Report Options

```bash
# Include evidence citations
python -m cdas.cli report detailed detailed_analysis.pdf --include-evidence

# Include visualizations
python -m cdas.cli report summary financial_summary.pdf --include-visuals

# Select specific document types to include
python -m cdas.cli report summary financial_summary.pdf --doc-types payment_app,change_order

# Export in different formats
python -m cdas.cli report summary financial_summary.html --format html
python -m cdas.cli report summary financial_summary.xlsx --format excel
```

## AI-Assisted Analysis

CDAS provides AI-powered tools for deeper analysis. These features require API keys configured in your `.env` file.

### Natural Language Querying

```bash
# Ask questions about your document set
python -m cdas.cli query ask "When was the first time the contractor billed for elevator maintenance?"

# Ask with specific document context
python -m cdas.cli query ask "What evidence suggests double-billing?" --context doc_123abc
```

### AI Investigation

```bash
# Have AI investigate a specific question
python -m cdas.cli investigate "What evidence suggests the contractor double-billed for HVAC equipment?"

# Run AI investigation with specific document focus
python -m cdas.cli investigate "Are there any suspicious change orders?" --focus change_order
```

### AI-Enhanced Reporting

```bash
# Generate a narrative report using AI
python -m cdas.cli report narrative narrative_report.pdf

# Generate a focused narrative on specific issues
python -m cdas.cli report narrative hvac_issues.pdf --focus "HVAC billing issues"
```

## Common Workflows

Here are some common workflows to help you get started:

### Basic Document Import and Analysis

```bash
# 1. Import documents
python -m cdas.cli doc ingest contract.pdf --type contract --party district
python -m cdas.cli doc ingest invoice1.pdf --type invoice --party contractor
python -m cdas.cli doc ingest invoice2.pdf --type invoice --party contractor

# 2. Run basic analysis
python -m cdas.cli analyze patterns

# 3. Generate summary report
python -m cdas.cli report summary summary_report.pdf
```

### Investigating a Specific Issue

```bash
# 1. Search for relevant documents
python -m cdas.cli query search "HVAC installation" --type invoice

# 2. Analyze specific amount
python -m cdas.cli analyze amount 23456.78

# 3. Generate evidence chain
python -m cdas.cli report evidence 23456.78 evidence_chain.pdf

# 4. Ask AI to investigate
python -m cdas.cli investigate "Was the HVAC installation billed multiple times?"
```

### Full Project Analysis

```bash
# 1. Import all documents from a CSV manifest
python -m cdas.cli doc batch-import project_manifest.csv

# 2. Run comprehensive analysis
python -m cdas.cli analyze patterns
python -m cdas.cli network visualize --output network.png

# 3. Find suspicious patterns
python -m cdas.cli analyze suspicious

# 4. Generate detailed report
python -m cdas.cli report detailed detailed_analysis.pdf --include-evidence --include-visuals
```

## Troubleshooting

### Common Issues

**Database Connection Issues:**
```bash
# Check database connection
python -m cdas.cli db check-connection

# Reset database (development only - will erase all data)
python -m cdas.db.reset
```

**Document Processing Issues:**
```bash
# Enable verbose logging for more details
python -m cdas.cli --verbose doc ingest problematic_document.pdf

# Check document processing capabilities
python -m cdas.cli doc check-capabilities
```

**API Connection Issues:**
```bash
# Verify API keys (for AI features)
python -m cdas.cli api verify

# Test API connection
python -m cdas.cli api test
```

**Performance Issues:**
```bash
# Check database statistics
python -m cdas.cli db stats

# Clean up temporary files
python -m cdas.cli cleanup
```

### Getting Help

For detailed help on any command:
```bash
python -m cdas.cli --help
python -m cdas.cli doc --help
python -m cdas.cli analyze --help
python -m cdas.cli report --help
```

For additional assistance, please refer to the documentation or contact support.