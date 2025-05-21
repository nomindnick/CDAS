# Construction Document Analysis System (CDAS) - Usage Guide

This guide provides comprehensive instructions for using the Construction Document Analysis System (CDAS), a specialized tool designed for attorneys representing public agencies in construction disputes.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Command-Line Interface](#command-line-interface)
4. [Interactive Shell](#interactive-shell)
5. [Document Management](#document-management)
6. [Financial Analysis](#financial-analysis)
7. [Network Analysis](#network-analysis)
8. [Reporting](#reporting)
9. [AI-Assisted Analysis](#ai-assisted-analysis)
10. [Common Workflows](#common-workflows)
11. [Troubleshooting](#troubleshooting)

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

## Command-Line Interface

CDAS provides a comprehensive command-line interface for all system operations. The CLI follows a consistent structure:

```bash
python -m cdas.cli [GLOBAL_OPTIONS] COMMAND [SUBCOMMAND] [ARGUMENTS] [OPTIONS]
```

### Global Options

```bash
--user USER         User identifier for tracking who created reports
--project PROJECT   Project identifier
--verbose, -v       Enable verbose logging
```

### Main Commands

- `doc` - Document management commands
- `analyze` - Financial analysis commands
- `query` - Querying and search commands
- `report` - Reporting commands
- `network` - Network analysis commands
- `shell` - Start the interactive shell

### Getting Help

```bash
# Get general help
python -m cdas.cli --help

# Get help on a specific command
python -m cdas.cli doc --help
python -m cdas.cli analyze --help
python -m cdas.cli report --help
```

## Interactive Shell

CDAS provides an interactive shell interface that offers a more user-friendly experience with features like command history, tab completion, and contextual help.

### Starting the Shell

```bash
# Start the interactive shell
python -m cdas.cli shell

# Start with project context
python -m cdas.cli shell --project school_123
```

### Shell Features

- **Command History**: Use up and down arrow keys to navigate through previous commands
- **Tab Completion**: Press Tab to complete commands, document IDs, file paths, etc.
- **Contextual Help**: Get detailed guidance on using commands and their options
- **Context Management**: Set and maintain project context between commands
- **Tutorials and Examples**: Access built-in tutorials and command examples

### Basic Shell Usage

Once in the shell, you can run all the same commands as the CLI but without the `python -m cdas.cli` prefix:

```
Construction Document Analysis System (CDAS) - Interactive Shell
----------------------------------------------------------------
Type 'help' or '?' to list commands.
Type 'help <command>' for detailed help on a specific command.
Type 'quit' or 'exit' to exit.

Common commands:
  ingest - Process and ingest a document into the system
  list   - List documents in the system
  show   - Show details of a specific document
  search - Search for text in documents
  ask    - Ask a natural language question about the data
  report - Generate various types of reports

cdas> ingest contract.pdf --type contract --party district
Document: contract.pdf
Type: contract
Party: district
Line items: 15

cdas> list --type contract
Found 1 documents:
ID       | Type    | Party    | Date       | File Name
----------------------------------------------
doc_123a | contract | district | 2023-05-15 | contract.pdf
```

### Getting Help in the Shell

The interactive shell provides several ways to get help:

```
# List all available commands
cdas> help

# Get help for a specific command
cdas> help ingest

# Show a tutorial with examples
cdas> tutorial

# Show tutorials on a specific topic
cdas> tutorial documents

# Show examples of a specific command
cdas> examples report
```

### Context Management

You can set a project context to work within a specific project:

```
# Set project context
cdas> project school_123
Project context set to: school_123

# Now prompt shows the context
cdas:school_123> list
...

# Show current context
cdas:school_123> context
Current context:
  project: school_123

# Clear context
cdas:school_123> context clear
Context cleared
cdas>
```

### Exiting the Shell

```
# Exit methods
cdas> exit
cdas> quit
# Or press Ctrl+D
```

For more detailed information on using the interactive shell, see [Interactive Shell Documentation](docs/interactive_shell.md).

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

For easier troubleshooting, use the interactive shell which provides better guidance and command completion:
```bash
python -m cdas.cli shell
cdas> help
```

You can also use the shell's built-in tutorials and examples:
```bash
cdas> tutorial
cdas> examples
```

For additional assistance, please refer to the documentation or contact support.