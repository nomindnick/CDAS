# Construction Document Analysis System (CDAS) - Usage Guide

This guide provides comprehensive instructions for using the Construction Document Analysis System (CDAS), a specialized tool designed for attorneys representing public agencies in construction disputes.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Project Management](#project-management)
4. [Command-Line Interface](#command-line-interface)
5. [Interactive Shell](#interactive-shell)
6. [Document Management](#document-management)
7. [Financial Analysis](#financial-analysis)
8. [Network Analysis](#network-analysis)
9. [Reporting](#reporting)
10. [AI-Assisted Analysis](#ai-assisted-analysis)
11. [Common Workflows](#common-workflows)
12. [Troubleshooting](#troubleshooting)

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

# Or migrate all existing projects (if any)
alembic upgrade head --all-projects
```

### Step 4: Verify Installation

```bash
# Run the CLI application to verify installation
python -m cdas.cli --help
```

## Project Management

CDAS supports project-based data isolation, allowing you to manage multiple construction disputes as separate projects. Each project maintains its own database, ensuring complete data separation between different cases.

### Creating and Managing Projects

```bash
# List all projects
python -m cdas.cli project list

# Create a new project
python -m cdas.cli project create school_renovation_2024

# Switch to a project (sets it as current for commands)
python -m cdas.cli project use school_renovation_2024

# Delete a project (removes all data - use with caution)
python -m cdas.cli project delete old_project_name
```

### Working with Projects

Once you have created a project, you can work within its context:

```bash
# Use project flag with any command
python -m cdas.cli --project school_renovation_2024 doc ingest contract.pdf --type contract --party district

# Or set the current project and omit the flag
python -m cdas.cli project use school_renovation_2024
python -m cdas.cli doc ingest contract.pdf --type contract --party district
```

### Project Database Isolation

Each project maintains:
- Separate SQLite database file
- Independent document storage and metadata
- Isolated financial analysis results
- Project-specific reporting data
- Separate AI embeddings and search indices

This ensures that:
- Different construction disputes remain completely separate
- You can reset/delete one project without affecting others
- Development and testing can be done safely on isolated data
- Multiple attorneys can work on different projects simultaneously

## Command-Line Interface

CDAS provides a comprehensive command-line interface for all system operations. The CLI follows a consistent structure:

```bash
python -m cdas.cli [GLOBAL_OPTIONS] COMMAND [SUBCOMMAND] [ARGUMENTS] [OPTIONS]
```

### Global Options

```bash
--user USER         User identifier for tracking who created reports
--project PROJECT   Project identifier (overrides current project)
--verbose, -v       Enable verbose logging
```

### Main Commands

- `project` - Project management commands
- `doc` - Document management commands
- `analyze` - Financial analysis commands
- `query` - Querying and search commands
- `report` - Reporting commands
- `interactive` - Start the interactive shell

### Getting Help

```bash
# Get general help
python -m cdas.cli --help

# Get help on a specific command
python -m cdas.cli project --help
python -m cdas.cli doc --help
python -m cdas.cli analyze --help
python -m cdas.cli report --help
```

## Interactive Shell

CDAS provides an interactive shell interface that offers a more user-friendly experience with features like command history, tab completion, and contextual help.

### Starting the Shell

```bash
# Start the interactive shell
python -m cdas.cli interactive

# Start with project context
python -m cdas.cli interactive --project school_renovation_2024
```

### Shell Features

- **Command History**: Use up and down arrow keys to navigate through previous commands
- **Tab Completion**: Press Tab to complete commands, document IDs, file paths, etc.
- **Contextual Help**: Get detailed guidance on using commands and their options
- **Project Context Management**: Set and maintain project context between commands
- **Project Commands**: Create, switch, and manage projects directly from the shell
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

### Project Management in the Shell

The interactive shell provides convenient project management commands:

```
# List all projects
cdas> projects
Available projects:
- school_renovation_2024 (current)
- district_vs_contractor_2023
- bridge_project_2024

# Create a new project
cdas> project create new_school_project
Project 'new_school_project' created successfully

# Switch to a different project
cdas> project new_school_project
Project context set to: new_school_project

# Now prompt shows the project context
cdas:new_school_project> list
...

# Delete a project (with confirmation)
cdas> project delete old_project
Are you sure you want to delete project 'old_project'? This will remove all data. (y/N): y
Project 'old_project' deleted successfully

# Show current context
cdas:new_school_project> context
Current context:
  project: new_school_project

# Clear context (return to no project)
cdas:new_school_project> context clear
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
# Import a single document (with project context)
python -m cdas.cli --project school_renovation_2024 doc ingest path/to/document.pdf --type payment_app --party contractor

# Import multiple documents from a directory
python -m cdas.cli doc ingest /path/to/documents/ --type invoice --party contractor --recursive

# Import specific file types from a directory
python -m cdas.cli doc ingest /path/to/documents/ --type payment_app --party contractor --extensions .pdf,.xlsx

# Import using current project (if set with 'project use')
python -m cdas.cli project use school_renovation_2024
python -m cdas.cli doc ingest path/to/document.pdf --type payment_app --party contractor
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
python -m cdas.cli doc list --start-date 2023-01-01 --end-date 2023-06-30
```

### Viewing Document Details

```bash
# View document details (use ID from 'doc list' command)
python -m cdas.cli doc show doc_123abc

# View document details with line items
python -m cdas.cli doc show doc_123abc --items
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
```

### Analyzing Documents

```bash
# Analyze a specific document
python -m cdas.cli analyze document doc_123abc
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

# Ask with verbose output to see source documents
python -m cdas.cli query ask "What evidence suggests double-billing?" --verbose
```



## Common Workflows

Here are some common workflows to help you get started:

### Setting Up a New Project

```bash
# 1. Create a new project for your case
python -m cdas.cli project create school_renovation_2024

# 2. Set it as the current project
python -m cdas.cli project use school_renovation_2024

# 3. Import documents
python -m cdas.cli doc ingest contract.pdf --type contract --party district
python -m cdas.cli doc ingest invoice1.pdf --type invoice --party contractor
python -m cdas.cli doc ingest invoice2.pdf --type invoice --party contractor

# 4. Run basic analysis
python -m cdas.cli analyze patterns

# 5. Generate summary report
python -m cdas.cli report summary summary_report.pdf
```

### Working with Multiple Projects

```bash
# List all projects to see what you're working with
python -m cdas.cli project list

# Work on one project
python -m cdas.cli project use school_renovation_2024
python -m cdas.cli doc list
python -m cdas.cli analyze patterns

# Switch to another project
python -m cdas.cli project use bridge_project_2023
python -m cdas.cli doc list
python -m cdas.cli report summary bridge_summary.pdf

# Or use project flag without switching
python -m cdas.cli --project school_renovation_2024 doc list
python -m cdas.cli --project bridge_project_2023 doc list
```

### Investigating a Specific Issue

```bash
# 1. Search for relevant documents
python -m cdas.cli query search "HVAC installation" --type invoice

# 2. Analyze specific amount
python -m cdas.cli analyze amount 23456.78

# 3. Generate evidence chain
python -m cdas.cli report evidence 23456.78 evidence_chain.pdf

# 4. Use AI to ask questions about the data
python -m cdas.cli query ask "Was the HVAC installation billed multiple times?"
```

### Full Project Analysis

```bash
# 1. Import documents
python -m cdas.cli doc ingest /path/to/documents/ --type invoice --party contractor --recursive

# 2. Run comprehensive analysis
python -m cdas.cli analyze patterns

# 3. Generate detailed report
python -m cdas.cli report detailed detailed_analysis.pdf --include-evidence
```

## Troubleshooting

### Common Issues

**Database Connection Issues:**
```bash
# Reset database (development only - will erase all data)
python -m cdas.db.reset
```

**Document Processing Issues:**
```bash
# Enable verbose logging for more details
python -m cdas.cli --verbose doc ingest problematic_document.pdf
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
python -m cdas.cli interactive
cdas> help
```

You can also use the shell's built-in tutorials and examples:
```bash
cdas> tutorial
cdas> examples
```

For additional assistance, please refer to the documentation or contact support.