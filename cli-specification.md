# Command Line Interface Specification

## Overview

The Command Line Interface (CLI) is the primary interaction point for users of the Construction Document Analysis System. It provides a comprehensive set of commands for document ingestion, analysis, querying, and report generation. The CLI is designed to be intuitive for attorneys and legal professionals who may not have extensive technical backgrounds, while still offering powerful features for advanced users.

## Key Features

1. **Project Management**: Create and manage isolated project databases
2. **Document Management**: Ingest and organize construction documents (single files or directories)
3. **Analysis Commands**: Trigger various types of financial analysis
4. **Query Interface**: Search and query the document database
5. **Report Generation**: Generate comprehensive reports in multiple formats
6. **Interactive Shell**: User-friendly shell with command history and tab completion
7. **AI Integration**: Natural language querying and AI-powered analysis
8. **Help & Documentation**: Provide helpful guidance and examples

## Implementation

The CLI is implemented in `cdas/cli.py` using Python's built-in argparse library, providing a robust command-line interface with subcommands. The interactive shell is implemented in `cdas/interactive_shell.py` using the cmd module.

## Core CLI Structure

```bash
python -m cdas.cli [GLOBAL_OPTIONS] COMMAND [SUBCOMMAND] [ARGUMENTS] [OPTIONS]
```

### Global Options
- `--user USER`: User identifier for tracking who created reports
- `--project PROJECT`: Project identifier (overrides current project)
- `--verbose, -v`: Enable verbose logging

### Main Commands
- `project`: Project management commands
- `doc`: Document management commands  
- `analyze`: Financial analysis commands
- `query`: Querying and search commands
- `report`: Reporting commands
- `interactive`: Start the interactive shell

## Command Details

### Project Management Commands

```bash
# List all projects
python -m cdas.cli project list

# Create a new project
python -m cdas.cli project create PROJECT_ID

# Set the current project
python -m cdas.cli project use PROJECT_ID

# Delete a project (with confirmation)
python -m cdas.cli project delete PROJECT_ID [--force]
```

### Document Management Commands

```bash
# Ingest a single document
python -m cdas.cli doc ingest FILE_PATH --type TYPE --party PARTY [OPTIONS]

# Ingest a directory of documents
python -m cdas.cli doc ingest DIR_PATH --type TYPE --party PARTY --recursive [OPTIONS]

# List documents
python -m cdas.cli doc list [--type TYPE] [--party PARTY] [--start-date DATE] [--end-date DATE]

# Show document details
python -m cdas.cli doc show DOC_ID [--items] [--pages]
```

Options for document ingestion:
- `--recursive`: Process subdirectories
- `--extensions`: Comma-separated file extensions (e.g., ".pdf,.xlsx")
- `--no-db`: Don't save to database
- `--no-handwriting`: Skip handwriting extraction
- `--no-tables`: Skip table extraction

### Analysis Commands  

```bash
# Run pattern analysis
python -m cdas.cli analyze patterns [--min-confidence FLOAT]

# Analyze a specific amount
python -m cdas.cli analyze amount AMOUNT [--tolerance FLOAT]

# Analyze a document
python -m cdas.cli analyze document DOC_ID
```

### Query Commands

```bash
# Search for text in documents
python -m cdas.cli query search TEXT [--type TYPE] [--party PARTY]

# Find line items by amount range
python -m cdas.cli query find [--min AMOUNT] [--max AMOUNT] [--desc TEXT]

# Ask a natural language question
python -m cdas.cli query ask "QUESTION" [--verbose]
```

### Report Commands

```bash
# Generate summary report
python -m cdas.cli report summary OUTPUT_PATH [--format FORMAT]

# Generate detailed report
python -m cdas.cli report detailed OUTPUT_PATH [--format FORMAT] [--include-evidence]

# Generate evidence report for an amount
python -m cdas.cli report evidence AMOUNT OUTPUT_PATH [--format FORMAT]
```

Supported formats: pdf, html, md, excel

### Interactive Shell

```bash
# Start interactive shell
python -m cdas.cli interactive [--project PROJECT_ID]
```

The interactive shell provides:
- Command history (arrow keys)
- Tab completion
- Built-in help and tutorials
- Project context management
- Simplified command syntax (no need for "python -m cdas.cli" prefix)

## Document Types

The system supports the following document types:
- `contract`: Construction contracts
- `change_order`: Change orders
- `payment_app`: Payment applications  
- `invoice`: Invoices
- `schedule`: Project schedules
- `correspondence`: Letters and emails
- `other`: Miscellaneous documents

## Party Types

The system recognizes the following parties:
- `district`: School district or public agency
- `contractor`: General contractor
- `subcontractor`: Subcontractor
- `architect`: Architect or engineer
- `other`: Other parties

## Implementation Guidelines

1. **User Experience**: Optimize for attorneys with limited technical experience
2. **Error Handling**: Provide clear error messages and recovery options
3. **Documentation**: Include comprehensive help text and examples
4. **Progress Indicators**: Show progress for long-running operations
5. **Output Formatting**: Use consistent, clear formatting for output
6. **Configurability**: Allow customization through config files and CLI options

## Security Considerations

1. **Input Validation**: Validate all user input
2. **Path Traversal Prevention**: Prevent directory traversal attacks
3. **SQL Injection Prevention**: Use parameterized queries
4. **Error Messages**: Avoid exposing sensitive information in error messages
5. **Project Isolation**: Ensure complete data isolation between projects