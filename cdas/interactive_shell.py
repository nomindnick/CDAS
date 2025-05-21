#!/usr/bin/env python
"""
Interactive Shell for the Construction Document Analysis System.

This module provides an interactive shell interface for CDAS,
allowing users to run commands in a REPL environment with features
like command history, autocompletion, and contextual help.
"""

import os
import sys
import cmd
import shlex
import argparse
import logging
import platform
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from cdas.db.session import get_session, session_scope
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType
from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.config import get_config

# Terminal colors for better UX
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def supports_color():
        """Check if the terminal supports colors."""
        plat = platform.system()
        supported_platform = plat != 'Windows' or 'ANSICON' in os.environ
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        return supported_platform and is_a_tty
from cdas.cli import (
    ingest_document, list_documents, show_document,
    analyze_patterns, analyze_amount, analyze_document,
    search_documents, find_line_items, ask_question,
    generate_report
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CdasShell(cmd.Cmd):
    """Interactive shell for the Construction Document Analysis System."""

    # Default intro without colors
    _intro_text = """
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
    
    Additional help:
      tutorial - Show a CDAS tutorial with examples
      examples - Show command examples
    """
    
    # Colored intro (will be set in __init__ if color is supported)
    intro = _intro_text
    prompt = 'cdas> '
    current_context = None  # Can be used to store current document, project, etc.

    def __init__(self):
        """Initialize the CDAS shell."""
        super().__init__()
        self.session = get_session()
        
        # Setup colors if supported
        self.use_colors = Colors.supports_color()
        if self.use_colors:
            # Create colorful intro
            colored_intro = self._intro_text.replace(
                "Construction Document Analysis System (CDAS) - Interactive Shell",
                f"{Colors.BOLD}{Colors.BLUE}Construction Document Analysis System (CDAS) - Interactive Shell{Colors.ENDC}"
            )
            colored_intro = colored_intro.replace(
                "Common commands:",
                f"{Colors.BOLD}Common commands:{Colors.ENDC}"
            )
            colored_intro = colored_intro.replace(
                "Additional help:",
                f"{Colors.BOLD}Additional help:{Colors.ENDC}"
            )
            
            # Add color to command names
            for cmd in ["ingest", "list", "show", "search", "ask", "report", "tutorial", "examples"]:
                colored_intro = colored_intro.replace(
                    f"  {cmd}", f"  {Colors.GREEN}{cmd}{Colors.ENDC}"
                )
            
            self.intro = colored_intro
            self.prompt = f"{Colors.CYAN}cdas>{Colors.ENDC} "
        
        # Store history
        history_path = Path.home() / '.cdas_history'
        try:
            import readline
            if os.path.exists(history_path):
                readline.read_history_file(history_path)
            readline.set_history_length(1000)
            self.has_readline = True
        except (ImportError, IOError):
            self.has_readline = False
            logger.warning("readline module not available. Command history will not be saved.")
        
        self.history_path = history_path
        self.doc_types = [t.value for t in DocumentType]
        self.party_types = [p.value for p in PartyType]
    
    def colorize(self, text, color_code):
        """Add color to text if colors are supported."""
        if self.use_colors:
            return f"{color_code}{text}{Colors.ENDC}"
        return text

    def save_history(self):
        """Save command history to a file."""
        if self.has_readline:
            import readline
            try:
                readline.write_history_file(self.history_path)
            except IOError:
                logger.warning(f"Could not write history to {self.history_path}")

    def parse_args(self, args_str, parser):
        """Parse arguments for a command using an argparse parser.
        
        Args:
            args_str: String containing command arguments
            parser: argparse.ArgumentParser instance
            
        Returns:
            Parsed arguments or None if parsing failed
        """
        try:
            args = parser.parse_args(shlex.split(args_str))
            return args
        except SystemExit:
            return None  # Parser printed help or error

    # Document commands
    def do_ingest(self, arg):
        """Ingest a document into the system.
        
        Usage: ingest FILE_PATH --type DOCTYPE --party PARTY [--project PROJECT] 
               [--no-db] [--no-handwriting] [--no-tables]
        
        Examples:
            ingest /path/to/doc.pdf --type invoice --party contractor --project school_123
            ingest contract.pdf --type contract --party district
        """
        parser = argparse.ArgumentParser(prog='ingest', description='Ingest a document')
        parser.add_argument('file_path', help='Path to the document file')
        parser.add_argument('--type', choices=self.doc_types, required=True,
                            help='Type of document')
        parser.add_argument('--party', choices=self.party_types, required=True,
                            help='Party associated with the document')
        parser.add_argument('--project', help='Project identifier')
        parser.add_argument('--no-db', action='store_true',
                            help='Do not save to database')
        parser.add_argument('--no-handwriting', action='store_true',
                            help='Do not extract handwritten text')
        parser.add_argument('--no-tables', action='store_true',
                            help='Do not extract tables')
        
        args = self.parse_args(arg, parser)
        if args:
            ingest_document(args)

    def do_list(self, arg):
        """List documents in the system.
        
        Usage: list [--type DOCTYPE] [--party PARTY] [--project PROJECT]
               [--start-date START_DATE] [--end-date END_DATE]
        
        Examples:
            list
            list --type invoice --party contractor
            list --start-date 2023-01-01 --end-date 2023-12-31
        """
        parser = argparse.ArgumentParser(prog='list', description='List documents')
        parser.add_argument('--type', choices=self.doc_types,
                            help='Filter by document type')
        parser.add_argument('--party', choices=self.party_types,
                            help='Filter by party')
        parser.add_argument('--project', help='Filter by project')
        parser.add_argument('--start-date', help='Filter by start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', help='Filter by end date (YYYY-MM-DD)')
        
        args = self.parse_args(arg, parser)
        if args:
            list_documents(args)

    def do_show(self, arg):
        """Show document details.
        
        Usage: show DOC_ID [--items] [--pages]
        
        Examples:
            show doc_123abc
            show doc_123abc --items
            show doc_123abc --pages
        """
        parser = argparse.ArgumentParser(prog='show', description='Show document details')
        parser.add_argument('doc_id', help='Document ID')
        parser.add_argument('--items', action='store_true',
                            help='Include line items')
        parser.add_argument('--pages', action='store_true',
                            help='Include page content')
        
        args = self.parse_args(arg, parser)
        if args:
            show_document(args)
            # Update current context
            self.current_context = {'doc_id': args.doc_id}
            self.prompt = f'cdas:{args.doc_id}> '

    # Analysis commands
    def do_patterns(self, arg):
        """Detect financial patterns.
        
        Usage: patterns [--min-confidence CONFIDENCE] [--doc-id DOC_ID]
        
        Examples:
            patterns
            patterns --min-confidence 0.85
            patterns --doc-id doc_123abc
        """
        parser = argparse.ArgumentParser(prog='patterns', description='Detect financial patterns')
        parser.add_argument('--min-confidence', type=float, default=0.7,
                            help='Minimum confidence level')
        parser.add_argument('--doc-id', help='Analyze a specific document')
        
        args = self.parse_args(arg, parser)
        if args:
            analyze_patterns(args)

    def do_amount(self, arg):
        """Analyze a specific amount.
        
        Usage: amount AMOUNT [--tolerance TOLERANCE]
        
        Examples:
            amount 12345.67
            amount 12345.67 --tolerance 0.05
        """
        parser = argparse.ArgumentParser(prog='amount', description='Analyze a specific amount')
        parser.add_argument('amount', type=float, help='Amount to analyze')
        parser.add_argument('--tolerance', type=float, default=0.01,
                            help='Matching tolerance')
        
        args = self.parse_args(arg, parser)
        if args:
            analyze_amount(args)

    def do_analyze(self, arg):
        """Analyze a document.
        
        Usage: analyze DOC_ID
        
        Examples:
            analyze doc_123abc
        """
        parser = argparse.ArgumentParser(prog='analyze', description='Analyze a document')
        parser.add_argument('doc_id', help='Document ID')
        
        args = self.parse_args(arg, parser)
        if args:
            analyze_document(args)

    # Query commands
    def do_search(self, arg):
        """Search for text in documents.
        
        Usage: search TEXT [--type DOCTYPE] [--party PARTY]
        
        Examples:
            search "HVAC installation"
            search "change order" --type correspondence
        """
        parser = argparse.ArgumentParser(prog='search', description='Search for text in documents')
        parser.add_argument('text', help='Text to search for')
        parser.add_argument('--type', choices=self.doc_types,
                            help='Filter by document type')
        parser.add_argument('--party', choices=self.party_types,
                            help='Filter by party')
        
        args = self.parse_args(arg, parser)
        if args:
            search_documents(args)

    def do_find(self, arg):
        """Find line items by amount range.
        
        Usage: find [--min MIN] [--max MAX] [--desc DESCRIPTION]
        
        Examples:
            find --min 5000 --max 10000
            find --desc "HVAC"
            find --min 5000 --desc "electrical"
        """
        parser = argparse.ArgumentParser(prog='find', description='Find line items by amount range')
        parser.add_argument('--min', type=float, help='Minimum amount')
        parser.add_argument('--max', type=float, help='Maximum amount')
        parser.add_argument('--desc', help='Description keyword')
        
        args = self.parse_args(arg, parser)
        if args:
            find_line_items(args)

    def do_ask(self, arg):
        """Ask a natural language question.
        
        Usage: ask QUESTION [--verbose]
        
        Examples:
            ask "When was the first time the contractor billed for elevator maintenance?"
            ask "What evidence supports the district's rejection of CO #3?" --verbose
        """
        parser = argparse.ArgumentParser(prog='ask', description='Ask a natural language question')
        parser.add_argument('question', help='Question to ask')
        parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show more detailed information in the answer')
        
        args = self.parse_args(arg, parser)
        if args:
            ask_question(args)

    # Report commands
    def do_report(self, arg):
        """Generate different types of reports.
        
        Usage: report TYPE OUTPUT_PATH [--format FORMAT] [--include-evidence] [--amount AMOUNT]
        
        Types:
            summary - Generate a summary report
            detailed - Generate a detailed report
            evidence - Generate an evidence report for a specific amount
        
        Examples:
            report summary report.pdf
            report detailed report.pdf --format html --include-evidence
            report evidence 12345.67 evidence.pdf
        """
        parser = argparse.ArgumentParser(prog='report', description='Generate a report')
        subparsers = parser.add_subparsers(dest='report_command', help='Report type')
        
        # Summary report
        summary_parser = subparsers.add_parser('summary', help='Generate summary report')
        summary_parser.add_argument('output_path', help='Output file path')
        summary_parser.add_argument('--format', choices=['pdf', 'html', 'md', 'excel'],
                                    default='pdf', help='Output format')
        
        # Detailed report
        detailed_parser = subparsers.add_parser('detailed', help='Generate detailed report')
        detailed_parser.add_argument('output_path', help='Output file path')
        detailed_parser.add_argument('--format', choices=['pdf', 'html', 'md', 'excel'],
                                     default='pdf', help='Output format')
        detailed_parser.add_argument('--include-evidence', action='store_true',
                                     help='Include evidence citations and screenshots')
        
        # Evidence report
        evidence_parser = subparsers.add_parser('evidence', help='Generate evidence report for amount')
        evidence_parser.add_argument('amount', type=float, help='Amount to analyze')
        evidence_parser.add_argument('output_path', help='Output file path')
        evidence_parser.add_argument('--format', choices=['pdf', 'html', 'md', 'excel'],
                                     default='pdf', help='Output format')
        
        args = self.parse_args(arg, parser)
        if args and args.report_command:
            generate_report(args, args.report_command)

    # Context management
    def do_context(self, arg):
        """Show or clear the current context.
        
        Usage: context [clear]
        
        Examples:
            context
            context clear
        """
        if arg.strip().lower() == 'clear':
            self.current_context = None
            self.prompt = 'cdas> '
            print("Context cleared")
        else:
            if self.current_context:
                print("Current context:")
                for key, value in self.current_context.items():
                    print(f"  {key}: {value}")
            else:
                print("No current context set")
                
    def do_tutorial(self, arg):
        """Show a tutorial for CDAS with examples.
        
        Usage: tutorial [topic]
        
        Available topics:
          basic      - Basic CDAS usage
          documents  - Document management
          analysis   - Financial analysis
          search     - Searching and querying
          reporting  - Report generation
          
        Examples:
            tutorial
            tutorial basic
            tutorial documents
        """
        topics = {
            'basic': """
CDAS Basic Usage Tutorial
========================

The Construction Document Analysis System (CDAS) helps attorneys analyze
construction documents and find financial anomalies and evidence.

Basic Workflow:
1. Ingest documents into the system
2. List and view documents
3. Analyze financial patterns
4. Generate reports with evidence

Example Session:
--------------
# Ingest a document
ingest /path/to/invoice.pdf --type invoice --party contractor

# List documents
list --type invoice

# Show details for a document
show doc_123abc

# Search for text
search "HVAC installation"

# Analyze an amount
amount 12345.67

# Generate a report
report summary output.pdf
""",
            'documents': """
Document Management Tutorial
==========================

CDAS can process various document types including:
- Contracts
- Change orders
- Payment applications
- Invoices
- Correspondence

Document Ingestion:
-----------------
# Basic ingestion
ingest contract.pdf --type contract --party district

# Ingestion with options
ingest invoice.pdf --type invoice --party contractor --project school_123 --no-handwriting

# Listing documents
list
list --type invoice --party contractor
list --start-date 2023-01-01 --end-date 2023-12-31

# Viewing document details
show doc_123abc
show doc_123abc --items  # Show line items
show doc_123abc --pages  # Show page content
""",
            'analysis': """
Financial Analysis Tutorial
=========================

CDAS provides tools to analyze financial data and detect patterns and anomalies.

Pattern Detection:
----------------
# Find suspicious patterns
patterns
patterns --min-confidence 0.85

# Analyze a specific document
analyze doc_123abc

Amount Analysis:
--------------
# Trace an amount through different documents
amount 12345.67
amount 12345.67 --tolerance 0.05  # Allow for small variations

# Find line items in a range
find --min 5000 --max 10000
find --desc "electrical" --min 5000
""",
            'search': """
Searching and Querying Tutorial
=============================

CDAS provides powerful search capabilities for finding information.

Text Search:
----------
# Basic search
search "change order"
search "HVAC installation" --type invoice

# Finding line items
find --min 5000 --max 10000
find --desc "electrical"

Natural Language Query:
--------------------
# Ask questions about the data
ask "When was the first time the contractor billed for elevator maintenance?"
ask "What evidence supports the district's rejection of CO #3?" --verbose
""",
            'reporting': """
Report Generation Tutorial
========================

CDAS can generate various types of reports with evidence.

Report Types:
-----------
# Summary report
report summary summary_report.pdf

# Detailed report with evidence
report detailed detailed_report.pdf --include-evidence

# Evidence chain for a specific amount
report evidence 12345.67 evidence_report.pdf

Output Formats:
------------
# HTML format
report summary report.html --format html

# Markdown format
report detailed report.md --format md

# Excel format (for data export)
report summary data.xlsx --format excel
"""
        }
        
        arg = arg.strip().lower()
        
        if not arg:
            # Show general tutorial info and list topics
            print("""
CDAS Interactive Shell Tutorial
==============================

This tutorial will help you get started with CDAS. 
Choose a specific topic for more detailed information.

Available topics:
  basic      - Basic CDAS usage
  documents  - Document management
  analysis   - Financial analysis
  search     - Searching and querying
  reporting  - Report generation

Example: tutorial basic
""")
        elif arg in topics:
            print(topics[arg])
        else:
            print(f"Unknown topic: {arg}")
            print("Available topics: basic, documents, analysis, search, reporting")
    
    def do_examples(self, arg):
        """Show examples of common CDAS commands.
        
        Usage: examples [command]
        
        Examples:
            examples
            examples ingest
            examples report
        """
        examples = {
            'ingest': """
Ingest Command Examples:
======================
# Basic document ingestion
ingest contract.pdf --type contract --party district

# Detailed ingestion with options
ingest invoice.pdf --type invoice --party contractor --project school_123

# Skip handwriting extraction (faster)
ingest letter.pdf --type correspondence --party contractor --no-handwriting

# Skip table extraction (if tables cause issues)
ingest spreadsheet.pdf --type payment_app --party contractor --no-tables

# Just analyze without saving to database
ingest document.pdf --type change_order --party contractor --no-db
""",
            'list': """
List Command Examples:
====================
# List all documents
list

# Filter by document type
list --type invoice

# Filter by party
list --type change_order --party contractor

# Filter by date range
list --start-date 2023-01-01 --end-date 2023-12-31

# Filter by project
list --project school_123
""",
            'show': """
Show Command Examples:
====================
# Show basic document info
show doc_123abc

# Show document with line items
show doc_123abc --items

# Show document with page content
show doc_123abc --pages

# Show document with both items and pages
show doc_123abc --items --pages
""",
            'patterns': """
Patterns Command Examples:
========================
# Find patterns with default confidence (0.7)
patterns

# Find patterns with higher confidence
patterns --min-confidence 0.85

# Find patterns in a specific document
patterns --doc-id doc_123abc
""",
            'amount': """
Amount Command Examples:
=====================
# Analyze an amount with default tolerance (0.01)
amount 12345.67

# Analyze with higher tolerance for fuzzy matching
amount 12345.67 --tolerance 0.05
""",
            'analyze': """
Analyze Command Examples:
======================
# Analyze a document for patterns and anomalies
analyze doc_123abc
""",
            'search': """
Search Command Examples:
=====================
# Search for text in all documents
search "HVAC installation"

# Search in specific document types
search "change order" --type correspondence

# Search for submissions from a specific party
search "electrical work" --party contractor
""",
            'find': """
Find Command Examples:
===================
# Find items in an amount range
find --min 5000 --max 10000

# Find items with a description
find --desc "electrical"

# Combined search
find --min 5000 --desc "HVAC"
""",
            'ask': """
Ask Command Examples:
==================
# Ask a simple question
ask "When was the first change order submitted?"

# Ask a complex question
ask "What evidence suggests the contractor double-billed for HVAC equipment?"

# Get more detailed answer
ask "Summarize all rejected change orders" --verbose
""",
            'report': """
Report Command Examples:
=====================
# Generate a summary report
report summary report.pdf

# Generate a detailed report with evidence
report detailed detailed_report.pdf --include-evidence

# Generate an evidence chain for a specific amount
report evidence 12345.67 evidence_report.pdf

# Output in different formats
report summary report.html --format html
report detailed report.md --format md
report summary data.xlsx --format excel
"""
        }
        
        arg = arg.strip().lower()
        
        if not arg:
            # Show general examples menu
            print("""
CDAS Command Examples
====================

Available command examples:
  ingest   - Document ingestion
  list     - Listing documents
  show     - Viewing document details
  patterns - Pattern detection
  amount   - Amount analysis
  analyze  - Document analysis
  search   - Text search
  find     - Finding line items
  ask      - Natural language queries
  report   - Report generation

Example: examples ingest
""")
        elif arg in examples:
            print(examples[arg])
        else:
            print(f"Unknown command: {arg}")
            print("Available commands: ingest, list, show, patterns, amount, analyze, search, find, ask, report")

    def do_project(self, arg):
        """Set the current project.
        
        Usage: project PROJECT_ID
        
        Examples:
            project school_123
        """
        arg = arg.strip()
        if arg:
            self.current_context = self.current_context or {}
            self.current_context['project'] = arg
            
            # Set prompt with or without colors
            if self.use_colors:
                self.prompt = f"{Colors.CYAN}cdas:{Colors.YELLOW}{arg}{Colors.CYAN}>{Colors.ENDC} "
            else:
                self.prompt = f'cdas:{arg}> '
                
            print(f"Project context set to: {self.colorize(arg, Colors.YELLOW)}")
        else:
            print("Please provide a project ID")

    # Shell management commands
    def do_exit(self, arg):
        """Exit the shell."""
        print("Exiting CDAS shell...")
        self.save_history()
        return True

    def do_quit(self, arg):
        """Exit the shell."""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl+D."""
        print()  # Add a newline for better output
        return self.do_exit(arg)
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: {line.split()[0] if line else ''}")
        print("Type 'help' or '?' to see available commands.")
        
    def completedefault(self, text, line, begidx, endidx):
        """Default completion handler."""
        return []
        
    def completenames(self, text, *ignored):
        """Complete command names."""
        commands = [name[3:] for name in self.get_names() if name.startswith('do_')]
        return [name for name in commands if name.startswith(text)]

    # Custom tab completion methods
    def complete_show(self, text, line, begidx, endidx):
        """Complete show command with document IDs."""
        try:
            with session_scope() as session:
                from cdas.db.models import Document
                
                # Query document IDs
                docs = session.query(Document.doc_id).limit(100).all()
                doc_ids = [doc[0] for doc in docs]
                
                # Filter by partial match
                return [doc_id for doc_id in doc_ids if doc_id.startswith(text)]
        except Exception as e:
            logger.warning(f"Error completing document IDs: {str(e)}")
            return []

    def complete_type(self, text, line, begidx, endidx):
        """Complete document types."""
        return [t for t in self.doc_types if t.startswith(text)]

    def complete_party(self, text, line, begidx, endidx):
        """Complete party types."""
        return [p for p in self.party_types if p.startswith(text)]

    def complete_project(self, text, line, begidx, endidx):
        """Complete project IDs."""
        try:
            with session_scope() as session:
                from sqlalchemy.sql import distinct
                from cdas.db.models import Document
                
                # Query distinct project IDs from metadata
                query = session.query(distinct(Document.metadata['project_id'].astext))
                projects = [row[0] for row in query if row[0]]
                
                # Filter by partial match
                return [p for p in projects if p.startswith(text)]
        except Exception as e:
            logger.warning(f"Error completing project IDs: {str(e)}")
            return []

    def _complete_path(self, text, line, begidx, endidx):
        """Complete with filesystem paths."""
        # Handle both absolute and relative paths
        if not text:
            completions = os.listdir('.')
        elif os.path.isdir(text):
            # If the text is a directory, list its contents
            if text[-1] == os.sep:
                completions = os.listdir(text)
            else:
                completions = os.listdir(text + os.sep)
        else:
            # Get the directory component of the path
            dirname = os.path.dirname(text)
            if not dirname:
                dirname = '.'
            
            # List files in that directory
            try:
                completions = os.listdir(dirname)
            except OSError:
                completions = []
            
            # Filter by the basename
            basename = os.path.basename(text)
            completions = [c for c in completions if c.startswith(basename)]
            
            # Add the directory prefix back
            if dirname != '.':
                completions = [os.path.join(dirname, c) for c in completions]
        
        # Add trailing slash to directories
        for i, completion in enumerate(completions):
            if os.path.isdir(os.path.join(dirname if dirname != '.' else '', completion)):
                completions[i] = completion + os.sep
                
        return completions

    # Command argument completion
    def complete_ingest(self, text, line, begidx, endidx):
        """Complete ingest command arguments."""
        words = line.split()
        
        # Handle file completion
        if (len(words) == 2 and not text) or (len(words) == 1 and text):
            # Right after command, show a file list
            return self._complete_path(text, line, begidx, endidx)
        
        # Handle flags
        if text.startswith('-'):
            options = ['--type', '--party', '--project', '--no-db', '--no-handwriting', '--no-tables']
            return [opt for opt in options if opt.startswith(text)]
        
        # Handle flag values
        prev_word = words[words.index(text) - 1] if text in words else words[-1]
        
        if prev_word == '--type':
            return self.complete_type(text, line, begidx, endidx)
        
        if prev_word == '--party':
            return self.complete_party(text, line, begidx, endidx)
        
        if prev_word == '--project':
            return self.complete_project(text, line, begidx, endidx)
        
        return []
        
    def complete_list(self, text, line, begidx, endidx):
        """Complete list command arguments."""
        words = line.split()
        
        # Handle flags
        if text.startswith('-'):
            options = ['--type', '--party', '--project', '--start-date', '--end-date']
            return [opt for opt in options if opt.startswith(text)]
        
        # Handle flag values
        prev_word = words[words.index(text) - 1] if text in words else words[-1]
        
        if prev_word == '--type':
            return self.complete_type(text, line, begidx, endidx)
        
        if prev_word == '--party':
            return self.complete_party(text, line, begidx, endidx)
        
        if prev_word == '--project':
            return self.complete_project(text, line, begidx, endidx)
        
        return []
        
    def complete_analyze(self, text, line, begidx, endidx):
        """Complete analyze command with document IDs."""
        return self.complete_show(text, line, begidx, endidx)
        
    def complete_report(self, text, line, begidx, endidx):
        """Complete report command arguments."""
        words = line.split()
        
        if len(words) == 2 and not text:
            # First argument after command
            return ['summary', 'detailed', 'evidence']
        
        if len(words) >= 2 and words[1] in ['summary', 'detailed', 'evidence']:
            # Complete file path for output
            if len(words) == 3 and not text:
                # Complete output path
                return self._complete_path(text, line, begidx, endidx)
            
            # Complete flags
            if text.startswith('-'):
                options = ['--format']
                if words[1] == 'detailed':
                    options.append('--include-evidence')
                return [opt for opt in options if opt.startswith(text)]
            
            # Complete flag values
            prev_word = words[words.index(text) - 1] if text in words else words[-1]
            
            if prev_word == '--format':
                formats = ['pdf', 'html', 'md', 'excel']
                return [fmt for fmt in formats if fmt.startswith(text)]
        
        return []


def main():
    """Main entry point for the CDAS interactive shell."""
    shell = CdasShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Exiting...")
    finally:
        shell.save_history()


if __name__ == '__main__':
    main()