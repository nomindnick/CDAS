# Command Line Interface Specification

## Overview

The Command Line Interface (CLI) is the primary interaction point for users of the Construction Document Analysis System. It provides a comprehensive set of commands for document ingestion, analysis, querying, and report generation. The CLI is designed to be intuitive for attorneys and legal professionals who may not have extensive technical backgrounds, while still offering powerful features for advanced users.

## Key Features

1. **Document Management**: Ingest, tag, and organize construction documents
2. **Analysis Commands**: Trigger various types of financial analysis
3. **Query Interface**: Search and query the document database
4. **Report Generation**: Generate comprehensive reports
5. **Workflow Management**: Manage analysis workflows and projects
6. **Batch Processing**: Process multiple documents or run multiple analyses
7. **Configuration**: Configure system behavior and preferences
8. **Help & Documentation**: Provide helpful guidance and examples

## Component Architecture

```
cli/
├─ __init__.py
├─ main.py                   # Main CLI entry point
├─ commands/                 # Command implementations
│   ├─ __init__.py
│   ├─ document.py           # Document management commands
│   ├─ analysis.py           # Analysis commands
│   ├─ query.py              # Query commands
│   ├─ report.py             # Report generation commands
│   └─ config.py             # Configuration commands
├─ utils/                    # CLI utilities
│   ├─ __init__.py
│   ├─ formatting.py         # Output formatting utilities
│   ├─ validation.py         # Input validation utilities
│   └─ progress.py           # Progress indicators
└─ templates/                # Command templates
    ├─ __init__.py
    └─ examples.py           # Command examples
```

## Core CLI Structure

The CLI will be built using the Typer framework, which provides a clean, decorator-based approach to building command-line applications.

```python
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from pathlib import Path
import os
import sys

from .commands import document, analysis, query, report, config

# Create Typer app
app = typer.Typer(
    name="cdas",
    help="Construction Document Analysis System - A tool for analyzing construction dispute documents",
    add_completion=True
)

# Initialize console
console = Console()

# Register command groups
app.add_typer(
    document.app,
    name="doc",
    help="Document management commands"
)

app.add_typer(
    analysis.app,
    name="analyze",
    help="Analysis commands"
)

app.add_typer(
    query.app,
    name="query",
    help="Query and search commands"
)

app.add_typer(
    report.app,
    name="report",
    help="Report generation commands"
)

app.add_typer(
    config.app,
    name="config",
    help="Configuration commands"
)

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config_file: Path = typer.Option(None, "--config", "-c", help="Path to configuration file")
):
    """
    Construction Document Analysis System
    
    A comprehensive tool for analyzing construction dispute documents
    """
    # Set up global options
    if verbose:
        os.environ["CDAS_VERBOSE"] = "1"
    
    if config_file:
        if not config_file.exists():
            console.print(f"[red]Error: Config file {config_file} not found[/red]")
            sys.exit(1)
        os.environ["CDAS_CONFIG_FILE"] = str(config_file)

def run():
    """Run the CLI application."""
    app()
```

## Command Groups

### Document Management Commands

```python
import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from pathlib import Path
import os
import glob

from ...document_processor.processor import DocumentProcessor
from ...db.session import get_session

# Create Typer app
app = typer.Typer(help="Document management commands")

# Initialize console
console = Console()

@app.command("ingest")
def ingest_document(
    file_path: Path = typer.Argument(..., help="Path to the document file or directory"),
    doc_type: str = typer.Option(None, "--type", "-t", help="Document type (payment_app, change_order, etc.)"),
    party: str = typer.Option(None, "--party", "-p", help="Document party (district, contractor)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Batch processing mode (minimal output)"),
    tags: list[str] = typer.Option([], "--tag", help="Add tags to the document")
):
    """
    Ingest a document or directory of documents into the system
    """
    # Create session
    session = get_session()
    
    # Initialize document processor
    processor = DocumentProcessor(session)
    
    # Process file or directory
    if file_path.is_dir():
        # Process directory
        pattern = "**/*" if recursive else "*"
        files = [f for f in file_path.glob(pattern) if f.is_file() and f.suffix.lower() in (".pdf", ".xlsx", ".xls", ".csv")]
        
        if not files:
            console.print(f"[yellow]No compatible files found in {file_path}[/yellow]")
            return
        
        # Process multiple files with progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {len(files)} files...", total=len(files))
            
            for file in files:
                if not batch:
                    progress.update(task, description=f"Processing {file.name}")
                
                try:
                    # Process file
                    result = processor.process_document(
                        file,
                        doc_type=doc_type,
                        party=party,
                        metadata={"tags": tags}
                    )
                    
                    if not batch:
                        progress.console.print(f"[green]Processed {file.name} (ID: {result.doc_id})[/green]")
                except Exception as e:
                    if not batch:
                        progress.console.print(f"[red]Error processing {file.name}: {str(e)}[/red]")
                
                progress.update(task, advance=1)
    else:
        # Process single file
        if not file_path.exists():
            console.print(f"[red]Error: File {file_path} not found[/red]")
            return
        
        if file_path.suffix.lower() not in (".pdf", ".xlsx", ".xls", ".csv"):
            console.print(f"[red]Error: Unsupported file type: {file_path.suffix}[/red]")
            return
        
        try:
            # Process file
            result = processor.process_document(
                file_path,
                doc_type=doc_type,
                party=party,
                metadata={"tags": tags}
            )
            
            console.print(f"[green]Processed {file_path.name} (ID: {result.doc_id})[/green]")
            
            # Display document information
            table = Table(title=f"Document Information: {file_path.name}")
            table.add_column("Field")
            table.add_column("Value")
            
            table.add_row("Document ID", result.doc_id)
            table.add_row("Type", result.doc_type or "Unknown")
            table.add_row("Party", result.party or "Unknown")
            table.add_row("Date Created", str(result.date_created) if result.date_created else "Unknown")
            table.add_row("Tags", ", ".join(tags) if tags else "None")
            
            console.print(table)
            
            # Display line item summary
            if hasattr(result, 'line_items') and result.line_items:
                line_item_table = Table(title=f"Line Items: {len(result.line_items)} items extracted")
                line_item_table.add_column("Description")
                line_item_table.add_column("Amount", justify="right")
                
                # Display up to 10 line items
                for item in result.line_items[:10]:
                    line_item_table.add_row(
                        item.description[:50] + "..." if len(item.description) > 50 else item.description,
                        f"${item.amount:,.2f}" if item.amount is not None else "N/A"
                    )
                
                if len(result.line_items) > 10:
                    line_item_table.add_row("...", "...")
                
                console.print(line_item_table)
        
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")

@app.command("list")
def list_documents(
    doc_type: str = typer.Option(None, "--type", "-t", help="Filter by document type"),
    party: str = typer.Option(None, "--party", "-p", help="Filter by party"),
    tag: str = typer.Option(None, "--tag", help="Filter by tag"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of documents to display")
):
    """
    List documents in the system
    """
    # Create session
    session = get_session()
    
    # Build query
    query = "SELECT d.doc_id, d.file_name, d.doc_type, d.party, d.date_created, d.date_processed, d.metadata FROM documents d WHERE 1=1"
    params = []
    
    if doc_type:
        query += " AND d.doc_type = %s"
        params.append(doc_type)
    
    if party:
        query += " AND d.party = %s"
        params.append(party)
    
    if tag:
        query += " AND d.metadata->>'tags' ? %s"
        params.append(tag)
    
    query += " ORDER BY d.date_processed DESC LIMIT %s"
    params.append(limit)
    
    # Execute query
    results = session.execute(query, params).fetchall()
    
    if not results:
        console.print("[yellow]No documents found matching the criteria[/yellow]")
        return
    
    # Display results
    table = Table(title=f"Documents ({len(results)} results)")
    table.add_column("ID")
    table.add_column("Filename")
    table.add_column("Type")
    table.add_column("Party")
    table.add_column("Created")
    table.add_column("Processed")
    table.add_column("Tags")
    
    for row in results:
        doc_id, file_name, doc_type, party, date_created, date_processed, metadata = row
        
        tags = []
        if metadata and "tags" in metadata:
            tags = metadata["tags"]
        
        table.add_row(
            doc_id,
            file_name,
            doc_type or "Unknown",
            party or "Unknown",
            str(date_created) if date_created else "Unknown",
            str(date_processed) if date_processed else "Unknown",
            ", ".join(tags) if tags else "None"
        )
    
    console.print(table)

@app.command("show")
def show_document(
    doc_id: str = typer.Argument(..., help="Document ID to show"),
    show_line_items: bool = typer.Option(False, "--items", "-i", help="Show line items"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show document content"),
    max_items: int = typer.Option(20, "--max-items", "-m", help="Maximum number of line items to display")
):
    """
    Show detailed information about a document
    """
    # Create session
    session = get_session()
    
    # Get document
    query = "SELECT * FROM documents WHERE doc_id = %s"
    result = session.execute(query, (doc_id,)).fetchone()
    
    if not result:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        return
    
    # Display document information
    doc = dict(zip(result.keys(), result))
    
    table = Table(title=f"Document: {doc['file_name']}")
    table.add_column("Field")
    table.add_column("Value")
    
    for field, value in doc.items():
        if field != "content" or show_content:
            table.add_row(field, str(value))
    
    console.print(table)
    
    # Show line items if requested
    if show_line_items:
        query = "SELECT * FROM line_items WHERE doc_id = %s ORDER BY item_id LIMIT %s"
        items = session.execute(query, (doc_id, max_items)).fetchall()
        
        if items:
            item_table = Table(title=f"Line Items ({len(items)} of {len(items)} total)")
            
            # Get column names
            columns = items[0].keys()
            for col in columns:
                item_table.add_column(col)
            
            # Add rows
            for item in items:
                item_table.add_row(*[str(value) for value in item.values()])
            
            console.print(item_table)
        else:
            console.print("[yellow]No line items found for this document[/yellow]")

@app.command("update")
def update_document(
    doc_id: str = typer.Argument(..., help="Document ID to update"),
    doc_type: str = typer.Option(None, "--type", "-t", help="Update document type"),
    party: str = typer.Option(None, "--party", "-p", help="Update document party"),
    add_tags: list[str] = typer.Option([], "--add-tag", help="Add tags to the document"),
    remove_tags: list[str] = typer.Option([], "--remove-tag", help="Remove tags from the document")
):
    """
    Update document metadata
    """
    # Create session
    session = get_session()
    
    # Get document
    query = "SELECT * FROM documents WHERE doc_id = %s"
    result = session.execute(query, (doc_id,)).fetchone()
    
    if not result:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        return
    
    # Prepare update fields
    updates = []
    params = []
    
    if doc_type is not None:
        updates.append("doc_type = %s")
        params.append(doc_type)
    
    if party is not None:
        updates.append("party = %s")
        params.append(party)
    
    # Handle tag updates
    if add_tags or remove_tags:
        # Get current metadata
        metadata = result.metadata or {}
        
        # Update tags
        current_tags = set(metadata.get("tags", []))
        current_tags.update(add_tags)
        current_tags.difference_update(remove_tags)
        
        metadata["tags"] = list(current_tags)
        
        updates.append("metadata = %s")
        params.append(json.dumps(metadata))
    
    if not updates:
        console.print("[yellow]No updates specified[/yellow]")
        return
    
    # Build update query
    update_query = f"UPDATE documents SET {', '.join(updates)} WHERE doc_id = %s"
    params.append(doc_id)
    
    # Execute update
    session.execute(update_query, params)
    session.commit()
    
    console.print(f"[green]Document {doc_id} updated successfully[/green]")

@app.command("delete")
def delete_document(
    doc_id: str = typer.Argument(..., help="Document ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation")
):
    """
    Delete a document and its associated data
    """
    # Create session
    session = get_session()
    
    # Get document
    query = "SELECT file_name FROM documents WHERE doc_id = %s"
    result = session.execute(query, (doc_id,)).fetchone()
    
    if not result:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        return
    
    file_name = result[0]
    
    # Confirm deletion
    if not force:
        confirmed = typer.confirm(f"Are you sure you want to delete document '{file_name}' ({doc_id})?")
        if not confirmed:
            console.print("[yellow]Deletion cancelled[/yellow]")
            return
    
    # Delete document and associated data
    try:
        # Delete line items
        session.execute("DELETE FROM line_items WHERE doc_id = %s", (doc_id,))
        
        # Delete annotations
        session.execute("DELETE FROM annotations WHERE doc_id = %s", (doc_id,))
        
        # Delete pages
        session.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        
        # Delete document
        session.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        
        # Commit changes
        session.commit()
        
        console.print(f"[green]Document {doc_id} deleted successfully[/green]")
    
    except Exception as e:
        session.rollback()
        console.print(f"[red]Error deleting document: {str(e)}[/red]")
```

### Analysis Commands

```python
import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from pathlib import Path
import json

from ...financial_analysis.engine import FinancialAnalysisEngine
from ...db.session import get_session

# Create Typer app
app = typer.Typer(help="Analysis commands")

# Initialize console
console = Console()

@app.command("patterns")
def find_patterns(
    doc_id: str = typer.Option(None, "--doc", "-d", help="Analyze specific document"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", "-c", help="Minimum confidence threshold (0.0-1.0)"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file for results (JSON)"),
    pattern_type: str = typer.Option(None, "--type", "-t", 
                                    help="Pattern type (recurring_amount, reappearing_amount, inconsistent_markup)")
):
    """
    Find suspicious financial patterns
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Run analysis
    console.print("[blue]Running pattern analysis...[/blue]")
    
    try:
        if doc_id:
            # Analyze specific document
            results = engine.pattern_detector.detect_patterns(doc_id)
        else:
            # Analyze all documents
            results = engine.pattern_detector.detect_patterns()
        
        # Filter results
        filtered_results = []
        for pattern in results:
            if pattern.get('confidence', 0) >= min_confidence:
                if pattern_type is None or pattern.get('type') == pattern_type:
                    filtered_results.append(pattern)
        
        # Display results
        if not filtered_results:
            console.print("[yellow]No patterns found matching the criteria[/yellow]")
            return
        
        console.print(f"[green]Found {len(filtered_results)} suspicious patterns[/green]")
        
        # Display pattern summary
        table = Table(title=f"Suspicious Patterns ({len(filtered_results)} results)")
        table.add_column("Type")
        table.add_column("Amount", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Explanation")
        
        for pattern in filtered_results:
            table.add_row(
                pattern.get('type', 'Unknown'),
                f"${pattern.get('amount', 0):,.2f}",
                f"{pattern.get('confidence', 0):.2f}",
                pattern.get('explanation', '')[:80] + ('...' if len(pattern.get('explanation', '')) > 80 else '')
            )
        
        console.print(table)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(filtered_results, f, indent=2)
            
            console.print(f"[green]Results saved to {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")

@app.command("amount")
def analyze_amount(
    amount: float = typer.Argument(..., help="Amount to analyze"),
    tolerance: float = typer.Option(0.01, "--tolerance", "-t", help="Matching tolerance"),
    fuzzy: bool = typer.Option(False, "--fuzzy", "-f", help="Use fuzzy matching"),
    fuzzy_threshold: float = typer.Option(0.1, "--fuzzy-threshold", help="Threshold for fuzzy matching (0.0-1.0)")
):
    """
    Analyze a specific amount across all documents
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Run analysis
    console.print(f"[blue]Analyzing amount: ${amount:,.2f}[/blue]")
    
    try:
        if fuzzy:
            matches = engine.amount_matcher.find_fuzzy_matches(amount, fuzzy_threshold)
        else:
            matches = engine.amount_matcher.find_matches_by_amount(amount, tolerance)
        
        if not matches:
            console.print(f"[yellow]No matches found for amount ${amount:,.2f}[/yellow]")
            return
        
        console.print(f"[green]Found {len(matches)} matches for amount ${amount:,.2f}[/green]")
        
        # Display matches
        table = Table(title=f"Matches for ${amount:,.2f} ({len(matches)} results)")
        table.add_column("Document ID")
        table.add_column("Document Type")
        table.add_column("Party")
        table.add_column("Amount", justify="right")
        table.add_column("Description")
        table.add_column("Date")
        
        for match in matches:
            table.add_row(
                match.get('doc_id', ''),
                match.get('doc_type', 'Unknown'),
                match.get('party', 'Unknown'),
                f"${match.get('amount', 0):,.2f}",
                match.get('description', '')[:50] + ('...' if len(match.get('description', '')) > 50 else ''),
                str(match.get('date', 'Unknown'))
            )
        
        console.print(table)
        
        # Analyze amount history
        history = engine.chronology_analyzer.analyze_amount_history(matches)
        
        if history.get('key_events'):
            console.print("\n[bold]Key Events:[/bold]")
            
            for event in history.get('key_events'):
                if event.get('type') == 'first_occurrence':
                    console.print(f"[cyan]First Occurrence:[/cyan] ${event['event']['amount']:,.2f} in {event['event']['doc_type']} ({event['event']['date']})")
                elif event.get('type') == 'status_change':
                    console.print(f"[cyan]Status Change:[/cyan] From {event['from_event']['doc_type']} to {event['to_event']['doc_type']} ({event['to_event']['date']})")
    
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")

@app.command("document")
def analyze_document(
    doc_id: str = typer.Argument(..., help="Document ID to analyze"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file for results (JSON)")
):
    """
    Perform comprehensive analysis on a document
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Run analysis
    console.print(f"[blue]Analyzing document: {doc_id}[/blue]")
    
    try:
        results = engine.analyze_document(doc_id)
        
        # Display patterns
        if results.get('patterns'):
            pattern_table = Table(title=f"Patterns in Document: {doc_id}")
            pattern_table.add_column("Type")
            pattern_table.add_column("Amount", justify="right")
            pattern_table.add_column("Confidence", justify="right")
            pattern_table.add_column("Explanation")
            
            for pattern in results.get('patterns'):
                pattern_table.add_column(
                    pattern.get('type', 'Unknown'),
                    f"${pattern.get('amount', 0):,.2f}",
                    f"{pattern.get('confidence', 0):.2f}",
                    pattern.get('explanation', '')
                )
            
            console.print(pattern_table)
        
        # Display anomalies
        if results.get('anomalies'):
            anomaly_table = Table(title=f"Anomalies in Document: {doc_id}")
            anomaly_table.add_column("Type")
            anomaly_table.add_column("Confidence", justify="right")
            anomaly_table.add_column("Description")
            
            for anomaly in results.get('anomalies'):
                anomaly_table.add_row(
                    anomaly.get('type', 'Unknown'),
                    f"{anomaly.get('confidence', 0):.2f}",
                    anomaly.get('description', '')
                )
            
            console.print(anomaly_table)
        
        # Display amount matches
        if results.get('amount_matches'):
            match_table = Table(title=f"Amount Matches: {doc_id}")
            match_table.add_column("Amount", justify="right")
            match_table.add_column("Description")
            match_table.add_column("Match Count", justify="right")
            
            for match in results.get('amount_matches'):
                match_table.add_row(
                    f"${match.get('amount', 0):,.2f}",
                    match.get('description', '')[:50] + ('...' if len(match.get('description', '')) > 50 else ''),
                    str(len(match.get('matches', [])))
                )
            
            console.print(match_table)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[green]Results saved to {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")

@app.command("ai")
def ai_investigation(
    question: str = typer.Argument(..., help="Question to investigate"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file for results (markdown)"),
    context: str = typer.Option(None, "--context", "-c", help="Additional context for the investigation")
):
    """
    Use AI to investigate a question about the construction dispute
    """
    # Create session
    session = get_session()
    
    # Initialize AI components
    from ...ai.agents.investigator import InvestigatorAgent
    from ...ai.llm import LLMManager
    
    # Load configuration
    config = {}
    
    # Initialize LLM manager
    llm_manager = LLMManager(config)
    
    # Initialize investigator agent
    agent = InvestigatorAgent(session, llm_manager, config)
    
    # Run investigation
    console.print(f"[blue]Investigating: {question}[/blue]")
    
    try:
        results = agent.investigate(question, context)
        
        # Display final report
        console.print("\n[bold]Investigation Results:[/bold]\n")
        console.print(results.get('final_report', 'No results generated'))
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("# Investigation Report\n\n")
                f.write(f"## Question\n\n{question}\n\n")
                if context:
                    f.write(f"## Context\n\n{context}\n\n")
                f.write("## Findings\n\n")
                f.write(results.get('final_report', 'No results generated'))
            
            console.print(f"[green]Results saved to {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during investigation: {str(e)}[/red]")
```

### Query Commands

```python
import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
import json

from ...db.session import get_session
from ...ai.semantic_search import semantic_search

# Create Typer app
app = typer.Typer(help="Query and search commands")

# Initialize console
console = Console()

@app.command("sql")
def run_sql(
    query: str = typer.Argument(..., help="SQL query to execute"),
    params: list[str] = typer.Option([], "--param", "-p", help="Query parameters"),
    output_file: str = typer.Option(None, "--output", "-o", help="Output file for results (CSV, JSON)")
):
    """
    Execute a SQL query against the database
    """
    # Create session
    session = get_session()
    
    # Display the query
    console.print("[bold]Executing query:[/bold]")
    console.print(Syntax(query, "sql"))
    
    try:
        # Execute query
        results = session.execute(query, params).fetchall()
        
        if not results:
            console.print("[yellow]Query returned no results[/yellow]")
            return
        
        # Display results
        column_names = results[0].keys()
        
        table = Table(title=f"Query Results ({len(results)} rows)")
        
        for column in column_names:
            table.add_column(column)
        
        for row in results:
            table.add_row(*[str(value) for value in row])
        
        console.print(table)
        
        # Save results if requested
        if output_file:
            if output_file.endswith(".csv"):
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(column_names)
                    for row in results:
                        writer.writerow(row)
            elif output_file.endswith(".json"):
                with open(output_file, 'w') as f:
                    json.dump([dict(row) for row in results], f, indent=2, default=str)
            else:
                console.print("[red]Unsupported output format. Use .csv or .json[/red]")
                return
            
            console.print(f"[green]Results saved to {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error executing query: {str(e)}[/red]")

@app.command("search")
def search_documents(
    query: str = typer.Argument(..., help="Search query"),
    doc_type: str = typer.Option(None, "--type", "-t", help="Filter by document type"),
    party: str = typer.Option(None, "--party", "-p", help="Filter by party"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Use semantic search"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results")
):
    """
    Search for documents by content
    """
    # Create session
    session = get_session()
    
    try:
        if semantic:
            # Use semantic search
            search_results = semantic_search(session, query, limit=limit, doc_type=doc_type, party=party)
        else:
            # Use full-text search
            conditions = ["to_tsvector('english', content) @@ plainto_tsquery('english', %s)"]
            params = [query]
            
            if doc_type:
                conditions.append("doc_type = %s")
                params.append(doc_type)
            
            if party:
                conditions.append("party = %s")
                params.append(party)
            
            search_query = f"""
                SELECT 
                    d.doc_id, 
                    d.file_name, 
                    d.doc_type, 
                    d.party,
                    p.page_number,
                    p.content,
                    ts_rank(to_tsvector('english', p.content), plainto_tsquery('english', %s)) AS rank
                FROM 
                    documents d
                JOIN
                    pages p ON d.doc_id = p.doc_id
                WHERE 
                    {' AND '.join(conditions)}
                ORDER BY 
                    rank DESC
                LIMIT 
                    %s
            """
            
            params.append(query)  # Add query again for ts_rank
            params.append(limit)
            
            search_results = session.execute(search_query, params).fetchall()
        
        if not search_results:
            console.print(f"[yellow]No results found for query: {query}[/yellow]")
            return
        
        # Display results
        console.print(f"[green]Found {len(search_results)} results for query: {query}[/green]")
        
        table = Table(title=f"Search Results ({len(search_results)} results)")
        table.add_column("Document ID")
        table.add_column("Filename")
        table.add_column("Type")
        table.add_column("Party")
        table.add_column("Page")
        table.add_column("Relevance", justify="right")
        table.add_column("Content")
        
        for result in search_results:
            if semantic:
                doc_id = result.get('doc_id')
                filename = result.get('file_name')
                doc_type = result.get('doc_type')
                party = result.get('party')
                page = result.get('page_number')
                relevance = f"{result.get('similarity', 0):.2f}"
                content = result.get('content', '')[:100] + '...'
            else:
                doc_id, filename, doc_type, party, page, content, relevance = result
                relevance = f"{relevance:.2f}"
                content = content[:100] + '...' if content and len(content) > 100 else content
            
            table.add_row(
                doc_id,
                filename,
                doc_type or "Unknown",
                party or "Unknown",
                str(page),
                relevance,
                content
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error during search: {str(e)}[/red]")

@app.command("find")
def find_financial_items(
    description: str = typer.Option(None, "--desc", "-d", help="Search by description"),
    min_amount: float = typer.Option(None, "--min", help="Minimum amount"),
    max_amount: float = typer.Option(None, "--max", help="Maximum amount"),
    amount: float = typer.Option(None, "--amount", "-a", help="Exact amount (with small tolerance)"),
    tolerance: float = typer.Option(0.01, "--tolerance", "-t", help="Tolerance for exact amount matching"),
    doc_type: str = typer.Option(None, "--type", help="Filter by document type"),
    party: str = typer.Option(None, "--party", "-p", help="Filter by party"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of results")
):
    """
    Find financial line items matching criteria
    """
    # Create session
    session = get_session()
    
    # Build query
    conditions = ["1=1"]
    params = []
    
    if description:
        conditions.append("li.description ILIKE %s")
        params.append(f"%{description}%")
    
    if amount is not None:
        conditions.append("li.amount BETWEEN %s AND %s")
        params.append(amount - tolerance)
        params.append(amount + tolerance)
    else:
        if min_amount is not None:
            conditions.append("li.amount >= %s")
            params.append(min_amount)
        
        if max_amount is not None:
            conditions.append("li.amount <= %s")
            params.append(max_amount)
    
    if doc_type:
        conditions.append("d.doc_type = %s")
        params.append(doc_type)
    
    if party:
        conditions.append("d.party = %s")
        params.append(party)
    
    query = f"""
        SELECT 
            li.item_id,
            li.doc_id,
            d.file_name,
            d.doc_type,
            d.party,
            li.description,
            li.amount,
            li.status
        FROM 
            line_items li
        JOIN
            documents d ON li.doc_id = d.doc_id
        WHERE 
            {' AND '.join(conditions)}
        ORDER BY 
            li.amount DESC
        LIMIT 
            %s
    """
    
    params.append(limit)
    
    try:
        # Execute query
        results = session.execute(query, params).fetchall()
        
        if not results:
            console.print("[yellow]No line items found matching the criteria[/yellow]")
            return
        
        # Display results
        console.print(f"[green]Found {len(results)} line items matching the criteria[/green]")
        
        table = Table(title=f"Financial Items ({len(results)} results)")
        table.add_column("Item ID")
        table.add_column("Document")
        table.add_column("Type")
        table.add_column("Party")
        table.add_column("Amount", justify="right")
        table.add_column("Status")
        table.add_column("Description")
        
        for row in results:
            item_id, doc_id, filename, doc_type, party, description, amount, status = row
            
            table.add_row(
                str(item_id),
                f"{filename} ({doc_id})",
                doc_type or "Unknown",
                party or "Unknown",
                f"${amount:,.2f}" if amount is not None else "N/A",
                status or "Unknown",
                description[:50] + '...' if description and len(description) > 50 else description or "N/A"
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error during search: {str(e)}[/red]")

@app.command("ask")
def ask_question(
    question: str = typer.Argument(..., help="Question to ask about the documents"),
    doc_ids: list[str] = typer.Option([], "--doc", "-d", help="Limit to specific documents")
):
    """
    Ask a natural language question about the documents
    """
    # Create session
    session = get_session()
    
    # Initialize AI components
    from ...ai.llm import LLMManager
    
    # Load configuration
    config = {}
    
    # Initialize LLM manager
    llm_manager = LLMManager(config)
    
    try:
        # Get relevant documents
        if doc_ids:
            # Use specified documents
            docs_filter = f"WHERE d.doc_id IN ({', '.join(['%s'] * len(doc_ids))})"
            docs = session.execute(f"""
                SELECT 
                    d.doc_id, 
                    d.file_name, 
                    d.doc_type, 
                    d.party,
                    p.page_number,
                    p.content
                FROM 
                    documents d
                JOIN
                    pages p ON d.doc_id = p.doc_id
                {docs_filter}
                ORDER BY
                    d.doc_id, p.page_number
            """, doc_ids).fetchall()
        else:
            # Use semantic search to find relevant documents
            docs = semantic_search(session, question, limit=5)
        
        if not docs:
            console.print("[yellow]No relevant documents found[/yellow]")
            return
        
        # Prepare context for LLM
        context = "I have the following documents:\n\n"
        
        for doc in docs:
            if isinstance(doc, dict):  # Handle semantic search results
                context += f"Document: {doc.get('file_name')} (Type: {doc.get('doc_type')}, Party: {doc.get('party')})\n"
                context += f"Page {doc.get('page_number')}:\n{doc.get('content')[:1000]}...\n\n"
            else:  # Handle SQL results
                doc_id, filename, doc_type, party, page, content = doc
                context += f"Document: {filename} (Type: {doc_type}, Party: {party})\n"
                context += f"Page {page}:\n{content[:1000]}...\n\n"
        
        # Prepare prompt
        prompt = f"""Based on the following documents from a construction dispute, please answer this question:

Question: {question}

Context:
{context}

Please provide a detailed answer based only on the information in these documents. If the answer cannot be determined from the documents, please say so.
"""
        
        # Generate answer
        console.print("[blue]Analyzing documents and generating answer...[/blue]")
        answer = llm_manager.generate(prompt)
        
        # Display answer
        console.print("\n[bold]Answer:[/bold]\n")
        console.print(answer)
    
    except Exception as e:
        console.print(f"[red]Error processing question: {str(e)}[/red]")
```

### Report Commands

```python
import typer
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path
import json

from ...db.session import get_session
from ...financial_analysis.engine import FinancialAnalysisEngine
from ...ai.llm import LLMManager
from ...reporting.generator import ReportGenerator

# Create Typer app
app = typer.Typer(help="Report generation commands")

# Initialize console
console = Console()

@app.command("summary")
def generate_summary(
    output_file: Path = typer.Argument(..., help="Output file path (PDF, HTML, or Markdown)"),
    title: str = typer.Option("Financial Summary Report", "--title", "-t", help="Report title"),
    doc_ids: list[str] = typer.Option([], "--doc", "-d", help="Include specific documents"),
    include_patterns: bool = typer.Option(True, "--patterns/--no-patterns", help="Include suspicious patterns"),
    include_anomalies: bool = typer.Option(True, "--anomalies/--no-anomalies", help="Include anomalies"),
    include_disputes: bool = typer.Option(True, "--disputes/--no-disputes", help="Include disputed amounts")
):
    """
    Generate a summary report of financial analysis
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Initialize report generator
    report_generator = ReportGenerator()
    
    console.print("[blue]Generating financial summary report...[/blue]")
    
    try:
        # Run financial analysis
        console.print("Running financial analysis...")
        
        # Generate report data
        report_data = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {},
            "suspicious_patterns": [],
            "anomalies": [],
            "disputed_amounts": []
        }
        
        # Get summary statistics
        console.print("Calculating summary statistics...")
        summary_stats = engine._generate_summary_stats()
        report_data["summary"] = summary_stats
        
        # Get suspicious patterns if requested
        if include_patterns:
            console.print("Detecting suspicious patterns...")
            patterns = engine.find_suspicious_patterns()
            report_data["suspicious_patterns"] = patterns
        
        # Get anomalies if requested
        if include_anomalies:
            console.print("Detecting anomalies...")
            anomalies = engine.anomaly_detector.detect_anomalies()
            report_data["anomalies"] = anomalies
        
        # Get disputed amounts if requested
        if include_disputes:
            console.print("Finding disputed amounts...")
            disputed_amounts = engine._find_disputed_amounts()
            report_data["disputed_amounts"] = disputed_amounts
        
        # Generate the report
        console.print("Generating report...")
        report_content = report_generator.generate_summary_report(report_data)
        
        # Save the report
        file_extension = output_file.suffix.lower()
        
        if file_extension == ".md":
            # Save as Markdown
            with open(output_file, "w") as f:
                f.write(report_content)
        elif file_extension == ".html":
            # Save as HTML
            html_content = report_generator.convert_markdown_to_html(report_content)
            with open(output_file, "w") as f:
                f.write(html_content)
        elif file_extension == ".pdf":
            # Save as PDF
            report_generator.generate_pdf(report_content, output_file)
        else:
            console.print(f"[red]Unsupported output format: {file_extension}. Use .md, .html, or .pdf[/red]")
            return
        
        console.print(f"[green]Report generated successfully: {output_file}[/green]")
        
        # Preview report (for Markdown)
        if file_extension == ".md":
            console.print("\n[bold]Report Preview:[/bold]\n")
            console.print(Markdown(report_content[:1000] + "...\n\n(Preview truncated)"))
    
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")

@app.command("detailed")
def generate_detailed_report(
    output_file: Path = typer.Argument(..., help="Output file path (PDF, HTML, or Markdown)"),
    title: str = typer.Option("Detailed Financial Analysis Report", "--title", "-t", help="Report title"),
    doc_ids: list[str] = typer.Option([], "--doc", "-d", help="Include specific documents"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", "-c", help="Minimum confidence threshold (0.0-1.0)"),
    include_evidence: bool = typer.Option(True, "--evidence/--no-evidence", help="Include evidence details"),
    include_timeline: bool = typer.Option(True, "--timeline/--no-timeline", help="Include chronological timeline")
):
    """
    Generate a detailed financial analysis report
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Initialize report generator
    report_generator = ReportGenerator()
    
    console.print("[blue]Generating detailed financial analysis report...[/blue]")
    
    try:
        # Run financial analysis
        console.print("Running comprehensive financial analysis...")
        
        # Generate the report
        console.print("Generating detailed report...")
        
        # [Implementation similar to summary report but with more detail]
        
        console.print(f"[green]Detailed report generated successfully: {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")

@app.command("custom")
def generate_custom_report(
    question: str = typer.Argument(..., help="Specific question or focus for the report"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    title: str = typer.Option(None, "--title", "-t", help="Report title"),
    audience: str = typer.Option("attorney", "--audience", "-a", 
                               help="Target audience (attorney, client, expert, mediator)"),
    format: str = typer.Option("detailed", "--format", "-f", 
                             help="Report format (summary, detailed, presentation)")
):
    """
    Generate a custom report focused on a specific question
    """
    # Create session
    session = get_session()
    
    # Initialize AI components
    llm_manager = LLMManager()
    
    # Initialize reporter agent
    from ...ai.agents.reporter import ReporterAgent
    reporter = ReporterAgent(session, llm_manager)
    
    console.print(f"[blue]Generating custom report for question: {question}[/blue]")
    
    try:
        # Generate the report
        report_content = reporter.generate_report(
            question=question,
            title=title or f"Analysis: {question}",
            audience=audience,
            format=format
        )
        
        # Save the report
        file_extension = output_file.suffix.lower()
        
        if file_extension == ".md":
            # Save as Markdown
            with open(output_file, "w") as f:
                f.write(report_content)
        elif file_extension == ".html":
            # Save as HTML
            html_content = reporter.convert_to_html(report_content)
            with open(output_file, "w") as f:
                f.write(html_content)
        elif file_extension == ".pdf":
            # Save as PDF
            reporter.convert_to_pdf(report_content, output_file)
        else:
            console.print(f"[red]Unsupported output format: {file_extension}. Use .md, .html, or .pdf[/red]")
            return
        
        console.print(f"[green]Custom report generated successfully: {output_file}[/green]")
        
        # Preview report (for Markdown)
        if file_extension == ".md":
            console.print("\n[bold]Report Preview:[/bold]\n")
            console.print(Markdown(report_content[:1000] + "...\n\n(Preview truncated)"))
    
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")

@app.command("evidence")
def generate_evidence_report(
    amount: float = typer.Argument(..., help="Amount to create evidence chain for"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    tolerance: float = typer.Option(0.01, "--tolerance", "-t", help="Matching tolerance"),
    include_images: bool = typer.Option(True, "--images/--no-images", help="Include document images")
):
    """
    Generate an evidence chain report for a specific amount
    """
    # Create session
    session = get_session()
    
    # Initialize analysis engine
    engine = FinancialAnalysisEngine(session)
    
    # Initialize report generator
    report_generator = ReportGenerator()
    
    console.print(f"[blue]Generating evidence chain report for amount: ${amount:,.2f}[/blue]")
    
    try:
        # Find matches for this amount
        matches = engine.amount_matcher.find_matches_by_amount(amount, tolerance)
        
        if not matches:
            console.print(f"[yellow]No matches found for amount ${amount:,.2f}[/yellow]")
            return
        
        console.print(f"[green]Found {len(matches)} matches for amount ${amount:,.2f}[/green]")
        
        # Analyze amount history
        history = engine.chronology_analyzer.analyze_amount_history(matches)
        
        # Generate evidence report
        report_content = report_generator.generate_evidence_report(
            amount=amount,
            matches=matches,
            history=history,
            include_images=include_images
        )
        
        # Save the report
        file_extension = output_file.suffix.lower()
        
        if file_extension == ".md":
            # Save as Markdown
            with open(output_file, "w") as f:
                f.write(report_content)
        elif file_extension == ".html":
            # Save as HTML
            html_content = report_generator.convert_markdown_to_html(report_content)
            with open(output_file, "w") as f:
                f.write(html_content)
        elif file_extension == ".pdf":
            # Save as PDF
            report_generator.generate_pdf(report_content, output_file)
        else:
            console.print(f"[red]Unsupported output format: {file_extension}. Use .md, .html, or .pdf[/red]")
            return
        
        console.print(f"[green]Evidence report generated successfully: {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")
```

## Implementation Guidelines

1. **User Experience**: Optimize for attorneys with limited technical experience
2. **Error Handling**: Provide clear error messages and recovery options
3. **Documentation**: Include comprehensive help text and examples
4. **Progress Indicators**: Show progress for long-running operations
5. **Output Formatting**: Use Rich library for clear, structured output
6. **Configurability**: Allow customization through config files and CLI options

## Command Examples

### Document Ingestion

```bash
# Ingest a single document
cdas doc ingest contract.pdf --type change_order --party contractor

# Ingest all PDFs in a directory
cdas doc ingest ./documents/ --recursive

# Batch process with specific document type
cdas doc ingest ./change_orders/ --type change_order --party contractor --batch
```

### Financial Analysis

```bash
# Find suspicious patterns
cdas analyze patterns --min-confidence 0.8

# Analyze a specific amount
cdas analyze amount 12345.67

# Analyze a specific document
cdas analyze document doc_123abc

# Use AI to investigate a question
cdas analyze ai "What evidence suggests the contractor double-billed for the HVAC system?"
```

### Querying

```bash
# Run a SQL query
cdas query sql "SELECT * FROM line_items WHERE amount > 10000 ORDER BY amount DESC"

# Search for documents by content
cdas query search "HVAC installation" --type payment_app

# Find financial items by criteria
cdas query find --min 5000 --max 10000 --desc "electrical"

# Ask a natural language question
cdas query ask "When was the first time the contractor billed for elevator maintenance?"
```

### Reporting

```bash
# Generate a summary report
cdas report summary financial_summary.pdf --title "Financial Summary Report"

# Generate a detailed report
cdas report detailed detailed_analysis.pdf --include-evidence

# Generate a custom report for a specific question
cdas report custom "What evidence suggests improper billing practices?" report.pdf --audience mediator

# Generate an evidence chain report for a specific amount
cdas report evidence 23456.78 evidence_chain.pdf
```

## Testing Strategy

1. **Unit Tests**: Test individual command functions
2. **Integration Tests**: Test command interactions with the database
3. **End-to-End Tests**: Test complete workflows
4. **User Acceptance Tests**: Test with actual users

## Security Considerations

1. **Input Validation**: Validate all user input
2. **Path Traversal Prevention**: Prevent directory traversal attacks
3. **SQL Injection Prevention**: Use parameterized queries
4. **Error Messages**: Avoid exposing sensitive information in error messages
