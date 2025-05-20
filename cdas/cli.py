#!/usr/bin/env python
"""
Command-line interface for the Construction Document Analysis System.

This module provides the main entry point for the CDAS CLI, with commands
for document management, financial analysis, and reporting.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import datetime
import markdown

from cdas.db.session import get_session, session_scope
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType
from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_document_commands(subparsers):
    """Set up document management commands.
    
    Args:
        subparsers: argparse subparsers object
    """
    # Document command group
    doc_parser = subparsers.add_parser('doc', help='Document management commands')
    doc_subparsers = doc_parser.add_subparsers(dest='doc_command', help='Document command')
    
    # Ingest command
    ingest_parser = doc_subparsers.add_parser('ingest', help='Ingest a document')
    ingest_parser.add_argument('file_path', help='Path to the document file')
    ingest_parser.add_argument('--type', 
                            choices=[t.value for t in DocumentType], 
                            required=True,
                            help='Type of document')
    ingest_parser.add_argument('--party', 
                            choices=[p.value for p in PartyType], 
                            required=True,
                            help='Party associated with the document')
    ingest_parser.add_argument('--project', 
                            help='Project identifier')
    ingest_parser.add_argument('--no-db', 
                            action='store_true',
                            help='Do not save to database')
    ingest_parser.add_argument('--no-handwriting', 
                            action='store_true',
                            help='Do not extract handwritten text')
    ingest_parser.add_argument('--no-tables', 
                            action='store_true',
                            help='Do not extract tables')
    
    # List command
    list_parser = doc_subparsers.add_parser('list', help='List documents')
    list_parser.add_argument('--type', 
                           choices=[t.value for t in DocumentType],
                           help='Filter by document type')
    list_parser.add_argument('--party', 
                           choices=[p.value for p in PartyType],
                           help='Filter by party')
    list_parser.add_argument('--project', 
                           help='Filter by project')
    list_parser.add_argument('--start-date',
                           help='Filter by start date (YYYY-MM-DD)')
    list_parser.add_argument('--end-date',
                           help='Filter by end date (YYYY-MM-DD)')
    
    # Show command
    show_parser = doc_subparsers.add_parser('show', help='Show document details')
    show_parser.add_argument('doc_id', help='Document ID')
    show_parser.add_argument('--items', 
                           action='store_true',
                           help='Include line items')
    show_parser.add_argument('--pages', 
                           action='store_true',
                           help='Include page content')


def setup_analyze_commands(subparsers):
    """Set up financial analysis commands.
    
    Args:
        subparsers: argparse subparsers object
    """
    # Analysis command group
    analyze_parser = subparsers.add_parser('analyze', help='Financial analysis commands')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_command', help='Analysis command')
    
    # Patterns command
    patterns_parser = analyze_subparsers.add_parser('patterns', help='Detect financial patterns')
    patterns_parser.add_argument('--min-confidence', 
                               type=float,
                               default=0.7,
                               help='Minimum confidence level')
    patterns_parser.add_argument('--doc-id',
                               help='Analyze a specific document')
    
    # Amount command
    amount_parser = analyze_subparsers.add_parser('amount', help='Analyze a specific amount')
    amount_parser.add_argument('amount', 
                             type=float,
                             help='Amount to analyze')
    amount_parser.add_argument('--tolerance', 
                             type=float,
                             default=0.01,
                             help='Matching tolerance')
    
    # Document command
    doc_parser = analyze_subparsers.add_parser('document', help='Analyze a document')
    doc_parser.add_argument('doc_id', help='Document ID')


def setup_query_commands(subparsers):
    """Set up querying commands.
    
    Args:
        subparsers: argparse subparsers object
    """
    # Query command group
    query_parser = subparsers.add_parser('query', help='Querying commands')
    query_subparsers = query_parser.add_subparsers(dest='query_command', help='Query command')
    
    # Search command
    search_parser = query_subparsers.add_parser('search', help='Search for text in documents')
    search_parser.add_argument('text', help='Text to search for')
    search_parser.add_argument('--type', 
                             choices=[t.value for t in DocumentType],
                             help='Filter by document type')
    search_parser.add_argument('--party', 
                             choices=[p.value for p in PartyType],
                             help='Filter by party')
    
    # Find command
    find_parser = query_subparsers.add_parser('find', help='Find line items by amount range')
    find_parser.add_argument('--min', 
                           type=float,
                           help='Minimum amount')
    find_parser.add_argument('--max', 
                           type=float,
                           help='Maximum amount')
    find_parser.add_argument('--desc', 
                           help='Description keyword')
    
    # Ask command
    ask_parser = query_subparsers.add_parser('ask', help='Ask a natural language question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--verbose', '-v', action='store_true', 
                         help='Show more detailed information in the answer')


def setup_report_commands(subparsers):
    """Set up reporting commands.
    
    Args:
        subparsers: argparse subparsers object
    """
    # Report command group
    report_parser = subparsers.add_parser('report', help='Reporting commands')
    report_subparsers = report_parser.add_subparsers(dest='report_command', help='Report command')
    
    # Summary report command
    summary_parser = report_subparsers.add_parser('summary', help='Generate summary report')
    summary_parser.add_argument('output_path', help='Output file path')
    summary_parser.add_argument('--format', 
                              choices=['pdf', 'html', 'md', 'excel'],
                              default='pdf',
                              help='Output format')
    
    # Detailed report command
    detailed_parser = report_subparsers.add_parser('detailed', help='Generate detailed report')
    detailed_parser.add_argument('output_path', help='Output file path')
    detailed_parser.add_argument('--format', 
                               choices=['pdf', 'html', 'md', 'excel'],
                               default='pdf',
                               help='Output format')
    detailed_parser.add_argument('--include-evidence', 
                               action='store_true',
                               help='Include evidence citations and screenshots')
    
    # Evidence report command
    evidence_parser = report_subparsers.add_parser('evidence', help='Generate evidence report for amount')
    evidence_parser.add_argument('amount', 
                               type=float,
                               help='Amount to analyze')
    evidence_parser.add_argument('output_path', help='Output file path')
    evidence_parser.add_argument('--format', 
                               choices=['pdf', 'html', 'md', 'excel'],
                               default='pdf',
                               help='Output format')


def ingest_document(args):
    """Ingest a document.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Ingesting document: {args.file_path}")
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    with session_scope() as session:
        # Create document processor
        factory = DocumentProcessorFactory()
        processor = factory.create_processor(session)
        
        # Process document
        result = processor.process_document(
            str(file_path),
            args.type,
            args.party,
            project_id=args.project,
            save_to_db=not args.no_db,
            extract_handwriting=not args.no_handwriting,
            extract_tables=not args.no_tables
        )
        
        if result.success:
            logger.info(f"Document processed successfully")
            if result.document_id:
                logger.info(f"Document ID: {result.document_id}")
                
            # Print basic information about the document
            print(f"Document: {file_path.name}")
            print(f"Type: {args.type}")
            print(f"Party: {args.party}")
            
            if result.metadata.get('document_date'):
                print(f"Date: {result.metadata['document_date']}")
                
            if 'page_count' in result.metadata:
                print(f"Pages: {result.metadata['page_count']}")
                
            # Print line item count
            if result.line_items:
                print(f"Line items: {len(result.line_items)}")
                
                # Print a sample of line items
                if len(result.line_items) > 0:
                    print("\nSample line items:")
                    for item in result.line_items[:5]:
                        description = item.get('description', '')
                        if len(description) > 50:
                            description = description[:47] + "..."
                        amount = item.get('amount')
                        print(f"  - {description}: ${amount:.2f}" if amount else f"  - {description}")
                    
                    if len(result.line_items) > 5:
                        print(f"  ... and {len(result.line_items) - 5} more items")
        else:
            logger.error(f"Error processing document: {result.error}")


def list_documents(args):
    """List documents.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Listing documents")
    
    # Parse date filters
    start_date = None
    end_date = None
    
    if hasattr(args, 'start_date') and args.start_date:
        try:
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid start date format: {args.start_date}. Expected YYYY-MM-DD.")
    
    if hasattr(args, 'end_date') and args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid end date format: {args.end_date}. Expected YYYY-MM-DD.")
    
    with session_scope() as session:
        from cdas.db.operations import search_documents
        
        documents = search_documents(
            session,
            doc_type=args.type,
            party=args.party,
            date_start=start_date,
            date_end=end_date
        )
        
        # Filter by project if provided
        if hasattr(args, 'project') and args.project:
            documents = [doc for doc in documents if 
                         doc.metadata and 
                         doc.metadata.get("project_id") == args.project]
        
        if not documents:
            print("No documents found matching criteria")
            return
        
        print(f"Found {len(documents)} documents:")
        
        # Print document list in table format
        headers = ["ID", "Type", "Party", "Date", "File Name"]
        # Calculate maximum width for each column
        max_widths = [len(header) for header in headers]
        
        for doc in documents:
            doc_values = [
                doc.doc_id,
                doc.doc_type or "?",
                doc.party or "?",
                doc.date_created.strftime("%Y-%m-%d") if doc.date_created else "?",
                doc.file_name
            ]
            
            # Update maximum widths
            for i, value in enumerate(doc_values):
                max_widths[i] = max(max_widths[i], len(str(value)))
        
        # Print headers
        header_row = " | ".join(f"{header:{width}}" for header, width in zip(headers, max_widths))
        print(header_row)
        print("-" * len(header_row))
        
        # Print document rows
        for doc in documents:
            date_str = doc.date_created.strftime("%Y-%m-%d") if hasattr(doc, 'date_created') and doc.date_created else "?"
            row = [
                f"{doc.doc_id:{max_widths[0]}}",
                f"{doc.doc_type or '?':{max_widths[1]}}",
                f"{doc.party or '?':{max_widths[2]}}",
                f"{date_str:{max_widths[3]}}",
                f"{doc.file_name:{max_widths[4]}}"
            ]
            print(" | ".join(row))


def show_document(args):
    """Show document details.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Showing document: {args.doc_id}")
    
    with session_scope() as session:
        from sqlalchemy.orm import joinedload
        from cdas.db.models import Document, Page, LineItem
        
        # Get document with optional relationships
        query = session.query(Document).filter(Document.doc_id == args.doc_id)
        
        if args.pages:
            query = query.options(joinedload(Document.pages))
        
        if args.items:
            query = query.options(joinedload(Document.line_items))
        
        document = query.first()
        
        if not document:
            logger.error(f"Document not found: {args.doc_id}")
            return
        
        # Print document details
        print(f"Document ID: {document.doc_id}")
        print(f"File: {document.file_name}")
        print(f"Type: {document.doc_type}")
        print(f"Party: {document.party}")
        
        if document.date_created:
            print(f"Date created: {document.date_created.strftime('%Y-%m-%d')}")
        
        if document.date_received:
            print(f"Date received: {document.date_received.strftime('%Y-%m-%d')}")
        
        if document.metadata:
            print("\nMetadata:")
            for key, value in document.metadata.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"  {key}: {value}")
        
        # Print pages
        if args.pages and document.pages:
            pages = sorted(document.pages, key=lambda p: p.page_number)
            print(f"\nPages: {len(pages)}")
            for page in pages:
                print(f"\nPage {page.page_number}:")
                print(f"  Tables: {'Yes' if page.has_tables else 'No'}")
                print(f"  Handwriting: {'Yes' if page.has_handwriting else 'No'}")
                print(f"  Financial data: {'Yes' if page.has_financial_data else 'No'}")
                
                # Print page content (truncated if too long)
                if page.content:
                    content_lines = page.content.strip().split('\n')
                    if len(content_lines) > 10:
                        content_preview = '\n  '.join(content_lines[:5] + ["..."] + content_lines[-5:])
                    else:
                        content_preview = '\n  '.join(content_lines)
                    
                    print(f"\n  Content:\n  {content_preview}")
        
        # Print line items
        if args.items and document.line_items:
            items = sorted(document.line_items, key=lambda i: i.item_id)
            print(f"\nLine items: {len(items)}")
            for i, item in enumerate(items):
                description = item.description
                if description and len(description) > 50:
                    description = description[:47] + "..."
                
                print(f"\nItem {i+1}:")
                print(f"  Description: {description}")
                
                if item.amount is not None:
                    print(f"  Amount: ${item.amount:.2f}")
                
                if item.quantity is not None:
                    print(f"  Quantity: {item.quantity}")
                
                if item.unit_price is not None:
                    print(f"  Unit price: ${item.unit_price:.2f}")
                
                if item.cost_code:
                    print(f"  Cost code: {item.cost_code}")
                
                if item.category:
                    print(f"  Category: {item.category}")


def analyze_patterns(args):
    """Detect financial patterns.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Detecting financial patterns (min confidence: {args.min_confidence})")
    
    with session_scope() as session:
        # Create financial analysis engine
        engine = FinancialAnalysisEngine(session)
        
        # Detect patterns
        patterns = engine.find_suspicious_patterns(args.min_confidence)
        
        print("Pattern analysis results:")
        
        if not patterns:
            print("No suspicious patterns found")
            return
        
        print(f"Found {len(patterns)} suspicious patterns:")
        
        for i, pattern in enumerate(patterns):
            print(f"\nPattern {i+1}:")
            print(f"Type: {pattern.get('type', 'Unknown')}")
            print(f"Description: {pattern.get('description', 'No description')}")
            print(f"Confidence: {pattern.get('confidence', 0.0):.2f}")
            
            # Print additional information based on pattern type
            if pattern.get('type') == 'recurring_amount' and 'amount' in pattern:
                print(f"Amount: ${pattern['amount']:.2f}")
                
                if 'document_types' in pattern:
                    print(f"Document types: {', '.join(pattern['document_types'])}")
            
            elif pattern.get('type') == 'circular_reference' and 'nodes' in pattern:
                print(f"Documents in cycle: {' -> '.join(pattern['nodes'])} -> {pattern['nodes'][0]}")


def analyze_amount(args):
    """Analyze a specific amount.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Analyzing amount: ${args.amount} (tolerance: {args.tolerance})")
    
    with session_scope() as session:
        # Create financial analysis engine
        engine = FinancialAnalysisEngine(session)
        
        # Analyze amount
        results = engine.analyze_amount(args.amount, args.tolerance)
        
        if not results.get('matches'):
            print(f"No matches found for amount ${args.amount:.2f}")
            return
        
        matches = results.get('matches', [])
        print(f"Found {len(matches)} matches for amount ${args.amount:.2f}:")
        
        for i, match in enumerate(matches):
            print(f"\nMatch {i+1}:")
            
            if 'doc_id' in match:
                print(f"Document: {match['doc_id']}")
            
            if 'doc_type' in match:
                print(f"Document type: {match['doc_type']}")
            
            if 'party' in match:
                print(f"Party: {match['party']}")
            
            if 'description' in match:
                description = match['description']
                if len(description) > 100:
                    description = description[:97] + "..."
                print(f"Description: {description}")
            
            if 'amount' in match:
                print(f"Amount: ${match['amount']:.2f}")
            
            if 'date' in match and match['date']:
                print(f"Date: {match['date']}")
        
        # Print anomalies
        if 'anomalies' in results and results['anomalies']:
            anomalies = results['anomalies']
            print(f"\nDetected {len(anomalies)} anomalies for this amount:")
            
            for i, anomaly in enumerate(anomalies):
                print(f"\nAnomaly {i+1}:")
                print(f"Type: {anomaly.get('type', 'Unknown')}")
                print(f"Confidence: {anomaly.get('confidence', 0.0):.2f}")
                
                if 'explanation' in anomaly:
                    print(f"Explanation: {anomaly['explanation']}")


def analyze_document(args):
    """Analyze a document.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Analyzing document: {args.doc_id}")
    
    with session_scope() as session:
        # Create financial analysis engine
        engine = FinancialAnalysisEngine(session)
        
        # Analyze document
        results = engine.analyze_document(args.doc_id)
        
        # Print patterns
        if 'patterns' in results and results['patterns']:
            patterns = results['patterns']
            print(f"Found {len(patterns)} patterns:")
            
            for i, pattern in enumerate(patterns):
                print(f"\nPattern {i+1}:")
                print(f"Type: {pattern.get('type', 'Unknown')}")
                print(f"Description: {pattern.get('description', 'No description')}")
                print(f"Confidence: {pattern.get('confidence', 0.0):.2f}")
        else:
            print("No patterns found")
        
        # Print anomalies
        if 'anomalies' in results and results['anomalies']:
            anomalies = results['anomalies']
            print(f"\nFound {len(anomalies)} anomalies:")
            
            for i, anomaly in enumerate(anomalies):
                print(f"\nAnomaly {i+1}:")
                print(f"Type: {anomaly.get('type', 'Unknown')}")
                print(f"Confidence: {anomaly.get('confidence', 0.0):.2f}")
                
                if 'explanation' in anomaly:
                    print(f"Explanation: {anomaly['explanation']}")
        else:
            print("\nNo anomalies found")
        
        # Print amount matches
        if 'amount_matches' in results and results['amount_matches']:
            matches = results['amount_matches']
            print(f"\nFound {len(matches)} amount matches:")
            
            # Group by amount
            amount_groups = {}
            for match in matches:
                amount = match.get('amount')
                if amount not in amount_groups:
                    amount_groups[amount] = []
                amount_groups[amount].append(match)
            
            for amount, group in sorted(amount_groups.items()):
                print(f"\nAmount ${amount:.2f}: {len(group)} matches")
                
                for i, match in enumerate(group[:3]):  # Show only first 3 matches per amount
                    print(f"  - {match.get('description', 'No description')} ({match.get('doc_type', '?')})")
                
                if len(group) > 3:
                    print(f"  ... and {len(group) - 3} more matches")
        else:
            print("\nNo amount matches found")


def search_documents(args):
    """Search for text in documents.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Searching for text: {args.text}")
    
    try:
        with session_scope() as session:
            # Try to use semantic search if available
            try:
                from cdas.ai.embeddings import EmbeddingManager
                from cdas.ai.semantic_search.search import semantic_search
                from cdas.config import get_config
                
                config = get_config()
                ai_config = config.get('ai', {})
                
                # Create embedding manager
                embedding_manager = EmbeddingManager(session, ai_config.get('embeddings', {}))
                
                print(f"Performing semantic search for: {args.text}")
                print("This may take a moment...")
                
                # Perform semantic search
                results = semantic_search(
                    session, 
                    embedding_manager, 
                    args.text, 
                    limit=10,
                    doc_type=args.type,
                    party=args.party
                )
                
                if not results:
                    print("No results found")
                    return
                
                print(f"Found {len(results)} semantically relevant documents:")
                
                for i, result in enumerate(results):
                    doc_info = result['document']
                    similarity = result['similarity']
                    
                    print(f"\nResult {i+1} (relevance: {similarity:.2f}):")
                    print(f"Document: {doc_info['title']}")
                    print(f"Type: {doc_info['doc_type']}")
                    print(f"Party: {doc_info['party']}")
                    print(f"Date: {doc_info['date']}")
                    print(f"Page: {result['page_number']}")
                    
                    # Print context (truncated)
                    context = result['context']
                    if len(context) > 300:
                        context = context[:297] + "..."
                    print(f"Content: {context}")
                
                return
                
            except ImportError:
                logger.warning("Semantic search components not available. Falling back to keyword search.")
            
            # Fallback to regular search
            from cdas.db.operations import search_document_content
            
            results = search_document_content(
                session,
                keyword=args.text,
                doc_type=args.type,
                party=args.party
            )
            
            if not results:
                print("No results found")
                return
            
            print(f"Found {len(results)} documents containing '{args.text}':")
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Document: {result['title']}")
                print(f"Type: {result['doc_type']}")
                print(f"Party: {result['party']}")
                print(f"Date: {result['date']}")
                
                # Print matching context if available
                if 'context' in result:
                    context = result['context']
                    if len(context) > 300:
                        context = context[:297] + "..."
                    print(f"Content: {context}")
    
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        logger.error(f"Error in search_documents: {str(e)}")


def find_line_items(args):
    """Find line items by amount range.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Finding line items in range: ${args.min or '?'} - ${args.max or '?'}")
    
    with session_scope() as session:
        from cdas.db.operations import search_line_items
        
        items = search_line_items(
            session,
            description_keyword=args.desc,
            min_amount=args.min,
            max_amount=args.max
        )
        
        if not items:
            print("No line items found matching criteria")
            return
        
        print(f"Found {len(items)} line items:")
        
        for i, item in enumerate(items[:20]):  # Limit to first 20 items
            description = item.description
            if description and len(description) > 50:
                description = description[:47] + "..."
            
            print(f"\nItem {i+1}:")
            print(f"  Document: {item.doc_id}")
            print(f"  Description: {description}")
            
            if item.amount is not None:
                print(f"  Amount: ${item.amount:.2f}")
            
            if item.cost_code:
                print(f"  Cost code: {item.cost_code}")
        
        if len(items) > 20:
            print(f"\n... and {len(items) - 20} more items")


def ask_question(args):
    """Ask a natural language question.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Question: {args.question}")
    
    try:
        from cdas.ai.question_answering import answer_question
        
        print(f"Analyzing question: {args.question}")
        print("This may take a moment...")
        
        result = answer_question(args.question)
        
        print("\nAnswer:")
        print(result['answer'])
        
        if args.verbose and 'search_results' in result and result['search_results']:
            print("\nRelevant documents:")
            for i, doc in enumerate(result['search_results'][:3]):  # Show top 3
                doc_info = doc['document']
                print(f"  {i+1}. {doc_info['title']} ({doc_info['doc_type']} from {doc_info['party']}, dated {doc_info['date']})")
    
    except ImportError as e:
        print("AI components not available. Make sure all required packages are installed.")
        logger.error(f"Error importing AI components: {str(e)}")
    except Exception as e:
        print("An error occurred while answering the question.")
        print(f"Error: {str(e)}")
        logger.error(f"Error in ask_question: {str(e)}")


def generate_report(args, report_type):
    """Generate a report.
    
    Args:
        args: Command-line arguments
        report_type: Type of report to generate
    """
    logger.info(f"Generating {report_type} report: {args.output_path}")
    
    with session_scope() as session:
        from cdas.reporting.generator import ReportGenerator
        
        # Create report generator
        report_generator = ReportGenerator(session)
        
        # Generate report based on type
        if report_type == 'summary':
            result = report_generator.generate_summary_report(
                args.output_path,
                project_id=getattr(args, 'project', None),
                format=args.format,
                include_evidence=getattr(args, 'include_evidence', False),
                created_by=getattr(args, 'user', None)
            )
        elif report_type == 'detailed':
            result = report_generator.generate_detailed_report(
                args.output_path,
                project_id=getattr(args, 'project', None),
                format=args.format,
                include_evidence=getattr(args, 'include_evidence', False),
                created_by=getattr(args, 'user', None)
            )
        elif report_type == 'evidence':
            result = report_generator.generate_evidence_report(
                args.amount,
                args.output_path,
                format=args.format,
                created_by=getattr(args, 'user', None)
            )
        else:
            logger.error(f"Unknown report type: {report_type}")
            print(f"Error: Unknown report type '{report_type}'")
            return
        
        print(f"{report_type.capitalize()} report generated: {args.output_path}")
        print(f"Report ID: {result.get('report_id')}")
        print(f"Format: {result.get('format')}")
        print(f"Content length: {result.get('content_length')} bytes")

        # Try to get file size
        try:
            file_size = os.path.getsize(args.output_path)
            print(f"File size: {file_size} bytes")
        except (FileNotFoundError, PermissionError):
            pass


def parse_args(args=None):
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Construction Document Analysis System')
    
    # Add subcommand parsers
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Set up command groups
    setup_document_commands(subparsers)
    setup_analyze_commands(subparsers)
    setup_query_commands(subparsers)
    setup_report_commands(subparsers)
    
    # Add global arguments
    parser.add_argument('--user', help='User identifier for tracking who created reports')
    parser.add_argument('--project', help='Project identifier')
    
    # Parse arguments
    return parser.parse_args(args)


def main():
    """Main entry point for the CDAS CLI."""
    args = parse_args()
    
    # Handle commands
    if args.command == 'doc':
        if args.doc_command == 'ingest':
            ingest_document(args)
        elif args.doc_command == 'list':
            list_documents(args)
        elif args.doc_command == 'show':
            show_document(args)
        else:
            parser.print_help()
    elif args.command == 'analyze':
        if args.analyze_command == 'patterns':
            analyze_patterns(args)
        elif args.analyze_command == 'amount':
            analyze_amount(args)
        elif args.analyze_command == 'document':
            analyze_document(args)
        else:
            parser.print_help()
    elif args.command == 'query':
        if args.query_command == 'search':
            search_documents(args)
        elif args.query_command == 'find':
            find_line_items(args)
        elif args.query_command == 'ask':
            ask_question(args)
        else:
            parser.print_help()
    elif args.command == 'report':
        if args.report_command == 'summary':
            generate_report(args, 'summary')
        elif args.report_command == 'detailed':
            generate_report(args, 'detailed')
        elif args.report_command == 'evidence':
            generate_report(args, 'evidence')
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()