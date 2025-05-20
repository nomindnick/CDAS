#!/usr/bin/env python
"""
Import documents from a CSV file into the CDAS system.

This script reads a CSV file with document metadata and imports each document
into the CDAS system using the document processor.
"""

import os
import sys
import csv
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import CDAS modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cdas.db.session import session_scope
from cdas.db.init import init_database
from cdas.db.reset import reset_database
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("import_documents")

def parse_date(date_str):
    """Parse date string into a datetime object.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # Try various date formats
    formats = [
        '%B %d, %Y',      # March 15, 2023
        '%b %d, %Y',      # Mar 15, 2023
        '%m/%d/%Y',       # 03/15/2023
        '%Y-%m-%d',       # 2023-03-15
        '%d-%m-%Y',       # 15-03-2023
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract date using regex
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 
        'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Match patterns like "March 15, 2023" or "March 2023"
    pattern = r'([a-zA-Z]+)\s+(\d{1,2})?,?\s+(\d{4})'
    match = re.search(pattern, date_str)
    if match:
        month_str, day_str, year_str = match.groups()
        month = month_names.get(month_str.lower())
        day = int(day_str) if day_str else 1
        year = int(year_str)
        
        if month and 1 <= day <= 31 and year >= 1900:
            return datetime(year, month, day)
    
    # If all parsing attempts fail
    logger.warning(f"Could not parse date: {date_str}")
    return None


def extract_document_date(file_path):
    """Extract document date from the file content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Tuple of (creation_date, received_date) as datetime objects
    """
    creation_date = None
    received_date = None
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Look for date patterns in the content
            date_patterns = [
                r'DATE:?\s*([A-Za-z]+ \d{1,2}, \d{4})',
                r'PERIOD FROM:?\s*([A-Za-z]+ \d{1,2}, \d{4})',
                r'CONTRACT DATE:?\s*([A-Za-z]+ \d{1,2}, \d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    date_str = match.group(1)
                    parsed_date = parse_date(date_str)
                    if parsed_date:
                        creation_date = parsed_date
                        break
    except Exception as e:
        logger.warning(f"Error extracting date from {file_path}: {str(e)}")
    
    return creation_date, received_date


def import_documents(csv_file, reset=False):
    """Import documents from a CSV file."""
    
    # Reset database if requested
    if reset:
        logger.info("Resetting database...")
        reset_database()
        # The reset_database function already initializes the database, so we don't need to call init_database separately
    
    # Create document processor factory
    factory = DocumentProcessorFactory()
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        # Process each row
        for i, row in enumerate(reader):
            try:
                file_path = row.get('file_path')
                doc_type = row.get('doc_type')
                party = row.get('party')
                project_id = row.get('project_id')
                notes = row.get('notes', '')
                
                # Validate required fields
                if not all([file_path, doc_type, party]):
                    logger.error(f"Row {i+2}: Missing required field(s)")
                    continue
                
                # Validate file exists
                if not os.path.exists(file_path):
                    logger.error(f"Row {i+2}: File does not exist: {file_path}")
                    continue
                
                logger.info(f"Processing {file_path} ({doc_type}, {party})...")
                
                # Extract document dates
                creation_date, received_date = extract_document_date(file_path)
                
                # Process the document
                with session_scope() as session:
                    # Add project_id and notes to metadata if provided
                    kwargs = {}
                    if project_id:
                        kwargs['project_id'] = project_id
                    
                    metadata = {'notes': notes} if notes else {}
                    kwargs['metadata'] = metadata
                    
                    # Handle document dates
                    if creation_date:
                        kwargs['date_created'] = creation_date
                    if received_date:
                        kwargs['date_received'] = received_date
                    
                    # Convert strings to enum values
                    doc_type_enum = DocumentType(doc_type) if doc_type else None
                    party_enum = PartyType(party) if party else None
                    
                    # Process the document directly
                    processor = factory.create_processor(session)
                    
                    # Add more verbose logging for debugging
                    logger.info(f"Processing document: {file_path}, type: {doc_type}, party: {party}")
                    
                    try:
                        result = processor.process_document(
                            file_path,
                            doc_type_enum,
                            party_enum,
                            save_to_db=True,
                            **kwargs
                        )
                        
                        # More detailed success/failure logging
                        if result.success:
                            logger.info(f"Successfully processed with ID: {result.document_id}")
                            logger.info(f"Extracted metadata: {result.metadata}")
                            logger.info(f"Found {len(result.line_items)} line items")
                        else:
                            logger.error(f"Failed to process: {result.error}")
                    except Exception as e:
                        logger.exception(f"Exception during document processing: {str(e)}")
                        raise
            
            except Exception as e:
                logger.exception(f"Error processing row {i+2}: {str(e)}")
    
    logger.info("Import completed")

def main():
    parser = argparse.ArgumentParser(description="Import documents from a CSV file")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--reset", action="store_true", help="Reset the database before importing")
    
    args = parser.parse_args()
    
    import_documents(args.csv_file, args.reset)

if __name__ == "__main__":
    main()