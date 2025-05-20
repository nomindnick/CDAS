#!/usr/bin/env python3
"""
Example usage of the document processor module.
"""
import os
import sys
import logging
import re
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cdas.db.session import get_session, session_scope
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType


def parse_date(date_str):
    """Convert date string to a datetime object."""
    if not date_str:
        return None
        
    # List of date formats to try
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
            
    # Try to extract date using regex for common patterns
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
    
    return None


def extract_document_date(file_path):
    """Extract document date from the file content."""
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
        pass
    
    return creation_date, received_date


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def process_from_arguments():
    """Process a document from command line arguments."""
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python document_processor_example.py <file_path> [doc_type] [party]")
        print("\nAvailable document types:")
        print("  payment_app, change_order, invoice, contract, schedule, correspondence, other")
        print("\nAvailable parties:")
        print("  contractor, subcontractor, owner, architect, engineer, other")
        return
    
    # Get file path from command line
    file_path = Path(sys.argv[1])
    
    # Get optional document type (default: other)
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "other"
    
    # Get optional party (default: other)
    party = sys.argv[3] if len(sys.argv) > 3 else "other"
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Processing document: {file_path}")
    logger.info(f"Document type: {doc_type}")
    logger.info(f"Party: {party}")
    
    # Extract document dates
    creation_date, received_date = extract_document_date(file_path)
    
    # Create a database session
    with session_scope() as session:
        # Create document processor factory
        factory = DocumentProcessorFactory()
        
        # Create the processor
        processor = factory.create_processor(session)
        
        # Convert string types to enums
        doc_type_enum = getattr(DocumentType, doc_type.upper()) if hasattr(DocumentType, doc_type.upper()) else doc_type
        party_type_enum = getattr(PartyType, party.upper()) if hasattr(PartyType, party.upper()) else party
        
        # Process the document with date objects
        result = processor.process_document(
            file_path=file_path,
            doc_type=doc_type_enum,
            party=party_type_enum,
            date_created=creation_date,
            date_received=received_date,
            save_to_db=True
        )
        
        # Check if processing was successful
        if result.success:
            logger.info("Document processed successfully")
            logger.info(f"Document ID: {result.document_id}")
            
            # Print some information about the document
            print("\nDocument Information:")
            print(f"Document ID: {result.document_id}")
            for key, value in result.metadata.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"{key}: {value}")
            
            # Print extracted line items
            if result.line_items:
                print(f"\nExtracted {len(result.line_items)} line items:")
                for i, item in enumerate(result.line_items[:10]):  # Show only first 10 items
                    description = item.get('description', 'No description')
                    if len(description) > 50:
                        description = description[:47] + "..."
                    
                    amount = item.get('amount')
                    if amount is not None:
                        print(f"{i+1}. {description}: ${amount:.2f}")
                    else:
                        print(f"{i+1}. {description}")
                
                if len(result.line_items) > 10:
                    print(f"... and {len(result.line_items) - 10} more items")
            else:
                print("\nNo line items extracted")
        else:
            logger.error(f"Error processing document: {result.error}")


def demonstrate_extractors():
    """Demonstrate using different extractors with example documents."""
    # Example document paths (replace with actual paths)
    pdf_path = Path('example_documents/payment_application.pdf')
    excel_path = Path('example_documents/change_order.xlsx')
    image_path = Path('example_documents/invoice_scan.jpg')
    
    # Process documents if they exist
    documents_to_process = [
        (pdf_path, 'payment_app', 'contractor'),
        (excel_path, 'change_order', 'contractor'),
        (image_path, 'invoice', 'subcontractor'),
    ]
    
    with session_scope() as session:
        # Create document processor factory
        factory = DocumentProcessorFactory()
        
        # Create document processor
        processor = factory.create_processor(session)
        
        for doc_path, doc_type, party in documents_to_process:
            if doc_path.exists():
                logger.info(f"Processing {doc_path}")
                
                # Extract document dates
                creation_date, received_date = extract_document_date(doc_path)
                
                # Convert string types to enums
                doc_type_enum = getattr(DocumentType, doc_type.upper()) if hasattr(DocumentType, doc_type.upper()) else doc_type
                party_type_enum = getattr(PartyType, party.upper()) if hasattr(PartyType, party.upper()) else party
                
                # Process the document
                result = processor.process_document(
                    file_path=doc_path,
                    doc_type=doc_type_enum,
                    party=party_type_enum,
                    date_created=creation_date,
                    date_received=received_date,
                    save_to_db=True
                )
                
                # Display results
                if result.success:
                    logger.info(f"Successfully processed document: {doc_path}")
                    logger.info(f"Document ID: {result.document_id}")
                    logger.info(f"Metadata: {result.metadata}")
                    logger.info(f"Extracted {len(result.line_items)} line items")
                    
                    # Print first 3 line items as example
                    for i, item in enumerate(result.line_items[:3]):
                        logger.info(f"Line item {i+1}: {item.get('description', 'No description')} - ${item.get('amount')}")
                else:
                    logger.error(f"Failed to process document: {result.error}")
            else:
                logger.warning(f"Document does not exist: {doc_path}")
    
    logger.info("Document processing example completed")


def main():
    """Main function to demonstrate document processor usage."""
    setup_logging()
    
    # If command-line arguments are provided, process a specific document
    if len(sys.argv) > 1:
        process_from_arguments()
    else:
        # Otherwise, run the demonstration with example documents
        demonstrate_extractors()


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    main()