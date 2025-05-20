#!/usr/bin/env python3
"""
Process a single synthetic test document for CDAS.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from cdas.db.session import session_scope
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType, DocumentFormat


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("synthetic_test")


def clean_metadata(metadata):
    """Clean metadata to ensure date fields are proper date objects."""
    if isinstance(metadata, dict):
        # Convert any date strings to date objects
        date_fields = ['date_created', 'date_received', 'document_date', 'creation_date', 
                     'submission_date', 'effective_date', 'period_start', 'period_end']
        
        for field in date_fields:
            if field in metadata and isinstance(metadata[field], str):
                try:
                    # If it's a date string like "March 15, 2023", parse it
                    from datetime import datetime
                    formats = ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']
                    
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(metadata[field], fmt)
                            metadata[field] = dt.date()
                            break
                        except ValueError:
                            continue
                    
                    # If still a string, remove it
                    if isinstance(metadata[field], str):
                        metadata[field] = None
                except Exception:
                    # If any parsing error, set to None
                    metadata[field] = None
        
        # Recursively clean nested dictionaries
        for key, value in metadata.items():
            if isinstance(value, dict):
                metadata[key] = clean_metadata(value)
    
    return metadata


def process_document(file_path, doc_type, party, logger):
    """Process a single document with proper date handling."""
    try:
        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
            
        # Create default dates
        creation_date = date(2023, 3, 15)  # Default to March 15, 2023
        received_date = date(2023, 3, 20)  # Default to March 20, 2023
        
        # Create session
        with session_scope() as session:
            factory = DocumentProcessorFactory()
            processor = factory.create_processor(session)
            
            # Convert doc_type and party to enums if needed
            if isinstance(doc_type, str):
                try:
                    doc_type = DocumentType(doc_type)
                except ValueError:
                    logger.warning(f"Invalid document type: {doc_type}, using as string")
                    
            if isinstance(party, str):
                try:
                    party = PartyType(party)
                except ValueError:
                    logger.warning(f"Invalid party: {party}, using as string")
            
            # Process the document
            logger.info(f"Processing document: {file_path}")
            logger.info(f"Document type: {doc_type}")
            logger.info(f"Party: {party}")
            
            # First extract metadata without saving to clean it
            extractor = processor._extractors.get((DocumentFormat.OTHER, doc_type), None)
            if extractor:
                try:
                    metadata = extractor.extract_metadata(file_path)
                    # Clean the metadata to ensure date fields are proper date objects
                    metadata = clean_metadata(metadata)
                except Exception as e:
                    logger.warning(f"Error extracting metadata: {e}")
                    metadata = {}
            else:
                metadata = {}
            
            # Now process with cleaned metadata
            kwargs = {
                'date_created': creation_date,
                'date_received': received_date,
                'metadata': metadata,
                'save_to_db': True
            }
            
            result = processor.process_document(
                file_path=file_path,
                doc_type=doc_type,
                party=party,
                **kwargs
            )
            
            if result.success:
                logger.info(f"Successfully processed document: {file_path}")
                logger.info(f"Document ID: {result.document_id}")
                return True
            else:
                logger.error(f"Error processing document: {result.error}")
                return False
                
    except Exception as e:
        logger.exception(f"Exception processing document: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process a single synthetic test document")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--doc-type", default="other", help="Document type (payment_app, change_order, etc.)")
    parser.add_argument("--party", default="contractor", help="Party (contractor, owner, etc.)")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    success = process_document(args.file_path, args.doc_type, args.party, logger)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())