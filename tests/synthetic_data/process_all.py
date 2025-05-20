#!/usr/bin/env python3
"""
Process all synthetic test documents for CDAS in a specific order.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from cdas.db.reset import reset_database


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("process_all")


def reset_db(args, logger):
    """Reset the database if requested."""
    if args.reset:
        try:
            logger.info("Resetting database...")
            reset_database()
            logger.info("Database reset successful")
        except Exception as e:
            logger.error(f"Failed to reset database: {str(e)}")
            return False
    return True


def process_documents(data_dir, logger):
    """Process all documents using the single document processor script."""
    # Get the script path
    script_path = Path(__file__).parent / "process_single.py"
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Define the document types to process in order
    document_types = [
        # Process contracts first
        {
            "dir": "contracts",
            "type": "contract",
            "party": "owner"
        },
        # Process change orders
        {
            "dir": "change_orders",
            "type": "change_order",
            "party": "contractor"
        },
        # Process payment applications
        {
            "dir": "payment_apps",
            "type": "payment_app",
            "party": "contractor"
        },
        # Process correspondence
        {
            "dir": "correspondence",
            "type": "correspondence",
            "party": "other"
        },
        # Process invoices
        {
            "dir": "invoices",
            "type": "invoice",
            "party": "subcontractor"
        }
    ]
    
    # Process each document type
    successful_docs = 0
    failed_docs = 0
    
    for doc_type in document_types:
        dir_path = data_dir / doc_type["dir"]
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
        
        logger.info(f"Processing documents in {dir_path}")
        
        # Get all .txt files in the directory
        files = list(dir_path.glob("*.txt"))
        
        if not files:
            logger.warning(f"No .txt files found in {dir_path}")
            continue
        
        # Process each file
        for file_path in files:
            logger.info(f"Processing {file_path}")
            
            # Build the command
            cmd = [
                sys.executable,
                str(script_path),
                str(file_path),
                "--doc-type", doc_type["type"],
                "--party", doc_type["party"]
            ]
            
            # Run the command
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Successfully processed {file_path}")
                    successful_docs += 1
                else:
                    logger.error(f"Failed to process {file_path}")
                    logger.error(f"Error: {result.stderr}")
                    failed_docs += 1
                    
            except Exception as e:
                logger.error(f"Exception processing {file_path}: {str(e)}")
                failed_docs += 1
    
    logger.info(f"Processing complete: {successful_docs} successful, {failed_docs} failed")
    return failed_docs == 0


def main():
    parser = argparse.ArgumentParser(description="Process all synthetic test documents")
    parser.add_argument("--reset", action="store_true", help="Reset the database before processing")
    parser.add_argument("--data-dir", default=None, help="Directory containing the test data")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent
    
    # Reset database if requested
    if not reset_db(args, logger):
        return 1
    
    # Process all documents
    success = process_documents(data_dir, logger)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())