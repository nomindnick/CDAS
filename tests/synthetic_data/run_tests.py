#!/usr/bin/env python
"""
Test runner script for CDAS synthetic data testing.

This script provides a more streamlined approach to testing with the synthetic data.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path to import CDAS modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("synthetic_test_runner")


def reset_database():
    """Reset the database to a clean state."""
    logger.info("Resetting the database...")
    try:
        from cdas.db.reset import reset_database as reset_db
        reset_db()
        logger.info("Database reset successful")
        return True
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        return False


def run_import_script():
    """Run the document import script."""
    logger.info("Running document import script...")
    
    import_script = Path(__file__).parent / "import_documents.py"
    csv_file = Path(__file__).parent / "document_import.csv"
    
    if not import_script.exists():
        logger.error(f"Import script not found: {import_script}")
        return False
        
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        return False
    
    try:
        # Use the import_documents function directly for better debugging
        from import_documents import import_documents
        
        # Set debug level to maximum
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('cdas').setLevel(logging.DEBUG)
        
        # Import the documents
        import_documents(str(csv_file), reset=False)  # We've already reset the database
        
        # Check how many documents were imported
        from cdas.db.session import session_scope
        from cdas.db.models import Document
        
        with session_scope() as session:
            doc_count = session.query(Document).count()
            logger.info(f"Documents imported: {doc_count}")
            
            if doc_count == 0:
                logger.error("No documents were imported!")
                return False
            elif doc_count < 16:  # We have 16 documents in the CSV
                logger.warning(f"Only {doc_count} out of 16 documents were imported")
            
        logger.info("Import script completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to run import script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_tests():
    """Run the financial analysis and network analysis tests."""
    logger.info("Running financial analysis tests...")
    
    try:
        from cdas.db.session import session_scope
        from cdas.financial_analysis.engine import FinancialAnalysisEngine
        from cdas.analysis.network import NetworkAnalyzer
        
        with session_scope() as session:
            # Check if any documents were imported
            from cdas.db.models import Document
            doc_count = session.query(Document).count()
            
            if doc_count == 0:
                logger.error("No documents found in the database. Tests cannot proceed.")
                return False
                
            logger.info(f"Found {doc_count} documents in the database.")
            
            # Run financial analysis
            engine = FinancialAnalysisEngine(session)
            
            logger.info("Running suspicious pattern detection...")
            patterns = engine.find_suspicious_patterns(min_confidence=0.5)
            
            logger.info(f"Found {len(patterns)} suspicious patterns")
            for pattern in patterns:
                # Handle different pattern formats
                pattern_type = pattern.get('type', 'unknown')
                description = pattern.get('description', pattern.get('explanation', 'No description available'))
                confidence = pattern.get('confidence', 0.0)
                
                logger.info(f"Pattern: {pattern_type} - {description}")
                logger.info(f"Confidence: {confidence:.2f}")
            
            # Analyze specific amounts from our known issues
            amounts_to_check = [
                35000.00,  # Duplicate HVAC equipment delivery
                24825.00,  # Rejected change order amount
                4875.00,   # Equipment pad (split change order)
                4850.00,   # Waterproofing (split change order)
                4825.00,   # Structural steel (split change order)
                4975.00    # Seismic bracing (split change order)
            ]
            
            for amount in amounts_to_check:
                logger.info(f"Analyzing amount: ${amount:.2f}")
                analysis = engine.analyze_amount(amount)
                
                if analysis.get('matches', []):
                    logger.info(f"Found {len(analysis['matches'])} matches for ${amount:.2f}")
                    
                    if analysis.get('anomalies', []):
                        logger.info(f"Detected {len(analysis['anomalies'])} anomalies")
                        for anomaly in analysis['anomalies']:
                            logger.info(f"  {anomaly['explanation']}")
            
            # Run network analysis
            analyzer = NetworkAnalyzer(session)
            
            logger.info("Building document network...")
            graph = analyzer.build_document_network()
            logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            logger.info("Finding circular references...")
            cycles = analyzer.find_circular_references()
            for cycle in cycles:
                logger.info(f"Circular reference: {' -> '.join(cycle)} -> {cycle[0]}")
            
            logger.info("Finding isolated documents...")
            isolated = analyzer.find_isolated_documents()
            logger.info(f"Found {len(isolated)} isolated documents")
            
            # Generate visualization
            output_path = analyzer.visualize_network(output_path="synthetic_test_network.png")
            logger.info(f"Network visualization saved to {output_path}")
            
        return True
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run synthetic tests for CDAS")
    parser.add_argument("--skip-reset", action="store_true", help="Skip database reset")
    parser.add_argument("--skip-import", action="store_true", help="Skip document import")
    
    args = parser.parse_args()
    
    # Step 1: Reset database
    if not args.skip_reset and not reset_database():
        logger.error("Database reset failed, aborting.")
        return 1
    
    # Step 2: Import documents
    if not args.skip_import and not run_import_script():
        logger.error("Document import failed, aborting.")
        return 1
    
    # Step 3: Run tests
    if not run_tests():
        logger.error("Tests failed.")
        return 1
    
    logger.info("Synthetic tests completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())