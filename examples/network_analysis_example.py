#!/usr/bin/env python3
"""
Example script demonstrating how to use the NetworkAnalyzer.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cdas.db.session import get_session, session_scope
from cdas.analysis.network import NetworkAnalyzer
from cdas.financial_analysis.engine import FinancialAnalysisEngine

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_financial_network():
    """Run the network analysis example."""
    
    # Create a database session
    with session_scope() as session:
        # Initialize the network analyzer
        logger.info("Initializing network analyzer...")
        analyzer = NetworkAnalyzer(session)
        
        # Build the financial network
        logger.info("Building financial network...")
        graph = analyzer.build_financial_network()
        logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Find suspicious patterns
        logger.info("Finding suspicious patterns...")
        patterns = analyzer.find_suspicious_patterns()
        if patterns:
            logger.info(f"Found {len(patterns)} suspicious patterns:")
            for i, pattern in enumerate(patterns, 1):
                logger.info(f"  Pattern {i}: {pattern['type']} - {pattern['description']} (confidence: {pattern['confidence']:.2f})")
        else:
            logger.info("No suspicious patterns found.")
        
        # Find circular references
        logger.info("Finding circular references...")
        cycles = analyzer.find_circular_references()
        if cycles:
            logger.info(f"Found {len(cycles)} circular references:")
            for i, cycle in enumerate(cycles, 1):
                logger.info(f"  Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")
        else:
            logger.info("No circular references found.")
        
        # Find isolated documents
        logger.info("Finding isolated documents...")
        isolated = analyzer.find_isolated_documents()
        if isolated:
            logger.info(f"Found {len(isolated)} isolated documents:")
            for i, doc in enumerate(isolated, 1):
                logger.info(f"  {i}. {doc}")
        else:
            logger.info("No isolated documents found.")
        
        # Visualize the network
        logger.info("Visualizing the network...")
        output_path = analyzer.visualize_network(output_path="financial_network.png")
        logger.info(f"Network visualization saved to {output_path}")
        
        # If there are suspicious patterns, create a visualization highlighting them
        if patterns:
            logger.info("Creating visualization with suspicious patterns highlighted...")
            
            highlight_nodes = []
            highlight_edges = []
            
            for pattern in patterns:
                if pattern["type"] == "circular_reference" and "nodes" in pattern:
                    # Add nodes in circular reference to highlight list
                    highlight_nodes.extend(pattern["nodes"])
                    
                    # Add edges in the cycle
                    for i in range(len(pattern["nodes"]) - 1):
                        highlight_edges.append((pattern["nodes"][i], pattern["nodes"][i + 1]))
                    # Add the closing edge
                    highlight_edges.append((pattern["nodes"][-1], pattern["nodes"][0]))
            
            output_path = analyzer.visualize_network(
                output_path="suspicious_patterns.png",
                highlight_nodes=highlight_nodes,
                highlight_edges=highlight_edges
            )
            logger.info(f"Suspicious patterns visualization saved to {output_path}")

def analyze_financial_data():
    """Run the financial analysis example."""
    
    with session_scope() as session:
        # Initialize the financial analysis engine
        logger.info("Initializing financial analysis engine...")
        engine = FinancialAnalysisEngine(session)
        
        # Find suspicious patterns
        logger.info("Finding suspicious patterns...")
        patterns = engine.find_suspicious_patterns(min_confidence=0.7)
        
        if patterns:
            logger.info(f"Found {len(patterns)} suspicious patterns:")
            for i, pattern in enumerate(patterns, 1):
                logger.info(f"  Pattern {i}: {pattern.get('type')} - {pattern.get('description')} (confidence: {pattern.get('confidence', 0):.2f})")
                
                # Print additional details based on pattern type
                if pattern.get('type') == 'recurring_amount' and 'amount' in pattern:
                    logger.info(f"    Amount: ${pattern['amount']:.2f}")
                    if 'document_types' in pattern:
                        logger.info(f"    Document types: {', '.join(pattern['document_types'])}")
                        
                elif pattern.get('type') == 'reappearing_amount' and 'rejected_date' in pattern and 'reappeared_date' in pattern:
                    logger.info(f"    Rejected on: {pattern['rejected_date']}")
                    logger.info(f"    Reappeared on: {pattern['reappeared_date']}")
        else:
            logger.info("No suspicious patterns found.")
        
        # Generate a financial report
        logger.info("Generating financial report...")
        report = engine.generate_financial_report()
        
        # Print report summary
        if report:
            logger.info("Financial Report Summary:")
            
            if 'summary' in report:
                summary = report['summary']
                logger.info(f"  Total documents: {summary.get('total_documents', 0)}")
                logger.info(f"  Total financial items: {summary.get('total_financial_items', 0)}")
                logger.info(f"  Total amount disputed: ${summary.get('total_amount_disputed', 0):.2f}")
            
            if 'disputed_amounts' in report and report['disputed_amounts']:
                logger.info(f"  Disputed amounts: {len(report['disputed_amounts'])}")
            
            if 'suspicious_patterns' in report and report['suspicious_patterns']:
                logger.info(f"  Suspicious patterns: {len(report['suspicious_patterns'])}")
            
            logger.info(f"  Report generated at: {report.get('generated_at')}")
        else:
            logger.info("No report data available.")

def main():
    """Main function to run examples."""
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'finance':
        analyze_financial_data()
    else:
        analyze_financial_network()
    
    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()