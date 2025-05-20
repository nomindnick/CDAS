"""
Example usage of the dependency injection system.

This module demonstrates how to use the dependency injection container
and registry for managing component creation and dependencies.
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from cdas.utils.di import container, registry
from cdas.financial_analysis.base import BaseAnalyzer, PatternAnalyzer
from cdas.document_processor.interfaces import DocumentProcessor, Extractor
from cdas.reporting.interfaces import ReportGenerator

# Example: Configuring the dependency injection container

def configure_container(db_session: Session, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure the dependency injection container with components.
    
    Args:
        db_session: Database session
        config: Optional configuration dictionary
    """
    # Register session and config
    container.register_instance('db_session', db_session)
    container.register_instance('config', config or {})
    
    # Register financial analysis components
    from cdas.financial_analysis.engine import (
        PatternDetector, AnomalyDetector, AmountMatcher,
        ChronologyAnalyzer, RelationshipAnalyzer, FinancialAnalysisEngine
    )
    
    container.register('pattern_detector', 
                      lambda db_session, config: PatternDetector(db_session, config))
    
    container.register('anomaly_detector', 
                      lambda db_session, config: AnomalyDetector(db_session, config))
    
    container.register('amount_matcher', 
                      lambda db_session, config: AmountMatcher(db_session, config))
    
    container.register('chronology_analyzer', 
                      lambda db_session, config: ChronologyAnalyzer(db_session, config))
    
    container.register('relationship_analyzer', 
                      lambda db_session, config: RelationshipAnalyzer(db_session, config))
    
    container.register('financial_analysis_engine', 
                      lambda db_session, config, pattern_detector, anomaly_detector, 
                      amount_matcher, chronology_analyzer, relationship_analyzer: 
                      FinancialAnalysisEngine(
                          db_session, 
                          config,
                          pattern_detector=pattern_detector,
                          anomaly_detector=anomaly_detector,
                          amount_matcher=amount_matcher,
                          chronology_analyzer=chronology_analyzer,
                          relationship_analyzer=relationship_analyzer
                      ))
    
    # Register document processor components
    from cdas.document_processor.processor import DocumentProcessor
    from cdas.document_processor.factory import DocumentProcessorFactory
    
    container.register('document_processor', 
                      lambda db_session, config: DocumentProcessor(db_session, config))
    
    container.register('document_processor_factory', 
                      lambda config: DocumentProcessorFactory(config))
    
    # Register reporting components
    from cdas.reporting.generator import ReportGenerator
    
    container.register('report_generator', 
                      lambda db_session, config: ReportGenerator(db_session, config))

# Example: Using the dependency injection container

def analyze_financial_patterns(doc_id: str) -> List[Dict[str, Any]]:
    """
    Analyze financial patterns in a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of patterns
    """
    # Get the financial analysis engine from the container
    engine = container.get('financial_analysis_engine')
    
    # Use the engine to analyze patterns
    analysis_result = engine.analyze_document(doc_id)
    
    return analysis_result.get('patterns', [])

def process_document(file_path: str, doc_type: str, party: str) -> str:
    """
    Process a document using the document processor.
    
    Args:
        file_path: Path to document
        doc_type: Document type
        party: Party type
        
    Returns:
        Document ID
    """
    # Get the document processor factory from the container
    factory = container.get('document_processor_factory')
    
    # Get database session from the container
    db_session = container.get('db_session')
    
    # Process the document
    result = factory.process_single_document(db_session, file_path, doc_type, party)
    
    return result.document_id

def generate_report(report_data: Dict[str, Any], output_path: str) -> str:
    """
    Generate a summary report.
    
    Args:
        report_data: Report data
        output_path: Output file path
        
    Returns:
        Path to generated report
    """
    # Get the report generator from the container
    generator = container.get('report_generator')
    
    # Generate the report
    result = generator.generate_summary_report(report_data, output_path=output_path)
    
    return str(result.report_path)

# Example: Configuring the component registry

def configure_registry() -> None:
    """Configure the component registry with extractors and analyzers."""
    # Register document extractors
    from cdas.document_processor.extractors.pdf import PDFExtractor
    from cdas.document_processor.extractors.excel import ExcelExtractor
    from cdas.document_processor.extractors.image import ImageExtractor
    from cdas.document_processor.extractors.text import TextExtractor
    
    registry.register('extractor', 'pdf', PDFExtractor)
    registry.register('extractor', 'excel', ExcelExtractor)
    registry.register('extractor', 'image', ImageExtractor)
    registry.register('extractor', 'text', TextExtractor)
    
    # Register financial analyzers
    from cdas.financial_analysis.patterns.recurring import RecurringPatternDetector
    from cdas.financial_analysis.patterns.sequencing import SequencingPatternDetector
    from cdas.financial_analysis.patterns.similarity import SimilarityPatternDetector
    
    registry.register('analyzer', 'recurring_pattern', RecurringPatternDetector)
    registry.register('analyzer', 'sequencing_pattern', SequencingPatternDetector)
    registry.register('analyzer', 'similarity_pattern', SimilarityPatternDetector)

# Example: Creating components using the registry

def create_extractor(extractor_type: str, **kwargs) -> Extractor:
    """
    Create an extractor from the registry.
    
    Args:
        extractor_type: Type of extractor
        **kwargs: Extractor configuration
        
    Returns:
        Extractor instance
    """
    # Get the extractor factory from the registry
    factory = registry.get_factory('extractor', extractor_type)
    
    if factory is None:
        raise ValueError(f"No extractor registered for type: {extractor_type}")
    
    # Create the extractor
    return factory(**kwargs)

def create_analyzer(analyzer_type: str, db_session: Session, config: Optional[Dict[str, Any]] = None) -> BaseAnalyzer:
    """
    Create an analyzer from the registry.
    
    Args:
        analyzer_type: Type of analyzer
        db_session: Database session
        config: Optional configuration
        
    Returns:
        Analyzer instance
    """
    # Get the analyzer factory from the registry
    factory = registry.get_factory('analyzer', analyzer_type)
    
    if factory is None:
        raise ValueError(f"No analyzer registered for type: {analyzer_type}")
    
    # Create the analyzer
    return factory(db_session, config or {})