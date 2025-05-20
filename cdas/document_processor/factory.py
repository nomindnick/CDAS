"""
Document processor factory module for creating and configuring document processors.
"""
import logging
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from sqlalchemy.orm import Session

from cdas.document_processor.processor import (
    DocumentProcessor, 
    BaseExtractor, 
    DocumentType, 
    DocumentFormat,
    PartyType,
    ProcessingResult
)
from cdas.document_processor.extractors.pdf import PDFExtractor
from cdas.document_processor.extractors.excel import ExcelExtractor
from cdas.document_processor.extractors.image import ImageExtractor
from cdas.document_processor.extractors.text import TextExtractor
from cdas.config import get_config

logger = logging.getLogger(__name__)


class DocumentProcessorFactory:
    """
    Factory for creating and configuring document processors.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the document processor factory.
        
        Args:
            config: Optional configuration dictionary
        """
        # Load configuration
        self.config = config or get_config()
        self.doc_processor_config = self.config.get('document_processor', {})
        
        # Get OCR engine configuration
        self.tesseract_cmd = self.doc_processor_config.get('tesseract_path')
        
        # Initialize extractors dictionary
        self._extractors: Dict[Tuple[DocumentFormat, DocumentType], BaseExtractor] = {}
        
        # Initialize extractors
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize and register all available extractors."""
        logger.info("Initializing document extractors")
        
        # Create extractors
        pdf_extractor = PDFExtractor()
        excel_extractor = ExcelExtractor()
        image_extractor = ImageExtractor(self.tesseract_cmd)
        text_extractor = TextExtractor()
        
        # Register PDF extractors for different document types
        self._register_extractor(DocumentFormat.PDF, DocumentType.PAYMENT_APP, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.CHANGE_ORDER, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.INVOICE, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.CONTRACT, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.SCHEDULE, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.CORRESPONDENCE, pdf_extractor)
        self._register_extractor(DocumentFormat.PDF, DocumentType.OTHER, pdf_extractor)
        
        # Register Excel extractors for different document types
        self._register_extractor(DocumentFormat.EXCEL, DocumentType.PAYMENT_APP, excel_extractor)
        self._register_extractor(DocumentFormat.EXCEL, DocumentType.CHANGE_ORDER, excel_extractor)
        self._register_extractor(DocumentFormat.EXCEL, DocumentType.INVOICE, excel_extractor)
        self._register_extractor(DocumentFormat.EXCEL, DocumentType.SCHEDULE, excel_extractor)
        self._register_extractor(DocumentFormat.EXCEL, DocumentType.OTHER, excel_extractor)
        
        # Register Image extractors for different document types
        self._register_extractor(DocumentFormat.IMAGE, DocumentType.PAYMENT_APP, image_extractor)
        self._register_extractor(DocumentFormat.IMAGE, DocumentType.CHANGE_ORDER, image_extractor)
        self._register_extractor(DocumentFormat.IMAGE, DocumentType.INVOICE, image_extractor)
        self._register_extractor(DocumentFormat.IMAGE, DocumentType.OTHER, image_extractor)
        
        # Register Text extractors for OTHER format documents and text files
        self._register_extractor(DocumentFormat.OTHER, DocumentType.PAYMENT_APP, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.CHANGE_ORDER, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.INVOICE, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.CONTRACT, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.SCHEDULE, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.CORRESPONDENCE, text_extractor)
        self._register_extractor(DocumentFormat.OTHER, DocumentType.OTHER, text_extractor)
        
        # Log registered extractors for debugging
        logger.info("Text extractor registered for all document types with format: OTHER")
        logger.info(f"Total registered extractors: {len(self._extractors)}")
        
        logger.info(f"Registered {len(self._extractors)} document extractors")
    
    def _register_extractor(self, doc_format: DocumentFormat, doc_type: DocumentType, extractor: BaseExtractor):
        """
        Register an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format
            doc_type: Document type
            extractor: Extractor instance
        """
        key = (doc_format, doc_type)
        self._extractors[key] = extractor
        logger.debug(f"Registered extractor for {doc_format.value}/{doc_type.value}")
    
    def create_processor(self, session: Session) -> DocumentProcessor:
        """
        Create and configure a document processor.
        
        Args:
            session: SQLAlchemy database session
            
        Returns:
            Configured DocumentProcessor instance
        """
        logger.info("Creating document processor")
        
        # Create processor
        processor = DocumentProcessor(session)
        
        # Register all extractors with the processor
        for (doc_format, doc_type), extractor in self._extractors.items():
            processor.register_extractor(doc_format, doc_type, extractor)
        
        logger.info("Document processor created and configured")
        return processor
    
    def create_specialized_processor(
        self, 
        session: Session, 
        doc_format: Union[str, DocumentFormat], 
        doc_type: Union[str, DocumentType]
    ) -> DocumentProcessor:
        """
        Create a processor specialized for a specific document format and type.
        
        Args:
            session: SQLAlchemy database session
            doc_format: Document format
            doc_type: Document type
            
        Returns:
            Specialized DocumentProcessor instance
        """
        logger.info(f"Creating specialized processor for {doc_format}/{doc_type}")
        
        # Convert string inputs to enum types if needed
        if isinstance(doc_format, str):
            doc_format = DocumentFormat(doc_format)
        
        if isinstance(doc_type, str):
            doc_type = DocumentType(doc_type)
        
        # Create processor
        processor = DocumentProcessor(session)
        
        # Get the appropriate extractor
        key = (doc_format, doc_type)
        extractor = self._extractors.get(key)
        
        if extractor:
            # Register the extractor
            processor.register_extractor(doc_format, doc_type, extractor)
            logger.info(f"Specialized processor created for {doc_format.value}/{doc_type.value}")
            return processor
        else:
            logger.warning(f"No extractor found for {doc_format.value}/{doc_type.value}")
            return self.create_processor(session)
    
    def process_single_document(
        self,
        session: Session,
        file_path: Union[str, Path],
        doc_type: Union[str, DocumentType],
        party: Union[str, PartyType],
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single document with the appropriate extractor.
        
        Args:
            session: SQLAlchemy database session
            file_path: Path to the document
            doc_type: Document type
            party: Party associated with the document
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessingResult object
        """
        logger.info(f"Processing document: {file_path}")
        
        # Create processor
        processor = self.create_processor(session)
        
        # Process document
        result = processor.process_document(file_path, doc_type, party, **kwargs)
        
        return result