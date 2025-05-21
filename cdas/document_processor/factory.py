"""
Document processor factory module for creating and configuring document processors.
"""
import logging
from typing import Dict, Optional, Tuple, Union, List
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
from cdas.db.project_manager import get_project_db_manager, get_session, session_scope

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
        
        # Initialize project database manager for project-specific sessions
        self.project_manager = get_project_db_manager()
        
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
        session: Optional[Session] = None,
        file_path: Union[str, Path] = None,
        doc_type: Union[str, DocumentType] = None,
        party: Union[str, PartyType] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single document with the appropriate extractor.
        
        Args:
            session: SQLAlchemy database session (optional if project_id is provided)
            file_path: Path to the document
            doc_type: Document type
            party: Party associated with the document
            project_id: Project ID (if not using an existing session)
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessingResult object
        """
        logger.info(f"Processing document: {file_path}")
        
        # Add project metadata if specified
        if project_id:
            kwargs['project_id'] = project_id
            # Store project ID in document metadata
            if 'metadata' not in kwargs:
                kwargs['metadata'] = {}
            kwargs['metadata']['project_id'] = project_id
        
        # Use provided session or create a new one for the specified project
        if session is None:
            # Use project-specific session
            with session_scope(project_id) as s:
                processor = self.create_processor(s)
                return processor.process_document(file_path, doc_type, party, **kwargs)
        else:
            # Use provided session
            processor = self.create_processor(session)
            return processor.process_document(file_path, doc_type, party, **kwargs)
        
    def process_directory(
        self,
        session: Optional[Session] = None,
        directory_path: Union[str, Path] = None,
        doc_type: Union[str, DocumentType] = None,
        party: Union[str, PartyType] = None,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, ProcessingResult]:
        """
        Process all documents in a directory with the appropriate extractor.
        
        Args:
            session: SQLAlchemy database session (optional if project_id is provided)
            directory_path: Path to the directory containing documents
            doc_type: Document type for all documents in directory
            party: Party associated with all documents in directory
            recursive: Whether to process subdirectories recursively
            file_extensions: List of file extensions to process (e.g., ['.pdf', '.xlsx'])
                            If None, all supported file types will be processed
            project_id: Project ID (if not using an existing session)
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary mapping file paths to ProcessingResult objects
        """
        logger.info(f"Processing documents in directory: {directory_path}")
        
        # Convert to Path object if string
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
            
        # Ensure directory exists
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found or not a directory: {directory_path}")
            return {}
        
        # Add project metadata if specified
        if project_id:
            kwargs['project_id'] = project_id
            # Store project ID in document metadata
            if 'metadata' not in kwargs:
                kwargs['metadata'] = {}
            kwargs['metadata']['project_id'] = project_id
        
        # Create processor with appropriate session
        if session is None and project_id:
            # Use project context to get session
            session = get_session(project_id)
        elif session is None:
            # Use current project context (if any)
            try:
                session = get_session()
            except ValueError:
                logger.error("No project specified and no current project set. Use project_id parameter or set a project context.")
                return {}
                
        processor = self.create_processor(session)
            
        # Define default file extensions if not provided
        if file_extensions is None:
            file_extensions = ['.pdf', '.xlsx', '.xls', '.csv', '.docx', '.doc', 
                              '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.txt']
        
        # Find all files with supported extensions
        results = {}
        
        # Determine glob pattern based on recursive flag
        glob_pattern = '**/*' if recursive else '*'
        
        # Process all files with supported extensions
        for file_path in directory_path.glob(glob_pattern):
            # Skip directories
            if file_path.is_dir():
                continue
                
            # Check if file has supported extension
            if file_path.suffix.lower() in file_extensions:
                try:
                    # Process document
                    result = processor.process_document(
                        str(file_path), doc_type, party, **kwargs
                    )
                    results[str(file_path)] = result
                    
                    if result.success:
                        logger.info(f"Successfully processed: {file_path}")
                    else:
                        logger.warning(f"Failed to process: {file_path}. Error: {result.error}")
                        
                except Exception as e:
                    logger.exception(f"Error processing {file_path}: {str(e)}")
                    results[str(file_path)] = ProcessingResult(
                        success=False,
                        error=f"Unhandled error: {str(e)}"
                    )
        
        logger.info(f"Completed processing directory. Processed {len(results)} files.")
        return results