"""
Base document processor module for extracting structured data from construction documents.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import uuid
import os

from sqlalchemy.orm import Session

from cdas.db.operations import (
    register_document,
    register_page,
    store_line_items,
    store_annotations,
    register_change_order,
    register_payment_application
)

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    PAYMENT_APP = "payment_app"
    CHANGE_ORDER = "change_order"
    INVOICE = "invoice"
    CONTRACT = "contract"
    SCHEDULE = "schedule"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


class PartyType(Enum):
    CONTRACTOR = "contractor"
    SUBCONTRACTOR = "subcontractor"
    OWNER = "owner"
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    OTHER = "other"


class DocumentFormat(Enum):
    PDF = "pdf"
    EXCEL = "excel"
    WORD = "word"
    IMAGE = "image"
    OTHER = "other"


class ProcessingResult:
    """Container for document processing results."""
    
    def __init__(
        self,
        success: bool,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
        line_items: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.document_id = document_id
        self.metadata = metadata or {}
        self.extracted_data = extracted_data or {}
        self.line_items = line_items or []
        self.error = error


class BaseExtractor(ABC):
    """Base class for document extractors."""
    
    @abstractmethod
    def extract_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract data from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing extracted data
        """
        pass
    
    @abstractmethod
    def extract_line_items(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Extract financial line items from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            List of dictionaries, each representing a line item
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing document metadata
        """
        pass


class DocumentProcessor:
    """
    Main document processor that extracts structured data from construction documents.
    """
    
    def __init__(self, session: Session):
        """
        Initialize the document processor.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self._extractors = {}
    
    def register_extractor(self, doc_format: DocumentFormat, doc_type: DocumentType, extractor: BaseExtractor):
        """
        Register an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format (PDF, Excel, etc.)
            doc_type: Document type (Payment App, Change Order, etc.)
            extractor: Extractor instance
        """
        key = (doc_format, doc_type)
        self._extractors[key] = extractor
    
    def get_extractor(self, doc_format: DocumentFormat, doc_type: DocumentType) -> Optional[BaseExtractor]:
        """
        Get an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format (PDF, Excel, etc.)
            doc_type: Document type (Payment App, Change Order, etc.)
            
        Returns:
            Extractor instance if registered, None otherwise
        """
        key = (doc_format, doc_type)
        return self._extractors.get(key)
    
    def process_document(
        self,
        file_path: Union[str, Path],
        doc_type: Union[str, DocumentType],
        party: Union[str, PartyType],
        **kwargs
    ) -> ProcessingResult:
        """
        Process a document and extract structured data.
        
        Args:
            file_path: Path to the document
            doc_type: Type of document (payment_app, change_order, etc.)
            party: Party associated with the document (contractor, owner, etc.)
            **kwargs: Additional processing arguments
                - project_id: Optional project identifier
                - save_to_db: Whether to save results to database (default: True)
                - extract_handwriting: Whether to extract handwritten text (default: True)
                - extract_tables: Whether to extract tables (default: True)
            
        Returns:
            ProcessingResult object containing extracted data and metadata
        """
        try:
            # Convert string inputs to enum types if needed
            if isinstance(doc_type, str):
                doc_type = DocumentType(doc_type)
            
            if isinstance(party, str):
                party = PartyType(party)
                
            # Convert file_path to Path object
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # Process optional arguments
            project_id = kwargs.get('project_id')
            save_to_db = kwargs.get('save_to_db', True)
            
            # Determine document format based on file extension
            doc_format = self._get_document_format(file_path)
            
            # Get appropriate extractor
            extractor = self.get_extractor(doc_format, doc_type)
            if not extractor:
                logger.warning(f"No extractor found for {doc_format.value} format and {doc_type.value} type, falling back to TextExtractor")
                
                # Import here to avoid circular imports
                from cdas.document_processor.extractors.text import TextExtractor
                extractor = TextExtractor()
                
                # If still no extractor, return failure
                if not extractor:
                    return ProcessingResult(
                        success=False,
                        error=f"No extractor found for {doc_format.value} format and {doc_type.value} type"
                    )
            
            # Extract data, line items, and metadata
            metadata = extractor.extract_metadata(file_path, **kwargs)
            extracted_data = extractor.extract_data(file_path, **kwargs)
            line_items = extractor.extract_line_items(file_path, **kwargs)
            
            # Add party information to metadata
            metadata["party"] = party.value
            metadata["document_type"] = doc_type.value
            metadata["document_format"] = doc_format.value
            
            # Add project ID if provided
            if project_id:
                metadata["project_id"] = project_id
            
            # Save to database if requested
            document_id = None
            if save_to_db:
                document_id = self._save_to_database(
                    file_path, doc_type, party, metadata, extracted_data, line_items, project_id
                )
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                metadata=metadata,
                extracted_data=extracted_data,
                line_items=line_items
            )
            
        except Exception as e:
            logger.exception(f"Error processing document: {str(e)}")
            return ProcessingResult(
                success=False,
                error=f"Error processing document: {str(e)}"
            )
    
    def _get_document_format(self, file_path: Path) -> DocumentFormat:
        """
        Determine document format based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            DocumentFormat enum value
        """
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            return DocumentFormat.PDF
        elif extension in [".xlsx", ".xls", ".csv"]:
            return DocumentFormat.EXCEL
        elif extension in [".docx", ".doc"]:
            return DocumentFormat.WORD
        elif extension in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
            return DocumentFormat.IMAGE
        elif extension in [".txt", ".text", ""]:
            # Explicitly handle text files, including files with no extension
            return DocumentFormat.OTHER
        else:
            # Try to determine if it's a text file by reading the first few bytes
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.read(1024)  # Try to read as text
                # If we get here, it's probably a text file
                return DocumentFormat.OTHER
            except Exception:
                # If reading as text fails, default to OTHER
                return DocumentFormat.OTHER
            
    def _save_to_database(
        self,
        file_path: Path,
        doc_type: DocumentType,
        party: PartyType,
        metadata: Dict[str, Any],
        extracted_data: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        project_id: Optional[str] = None
    ) -> str:
        """
        Save the extracted document data to the database.
        
        Args:
            file_path: Path to the document file
            doc_type: Document type
            party: Party associated with the document
            metadata: Document metadata
            extracted_data: Extracted document data
            line_items: Extracted line items
            project_id: Optional project identifier
            
        Returns:
            Document ID
        """
        try:
            logger.info(f"Saving document to database: {file_path}")
            
            # Register the document
            metadata_with_project = dict(metadata)
            if project_id:
                metadata_with_project['project_id'] = project_id
                
            # Convert date strings to date objects if needed
            from datetime import datetime
            
            date_created = metadata.get('creation_date')
            date_received = metadata.get('document_date')
            
            # Convert string dates to datetime objects if they're strings
            if isinstance(date_created, str):
                try:
                    date_created = datetime.strptime(date_created, '%B %d, %Y').date()
                except (ValueError, TypeError):
                    date_created = None
            
            if isinstance(date_received, str):
                try:
                    date_received = datetime.strptime(date_received, '%B %d, %Y').date()
                except (ValueError, TypeError):
                    date_received = None
            
            # Register the document with proper date objects
            document = register_document(
                self.session,
                str(file_path.absolute()),
                doc_type.value,
                party.value,
                date_created=date_created,
                date_received=date_received,
                metadata=metadata_with_project
            )
            
            # Get the document ID
            doc_id = document.doc_id
            
            # Register pages
            if 'pages' in extracted_data:
                for page_data in extracted_data['pages']:
                    page_number = page_data.get('page_number', 0)
                    content = page_data.get('text', '')
                    has_tables = len(page_data.get('tables', [])) > 0
                    has_handwriting = 'handwritten_regions' in page_data and len(page_data['handwritten_regions']) > 0
                    has_financial_data = any(
                        'amount' in table and table['amount'] is not None
                        for table in page_data.get('tables', [])
                    )
                    
                    page = register_page(
                        self.session,
                        doc_id,
                        page_number,
                        content,
                        has_tables,
                        has_handwriting,
                        has_financial_data
                    )
                    
                    # Store page-specific annotations if available
                    if 'annotations' in page_data:
                        store_annotations(self.session, doc_id, page_data['annotations'])
            
            # Store line items
            if line_items:
                stored_items = store_line_items(self.session, doc_id, line_items)
            
            # Register document-specific data based on type
            if doc_type == DocumentType.CHANGE_ORDER:
                register_change_order(
                    self.session,
                    doc_id,
                    change_order_number=metadata.get('change_order_number'),
                    description=metadata.get('description'),
                    amount=metadata.get('amount'),
                    status=metadata.get('status'),
                    date_submitted=metadata.get('submission_date'),
                    date_responded=metadata.get('response_date'),
                    reason_code=metadata.get('reason_code'),
                    metadata=metadata
                )
            
            elif doc_type == DocumentType.PAYMENT_APP:
                register_payment_application(
                    self.session,
                    doc_id,
                    payment_app_number=metadata.get('payment_app_number'),
                    period_start=metadata.get('period_start'),
                    period_end=metadata.get('period_end'),
                    amount_requested=metadata.get('amount_requested'),
                    amount_approved=metadata.get('amount_approved'),
                    status=metadata.get('status'),
                    metadata=metadata
                )
            
            # Commit the session
            self.session.commit()
            
            logger.info(f"Document saved successfully with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.exception(f"Error saving document to database: {str(e)}")
            self.session.rollback()
            raise