"""
Interface definitions for the document processor system.

This module defines the core interfaces and protocols used by the document
processor components, ensuring consistent patterns across different extractors
and processors.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Protocol, runtime_checkable
from pathlib import Path
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

# Enums for document types and formats
class DocumentType(Enum):
    """Enumeration of document types."""
    PAYMENT_APP = "payment_app"
    CHANGE_ORDER = "change_order"
    INVOICE = "invoice"
    CONTRACT = "contract"
    SCHEDULE = "schedule"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"

class PartyType(Enum):
    """Enumeration of party types."""
    CONTRACTOR = "contractor"
    SUBCONTRACTOR = "subcontractor"
    OWNER = "owner"
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    OTHER = "other"

class DocumentFormat(Enum):
    """Enumeration of document formats."""
    PDF = "pdf"
    EXCEL = "excel"
    WORD = "word"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"

# Result objects
@dataclass
class ExtractorResult:
    """Container for extractor operation results."""
    
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @staticmethod
    def success(data: Dict[str, Any]) -> 'ExtractorResult':
        """Create a successful extractor result."""
        return ExtractorResult(success=True, data=data)
    
    @staticmethod
    def failure(error: str) -> 'ExtractorResult':
        """Create a failed extractor result."""
        return ExtractorResult(success=False, error=error)

@dataclass
class ProcessingResult:
    """Container for document processing results."""
    
    success: bool
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @staticmethod
    def success(document_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None,
               extracted_data: Optional[Dict[str, Any]] = None,
               line_items: Optional[List[Dict[str, Any]]] = None) -> 'ProcessingResult':
        """Create a successful processing result."""
        return ProcessingResult(
            success=True,
            document_id=document_id,
            metadata=metadata or {},
            extracted_data=extracted_data or {},
            line_items=line_items or []
        )
    
    @staticmethod
    def failure(error: str) -> 'ProcessingResult':
        """Create a failed processing result."""
        return ProcessingResult(success=False, error=error)

# Interface definitions
@runtime_checkable
class Extractor(Protocol):
    """Protocol for document extractors."""
    
    def extract_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract data from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing extracted data
        """
        ...
    
    def extract_line_items(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Extract financial line items from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            List of dictionaries, each representing a line item
        """
        ...
    
    def extract_metadata(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing document metadata
        """
        ...

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

@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol for document processors."""
    
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
            doc_type: Type of document
            party: Party associated with the document
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessingResult object containing extracted data and metadata
        """
        ...
    
    def register_extractor(
        self, 
        doc_format: DocumentFormat, 
        doc_type: DocumentType, 
        extractor: Extractor
    ) -> None:
        """
        Register an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format
            doc_type: Document type
            extractor: Extractor instance
        """
        ...
    
    def get_extractor(
        self, 
        doc_format: DocumentFormat, 
        doc_type: DocumentType
    ) -> Optional[Extractor]:
        """
        Get an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format
            doc_type: Document type
            
        Returns:
            Extractor instance if registered, None otherwise
        """
        ...

@runtime_checkable
class ExtractorFactory(Protocol):
    """Protocol for extractor factories."""
    
    def create_extractor(
        self, 
        doc_format: DocumentFormat, 
        doc_type: DocumentType, 
        **kwargs
    ) -> Optional[Extractor]:
        """
        Create an extractor for a specific document format and type.
        
        Args:
            doc_format: Document format
            doc_type: Document type
            **kwargs: Additional configuration
            
        Returns:
            Extractor instance if available, None otherwise
        """
        ...

@runtime_checkable
class ProcessorFactory(Protocol):
    """Protocol for document processor factories."""
    
    def create_processor(
        self, 
        session: Session, 
        **kwargs
    ) -> DocumentProcessor:
        """
        Create a document processor.
        
        Args:
            session: Database session
            **kwargs: Additional configuration
            
        Returns:
            Document processor instance
        """
        ...
    
    def process_single_document(
        self,
        session: Session,
        file_path: Union[str, Path],
        doc_type: Union[str, DocumentType],
        party: Union[str, PartyType],
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single document in one call.
        
        Args:
            session: Database session
            file_path: Path to the document
            doc_type: Type of document
            party: Party associated with the document
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessingResult object
        """
        ...

# Common configuration type
@dataclass
class ExtractorConfig:
    """Configuration for document extractors."""
    
    extract_handwriting: bool = True
    extract_tables: bool = True
    extract_images: bool = False
    confidence_threshold: float = 0.7
    max_pages: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'extract_handwriting': self.extract_handwriting,
            'extract_tables': self.extract_tables,
            'extract_images': self.extract_images,
            'confidence_threshold': self.confidence_threshold,
            'max_pages': self.max_pages
        }
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ExtractorConfig':
        """Create from dictionary."""
        return ExtractorConfig(
            extract_handwriting=config.get('extract_handwriting', True),
            extract_tables=config.get('extract_tables', True),
            extract_images=config.get('extract_images', False),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            max_pages=config.get('max_pages')
        )