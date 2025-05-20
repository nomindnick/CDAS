"""
Interface definitions for the reporting system.

This module defines the core interfaces and protocols used by the
reporting components of the Construction Document Analysis System.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Protocol, runtime_checkable, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

# Type variable for generic result object
T = TypeVar('T')

# Enums for report types and formats
class ReportType(Enum):
    """Enumeration of report types."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    EVIDENCE = "evidence"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Enumeration of report output formats."""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    EXCEL = "excel"
    JSON = "json"

# Result objects
@dataclass
class ReportResult(Generic[T]):
    """Container for report generation results."""
    
    success: bool
    report_path: Optional[Union[str, Path]] = None
    report_data: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @staticmethod
    def success(report_path: Union[str, Path], report_data: Optional[T] = None, 
               metadata: Optional[Dict[str, Any]] = None) -> 'ReportResult[T]':
        """Create a successful report result."""
        return ReportResult(
            success=True,
            report_path=report_path,
            report_data=report_data,
            metadata=metadata or {}
        )
    
    @staticmethod
    def failure(error: str) -> 'ReportResult[T]':
        """Create a failed report result."""
        return ReportResult(success=False, error=error)

# Interface definitions
@runtime_checkable
class ReportGenerator(Protocol):
    """Protocol for report generators."""
    
    def generate_report(
        self,
        report_type: Union[str, ReportType],
        data: Dict[str, Any],
        format: Union[str, ReportFormat] = ReportFormat.PDF,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ReportResult:
        """
        Generate a report.
        
        Args:
            report_type: Type of report to generate
            data: Report data
            format: Output format
            output_path: Path to save the report
            **kwargs: Additional report-specific parameters
            
        Returns:
            ReportResult object containing the report path and metadata
        """
        ...
    
    def generate_summary_report(
        self,
        data: Dict[str, Any],
        format: Union[str, ReportFormat] = ReportFormat.PDF,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ReportResult:
        """
        Generate a summary report.
        
        Args:
            data: Report data
            format: Output format
            output_path: Path to save the report
            **kwargs: Additional report-specific parameters
            
        Returns:
            ReportResult object
        """
        ...
    
    def generate_detailed_report(
        self,
        data: Dict[str, Any],
        format: Union[str, ReportFormat] = ReportFormat.PDF,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ReportResult:
        """
        Generate a detailed report.
        
        Args:
            data: Report data
            format: Output format
            output_path: Path to save the report
            **kwargs: Additional report-specific parameters
            
        Returns:
            ReportResult object
        """
        ...
    
    def generate_evidence_report(
        self,
        data: Dict[str, Any],
        format: Union[str, ReportFormat] = ReportFormat.PDF,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ReportResult:
        """
        Generate an evidence report.
        
        Args:
            data: Report data
            format: Output format
            output_path: Path to save the report
            **kwargs: Additional report-specific parameters
            
        Returns:
            ReportResult object
        """
        ...

@runtime_checkable
class ReportRenderer(Protocol):
    """Protocol for report renderers."""
    
    def render(
        self,
        template_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Render a report template with data.
        
        Args:
            template_name: Template name or path
            data: Template data
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered report content
        """
        ...

@runtime_checkable
class ReportExporter(Protocol):
    """Protocol for report exporters."""
    
    def export(
        self,
        content: str,
        output_path: Union[str, Path],
        format: Union[str, ReportFormat],
        **kwargs
    ) -> Path:
        """
        Export report content to a file.
        
        Args:
            content: Report content
            output_path: Path to save the report
            format: Output format
            **kwargs: Additional export parameters
            
        Returns:
            Path to the exported report file
        """
        ...

class BaseReportGenerator(ABC):
    """Base class for report generators."""
    
    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the report generator.
        
        Args:
            session: Database session
            config: Optional configuration dictionary
        """
        self.session = session
        self.config = config or {}
        self.renderers: Dict[str, ReportRenderer] = {}
        self.exporters: Dict[str, ReportExporter] = {}
    
    def register_renderer(self, name: str, renderer: ReportRenderer) -> None:
        """Register a renderer."""
        self.renderers[name] = renderer
    
    def register_exporter(self, format: Union[str, ReportFormat], exporter: ReportExporter) -> None:
        """Register an exporter for a format."""
        format_value = format.value if isinstance(format, ReportFormat) else format
        self.exporters[format_value] = exporter
    
    def get_renderer(self, name: str) -> Optional[ReportRenderer]:
        """Get a registered renderer by name."""
        return self.renderers.get(name)
    
    def get_exporter(self, format: Union[str, ReportFormat]) -> Optional[ReportExporter]:
        """Get a registered exporter for a format."""
        format_value = format.value if isinstance(format, ReportFormat) else format
        return self.exporters.get(format_value)
    
    @abstractmethod
    def generate_report(
        self,
        report_type: Union[str, ReportType],
        data: Dict[str, Any],
        format: Union[str, ReportFormat] = ReportFormat.PDF,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> ReportResult:
        """Generate a report."""
        pass

# Common configuration type
@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    templates_dir: Optional[Union[str, Path]] = None
    default_format: str = "pdf"
    include_timestamp: bool = True
    include_page_numbers: bool = True
    company_logo: Optional[Union[str, Path]] = None
    company_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'templates_dir': str(self.templates_dir) if self.templates_dir else None,
            'default_format': self.default_format,
            'include_timestamp': self.include_timestamp,
            'include_page_numbers': self.include_page_numbers,
            'company_logo': str(self.company_logo) if self.company_logo else None,
            'company_name': self.company_name
        }
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ReportConfig':
        """Create from dictionary."""
        return ReportConfig(
            templates_dir=config.get('templates_dir'),
            default_format=config.get('default_format', "pdf"),
            include_timestamp=config.get('include_timestamp', True),
            include_page_numbers=config.get('include_page_numbers', True),
            company_logo=config.get('company_logo'),
            company_name=config.get('company_name')
        )