"""
Type definitions for the Construction Document Analysis System.

This module provides common type definitions and aliases used throughout
the codebase to ensure consistent typing and better static analysis.
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Callable, Tuple, Set
from pathlib import Path
from datetime import datetime, date
import uuid

# Basic type aliases
PathLike = Union[str, Path]
JSON = Dict[str, Any]
Number = Union[int, float]
DateType = Union[date, datetime, str]
DocumentID = str
LineItemID = str
PartyID = str
ProjectID = str
UserID = str

# Function types
Processor = Callable[[Any], Any]
Filter = Callable[[Any], bool]
Mapper = Callable[[Any], Any]
ErrorHandler = Callable[[Exception], Any]
ProgressCallback = Callable[[int, int, str], None]  # current, total, message

# Document related types
class DocumentMetadata(TypedDict):
    """Document metadata type."""
    
    doc_id: str
    doc_type: str
    party: str
    project_id: Optional[str]
    date_created: Optional[DateType]
    date_received: Optional[DateType]
    filename: str
    title: Optional[str]
    status: Optional[str]
    description: Optional[str]
    tags: List[str]

class LineItem(TypedDict):
    """Line item type."""
    
    item_id: str
    doc_id: str
    line_number: int
    description: str
    amount: float
    quantity: Optional[float]
    unit_price: Optional[float]
    unit: Optional[str]
    category: Optional[str]
    status: Optional[str]
    metadata: Dict[str, Any]

class Document(TypedDict):
    """Document type."""
    
    doc_id: str
    doc_type: str
    party: str
    project_id: Optional[str]
    filename: str
    date_created: Optional[DateType]
    date_received: Optional[DateType]
    metadata: Dict[str, Any]
    line_items: List[LineItem]

# Analysis related types
class Pattern(TypedDict):
    """Pattern type."""
    
    type: str
    description: str
    items: List[Any]
    confidence: float
    confidence_level: str
    detected_at: str

class Anomaly(TypedDict):
    """Anomaly type."""
    
    type: str
    description: str
    items: List[Any]
    confidence: float
    confidence_level: str
    severity: float
    severity_level: str
    detected_at: str

class Match(TypedDict):
    """Match type."""
    
    source_item: LineItem
    matched_item: LineItem
    similarity: float
    context_similarity: Optional[float]
    match_type: str
    confidence: float

class Relationship(TypedDict):
    """Relationship type."""
    
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    metadata: Dict[str, Any]

class Timeline(TypedDict):
    """Timeline type."""
    
    items: List[Dict[str, Any]]
    start_date: DateType
    end_date: DateType
    milestones: List[Dict[str, Any]]

# Report related types
class ReportMetadata(TypedDict):
    """Report metadata type."""
    
    report_id: str
    report_type: str
    generated_at: DateType
    generated_by: str
    project_id: Optional[str]
    title: str
    description: Optional[str]
    parameters: Dict[str, Any]
    
class ReportSection(TypedDict):
    """Report section type."""
    
    title: str
    content: str
    type: str
    items: List[Any]
    metadata: Dict[str, Any]
    
class Report(TypedDict):
    """Report type."""
    
    metadata: ReportMetadata
    summary: str
    sections: List[ReportSection]
    data: Dict[str, Any]