"""
Base interfaces for the financial analysis components.

This module defines abstract base classes and common interfaces for the
financial analysis components of the Construction Document Analysis System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    
    success: bool
    data: Dict[str, Any]
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

# Generic type for the specific analysis result
T = TypeVar('T')

class BaseAnalyzer(ABC, Generic[T]):
    """Base class for all financial analysis components."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
    
    @abstractmethod
    def analyze(self, **kwargs) -> T:
        """Perform analysis operation.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Analysis result
        """
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback to default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)

class PatternAnalyzer(BaseAnalyzer[List[Dict[str, Any]]]):
    """Base class for pattern detection analyzers."""
    
    @abstractmethod
    def detect_patterns(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect patterns in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected patterns
        """
        pass
    
    def analyze(self, **kwargs) -> List[Dict[str, Any]]:
        """Implement the BaseAnalyzer interface.
        
        Args:
            **kwargs: Analysis parameters, including doc_id
            
        Returns:
            List of detected patterns
        """
        doc_id = kwargs.get('doc_id')
        return self.detect_patterns(doc_id)

class AnomalyAnalyzer(BaseAnalyzer[List[Dict[str, Any]]]):
    """Base class for anomaly detection analyzers."""
    
    @abstractmethod
    def detect_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect anomalies in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected anomalies
        """
        pass
    
    def analyze(self, **kwargs) -> List[Dict[str, Any]]:
        """Implement the BaseAnalyzer interface.
        
        Args:
            **kwargs: Analysis parameters, including doc_id
            
        Returns:
            List of detected anomalies
        """
        doc_id = kwargs.get('doc_id')
        return self.detect_anomalies(doc_id)

class MatchingAnalyzer(BaseAnalyzer[List[Dict[str, Any]]]):
    """Base class for amount matching analyzers."""
    
    @abstractmethod
    def find_matches(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find matching amounts in a document or across documents.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of matches
        """
        pass
    
    def analyze(self, **kwargs) -> List[Dict[str, Any]]:
        """Implement the BaseAnalyzer interface.
        
        Args:
            **kwargs: Analysis parameters, including doc_id
            
        Returns:
            List of matches
        """
        doc_id = kwargs.get('doc_id')
        return self.find_matches(doc_id)

class ChronologyAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """Base class for chronology analyzers."""
    
    @abstractmethod
    def analyze_timeline(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the timeline of documents.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            Timeline analysis
        """
        pass
    
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """Implement the BaseAnalyzer interface.
        
        Args:
            **kwargs: Analysis parameters, including doc_id
            
        Returns:
            Timeline analysis
        """
        doc_id = kwargs.get('doc_id')
        return self.analyze_timeline(doc_id)

class RelationshipAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """Base class for relationship analyzers."""
    
    @abstractmethod
    def analyze_relationships(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze relationships between entities.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            Relationship analysis
        """
        pass
    
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """Implement the BaseAnalyzer interface.
        
        Args:
            **kwargs: Analysis parameters, including doc_id
            
        Returns:
            Relationship analysis
        """
        doc_id = kwargs.get('doc_id')
        return self.analyze_relationships(doc_id)