"""
Base implementations for financial analysis components.

This module provides concrete base implementations for the financial analysis
components, reducing duplication across different analyzers.
"""

import logging
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union, Type
from datetime import datetime, date
from pathlib import Path
import json

from sqlalchemy.orm import Session
from sqlalchemy import text

from cdas.financial_analysis.base import (
    BaseAnalyzer, PatternAnalyzer, AnomalyAnalyzer,
    MatchingAnalyzer, ChronologyAnalyzer, RelationshipAnalyzer,
    AnalysisResult
)
from cdas.utils.common import (
    safe_get, ensure_list, parse_date, normalize_amount,
    merge_dicts, format_currency, calculate_percentage,
    round_to_nearest, find_amount_patterns
)

logger = logging.getLogger(__name__)

# Type variables for specialization
T = TypeVar('T')
U = TypeVar('U')

class AnalyzerBase(BaseAnalyzer[T]):
    """
    Base implementation for analyzers with common functionality.
    
    This class provides common methods used across different analyzer types
    to reduce code duplication.
    """
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        super().__init__(db_session, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document data from the database.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            # Use SQLAlchemy text to allow for more flexible queries
            query = text("""
                SELECT d.*, 
                       p.party_name, 
                       dt.type_name
                FROM documents d
                LEFT JOIN parties p ON d.party_id = p.party_id
                LEFT JOIN document_types dt ON d.doc_type_id = dt.doc_type_id
                WHERE d.doc_id = :doc_id
            """)
            
            result = self.db_session.execute(query, {"doc_id": doc_id}).fetchone()
            
            if not result:
                return None
                
            # Convert to dictionary
            doc = {column: value for column, value in result._mapping.items()}
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id}: {str(e)}")
            return None
    
    def get_line_items(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get line items from the database.
        
        Args:
            doc_id: Optional document ID to filter by
            
        Returns:
            List of line items
        """
        try:
            # Build the query based on whether doc_id is provided
            if doc_id:
                query = text("""
                    SELECT li.*, d.doc_type, d.party
                    FROM line_items li
                    JOIN documents d ON li.doc_id = d.doc_id
                    WHERE li.doc_id = :doc_id
                    ORDER BY li.line_number
                """)
                
                results = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
            else:
                query = text("""
                    SELECT li.*, d.doc_type, d.party
                    FROM line_items li
                    JOIN documents d ON li.doc_id = d.doc_id
                    ORDER BY li.doc_id, li.line_number
                """)
                
                results = self.db_session.execute(query).fetchall()
                
            # Convert to list of dictionaries
            items = []
            for row in results:
                item = {column: value for column, value in row._mapping.items()}
                items.append(item)
                
            return items
            
        except Exception as e:
            self.logger.error(f"Error getting line items: {str(e)}")
            return []
    
    def query_amounts(self, min_amount: Optional[float] = None, 
                     max_amount: Optional[float] = None,
                     tolerance: float = 0.01) -> List[Dict[str, Any]]:
        """
        Query amounts within a range.
        
        Args:
            min_amount: Optional minimum amount
            max_amount: Optional maximum amount
            tolerance: Tolerance for amount matching
            
        Returns:
            List of matching amounts with metadata
        """
        try:
            # Build query conditions based on parameters
            conditions = []
            params = {}
            
            if min_amount is not None:
                conditions.append("li.amount >= :min_amount")
                params["min_amount"] = min_amount
                
            if max_amount is not None:
                conditions.append("li.amount <= :max_amount")
                params["max_amount"] = max_amount
                
            # Build the full query
            query_str = """
                SELECT li.*, d.doc_type, d.party, d.doc_date
                FROM line_items li
                JOIN documents d ON li.doc_id = d.doc_id
            """
            
            if conditions:
                query_str += " WHERE " + " AND ".join(conditions)
                
            query_str += " ORDER BY li.amount, d.doc_date"
            
            query = text(query_str)
            
            # Execute the query
            results = self.db_session.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            items = []
            for row in results:
                item = {column: value for column, value in row._mapping.items()}
                items.append(item)
                
            return items
            
        except Exception as e:
            self.logger.error(f"Error querying amounts: {str(e)}")
            return []
    
    def find_recurring_amounts(self, amounts: List[float], 
                              tolerance: float = 0.01,
                              min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Find recurring amounts in a list.
        
        Args:
            amounts: List of amounts to analyze
            tolerance: Relative tolerance for matching
            min_occurrences: Minimum number of occurrences to consider recurring
            
        Returns:
            List of recurring amount patterns
        """
        # Use the utility function to find patterns
        recurring_patterns = find_amount_patterns(amounts, tolerance)
        
        # Filter by minimum occurrences
        filtered_patterns = [(amount, indices) for amount, indices in recurring_patterns 
                           if len(indices) >= min_occurrences]
        
        # Convert to result format
        results = []
        for amount, indices in filtered_patterns:
            results.append({
                'amount': amount,
                'occurrences': len(indices),
                'indices': indices,
                'confidence': min(0.95, 0.6 + (len(indices) / 10) * 0.35)
            })
            
        return results
    
    def get_confidence_level(self, strength: float) -> str:
        """
        Convert a confidence value to a level string.
        
        Args:
            strength: Confidence value (0.0-1.0)
            
        Returns:
            Confidence level string
        """
        if strength >= 0.9:
            return "Very High"
        elif strength >= 0.75:
            return "High"
        elif strength >= 0.6:
            return "Medium"
        elif strength >= 0.4:
            return "Low"
        else:
            return "Very Low"

class PatternAnalyzerBase(PatternAnalyzer):
    """
    Base implementation for pattern analyzers.
    
    This class provides common methods used by different pattern detection 
    components to reduce code duplication.
    """
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        super().__init__(db_session, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def detect_patterns(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect patterns in financial data.
        
        This base implementation delegates to specialized methods that should
        be implemented by subclasses.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Let each subclass add its specific patterns
        for method_name in self._get_detector_methods():
            try:
                method = getattr(self, method_name)
                results = method(doc_id)
                patterns.extend(ensure_list(results))
            except Exception as e:
                self.logger.error(f"Error in pattern detector method {method_name}: {str(e)}")
                
        return patterns
    
    def _get_detector_methods(self) -> List[str]:
        """
        Get all pattern detector methods.
        
        Subclasses should override this to return a list of method names
        that detect specific patterns.
        
        Returns:
            List of method names
        """
        return []
        
    def tag_pattern(self, pattern_type: str, 
                   description: str, 
                   items: List[Dict[str, Any]], 
                   confidence: float) -> Dict[str, Any]:
        """
        Create a standardized pattern object.
        
        Args:
            pattern_type: Type of pattern
            description: Description of the pattern
            items: List of items involved in the pattern
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Pattern object
        """
        return {
            'type': pattern_type,
            'description': description,
            'items': items,
            'confidence': confidence,
            'confidence_level': self.get_confidence_level(confidence),
            'detected_at': datetime.utcnow().isoformat()
        }
        
    def get_confidence_level(self, strength: float) -> str:
        """
        Convert a confidence value to a level string.
        
        Args:
            strength: Confidence value (0.0-1.0)
            
        Returns:
            Confidence level string
        """
        if strength >= 0.9:
            return "Very High"
        elif strength >= 0.75:
            return "High"
        elif strength >= 0.6:
            return "Medium"
        elif strength >= 0.4:
            return "Low"
        else:
            return "Very Low"

class AnomalyAnalyzerBase(AnomalyAnalyzer):
    """
    Base implementation for anomaly analyzers.
    
    This class provides common methods used by different anomaly detection
    components to reduce code duplication.
    """
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        super().__init__(db_session, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def detect_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in financial data.
        
        This base implementation delegates to specialized methods that should
        be implemented by subclasses.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Let each subclass add its specific anomalies
        for method_name in self._get_detector_methods():
            try:
                method = getattr(self, method_name)
                results = method(doc_id)
                anomalies.extend(ensure_list(results))
            except Exception as e:
                self.logger.error(f"Error in anomaly detector method {method_name}: {str(e)}")
                
        return anomalies
    
    def _get_detector_methods(self) -> List[str]:
        """
        Get all anomaly detector methods.
        
        Subclasses should override this to return a list of method names
        that detect specific anomalies.
        
        Returns:
            List of method names
        """
        return []
        
    def tag_anomaly(self, anomaly_type: str, 
                   description: str, 
                   items: List[Dict[str, Any]], 
                   confidence: float,
                   severity: float = 0.5) -> Dict[str, Any]:
        """
        Create a standardized anomaly object.
        
        Args:
            anomaly_type: Type of anomaly
            description: Description of the anomaly
            items: List of items involved in the anomaly
            confidence: Confidence level (0.0-1.0)
            severity: Severity level (0.0-1.0)
            
        Returns:
            Anomaly object
        """
        return {
            'type': anomaly_type,
            'description': description,
            'items': items,
            'confidence': confidence,
            'confidence_level': self.get_confidence_level(confidence),
            'severity': severity,
            'severity_level': self.get_severity_level(severity),
            'detected_at': datetime.utcnow().isoformat()
        }
        
    def get_confidence_level(self, strength: float) -> str:
        """
        Convert a confidence value to a level string.
        
        Args:
            strength: Confidence value (0.0-1.0)
            
        Returns:
            Confidence level string
        """
        if strength >= 0.9:
            return "Very High"
        elif strength >= 0.75:
            return "High"
        elif strength >= 0.6:
            return "Medium"
        elif strength >= 0.4:
            return "Low"
        else:
            return "Very Low"
            
    def get_severity_level(self, severity: float) -> str:
        """
        Convert a severity value to a level string.
        
        Args:
            severity: Severity value (0.0-1.0)
            
        Returns:
            Severity level string
        """
        if severity >= 0.8:
            return "Critical"
        elif severity >= 0.6:
            return "High"
        elif severity >= 0.4:
            return "Medium"
        elif severity >= 0.2:
            return "Low"
        else:
            return "Negligible"