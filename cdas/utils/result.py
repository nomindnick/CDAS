"""
Result objects and interfaces for the Construction Document Analysis System.

This module provides standardized result objects used across various components
of the system to ensure consistent error handling and result structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Generic, TypeVar, Union
from datetime import datetime

# Type variable for generic result objects
T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Result(Generic[T, E]):
    """
    Generic result object for operations that may succeed or fail.
    
    This class provides a consistent way to represent the results of operations,
    including success status, data, and error information.
    """
    
    success: bool
    data: Optional[T] = None
    error: Optional[E] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate the result object state."""
        if self.success and self.error is not None:
            raise ValueError("Cannot have error in successful result")
            
        if not self.success and self.error is None:
            raise ValueError("Must have error in failed result")

    @staticmethod
    def success(data: T, **metadata) -> 'Result[T, E]':
        """
        Create a success result.
        
        Args:
            data: Result data
            **metadata: Additional metadata
            
        Returns:
            Success result object
        """
        return Result(success=True, data=data, metadata=metadata)
        
    @staticmethod
    def failure(error: E, error_code: Optional[str] = None, **metadata) -> 'Result[T, E]':
        """
        Create a failure result.
        
        Args:
            error: Error information
            error_code: Optional error code
            **metadata: Additional metadata
            
        Returns:
            Failure result object
        """
        return Result(success=False, error=error, error_code=error_code, metadata=metadata)
        
    def on_success(self, success_fn):
        """
        Execute a function only if the result is successful.
        
        Args:
            success_fn: Function to execute with the result data
            
        Returns:
            Result of the function or self if result is not successful
        """
        if self.success:
            return success_fn(self.data)
        return self
        
    def on_failure(self, failure_fn):
        """
        Execute a function only if the result is a failure.
        
        Args:
            failure_fn: Function to execute with the error
            
        Returns:
            Result of the function or self if result is successful
        """
        if not self.success:
            return failure_fn(self.error)
        return self
        
    def map(self, mapper_fn):
        """
        Transform the result data if successful.
        
        Args:
            mapper_fn: Function to transform the data
            
        Returns:
            New result with transformed data or same result if failure
        """
        if not self.success:
            return self
            
        try:
            new_data = mapper_fn(self.data)
            return Result(
                success=True,
                data=new_data,
                metadata=self.metadata,
                timestamp=self.timestamp
            )
        except Exception as e:
            return Result(
                success=False,
                error=str(e),
                error_code="MAPPING_ERROR",
                metadata=self.metadata,
                timestamp=self.timestamp
            )

@dataclass
class DocumentResult(Result[Dict[str, Any], str]):
    """Result object for document processing operations."""
    
    document_id: Optional[str] = None
    
    @staticmethod
    def success(data: Dict[str, Any], document_id: str, **metadata) -> 'DocumentResult':
        """Create a successful document result."""
        return DocumentResult(success=True, data=data, document_id=document_id, metadata=metadata)
    
    @staticmethod
    def failure(error: str, error_code: Optional[str] = None, **metadata) -> 'DocumentResult':
        """Create a failed document result."""
        return DocumentResult(success=False, error=error, error_code=error_code, metadata=metadata)

@dataclass
class AnalysisResult(Result[Dict[str, Any], str]):
    """Result object for financial analysis operations."""
    
    confidence: float = 0.0
    
    @staticmethod
    def success(data: Dict[str, Any], confidence: float = 0.0, **metadata) -> 'AnalysisResult':
        """Create a successful analysis result."""
        return AnalysisResult(success=True, data=data, confidence=confidence, metadata=metadata)
    
    @staticmethod
    def failure(error: str, error_code: Optional[str] = None, **metadata) -> 'AnalysisResult':
        """Create a failed analysis result."""
        return AnalysisResult(success=False, error=error, error_code=error_code, metadata=metadata)

@dataclass
class BatchResult(Result[List[Result], List[Exception]]):
    """Result object for batch operations."""
    
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    
    @staticmethod
    def from_results(results: List[Result]) -> 'BatchResult':
        """
        Create a batch result from a list of individual results.
        
        Args:
            results: List of results
            
        Returns:
            BatchResult summarizing all results
        """
        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        
        success = failed == 0
        
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        
        error_data = [r.error for r in failures] if failures else None
        
        return BatchResult(
            success=success,
            data=results,
            error=error_data,
            total=len(results),
            succeeded=succeeded,
            failed=failed
        )