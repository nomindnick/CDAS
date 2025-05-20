"""
Common utility functions for the Construction Document Analysis System.

This module provides shared functionality used across various components
of the system, including result object handling, configuration management,
and data transformation utilities.
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Generic, Callable, Tuple
from enum import Enum
from datetime import datetime, date
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Type aliases for better type hints
JSON = Dict[str, Any]
Number = Union[int, float]
DateType = Union[date, datetime, str]

T = TypeVar('T')
U = TypeVar('U')

def safe_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested key from a dictionary.
    
    Args:
        data: Dictionary to search within
        key_path: Dot-separated path to the key (e.g., "parent.child.grandchild")
        default: Default value to return if key not found
        
    Returns:
        Value at the key path or default if not found
    """
    if not data or not isinstance(data, dict):
        return default
        
    keys = key_path.split('.')
    result = data
    
    try:
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    except Exception:
        return default

def ensure_list(value: Union[List[T], T]) -> List[T]:
    """
    Ensure a value is a list. If it's not, wrap it in a list.
    
    Args:
        value: Value to ensure is a list
        
    Returns:
        List containing the value or the original list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse a date string into a date object.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Date object or None if parsing fails
    """
    if not date_str:
        return None
        
    # Common date formats in construction documents
    formats = [
        '%B %d, %Y',  # January 15, 2023
        '%m/%d/%Y',   # 01/15/2023
        '%Y-%m-%d',   # 2023-01-15
        '%d-%b-%Y',   # 15-Jan-2023
        '%m-%d-%Y',   # 01-15-2023
        '%d/%m/%Y',   # 15/01/2023 (UK format)
    ]
    
    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
            
    # Log error but don't raise exception
    logger.warning(f"Failed to parse date string: {date_str}")
    return None

def normalize_amount(amount_str: str) -> Optional[float]:
    """
    Convert an amount string to a float, handling various formats.
    
    Args:
        amount_str: String representation of an amount
        
    Returns:
        Normalized float amount or None if conversion fails
    """
    if not amount_str:
        return None
        
    # If already a number, return it
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
        
    # Remove currency symbols, commas, and other non-numeric characters
    # Keep decimal points and minus signs
    amount_str = str(amount_str)
    
    try:
        # Handle parentheses for negative numbers (accounting notation)
        if amount_str.startswith('(') and amount_str.endswith(')'):
            amount_str = '-' + amount_str[1:-1]
            
        # Remove currency symbols, commas, and spaces
        for char in ['$', '€', '£', '¥', ',', ' ']:
            amount_str = amount_str.replace(char, '')
            
        # Convert to float
        return float(amount_str)
    except ValueError:
        logger.warning(f"Failed to normalize amount string: {amount_str}")
        return None

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, handling nested structures.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1 on conflicts)
        
    Returns:
        Merged dictionary
    """
    if not dict1:
        return dict2 or {}
    if not dict2:
        return dict1 or {}
        
    result = dict1.copy()
    
    for key, value in dict2.items():
        # If both values are dicts, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
            
    return result

def format_currency(amount: Optional[float]) -> str:
    """
    Format a number as a currency string.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return "N/A"
        
    return f"${amount:,.2f}"

def calculate_percentage(part: Number, whole: Number) -> Optional[float]:
    """
    Calculate a percentage safely.
    
    Args:
        part: Numerator
        whole: Denominator
        
    Returns:
        Percentage (0-100) or None if calculation fails
    """
    try:
        if whole == 0:
            return None
        return (part / whole) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return None

def round_to_nearest(value: float, nearest: float = 0.01) -> float:
    """
    Round a value to the nearest specified value.
    
    Args:
        value: Value to round
        nearest: Value to round to (default is 0.01 for cents)
        
    Returns:
        Rounded value
    """
    return round(value / nearest) * nearest

def safe_json_serialize(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, Path):
        return str(obj)
    return str(obj)

def safe_json_dumps(data: Any) -> str:
    """
    Convert data to a JSON string, handling non-serializable types.
    
    Args:
        data: Data to convert to JSON
        
    Returns:
        JSON string
    """
    return json.dumps(data, default=safe_json_serialize)

def batch_process(items: List[T], 
                 processor: Callable[[T], U], 
                 batch_size: int = 100) -> List[U]:
    """
    Process a list of items in batches.
    
    Args:
        items: List of items to process
        processor: Function to process each item
        batch_size: Number of items per batch
        
    Returns:
        List of processed items
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = [processor(item) for item in batch]
        results.extend(batch_results)
        
    return results

def find_amount_patterns(amounts: List[float], 
                         tolerance: float = 0.01) -> List[Tuple[float, List[int]]]:
    """
    Find patterns in a list of amounts (recurring, multiples, etc.).
    
    Args:
        amounts: List of financial amounts
        tolerance: Relative tolerance for matching
        
    Returns:
        List of (amount, [indices]) tuples for amounts that appear multiple times
    """
    if not amounts:
        return []
        
    # Dictionary to track amount occurrences
    amount_map: Dict[float, List[int]] = {}
    
    # Round amounts to avoid floating point precision issues
    rounded_amounts = [round_to_nearest(amount) for amount in amounts]
    
    # Find recurring amounts
    for i, amount in enumerate(rounded_amounts):
        found_match = False
        
        # Check if this amount matches any existing key within tolerance
        for key in list(amount_map.keys()):
            if abs(amount - key) <= tolerance * max(abs(amount), abs(key)):
                amount_map[key].append(i)
                found_match = True
                break
                
        # If no match found, add a new entry
        if not found_match:
            amount_map[amount] = [i]
    
    # Filter out amounts that only appear once
    recurring = [(amount, indices) for amount, indices in amount_map.items() 
                if len(indices) > 1]
    
    # Sort by number of occurrences (most frequent first)
    recurring.sort(key=lambda x: len(x[1]), reverse=True)
    
    return recurring