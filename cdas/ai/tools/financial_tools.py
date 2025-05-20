"""Financial analysis tools for agent use.

This module provides financial analysis tools that can be used by agents
for investigating suspicious patterns and analyzing financial discrepancies.
"""

import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


def search_line_items(db_session, description: Optional[str] = None, 
                      min_amount: Optional[float] = None, 
                      max_amount: Optional[float] = None,
                      amount: Optional[float] = None) -> List[Dict[str, Any]]:
    """Search for line items matching criteria.
    
    Args:
        db_session: Database session
        description: Optional description search
        min_amount: Optional minimum amount
        max_amount: Optional maximum amount
        amount: Optional exact amount to match (with small tolerance)
        
    Returns:
        List of matching line items
    """
    try:
        # Build SQL query
        sql_query = """
            SELECT 
                li.item_id,
                li.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE 
                1=1
        """
        
        params = []
        
        if description:
            sql_query += " AND li.description ILIKE %s"
            params.append(f"%{description}%")
        
        if min_amount is not None:
            sql_query += " AND li.amount >= %s"
            params.append(min_amount)
        
        if max_amount is not None:
            sql_query += " AND li.amount <= %s"
            params.append(max_amount)
        
        if amount is not None:
            # Search with tolerance (0.01%)
            tolerance = amount * 0.0001
            sql_query += " AND li.amount BETWEEN %s AND %s"
            params.append(amount - tolerance)
            params.append(amount + tolerance)
        
        sql_query += " ORDER BY d.date DESC, li.amount DESC"
        
        # Execute query
        results = db_session.execute(sql_query, params).fetchall()
        
        # Format results
        items = []
        for item_id, doc_id, title, doc_type, party, date, item_desc, amount, quantity, unit_price in results:
            items.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'document': {
                    'title': title,
                    'doc_type': doc_type,
                    'party': party,
                    'date': date.isoformat() if date else None
                },
                'description': item_desc,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price
            })
        
        return items
        
    except Exception as e:
        logger.error(f"Error searching line items: {str(e)}")
        raise


def find_similar_amounts(db_session, amount: float, tolerance_percentage: float = 0.1) -> List[Dict[str, Any]]:
    """Find similar amounts across different documents.
    
    Args:
        db_session: Database session
        amount: Amount to match
        tolerance_percentage: Tolerance percentage (default 0.1%)
        
    Returns:
        List of matching amounts
    """
    try:
        tolerance = amount * (tolerance_percentage / 100)
        
        # Build SQL query
        sql_query = """
            SELECT 
                li.item_id,
                li.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                li.description,
                li.amount
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.amount BETWEEN %s AND %s
            ORDER BY 
                d.date, d.doc_type, li.amount
        """
        
        # Execute query
        results = db_session.execute(sql_query, (amount - tolerance, amount + tolerance)).fetchall()
        
        # Format results
        items = []
        for item_id, doc_id, title, doc_type, party, date, description, item_amount in results:
            items.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'document': {
                    'title': title,
                    'doc_type': doc_type,
                    'party': party,
                    'date': date.isoformat() if date else None
                },
                'description': description,
                'amount': item_amount,
                'difference': item_amount - amount,
                'difference_percentage': (item_amount - amount) / amount * 100
            })
        
        return items
        
    except Exception as e:
        logger.error(f"Error finding similar amounts: {str(e)}")
        raise


def find_suspicious_patterns(db_session, pattern_type: Optional[str] = None, 
                             min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """Find suspicious financial patterns.
    
    Args:
        db_session: Database session
        pattern_type: Optional pattern type (recurring_amount, reappearing_amount, inconsistent_markup)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        List of suspicious patterns
    """
    try:
        # Import financial analysis engine
        from cdas.financial_analysis.engine import FinancialAnalysisEngine
        
        # Create analysis engine
        engine = FinancialAnalysisEngine(db_session)
        
        # Find suspicious patterns
        if pattern_type:
            patterns = engine.find_suspicious_patterns(pattern_type=pattern_type, min_confidence=min_confidence)
        else:
            patterns = engine.find_suspicious_patterns(min_confidence=min_confidence)
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error finding suspicious patterns: {str(e)}")
        raise


def analyze_amount(db_session, amount: float) -> Dict[str, Any]:
    """Analyze a specific amount.
    
    Args:
        db_session: Database session
        amount: Amount to analyze
        
    Returns:
        Analysis results
    """
    try:
        # Import financial analysis engine
        from cdas.financial_analysis.engine import FinancialAnalysisEngine
        
        # Create analysis engine
        engine = FinancialAnalysisEngine(db_session)
        
        # Analyze amount
        analysis = engine.analyze_amount(amount)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing amount: {str(e)}")
        raise
