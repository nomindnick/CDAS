"""
Exact amount matcher for the financial analysis engine.

This module provides exact matching of amounts across different documents,
supporting both direct matches and matches within a small tolerance.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class ExactMatcher:
    """Matches amounts exactly across documents with optional tolerance."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the exact amount matcher.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.default_tolerance = self.config.get('default_tolerance', 0.01)
        self.match_across_parties = self.config.get('match_across_parties', True)
        
    def _format_date(self, date_value):
        """Format a date value safely for output.
        
        Args:
            date_value: Date value (datetime, string, or None)
            
        Returns:
            Formatted date string or None
        """
        if date_value is None:
            return None
            
        # If it's already a string, return it
        if isinstance(date_value, str):
            return date_value
            
        # If it's a datetime object, format it
        try:
            return date_value.isoformat()
        except (AttributeError, TypeError):
            # If all else fails, convert to string
            return str(date_value)
        
    def find_matches(self, doc_id: str) -> List[Dict[str, Any]]:
        """Find matching amounts for items in a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of matches
        """
        logger.info(f"Finding exact matches for document: {doc_id}")
        
        # Get line items from the document
        query = text("""
            SELECT 
                item_id, amount, description
            FROM 
                line_items
            WHERE 
                doc_id = :doc_id
                AND amount IS NOT NULL
                AND amount > 0
        """)
        
        items = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        matches = []
        for item_id, amount, description in items:
            # Find matches for this amount
            item_matches = self.find_matches_by_amount(amount)
            
            # Filter out self-matches
            item_matches = [m for m in item_matches if m['item_id'] != item_id]
            
            if item_matches:
                matches.append({
                    'item_id': item_id,
                    'amount': float(amount) if amount is not None else None,
                    'description': description,
                    'matches': item_matches
                })
        
        logger.info(f"Found {len(matches)} items with exact matches")
        return matches
    
    def find_matches_by_amount(self, amount: float, tolerance: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find matches for a specific amount.
        
        Args:
            amount: Amount to match
            tolerance: Matching tolerance (defaults to self.default_tolerance)
            
        Returns:
            List of matches
        """
        logger.info(f"Finding exact matches for amount: {amount}")
        
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount >= :amount_min AND li.amount <= :amount_max
            ORDER BY
                d.date_created
        """)
        
        amount_min = float(amount) * (1 - tolerance) if tolerance else amount - 0.01
        amount_max = float(amount) * (1 + tolerance) if tolerance else amount + 0.01
        
        matches = self.db_session.execute(
            query, 
            {
                "amount_min": amount_min, 
                "amount_max": amount_max
            }
        ).fetchall()
        
        result = []
        for match in matches:
            item_id, doc_id, description, match_amount, doc_type, party, date_created = match
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created)
            })
        
        logger.info(f"Found {len(result)} exact matches for amount: {amount}")
        return result
    
    def find_all_occurrences(self, amount: float, tolerance: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find all occurrences of an amount across all documents.
        
        Args:
            amount: Amount to find
            tolerance: Matching tolerance (defaults to self.default_tolerance)
            
        Returns:
            List of occurrences
        """
        logger.info(f"Finding all occurrences of amount: {amount}")
        
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount BETWEEN :amount_min AND :amount_max
            ORDER BY
                d.date_created
        """)
        
        amount_min = amount - tolerance
        amount_max = amount + tolerance
        
        occurrences = self.db_session.execute(
            query, 
            {
                "amount_min": amount_min, 
                "amount_max": amount_max
            }
        ).fetchall()
        
        result = []
        for occurrence in occurrences:
            item_id, doc_id, description, match_amount, cost_code, doc_type, party, date_created = occurrence
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created)
            })
        
        logger.info(f"Found {len(result)} occurrences of amount: {amount}")
        return result
    
    def find_matches_between_documents(self, doc_id1: str, doc_id2: str, 
                                    tolerance: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find matching amounts between two specific documents.
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            tolerance: Matching tolerance (defaults to self.default_tolerance)
            
        Returns:
            List of matches between the documents
        """
        logger.info(f"Finding matches between documents: {doc_id1} and {doc_id2}")
        
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        
        # Get line items from both documents
        query1 = text("""
            SELECT 
                li.item_id,
                li.description,
                li.amount,
                li.cost_code
            FROM 
                line_items li
            WHERE 
                li.doc_id = :doc_id
                AND li.amount IS NOT NULL
                AND li.amount > 0
        """)
        
        items1 = self.db_session.execute(query1, {"doc_id": doc_id1}).fetchall()
        items2 = self.db_session.execute(query1, {"doc_id": doc_id2}).fetchall()
        
        # Create a lookup for items in doc2
        items2_by_amount = {}
        for item_id, description, amount, cost_code in items2:
            if amount not in items2_by_amount:
                items2_by_amount[amount] = []
                
            items2_by_amount[amount].append({
                'item_id': item_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code
            })
        
        # Find matches
        matches = []
        for item_id1, description1, amount1, cost_code1 in items1:
            # Look for matches within tolerance
            amount_matches = []
            for amount2 in items2_by_amount:
                if abs(amount1 - amount2) <= tolerance:
                    for item2 in items2_by_amount[amount2]:
                        amount_matches.append({
                            'doc1_item_id': item_id1,
                            'doc1_description': description1,
                            'doc1_amount': float(amount1) if amount1 is not None else None,
                            'doc1_cost_code': cost_code1,
                            'doc2_item_id': item2['item_id'],
                            'doc2_description': item2['description'],
                            'doc2_amount': float(item2['amount']) if item2['amount'] is not None else None,
                            'doc2_cost_code': item2['cost_code'],
                            'difference': float(abs(amount1 - item2['amount'])) if amount1 is not None and item2['amount'] is not None else None
                        })
            
            if amount_matches:
                matches.append({
                    'item_id': item_id1,
                    'description': description1,
                    'amount': float(amount1) if amount1 is not None else None,
                    'matches': amount_matches
                })
        
        logger.info(f"Found {len(matches)} items with matches between documents")
        return matches
    
    def find_matches_in_date_range(self, amount: float, start_date: datetime, 
                                end_date: datetime, tolerance: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find matches for an amount within a specific date range.
        
        Args:
            amount: Amount to match
            start_date: Start date for the range
            end_date: End date for the range
            tolerance: Matching tolerance (defaults to self.default_tolerance)
            
        Returns:
            List of matches within the date range
        """
        logger.info(f"Finding matches for amount {amount} between {start_date.isoformat()} and {end_date.isoformat()}")
        
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount BETWEEN :amount_min AND :amount_max
                AND d.date_created BETWEEN :start_date AND :end_date
            ORDER BY
                d.date_created
        """)
        
        amount_min = amount - tolerance
        amount_max = amount + tolerance
        
        matches = self.db_session.execute(
            query, 
            {
                "amount_min": amount_min, 
                "amount_max": amount_max,
                "start_date": start_date,
                "end_date": end_date
            }
        ).fetchall()
        
        result = []
        for match in matches:
            item_id, doc_id, description, match_amount, doc_type, party, date_created = match
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created)
            })
        
        logger.info(f"Found {len(result)} matches in date range")
        return result
    
    def register_amount_match(self, source_item_id: int, target_item_id: int, 
                           match_type: str = 'exact', confidence: float = 1.0) -> Dict[str, Any]:
        """Register a match between two line items in the database.
        
        Args:
            source_item_id: Source line item ID
            target_item_id: Target line item ID
            match_type: Type of match (e.g., 'exact', 'fuzzy', 'context')
            confidence: Confidence score for the match
            
        Returns:
            Dictionary with match information
        """
        logger.info(f"Registering {match_type} match between items {source_item_id} and {target_item_id}")
        
        # Get line item details
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.amount
            FROM 
                line_items li
            WHERE 
                li.item_id IN (:source_id, :target_id)
        """)
        
        items = self.db_session.execute(
            query, 
            {
                "source_id": source_item_id, 
                "target_id": target_item_id
            }
        ).fetchall()
        
        # Create a lookup by item_id
        item_lookup = {item[0]: {'doc_id': item[1], 'amount': item[2]} for item in items}
        
        # Verify both items exist
        if source_item_id not in item_lookup or target_item_id not in item_lookup:
            logger.warning(f"Cannot register match: one or both items not found")
            return {'error': 'One or both items not found'}
        
        # Calculate difference
        source_amount = item_lookup[source_item_id]['amount']
        target_amount = item_lookup[target_item_id]['amount']
        difference = abs(source_amount - target_amount) if source_amount is not None and target_amount is not None else None
        
        # Check if match already exists
        check_query = text("""
            SELECT 
                match_id
            FROM 
                amount_matches
            WHERE 
                source_item_id = :source_id
                AND target_item_id = :target_id
        """)
        
        existing = self.db_session.execute(
            check_query, 
            {
                "source_id": source_item_id, 
                "target_id": target_item_id
            }
        ).fetchone()
        
        if existing:
            logger.info(f"Match already exists (match_id: {existing[0]})")
            
            # Update existing match
            update_query = text("""
                UPDATE amount_matches 
                SET 
                    match_type = :match_type,
                    confidence = :confidence,
                    difference = :difference
                WHERE 
                    match_id = :match_id
                RETURNING match_id
            """)
            
            try:
                result = self.db_session.execute(
                    update_query, 
                    {
                        "match_id": existing[0],
                        "match_type": match_type,
                        "confidence": confidence,
                        "difference": difference
                    }
                ).fetchone()
                
                self.db_session.commit()
                
                return {
                    'match_id': result[0],
                    'source_item_id': source_item_id,
                    'target_item_id': target_item_id,
                    'match_type': match_type,
                    'confidence': confidence,
                    'difference': float(difference) if difference is not None else None,
                    'status': 'updated'
                }
            except Exception as e:
                self.db_session.rollback()
                logger.exception(f"Error updating match: {e}")
                return {'error': str(e)}
        else:
            # Create new match
            insert_query = text("""
                INSERT INTO amount_matches 
                    (source_item_id, target_item_id, match_type, confidence, difference) 
                VALUES 
                    (:source_id, :target_id, :match_type, :confidence, :difference)
                RETURNING match_id
            """)
            
            try:
                result = self.db_session.execute(
                    insert_query, 
                    {
                        "source_id": source_item_id,
                        "target_id": target_item_id,
                        "match_type": match_type,
                        "confidence": confidence,
                        "difference": difference
                    }
                ).fetchone()
                
                self.db_session.commit()
                
                return {
                    'match_id': result[0],
                    'source_item_id': source_item_id,
                    'target_item_id': target_item_id,
                    'match_type': match_type,
                    'confidence': confidence,
                    'difference': float(difference) if difference is not None else None,
                    'status': 'created'
                }
            except Exception as e:
                self.db_session.rollback()
                logger.exception(f"Error creating match: {e}")
                return {'error': str(e)}
    
    def get_existing_matches(self, item_id: Optional[int] = None, 
                          doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get existing matches for a line item or document.
        
        Args:
            item_id: Optional line item ID to filter matches
            doc_id: Optional document ID to filter matches
            
        Returns:
            List of existing matches
        """
        logger.info(f"Getting existing matches for item_id={item_id}, doc_id={doc_id}")
        
        # Construct query based on provided filters
        if item_id is not None:
            query = text("""
                SELECT 
                    am.match_id,
                    am.source_item_id,
                    am.target_item_id,
                    am.match_type,
                    am.confidence,
                    am.difference,
                    am.created_at,
                    s_li.doc_id AS source_doc_id,
                    t_li.doc_id AS target_doc_id,
                    s_li.amount AS source_amount,
                    t_li.amount AS target_amount
                FROM 
                    amount_matches am
                JOIN
                    line_items s_li ON am.source_item_id = s_li.item_id
                JOIN
                    line_items t_li ON am.target_item_id = t_li.item_id
                WHERE 
                    am.source_item_id = :item_id
                    OR am.target_item_id = :item_id
                ORDER BY
                    am.confidence DESC
            """)
            
            params = {"item_id": item_id}
            
        elif doc_id is not None:
            query = text("""
                SELECT 
                    am.match_id,
                    am.source_item_id,
                    am.target_item_id,
                    am.match_type,
                    am.confidence,
                    am.difference,
                    am.created_at,
                    s_li.doc_id AS source_doc_id,
                    t_li.doc_id AS target_doc_id,
                    s_li.amount AS source_amount,
                    t_li.amount AS target_amount
                FROM 
                    amount_matches am
                JOIN
                    line_items s_li ON am.source_item_id = s_li.item_id
                JOIN
                    line_items t_li ON am.target_item_id = t_li.item_id
                WHERE 
                    s_li.doc_id = :doc_id
                    OR t_li.doc_id = :doc_id
                ORDER BY
                    am.confidence DESC
            """)
            
            params = {"doc_id": doc_id}
            
        else:
            # If no filters provided, get top matches
            query = text("""
                SELECT 
                    am.match_id,
                    am.source_item_id,
                    am.target_item_id,
                    am.match_type,
                    am.confidence,
                    am.difference,
                    am.created_at,
                    s_li.doc_id AS source_doc_id,
                    t_li.doc_id AS target_doc_id,
                    s_li.amount AS source_amount,
                    t_li.amount AS target_amount
                FROM 
                    amount_matches am
                JOIN
                    line_items s_li ON am.source_item_id = s_li.item_id
                JOIN
                    line_items t_li ON am.target_item_id = t_li.item_id
                ORDER BY
                    am.confidence DESC
                LIMIT 100
            """)
            
            params = {}
        
        # Execute query
        try:
            matches = self.db_session.execute(query, params).fetchall()
        except Exception as e:
            logger.exception(f"Error fetching matches: {e}")
            return []
        
        # Convert to dictionaries
        result = []
        for match in matches:
            match_id, source_item_id, target_item_id, match_type, confidence, difference, created_at, \
            source_doc_id, target_doc_id, source_amount, target_amount = match
            
            result.append({
                'match_id': match_id,
                'source_item_id': source_item_id,
                'target_item_id': target_item_id,
                'match_type': match_type,
                'confidence': float(confidence) if confidence is not None else None,
                'difference': float(difference) if difference is not None else None,
                'created_at': created_at.isoformat() if created_at else None,
                'source_doc_id': source_doc_id,
                'target_doc_id': target_doc_id,
                'source_amount': float(source_amount) if source_amount is not None else None,
                'target_amount': float(target_amount) if target_amount is not None else None
            })
        
        logger.info(f"Found {len(result)} existing matches")
        return result