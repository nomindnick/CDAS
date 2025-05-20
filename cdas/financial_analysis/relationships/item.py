"""
Line item relationship analyzer for the financial analysis engine.

This module analyzes relationships between line items across different documents,
tracking how financial items evolve and relate to each other over time.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class ItemRelationshipAnalyzer:
    """Analyzes relationships between line items."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the item relationship analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.auto_register = self.config.get('auto_register_matches', True)
        self.default_tolerance = self.config.get('default_amount_tolerance', 0.01)
        
    def analyze_amount_relationships(self, amount: float, 
                                  matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships for a specific amount.
        
        Args:
            amount: Amount to analyze
            matches: List of matches for the amount
            
        Returns:
            Relationship analysis
        """
        logger.info(f"Analyzing amount relationships for amount: {amount}")
        
        # Skip if no matches
        if not matches:
            return {
                'amount': amount,
                'matches': [],
                'documents': [],
                'chronology': {},
                'relationships': []
            }
            
        # Sort matches by date
        sorted_matches = self._sort_by_date(matches)
        
        # Get document details for all matches
        doc_ids = list({match['doc_id'] for match in sorted_matches if match.get('doc_id')})
        documents = self._get_documents(doc_ids)
        
        # Build amount chronology
        chronology = self._build_amount_chronology(sorted_matches, documents)
        
        # Find registered amount matches
        registered_matches = self._get_registered_amount_matches(sorted_matches)
        
        # Identify potential relationships between line items
        potential_relationships = self._identify_item_relationships(sorted_matches, documents)
        
        # Register new matches if auto-registration is enabled
        if self.auto_register:
            self._register_new_item_matches(potential_relationships, registered_matches)
        
        # Combine all relationships
        all_relationships = registered_matches + [
            rel for rel in potential_relationships 
            if not any(
                rm['source_item_id'] == rel['source_item_id'] and 
                rm['target_item_id'] == rel['target_item_id'] 
                for rm in registered_matches
            )
        ]
        
        # Return analysis results
        result = {
            'amount': amount,
            'matches': sorted_matches,
            'documents': list(documents.values()),
            'chronology': chronology,
            'relationships': all_relationships
        }
        
        logger.info(f"Amount relationship analysis complete: {len(all_relationships)} relationships")
        return result
    
    def analyze_item_relationships(self, item_id: int) -> Dict[str, Any]:
        """Analyze relationships for a specific line item.
        
        Args:
            item_id: Line item ID
            
        Returns:
            Relationship analysis
        """
        logger.info(f"Analyzing item relationships for item ID: {item_id}")
        
        # Get item details
        item = self._get_item(item_id)
        if not item:
            logger.warning(f"Item not found: {item_id}")
            return {'error': 'Item not found'}
        
        # Get registered matches for this item
        registered_matches = self._get_registered_item_matches(item_id)
        
        # Find similar items
        similar_items = self._find_similar_items(item)
        
        # Get line items from the same document
        document_items = self._get_document_items(item['doc_id'])
        
        # Identify related items within the document
        related_items = self._identify_related_document_items(item, document_items)
        
        # Identify potential relationships with similar items
        potential_relationships = self._identify_potential_item_relationships(item, similar_items)
        
        # Register new matches if auto-registration is enabled
        if self.auto_register:
            self._register_new_item_matches(potential_relationships, registered_matches)
        
        # Combine all relationships
        all_relationships = registered_matches + [
            rel for rel in potential_relationships 
            if not any(
                rm['source_item_id'] == rel['source_item_id'] and 
                rm['target_item_id'] == rel['target_item_id'] 
                for rm in registered_matches
            )
        ]
        
        # Return analysis results
        result = {
            'item': item,
            'similar_items': similar_items,
            'related_document_items': related_items,
            'relationships': all_relationships
        }
        
        logger.info(f"Item relationship analysis complete: {len(all_relationships)} relationships")
        return result
    
    def analyze_document_item_relationships(self, doc_id: str) -> Dict[str, Any]:
        """Analyze relationships between items in a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Relationship analysis
        """
        logger.info(f"Analyzing document item relationships for document: {doc_id}")
        
        # Get document details
        document = self._get_document(doc_id)
        if not document:
            logger.warning(f"Document not found: {doc_id}")
            return {'error': 'Document not found'}
        
        # Get all line items in the document
        items = self._get_document_items(doc_id)
        
        # Skip if no items
        if not items:
            return {
                'document': document,
                'items': [],
                'item_relationships': [],
                'amount_groups': []
            }
            
        # Analyze relationships between items
        item_relationships = []
        for i, item1 in enumerate(items):
            for j in range(i+1, len(items)):
                item2 = items[j]
                
                # Check for potential parent-child relationship
                item_rel = self._check_item_relationship(item1, item2)
                if item_rel:
                    item_relationships.append(item_rel)
        
        # Group items by amount
        amount_groups = self._group_items_by_amount(items)
        
        # Return analysis results
        result = {
            'document': document,
            'items': items,
            'item_relationships': item_relationships,
            'amount_groups': amount_groups
        }
        
        logger.info(f"Document item relationship analysis complete: {len(item_relationships)} relationships")
        return result
    
    def find_amount_history(self, amount: float, tolerance: Optional[float] = None) -> Dict[str, Any]:
        """Find the history of an amount across documents.
        
        Args:
            amount: Amount to find
            tolerance: Optional tolerance (defaults to self.default_tolerance)
            
        Returns:
            Amount history
        """
        logger.info(f"Finding history for amount: {amount}")
        
        tolerance = tolerance if tolerance is not None else self.default_tolerance
        
        # Find items with this amount
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created,
                d.status
            FROM 
                line_items li
            JOIN 
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.amount BETWEEN :amount_min AND :amount_max
                AND d.date_created IS NOT NULL
            ORDER BY 
                d.date_created
        """)
        
        amount_min = amount - (amount * tolerance)
        amount_max = amount + (amount * tolerance)
        
        items = self.db_session.execute(
            query, 
            {
                "amount_min": amount_min, 
                "amount_max": amount_max
            }
        ).fetchall()
        
        # Convert to dictionaries
        item_dicts = []
        for item in items:
            item_id, doc_id, description, item_amount, cost_code, doc_type, party, date_created, status = item
            
            item_dicts.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(item_amount) if item_amount is not None else None,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created,
                'status': status
            })
        
        # Group by document type
        doc_type_groups = {}
        for item in item_dicts:
            doc_type = item['doc_type'] or 'unknown'
            
            if doc_type not in doc_type_groups:
                doc_type_groups[doc_type] = []
                
            doc_type_groups[doc_type].append(item)
        
        # Count occurrences by month
        monthly_counts = {}
        for item in item_dicts:
            if item['date']:
                date = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
                month_key = date.strftime('%Y-%m')
                
                if month_key not in monthly_counts:
                    monthly_counts[month_key] = 0
                    
                monthly_counts[month_key] += 1
        
        # Sort months chronologically
        sorted_months = sorted(monthly_counts.keys())
        
        # Build timeline
        timeline = [
            {'month': month, 'count': monthly_counts[month]}
            for month in sorted_months
        ]
        
        # Find registered matches between these items
        item_ids = [item['item_id'] for item in item_dicts]
        registered_matches = self._get_registered_matches_by_item_ids(item_ids)
        
        # Return history
        result = {
            'amount': amount,
            'tolerance': tolerance,
            'items': item_dicts,
            'doc_type_groups': doc_type_groups,
            'timeline': timeline,
            'registered_matches': registered_matches
        }
        
        logger.info(f"Found {len(item_dicts)} occurrences of amount {amount}")
        return result
    
    def register_amount_match(self, source_item_id: int, target_item_id: int, 
                         match_type: str = 'exact', confidence: float = 1.0) -> Dict[str, Any]:
        """Register a match between two line items.
        
        Args:
            source_item_id: Source line item ID
            target_item_id: Target line item ID
            match_type: Type of match (e.g., 'exact', 'fuzzy', 'context')
            confidence: Confidence score for the match
            
        Returns:
            Dictionary with match information
        """
        return self._register_amount_match(source_item_id, target_item_id, match_type, confidence)
    
    def _sort_by_date(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort matches by date.
        
        Args:
            matches: List of matches
            
        Returns:
            Sorted list of matches
        """
        # Convert ISO date strings to datetime objects for sorting
        matches_with_datetime = []
        for match in matches:
            match_copy = match.copy()
            
            # Handle date field which could be an ISO string
            date_str = match.get('date')
            if isinstance(date_str, str):
                try:
                    match_copy['date'] = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    match_copy['date'] = None
            
            matches_with_datetime.append(match_copy)
        
        # Sort by date (using datetime.min for None values)
        sorted_matches = sorted(
            matches_with_datetime, 
            key=lambda x: x.get('date') or datetime.min
        )
        
        # Convert back to ISO strings for the result
        for match in sorted_matches:
            if isinstance(match.get('date'), datetime):
                match['date'] = match['date'].isoformat()
        
        return sorted_matches
    
    def _get_documents(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get details for multiple documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Dictionary mapping document ID to document details
        """
        if not doc_ids:
            return {}
            
        # Create placeholders for query
        placeholders = ', '.join([f":doc_id_{i}" for i in range(len(doc_ids))])
        
        query = text(f"""
            SELECT 
                d.doc_id,
                d.file_name,
                d.doc_type,
                d.party,
                d.date_created,
                d.status
            FROM 
                documents d
            WHERE 
                d.doc_id IN ({placeholders})
        """)
        
        params = {}
        for i, doc_id in enumerate(doc_ids):
            params[f"doc_id_{i}"] = doc_id
            
        results = self.db_session.execute(query, params).fetchall()
        
        documents = {}
        for result in results:
            doc_id, file_name, doc_type, party, date_created, status = result
            
            documents[doc_id] = {
                'doc_id': doc_id,
                'file_name': file_name,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created,
                'status': status
            }
        
        return documents
    
    def _get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document details.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document details or None if not found
        """
        query = text("""
            SELECT 
                d.doc_id,
                d.file_name,
                d.doc_type,
                d.party,
                d.date_created,
                d.date_received,
                d.date_processed,
                d.status,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount
            FROM 
                documents d
            LEFT JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                d.doc_id = :doc_id
            GROUP BY
                d.doc_id, d.file_name, d.doc_type, d.party, d.date_created, d.date_received, d.date_processed, d.status
        """)
        
        result = self.db_session.execute(query, {"doc_id": doc_id}).fetchone()
        
        if not result:
            return None
            
        doc_id, file_name, doc_type, party, date_created, date_received, date_processed, status, item_count, total_amount = result
        
        return {
            'doc_id': doc_id,
            'file_name': file_name,
            'doc_type': doc_type,
            'party': party,
            'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created,
            'date_received': date_received.isoformat() if date_received else None,
            'date_processed': date_processed.isoformat() if date_processed else None,
            'status': status,
            'item_count': item_count,
            'total_amount': float(total_amount) if total_amount is not None else None
        }
    
    def _get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get line item details.
        
        Args:
            item_id: Line item ID
            
        Returns:
            Line item details or None if not found
        """
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.total,
                li.cost_code,
                li.category,
                li.status,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN 
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.item_id = :item_id
        """)
        
        result = self.db_session.execute(query, {"item_id": item_id}).fetchone()
        
        if not result:
            return None
            
        item_id, doc_id, description, amount, quantity, unit_price, total, cost_code, category, status, doc_type, party, date_created = result
        
        return {
            'item_id': item_id,
            'doc_id': doc_id,
            'description': description,
            'amount': float(amount) if amount is not None else None,
            'quantity': float(quantity) if quantity is not None else None,
            'unit_price': float(unit_price) if unit_price is not None else None,
            'total': float(total) if total is not None else None,
            'cost_code': cost_code,
            'category': category,
            'status': status,
            'doc_type': doc_type,
            'party': party,
            'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created
        }
    
    def _get_document_items(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all line items from a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of line items
        """
        query = text("""
            SELECT 
                li.item_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.total,
                li.cost_code,
                li.category,
                li.status
            FROM 
                line_items li
            WHERE 
                li.doc_id = :doc_id
        """)
        
        results = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        items = []
        for result in results:
            item_id, description, amount, quantity, unit_price, total, cost_code, category, status = result
            
            items.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(amount) if amount is not None else None,
                'quantity': float(quantity) if quantity is not None else None,
                'unit_price': float(unit_price) if unit_price is not None else None,
                'total': float(total) if total is not None else None,
                'cost_code': cost_code,
                'category': category,
                'status': status
            })
        
        return items
    
    def _build_amount_chronology(self, matches: List[Dict[str, Any]], 
                            documents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build a chronology of an amount's appearance across documents.
        
        Args:
            matches: List of matches sorted by date
            documents: Dictionary of document details
            
        Returns:
            Amount chronology
        """
        # Skip if no matches
        if not matches:
            return {
                'first_appearance': None,
                'last_appearance': None,
                'doc_type_sequence': [],
                'party_sequence': []
            }
            
        # Get first and last appearance
        first_appearance = {
            'item_id': matches[0]['item_id'],
            'doc_id': matches[0]['doc_id'],
            'doc_type': matches[0]['doc_type'],
            'party': matches[0]['party'],
            'date': matches[0]['date'],
            'amount': matches[0]['amount']
        }
        
        last_appearance = {
            'item_id': matches[-1]['item_id'],
            'doc_id': matches[-1]['doc_id'],
            'doc_type': matches[-1]['doc_type'],
            'party': matches[-1]['party'],
            'date': matches[-1]['date'],
            'amount': matches[-1]['amount']
        }
        
        # Build document type sequence
        doc_type_sequence = []
        last_doc_type = None
        
        for match in matches:
            doc_type = match['doc_type']
            
            if doc_type != last_doc_type:
                doc_type_sequence.append({
                    'doc_type': doc_type,
                    'date': match['date'],
                    'doc_id': match['doc_id'],
                    'item_id': match['item_id']
                })
                
                last_doc_type = doc_type
        
        # Build party sequence
        party_sequence = []
        last_party = None
        
        for match in matches:
            party = match['party']
            
            if party != last_party:
                party_sequence.append({
                    'party': party,
                    'date': match['date'],
                    'doc_id': match['doc_id'],
                    'item_id': match['item_id']
                })
                
                last_party = party
        
        return {
            'first_appearance': first_appearance,
            'last_appearance': last_appearance,
            'doc_type_sequence': doc_type_sequence,
            'party_sequence': party_sequence
        }
    
    def _get_registered_amount_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get registered matches for a list of matches.
        
        Args:
            matches: List of matches
            
        Returns:
            List of registered matches
        """
        # Extract item IDs
        item_ids = [match['item_id'] for match in matches if match.get('item_id')]
        
        if not item_ids:
            return []
            
        return self._get_registered_matches_by_item_ids(item_ids)
    
    def _get_registered_matches_by_item_ids(self, item_ids: List[int]) -> List[Dict[str, Any]]:
        """Get registered matches for a list of item IDs.
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            List of registered matches
        """
        if not item_ids:
            return []
            
        # Create placeholders for query
        placeholders = ', '.join([f":item_id_{i}" for i in range(len(item_ids))])
        
        query = text(f"""
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
                am.source_item_id IN ({placeholders})
                OR am.target_item_id IN ({placeholders})
        """)
        
        params = {}
        for i, item_id in enumerate(item_ids):
            params[f"item_id_{i}"] = item_id
            
        results = self.db_session.execute(query, params).fetchall()
        
        matches = []
        for result in results:
            match_id, source_item_id, target_item_id, match_type, confidence, difference, created_at, source_doc_id, target_doc_id, source_amount, target_amount = result
            
            matches.append({
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
                'target_amount': float(target_amount) if target_amount is not None else None,
                'is_registered': True
            })
        
        return matches
    
    def _get_registered_item_matches(self, item_id: int) -> List[Dict[str, Any]]:
        """Get registered matches for a specific item ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            List of registered matches
        """
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
        """)
        
        results = self.db_session.execute(query, {"item_id": item_id}).fetchall()
        
        matches = []
        for result in results:
            match_id, source_item_id, target_item_id, match_type, confidence, difference, created_at, source_doc_id, target_doc_id, source_amount, target_amount = result
            
            # Determine direction
            direction = "outgoing" if source_item_id == item_id else "incoming"
            
            matches.append({
                'match_id': match_id,
                'source_item_id': source_item_id,
                'target_item_id': target_item_id,
                'match_type': match_type,
                'direction': direction,
                'confidence': float(confidence) if confidence is not None else None,
                'difference': float(difference) if difference is not None else None,
                'created_at': created_at.isoformat() if created_at else None,
                'source_doc_id': source_doc_id,
                'target_doc_id': target_doc_id,
                'source_amount': float(source_amount) if source_amount is not None else None,
                'target_amount': float(target_amount) if target_amount is not None else None,
                'is_registered': True
            })
        
        return matches
    
    def _identify_item_relationships(self, matches: List[Dict[str, Any]], 
                               documents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential relationships between line items.
        
        Args:
            matches: List of matches sorted by date
            documents: Dictionary of document details
            
        Returns:
            List of potential item relationships
        """
        relationships = []
        
        # Need at least 2 matches to identify relationships
        if len(matches) < 2:
            return relationships
            
        # For each pair of matches, check if they might be related
        for i in range(len(matches) - 1):
            for j in range(i + 1, len(matches)):
                source_match = matches[i]
                target_match = matches[j]
                
                # Skip if missing required data
                if (not source_match.get('item_id') or 
                    not target_match.get('item_id') or 
                    not source_match.get('date') or 
                    not target_match.get('date')):
                    continue
                    
                # Parse dates
                source_date = None
                if source_match['date']:
                    try:
                        source_date = datetime.fromisoformat(source_match['date'].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue
                
                target_date = None
                if target_match['date']:
                    try:
                        target_date = datetime.fromisoformat(target_match['date'].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue
                
                # Skip if dates can't be parsed
                if not source_date or not target_date:
                    continue
                    
                # Ensure chronological order
                if source_date > target_date:
                    source_match, target_match = target_match, source_match
                    source_date, target_date = target_date, source_date
                
                # Calculate time difference
                time_diff_days = (target_date - source_date).days
                
                # Skip if too far apart in time
                if time_diff_days > 365:  # More than a year
                    continue
                    
                # Check if the items have the same amount
                amount_diff = abs(source_match['amount'] - target_match['amount'])
                amount_pct_diff = amount_diff / source_match['amount'] if source_match['amount'] else 1.0
                
                # Calculate confidence based on various factors
                confidence = self._calculate_item_relationship_confidence(
                    source_match, target_match, 
                    amount_pct_diff, time_diff_days
                )
                
                if confidence >= self.min_confidence:
                    relationship = {
                        'source_item_id': source_match['item_id'],
                        'target_item_id': target_match['item_id'],
                        'source_doc_id': source_match['doc_id'],
                        'target_doc_id': target_match['doc_id'],
                        'source_amount': source_match['amount'],
                        'target_amount': target_match['amount'],
                        'time_diff_days': time_diff_days,
                        'amount_diff': amount_diff,
                        'match_type': 'exact' if amount_pct_diff <= 0.01 else 'fuzzy',
                        'confidence': confidence,
                        'is_registered': False
                    }
                    
                    relationships.append(relationship)
        
        return relationships
    
    def _find_similar_items(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find items similar to a given item.
        
        Args:
            item: Line item
            
        Returns:
            List of similar items
        """
        # Skip if missing amount
        if item.get('amount') is None:
            return []
            
        # Find items with similar amounts
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.cost_code,
                li.category,
                li.status,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN 
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.item_id != :item_id
                AND li.amount BETWEEN :amount_min AND :amount_max
            ORDER BY 
                ABS(li.amount - :amount)
        """)
        
        # Use 5% tolerance for similar items
        tolerance = item['amount'] * 0.05
        amount_min = item['amount'] - tolerance
        amount_max = item['amount'] + tolerance
        
        results = self.db_session.execute(
            query, 
            {
                "item_id": item['item_id'], 
                "amount": item['amount'], 
                "amount_min": amount_min, 
                "amount_max": amount_max
            }
        ).fetchall()
        
        similar_items = []
        for result in results:
            item_id, doc_id, description, amount, cost_code, category, status, doc_type, party, date_created = result
            
            similar_items.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(amount) if amount is not None else None,
                'cost_code': cost_code,
                'category': category,
                'status': status,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created
            })
        
        return similar_items
    
    def _identify_related_document_items(self, item: Dict[str, Any], 
                                    document_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify items in the same document that might be related.
        
        Args:
            item: Line item
            document_items: List of all items in the document
            
        Returns:
            List of related items
        """
        related_items = []
        
        # Skip if missing amount
        if item.get('amount') is None:
            return related_items
            
        for other_item in document_items:
            # Skip self-comparison
            if other_item['item_id'] == item['item_id']:
                continue
                
            # Skip if missing amount
            if other_item.get('amount') is None:
                continue
                
            # Check for potential relationships
            
            # Potential components (items that sum to this item's amount)
            if abs(other_item['amount'] * 2 - item['amount']) / item['amount'] < 0.01:
                # Other item is roughly half of this item
                related_items.append({
                    'item_id': other_item['item_id'],
                    'relationship_type': 'potential_component',
                    'amount': other_item['amount'],
                    'reason': f"Amount (${other_item['amount']}) is approximately half of ${item['amount']}"
                })
            
            # Potential parent (item is a component of another item)
            if abs(item['amount'] * 2 - other_item['amount']) / other_item['amount'] < 0.01:
                # This item is roughly half of other item
                related_items.append({
                    'item_id': other_item['item_id'],
                    'relationship_type': 'potential_parent',
                    'amount': other_item['amount'],
                    'reason': f"Amount (${other_item['amount']}) is approximately double ${item['amount']}"
                })
                
            # Check if items have the same cost code
            if item.get('cost_code') and other_item.get('cost_code') and item['cost_code'] == other_item['cost_code']:
                related_items.append({
                    'item_id': other_item['item_id'],
                    'relationship_type': 'same_cost_code',
                    'amount': other_item['amount'],
                    'cost_code': other_item['cost_code'],
                    'reason': f"Items share the same cost code: {other_item['cost_code']}"
                })
        
        return related_items
    
    def _identify_potential_item_relationships(self, item: Dict[str, Any], 
                                         similar_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential relationships with similar items.
        
        Args:
            item: Line item
            similar_items: List of similar items
            
        Returns:
            List of potential item relationships
        """
        relationships = []
        
        # Skip if missing required data
        if not item.get('item_id') or not item.get('amount') or not item.get('date'):
            return relationships
            
        # Parse date
        item_date = None
        if item['date']:
            try:
                item_date = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return relationships
        
        if not item_date:
            return relationships
            
        for similar_item in similar_items:
            # Skip if missing required data
            if not similar_item.get('item_id') or not similar_item.get('amount') or not similar_item.get('date'):
                continue
                
            # Parse date
            similar_date = None
            if similar_item['date']:
                try:
                    similar_date = datetime.fromisoformat(similar_item['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    continue
            
            if not similar_date:
                continue
                
            # Determine source and target based on chronology
            if item_date <= similar_date:
                source_item_id = item['item_id']
                target_item_id = similar_item['item_id']
                source_doc_id = item['doc_id']
                target_doc_id = similar_item['doc_id']
                source_amount = item['amount']
                target_amount = similar_item['amount']
                time_diff_days = (similar_date - item_date).days
            else:
                source_item_id = similar_item['item_id']
                target_item_id = item['item_id']
                source_doc_id = similar_item['doc_id']
                target_doc_id = item['doc_id']
                source_amount = similar_item['amount']
                target_amount = item['amount']
                time_diff_days = (item_date - similar_date).days
            
            # Calculate amount difference
            amount_diff = abs(source_amount - target_amount)
            amount_pct_diff = amount_diff / source_amount if source_amount else 1.0
            
            # Calculate confidence
            confidence = self._calculate_item_relationship_confidence(
                {
                    'item_id': source_item_id,
                    'doc_id': source_doc_id,
                    'amount': source_amount,
                    'doc_type': item.get('doc_type') if source_item_id == item['item_id'] else similar_item.get('doc_type')
                },
                {
                    'item_id': target_item_id,
                    'doc_id': target_doc_id,
                    'amount': target_amount,
                    'doc_type': similar_item.get('doc_type') if target_item_id == similar_item['item_id'] else item.get('doc_type')
                },
                amount_pct_diff, 
                time_diff_days
            )
            
            if confidence >= self.min_confidence:
                relationship = {
                    'source_item_id': source_item_id,
                    'target_item_id': target_item_id,
                    'source_doc_id': source_doc_id,
                    'target_doc_id': target_doc_id,
                    'source_amount': source_amount,
                    'target_amount': target_amount,
                    'time_diff_days': time_diff_days,
                    'amount_diff': amount_diff,
                    'match_type': 'exact' if amount_pct_diff <= 0.01 else 'fuzzy',
                    'confidence': confidence,
                    'is_registered': False
                }
                
                relationships.append(relationship)
        
        return relationships
    
    def _check_item_relationship(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if two items might have a parent-child relationship.
        
        Args:
            item1: First line item
            item2: Second line item
            
        Returns:
            Relationship dictionary or None
        """
        # Skip if missing amounts
        if item1.get('amount') is None or item2.get('amount') is None:
            return None
            
        # Check for potential parent-child relationships
        
        # Check if one item's amount is roughly double the other
        if abs(item1['amount'] * 2 - item2['amount']) / item2['amount'] < 0.01:
            # Item1 is roughly half of item2
            return {
                'parent_item_id': item2['item_id'],
                'child_item_id': item1['item_id'],
                'parent_amount': item2['amount'],
                'child_amount': item1['amount'],
                'relationship_type': 'parent_child',
                'confidence': 0.9,
                'explanation': f"Item {item1['item_id']} (${item1['amount']}) appears to be half of item {item2['item_id']} (${item2['amount']})"
            }
        elif abs(item2['amount'] * 2 - item1['amount']) / item1['amount'] < 0.01:
            # Item2 is roughly half of item1
            return {
                'parent_item_id': item1['item_id'],
                'child_item_id': item2['item_id'],
                'parent_amount': item1['amount'],
                'child_amount': item2['amount'],
                'relationship_type': 'parent_child',
                'confidence': 0.9,
                'explanation': f"Item {item2['item_id']} (${item2['amount']}) appears to be half of item {item1['item_id']} (${item1['amount']})"
            }
        
        # Check if the sum of one amount plus a common percentage is close to the other
        common_percentages = [0.05, 0.0625, 0.07, 0.08, 0.1, 0.15, 0.2]
        
        for pct in common_percentages:
            # Check if item2 = item1 + percentage
            calculated = item1['amount'] * (1 + pct)
            if abs(calculated - item2['amount']) / item2['amount'] < 0.01:
                return {
                    'base_item_id': item1['item_id'],
                    'markup_item_id': item2['item_id'],
                    'base_amount': item1['amount'],
                    'markup_amount': item2['amount'],
                    'markup_percentage': pct * 100,
                    'relationship_type': 'markup',
                    'confidence': 0.9,
                    'explanation': f"Item {item2['item_id']} (${item2['amount']}) appears to be item {item1['item_id']} (${item1['amount']}) plus {pct*100}%"
                }
            
            # Check if item1 = item2 + percentage
            calculated = item2['amount'] * (1 + pct)
            if abs(calculated - item1['amount']) / item1['amount'] < 0.01:
                return {
                    'base_item_id': item2['item_id'],
                    'markup_item_id': item1['item_id'],
                    'base_amount': item2['amount'],
                    'markup_amount': item1['amount'],
                    'markup_percentage': pct * 100,
                    'relationship_type': 'markup',
                    'confidence': 0.9,
                    'explanation': f"Item {item1['item_id']} (${item1['amount']}) appears to be item {item2['item_id']} (${item2['amount']}) plus {pct*100}%"
                }
        
        # No relationship detected
        return None
    
    def _group_items_by_amount(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group line items by amount.
        
        Args:
            items: List of line items
            
        Returns:
            List of amount groups
        """
        # Skip if no items
        if not items:
            return []
            
        # Group by amount (within a small tolerance)
        amount_groups = {}
        
        for item in items:
            # Skip if missing amount
            if item.get('amount') is None:
                continue
                
            # Round to 2 decimal places to handle floating point issues
            amount = round(item['amount'] * 100) / 100
            
            # Check if this amount is already in a group (within tolerance)
            found_group = False
            for group_amount in list(amount_groups.keys()):
                if abs(amount - group_amount) / group_amount < 0.0001:  # Tiny tolerance for rounding
                    amount_groups[group_amount].append(item)
                    found_group = True
                    break
            
            if not found_group:
                amount_groups[amount] = [item]
        
        # Convert to list of groups
        groups = []
        for amount, group_items in amount_groups.items():
            # Only include groups with multiple items
            if len(group_items) > 1:
                groups.append({
                    'amount': amount,
                    'count': len(group_items),
                    'items': group_items
                })
        
        # Sort by count (most frequent first)
        groups.sort(key=lambda x: x['count'], reverse=True)
        
        return groups
    
    def _calculate_item_relationship_confidence(self, source_item: Dict[str, Any], 
                                           target_item: Dict[str, Any],
                                           amount_pct_diff: float,
                                           time_diff_days: int) -> float:
        """Calculate confidence score for an item relationship.
        
        Args:
            source_item: Source item
            target_item: Target item
            amount_pct_diff: Percentage difference in amounts
            time_diff_days: Time difference in days
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        if amount_pct_diff <= 0.01:
            # Exact match
            base_confidence = 0.9
        elif amount_pct_diff <= 0.05:
            # Close match
            base_confidence = 0.8
        else:
            # Fuzzy match
            base_confidence = 0.7
        
        # Adjust for time difference
        if time_diff_days <= 7:
            # Very close in time
            time_factor = 0.1
        elif time_diff_days <= 30:
            # Within a month
            time_factor = 0.05
        elif time_diff_days <= 90:
            # Within a quarter
            time_factor = 0.0
        else:
            # More than a quarter apart
            time_factor = -0.1
        
        # Adjust for document type
        doc_type_factor = 0.0
        
        if source_item.get('doc_type') and target_item.get('doc_type'):
            # Both document types are known
            source_type = source_item['doc_type'].lower() if source_item['doc_type'] else ''
            target_type = target_item['doc_type'].lower() if target_item['doc_type'] else ''
            
            # Common document type progressions
            if (source_type == 'change_order' and target_type == 'payment_application'):
                doc_type_factor = 0.1
            elif (source_type == 'invoice' and target_type == 'payment'):
                doc_type_factor = 0.1
            elif (source_type == 'proposal' and target_type == 'contract'):
                doc_type_factor = 0.1
            elif (source_type == target_type):
                # Same document type
                doc_type_factor = 0.05
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + time_factor + doc_type_factor)
        
        return confidence
    
    def _register_new_item_matches(self, potential_relationships: List[Dict[str, Any]], 
                             existing_matches: List[Dict[str, Any]]) -> None:
        """Register new item matches in the database.
        
        Args:
            potential_relationships: List of potential relationships
            existing_matches: List of existing matches
        """
        # Create a set of existing match keys
        existing_keys = set()
        for match in existing_matches:
            key = (match['source_item_id'], match['target_item_id'])
            existing_keys.add(key)
        
        # Register new relationships
        for rel in potential_relationships:
            key = (rel['source_item_id'], rel['target_item_id'])
            
            # Skip if already exists
            if key in existing_keys:
                continue
                
            # Register relationship
            if rel['confidence'] >= self.min_confidence:
                self._register_amount_match(
                    rel['source_item_id'],
                    rel['target_item_id'],
                    rel['match_type'],
                    rel['confidence']
                )
    
    def _register_amount_match(self, source_item_id: int, target_item_id: int, 
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