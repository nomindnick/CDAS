"""
Context-aware amount matcher for the financial analysis engine.

This module provides context-aware matching of amounts across different documents,
taking into account the semantic context (descriptions, cost codes, etc.)
in addition to the amount values.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class ContextMatcher:
    """Matches amounts with context awareness across documents."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the context-aware amount matcher.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.default_amount_tolerance = self.config.get('default_amount_tolerance', 0.05)  # 5% tolerance
        self.default_similarity_threshold = self.config.get('default_similarity_threshold', 0.6)  # 60% similarity
        self.match_across_parties = self.config.get('match_across_parties', True)
        self.match_across_doc_types = self.config.get('match_across_doc_types', True)
        
        # Weight factors for different context elements
        self.amount_weight = self.config.get('amount_weight', 0.4)
        self.description_weight = self.config.get('description_weight', 0.3)
        self.cost_code_weight = self.config.get('cost_code_weight', 0.2)
        self.doc_type_weight = self.config.get('doc_type_weight', 0.1)
        
    def find_matches(self, amount: float, description: str, 
                   similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find matches based on amount and contextual description similarity.
        
        Args:
            amount: Amount to match
            description: Description to match
            similarity_threshold: Threshold for overall similarity (defaults to self.default_similarity_threshold)
            
        Returns:
            List of context-aware matches
        """
        logger.info(f"Finding context matches for amount: {amount}, description: '{description}'")
        
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        
        # First, get potential matches based on amount
        potential_matches = self._get_potential_matches_by_amount(amount)
        
        # Calculate similarity for each potential match
        matches_with_similarity = []
        for match in potential_matches:
            # Calculate overall similarity score
            similarity = self._calculate_similarity(
                amount, description, 
                match['amount'], match['description'], 
                match['cost_code'], match['doc_type']
            )
            
            # Add similarity score to match
            match_with_similarity = {**match, 'similarity': similarity}
            matches_with_similarity.append(match_with_similarity)
        
        # Filter by similarity threshold
        filtered_matches = [
            match for match in matches_with_similarity 
            if match['similarity'] >= similarity_threshold
        ]
        
        # Sort by similarity (highest first)
        sorted_matches = sorted(filtered_matches, key=lambda x: x['similarity'], reverse=True)
        
        # Format for output
        result = []
        for match in sorted_matches:
            result.append({
                'item_id': match['item_id'],
                'doc_id': match['doc_id'],
                'description': match['description'],
                'amount': float(match['amount']) if match['amount'] is not None else None,
                'cost_code': match['cost_code'],
                'doc_type': match['doc_type'],
                'party': match['party'],
                'date': match['date'].isoformat() if match['date'] else None,
                'similarity': float(match['similarity']),
                'confidence': float(match['similarity']),  # Use similarity as confidence
                'match_type': 'context'
            })
        
        logger.info(f"Found {len(result)} context matches for amount: {amount}")
        return result
    
    def find_matches_for_item(self, item_id: int, 
                           similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find context-aware matches for a specific line item.
        
        Args:
            item_id: Line item ID
            similarity_threshold: Threshold for overall similarity (defaults to self.default_similarity_threshold)
            
        Returns:
            List of context-aware matches
        """
        logger.info(f"Finding context matches for item ID: {item_id}")
        
        # Get item details
        query = text("""
            SELECT 
                li.amount,
                li.description,
                li.cost_code,
                d.doc_type
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.item_id = :item_id
        """)
        
        result = self.db_session.execute(query, {"item_id": item_id}).fetchone()
        
        if not result:
            logger.warning(f"Item ID {item_id} not found")
            return []
            
        amount, description, cost_code, doc_type = result
        
        # Find matches using the item's attributes
        return self.find_matches(amount, description or "", similarity_threshold)
    
    def find_matches_between_documents(self, doc_id1: str, doc_id2: str, 
                                    similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find context-aware matches between two specific documents.
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            similarity_threshold: Threshold for overall similarity
            
        Returns:
            List of context-aware matches between the documents
        """
        logger.info(f"Finding context matches between documents: {doc_id1} and {doc_id2}")
        
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold
        
        # Get items from first document
        query = text("""
            SELECT 
                li.item_id,
                li.description,
                li.amount,
                li.cost_code,
                d.doc_type
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE 
                li.doc_id = :doc_id
                AND li.amount IS NOT NULL
                AND li.amount > 0
        """)
        
        items1 = self.db_session.execute(query, {"doc_id": doc_id1}).fetchall()
        items2 = self.db_session.execute(query, {"doc_id": doc_id2}).fetchall()
        
        # Convert to dictionaries for easier handling
        items1_dicts = []
        for item_id, description, amount, cost_code, doc_type in items1:
            items1_dicts.append({
                'item_id': item_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code,
                'doc_type': doc_type
            })
            
        items2_dicts = []
        for item_id, description, amount, cost_code, doc_type in items2:
            items2_dicts.append({
                'item_id': item_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code,
                'doc_type': doc_type
            })
        
        # Find matches between the documents
        matches = []
        for item1 in items1_dicts:
            item_matches = []
            
            for item2 in items2_dicts:
                # Calculate similarity
                similarity = self._calculate_similarity(
                    item1['amount'], item1['description'] or "",
                    item2['amount'], item2['description'] or "",
                    item2['cost_code'], item2['doc_type']
                )
                
                if similarity >= similarity_threshold:
                    item_matches.append({
                        'item_id': item2['item_id'],
                        'description': item2['description'],
                        'amount': float(item2['amount']) if item2['amount'] is not None else None,
                        'cost_code': item2['cost_code'],
                        'doc_type': item2['doc_type'],
                        'similarity': float(similarity),
                        'confidence': float(similarity)
                    })
            
            if item_matches:
                # Sort by similarity
                item_matches.sort(key=lambda x: x['similarity'], reverse=True)
                
                matches.append({
                    'item_id': item1['item_id'],
                    'description': item1['description'],
                    'amount': float(item1['amount']) if item1['amount'] is not None else None,
                    'cost_code': item1['cost_code'],
                    'matches': item_matches
                })
        
        logger.info(f"Found {len(matches)} items with context matches between documents")
        return matches
    
    def register_context_match(self, source_item_id: int, target_item_id: int, 
                            similarity: float) -> Dict[str, Any]:
        """Register a context-aware match between two line items in the database.
        
        Args:
            source_item_id: Source line item ID
            target_item_id: Target line item ID
            similarity: Similarity score
            
        Returns:
            Dictionary with match information
        """
        logger.info(f"Registering context match between items {source_item_id} and {target_item_id}")
        
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
                    match_type = 'context',
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
                        "confidence": similarity,
                        "difference": difference
                    }
                ).fetchone()
                
                self.db_session.commit()
                
                return {
                    'match_id': result[0],
                    'source_item_id': source_item_id,
                    'target_item_id': target_item_id,
                    'match_type': 'context',
                    'confidence': float(similarity),
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
                    (:source_id, :target_id, 'context', :confidence, :difference)
                RETURNING match_id
            """)
            
            try:
                result = self.db_session.execute(
                    insert_query, 
                    {
                        "source_id": source_item_id,
                        "target_id": target_item_id,
                        "confidence": similarity,
                        "difference": difference
                    }
                ).fetchone()
                
                self.db_session.commit()
                
                return {
                    'match_id': result[0],
                    'source_item_id': source_item_id,
                    'target_item_id': target_item_id,
                    'match_type': 'context',
                    'confidence': float(similarity),
                    'difference': float(difference) if difference is not None else None,
                    'status': 'created'
                }
            except Exception as e:
                self.db_session.rollback()
                logger.exception(f"Error creating match: {e}")
                return {'error': str(e)}
    
    def _get_potential_matches_by_amount(self, amount: float) -> List[Dict[str, Any]]:
        """Get potential matches based on amount similarity.
        
        Args:
            amount: Amount to match
            
        Returns:
            List of potential matches
        """
        # Calculate range for amount similarity
        min_amount = amount * (1 - self.default_amount_tolerance)
        max_amount = amount * (1 + self.default_amount_tolerance)
        
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
                li.amount >= :min_amount AND li.amount <= :max_amount
            ORDER BY
                ABS(li.amount - :amount)
        """)
        
        matches = self.db_session.execute(
            query, 
            {
                "amount": amount, 
                "min_amount": min_amount, 
                "max_amount": max_amount
            }
        ).fetchall()
        
        # Convert to dictionaries
        result = []
        for match in matches:
            item_id, doc_id, description, match_amount, cost_code, doc_type, party, date_created = match
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': match_amount,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            })
        
        return result
    
    def _calculate_similarity(self, amount1: float, description1: str, 
                          amount2: float, description2: str, 
                          cost_code2: Optional[str], doc_type2: Optional[str]) -> float:
        """Calculate overall similarity between two items based on context.
        
        Args:
            amount1: First amount
            description1: First description
            amount2: Second amount
            description2: Second description
            cost_code2: Second cost code
            doc_type2: Second document type
            
        Returns:
            Overall similarity score between 0.0 and 1.0
        """
        # Calculate amount similarity
        amount_similarity = self._calculate_amount_similarity(amount1, amount2)
        
        # Calculate description similarity
        description_similarity = self._calculate_text_similarity(description1, description2)
        
        # Calculate cost code similarity (if available)
        cost_code_similarity = 0.0
        
        # For simplicity, we'll just use exact matching for cost codes
        # A more sophisticated approach might use a hierarchy of cost codes
        if cost_code2 is not None:
            cost_code_parts1 = description1.lower().split()
            cost_code_parts2 = cost_code2.lower().split()
            
            # Check if any part of the cost code appears in the description
            if any(part in cost_code_parts1 for part in cost_code_parts2):
                cost_code_similarity = 1.0
        
        # Calculate document type similarity (if matching across doc types is enabled)
        doc_type_similarity = 0.0
        if self.match_across_doc_types:
            # Define similarity between different document types based on domain knowledge
            # For simplicity, we'll use a dummy implementation here
            doc_type_similarity = 0.5  # Medium similarity for different doc types
        
        # Calculate weighted average of all similarity scores
        overall_similarity = (
            (self.amount_weight * amount_similarity) +
            (self.description_weight * description_similarity) +
            (self.cost_code_weight * cost_code_similarity) +
            (self.doc_type_weight * doc_type_similarity)
        )
        
        return overall_similarity
    
    def _calculate_amount_similarity(self, amount1: float, amount2: float) -> float:
        """Calculate similarity between two amounts.
        
        Args:
            amount1: First amount
            amount2: Second amount
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if amount1 is None or amount2 is None or amount1 <= 0 or amount2 <= 0:
            return 0.0
            
        # Calculate difference percentage
        difference_percent = abs(amount1 - amount2) / max(amount1, amount2)
        
        # Convert to similarity (higher for lower difference)
        similarity = max(0.0, 1.0 - (difference_percent / self.default_amount_tolerance))
        
        return min(1.0, similarity)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
            
        # Clean and normalize texts
        text1_normalized = self._normalize_text(text1)
        text2_normalized = self._normalize_text(text2)
        
        # Tokenize into words
        words1 = set(text1_normalized.split())
        words2 = set(text2_normalized.split())
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        similarity = len(intersection) / len(union)
        
        return similarity
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace common separators with spaces
        for char in ['-', '_', '/', '\\', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}']:
            text = text.replace(char, ' ')
        
        # Remove duplicate spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
            
        # Trim spaces
        text = text.strip()
        
        return text