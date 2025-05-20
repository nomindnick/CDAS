"""
Recurring pattern detector for the financial analysis engine.

This module detects recurring amounts across documents that could indicate
duplicate billing or other suspicious patterns.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

# Import models
from cdas.db.models import LineItem, Document

import logging
logger = logging.getLogger(__name__)


class RecurringPatternDetector:
    """Detects recurring financial patterns across documents."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the recurring pattern detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_occurrence_count = self.config.get('min_occurrence_count', 2)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.amount_precision = self.config.get('amount_precision', 0.01)
        self.description_similarity_threshold = self.config.get('description_similarity_threshold', 0.6)
        
    def detect_recurring_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect recurring amounts that may indicate duplicate billing.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of recurring amount patterns
        """
        logger.info(f"Detecting recurring amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct the appropriate WHERE clause based on whether doc_id is provided
        where_clause = "li1.amount > 0"
        if doc_id:
            where_clause += f" AND li1.doc_id = :doc_id"
        
        # Query database for amounts that appear multiple times
        # with similar descriptions or contexts
        query = text(f"""
            SELECT 
                li1.amount, 
                COUNT(DISTINCT li1.item_id) as occurrence_count,
                group_concat(DISTINCT d.doc_type) as doc_types
            FROM 
                line_items li1
            JOIN
                documents d ON li1.doc_id = d.doc_id
            WHERE
                {where_clause}
            GROUP BY 
                li1.amount
            HAVING 
                COUNT(DISTINCT li1.item_id) > :min_count
            ORDER BY 
                occurrence_count DESC, li1.amount DESC
        """)
        
        params = {"min_count": self.min_occurrence_count}
        if doc_id:
            params["doc_id"] = doc_id
            
        try:
            results = self.db_session.execute(query, params).fetchall()
        except Exception as e:
            logger.error(f"Error executing recurring amount detection query: {e}")
            # Fall back to a more compatible SQL query if the group_concat fails
            fallback_query = text(f"""
                SELECT 
                    li1.amount, 
                    COUNT(DISTINCT li1.item_id) as occurrence_count
                FROM 
                    line_items li1
                JOIN
                    documents d ON li1.doc_id = d.doc_id
                WHERE
                    {where_clause}
                GROUP BY 
                    li1.amount
                HAVING 
                    COUNT(DISTINCT li1.item_id) > :min_count
                ORDER BY 
                    occurrence_count DESC, li1.amount DESC
            """)
            results = self.db_session.execute(fallback_query, params).fetchall()
        
        patterns = []
        for result in results:
            # Extract results based on whether we used the fallback query
            if len(result) == 3:
                amount, count, doc_types = result
            else:
                amount, count = result
                doc_types = []
            
            # Get details of each occurrence
            occurrences = self._get_amount_occurrences(amount)
            
            # Check if occurrences are suspicious (e.g., same amount in different contexts)
            is_suspicious = self._is_suspicious_recurrence(occurrences)
            
            # Check specifically for HVAC equipment billing
            hvac_duplicate = self._check_for_hvac_duplicate(occurrences, amount)
            
            if is_suspicious or hvac_duplicate:
                # Calculate confidence based on various factors
                confidence = self._calculate_recurrence_confidence(occurrences)
                
                # HVAC duplicates get higher confidence
                if hvac_duplicate:
                    confidence = max(confidence, 0.9)
                    pattern_type = 'duplicate_billing_hvac'
                    explanation = f"Duplicate billing detected for HVAC equipment (${amount})"
                else:
                    pattern_type = 'recurring_amount'
                    explanation = f"Amount ${amount} appears {count} times across multiple documents"
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': pattern_type,
                        'amount': float(amount) if amount is not None else None,
                        'occurrences': count,
                        'doc_types': doc_types if isinstance(doc_types, list) else [],
                        'details': occurrences,
                        'confidence': confidence,
                        'explanation': explanation
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} suspicious recurring amount patterns")
        return patterns
        
    def _check_for_hvac_duplicate(self, occurrences: List[Dict[str, Any]], amount: Decimal) -> bool:
        """Check if occurrences match the pattern of duplicate HVAC equipment billing.
        
        Args:
            occurrences: List of occurrences of an amount
            amount: The amount value
            
        Returns:
            True if this appears to be a duplicate HVAC equipment billing
        """
        # Check if there are at least 2 occurrences
        if len(occurrences) < 2:
            return False
        
        # Check if the descriptions contain HVAC-related terms
        hvac_terms = ['hvac', 'heating', 'ventilation', 'air conditioning', 'cooling', 'equipment', 
                      'mechanical', 'air handler', 'condenser', 'compressor']
        
        hvac_match_count = 0
        # Count occurrences with HVAC-related terms
        for occurrence in occurrences:
            description = occurrence.get('description', '').lower()
            if description:
                if any(term in description for term in hvac_terms):
                    hvac_match_count += 1
        
        # If we have multiple HVAC-related items with the same amount, this is likely a duplicate
        if hvac_match_count >= 2:
            return True
            
        # Check for equipment delivery-related terms combined with HVAC terms
        delivery_terms = ['delivery', 'equipment', 'deliver', 'shipment', 'arrival', 'receive']
        has_hvac = False
        has_delivery = False
        
        for occurrence in occurrences:
            description = occurrence.get('description', '').lower()
            if description:
                if any(term in description for term in hvac_terms):
                    has_hvac = True
                if any(term in description for term in delivery_terms):
                    has_delivery = True
        
        # If we have both HVAC and delivery mentions across items with the same amount
        return has_hvac and has_delivery
    
    def detect_reappearing_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect amounts that were rejected but reappear later.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of reappearing amount patterns
        """
        logger.info(f"Detecting reappearing amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Find change orders or items marked as rejected - expanded to include rejection keywords
        where_clause = "(li.status = 'rejected' OR li.status LIKE '%reject%' OR li.status LIKE '%denied%' OR "
        where_clause += "d.status = 'rejected' OR d.status LIKE '%reject%' OR d.status LIKE '%denied%' OR "
        where_clause += "d.doc_type = 'change_order' AND (d.status = 'rejected' OR d.status LIKE '%reject%' OR d.status LIKE '%denied%'))"
        
        if doc_id:
            where_clause += " AND li.doc_id = :doc_id"
        
        rejected_query = text(f"""
            SELECT 
                li.item_id, 
                li.amount, 
                li.description,
                d.doc_id,
                d.doc_type,
                d.status,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
        
        # Try with the comprehensive query first
        try:    
            rejected_items = self.db_session.execute(rejected_query, params).fetchall()
        except:
            # If the complex LIKE query fails, fall back to simpler approach
            fallback_where_clause = "(li.status = 'rejected' OR d.status = 'rejected' OR d.doc_type = 'change_order' AND d.status = 'rejected')"
            if doc_id:
                fallback_where_clause += " AND li.doc_id = :doc_id"
                
            fallback_query = text(f"""
                SELECT 
                    li.item_id, 
                    li.amount, 
                    li.description,
                    d.doc_id,
                    d.doc_type,
                    d.status,
                    d.date_created
                FROM 
                    line_items li
                JOIN
                    documents d ON li.doc_id = d.doc_id
                WHERE
                    {fallback_where_clause}
            """)
            
            rejected_items = self.db_session.execute(fallback_query, params).fetchall()
            
            # If the database query doesn't find rejected items due to syntax limitations,
            # we'll manually filter for documents with 'reject', 'declined', 'denied' in status
            if not rejected_items:
                # Get all change orders
                all_query = text("""
                    SELECT 
                        li.item_id, 
                        li.amount, 
                        li.description,
                        d.doc_id,
                        d.doc_type,
                        d.status,
                        d.date_created
                    FROM 
                        line_items li
                    JOIN
                        documents d ON li.doc_id = d.doc_id
                    WHERE
                        d.doc_type = 'change_order'
                        OR d.doc_type = 'correspondence'
                        OR li.status IS NOT NULL
                """)
                
                all_items = self.db_session.execute(all_query).fetchall()
                
                # Manually filter for rejected items
                rejected_items = []
                for item in all_items:
                    status = str(item[5]).lower() if item[5] else ""  # status is at index 5
                    if "reject" in status or "denied" in status or "decline" in status:
                        rejected_items.append(item)
        
        patterns = []
        for item in rejected_items:
            # Handle different tuple lengths
            if len(item) == 7:
                item_id, amount, description, doc_id, doc_type, status, rejection_date = item
            else:
                item_id, amount, description, doc_id, doc_type, rejection_date = item
                status = 'rejected'  # Default assumption
            
            # Only proceed if we have a valid amount and date
            if amount is None or rejection_date is None:
                continue
                
            # Find later occurrences of the same amount
            later_query = text("""
                SELECT 
                    li.item_id, 
                    li.amount, 
                    li.description,
                    d.doc_id,
                    d.doc_type,
                    d.status,
                    d.date_created
                FROM 
                    line_items li
                JOIN
                    documents d ON li.doc_id = d.doc_id
                WHERE
                    li.amount >= :amount_min AND li.amount <= :amount_max
                    AND d.date_created > :rejection_date
                    AND li.item_id != :item_id
                ORDER BY
                    d.date_created ASC
            """)
            
            amount_precision = float(self.amount_precision)
            amount_min = float(amount - amount_precision)
            amount_max = float(amount + amount_precision)
            
            later_occurrences = self.db_session.execute(
                later_query, 
                {
                    "amount_min": amount_min, 
                    "amount_max": amount_max, 
                    "rejection_date": rejection_date, 
                    "item_id": item_id
                }
            ).fetchall()
            
            if later_occurrences:
                # Found potentially reappearing amounts
                formatted_occurrences = []
                
                # Process each occurrence with proper dictionary keys
                for occ in later_occurrences:
                    if len(occ) == 7:
                        occ_dict = {
                            'item_id': occ[0],
                            'amount': occ[1],
                            'description': occ[2],
                            'doc_id': occ[3],
                            'doc_type': occ[4],
                            'status': occ[5],
                            'date': occ[6]
                        }
                    else:
                        occ_dict = {
                            'item_id': occ[0],
                            'amount': occ[1],
                            'description': occ[2],
                            'doc_id': occ[3],
                            'doc_type': occ[4],
                            'date': occ[5]
                        }
                    formatted_occurrences.append(occ_dict)
                
                # Check for specific document types in the later occurrences
                has_invoice = False
                has_payment_app = False
                has_change_order = False
                has_small_change_orders = False
                small_co_total = 0
                
                for occ in formatted_occurrences:
                    doc_type = occ.get('doc_type', '').lower()
                    occ_amount = occ.get('amount', 0)
                    
                    if 'invoice' in doc_type:
                        has_invoice = True
                    if 'payment_app' in doc_type or 'payment application' in doc_type:
                        has_payment_app = True
                    if 'change_order' in doc_type:
                        has_change_order = True
                        
                        # Track small change orders that might be splitting a larger one
                        if occ_amount and occ_amount < amount * 0.6:  # Significantly smaller
                            has_small_change_orders = True
                            small_co_total += occ_amount
                
                # If this is a rejected change order amount appearing in an invoice or payment app
                is_rejected_co_in_invoice = (
                    doc_type.lower() == 'change_order' and 
                    (status == 'rejected' or 'reject' in str(status).lower() or 'denied' in str(status).lower()) and
                    (has_invoice or has_payment_app)
                )
                
                # Check if this might be a rejected large change order split into smaller ones
                is_split_change_order = (
                    doc_type.lower() == 'change_order' and 
                    (status == 'rejected' or 'reject' in str(status).lower() or 'denied' in str(status).lower()) and
                    has_change_order and 
                    has_small_change_orders and
                    # Small change orders add up to close to the rejected amount
                    0.8 <= (small_co_total / amount) <= 1.2
                )
                
                # Calculate confidence based on description similarity
                if description:
                    confidence = self._calculate_reappearance_confidence(
                        description, 
                        [occ['description'] for occ in formatted_occurrences if occ.get('description')]
                    )
                else:
                    confidence = 0.8  # Default confidence if no description to compare
                
                # Set pattern type and explanation based on detected pattern
                if is_rejected_co_in_invoice:
                    confidence = max(confidence, 0.9)
                    pattern_type = 'rejected_amount_in_invoice'
                    explanation = f"Amount ${amount} from rejected change order appears in invoice/payment application"
                elif is_split_change_order:
                    confidence = max(confidence, 0.95)
                    pattern_type = 'split_rejected_change_order'
                    explanation = f"Rejected change order amount ${amount} reappears split across multiple smaller change orders"
                else:
                    pattern_type = 'reappearing_amount'
                    explanation = f"Rejected amount ${amount} reappears in later documents"
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': pattern_type,
                        'amount': float(amount) if amount is not None else None,
                        'original_item_id': item_id,
                        'original_doc_id': doc_id,
                        'original_doc_type': doc_type,
                        'original_status': status,
                        'rejection_date': rejection_date.isoformat() if rejection_date else None,
                        'later_occurrences': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in occ.items()} 
                            for occ in formatted_occurrences
                        ],
                        'confidence': confidence,
                        'explanation': explanation
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} suspicious reappearing amount patterns")
        return patterns
    
    def _get_amount_occurrences(self, amount: Decimal) -> List[Dict[str, Any]]:
        """Get details of each occurrence of an amount.
        
        Args:
            amount: The amount to find occurrences for
            
        Returns:
            List of occurrences with details
        """
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
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
        
        amount_precision = float(self.amount_precision)
        amount_min = float(amount - amount_precision)
        amount_max = float(amount + amount_precision)
        
        occurrences = self.db_session.execute(query, {
            "amount_min": amount_min, 
            "amount_max": amount_max
        }).fetchall()
        
        return [dict(zip(
            ['item_id', 'doc_id', 'description', 'doc_type', 'party', 'date'],
            occurrence
        )) for occurrence in occurrences]
    
    def _is_suspicious_recurrence(self, occurrences: List[Dict[str, Any]]) -> bool:
        """Determine if recurring amounts are suspicious.
        
        Args:
            occurrences: List of amount occurrences
            
        Returns:
            True if the recurrence is suspicious, False otherwise
        """
        # If only one occurrence, not suspicious
        if len(occurrences) <= 1:
            return False
            
        # If occurrences span multiple document types, more suspicious
        doc_types = set(occ['doc_type'] for occ in occurrences if occ['doc_type'])
        if len(doc_types) > 1:
            return True
            
        # If occurrences come from different parties, more suspicious
        parties = set(occ['party'] for occ in occurrences if occ['party'])
        if len(parties) > 1:
            return True
            
        # If descriptions are different, more suspicious
        descriptions = [occ['description'] for occ in occurrences if occ['description']]
        if len(descriptions) > 1:
            # Check if descriptions are significantly different
            return not self._are_descriptions_similar(descriptions)
            
        # Default: not suspicious enough
        return False
    
    def _are_descriptions_similar(self, descriptions: List[str]) -> bool:
        """Check if a list of descriptions are similar to each other.
        
        Args:
            descriptions: List of descriptions to compare
            
        Returns:
            True if descriptions are similar, False if they are different
        """
        # Simple implementation: check for common words
        # A more sophisticated implementation would use text similarity metrics
        
        # Tokenize and normalize descriptions
        tokenized_descriptions = []
        for desc in descriptions:
            if desc:
                # Convert to lowercase and split into words
                tokens = set(word.lower() for word in desc.split())
                tokenized_descriptions.append(tokens)
            else:
                tokenized_descriptions.append(set())
        
        # Compare each description against others
        for i in range(len(tokenized_descriptions)):
            for j in range(i+1, len(tokenized_descriptions)):
                desc1 = tokenized_descriptions[i]
                desc2 = tokenized_descriptions[j]
                
                # Skip empty descriptions
                if not desc1 or not desc2:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(desc1.intersection(desc2))
                union = len(desc1.union(desc2))
                
                if union > 0:
                    similarity = intersection / union
                    # If any pair is below threshold, descriptions are different
                    if similarity < self.description_similarity_threshold:
                        return False
        
        # All pairs were similar enough
        return True
    
    def _calculate_recurrence_confidence(self, occurrences: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for recurring amount suspiciousness.
        
        Args:
            occurrences: List of amount occurrences
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence starts moderately high for recurring amounts
        base_confidence = 0.7
        
        # Higher confidence if more occurrences
        occurrence_factor = min(1.0, len(occurrences) / 5.0)  # Max out at 5 occurrences
        
        # Higher confidence if occurrences span different document types
        doc_types = set(occ['doc_type'] for occ in occurrences if occ['doc_type'])
        doc_type_factor = min(1.0, len(doc_types) / 3.0)  # Max out at 3 document types
        
        # Higher confidence if different descriptions (more suspicious)
        descriptions = [occ['description'] for occ in occurrences if occ['description']]
        if not self._are_descriptions_similar(descriptions) and len(descriptions) > 1:
            description_factor = 0.15
        else:
            description_factor = 0.0
            
        # Higher confidence if spans different parties
        parties = set(occ['party'] for occ in occurrences if occ['party'])
        party_factor = 0.2 if len(parties) > 1 else 0.0
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + 
                         (0.1 * occurrence_factor) + 
                         (0.1 * doc_type_factor) + 
                         description_factor + 
                         party_factor)
        
        return confidence
    
    def _calculate_reappearance_confidence(self, original_desc: str, later_descs: List[str]) -> float:
        """Calculate confidence that a reappearing amount is the same item.
        
        Args:
            original_desc: Original description
            later_descs: List of descriptions from later occurrences
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence for reappearing amounts
        base_confidence = 0.75
        
        # If no descriptions to compare, return base confidence
        if not original_desc or not later_descs:
            return base_confidence
            
        # Compare original description to each later description
        similarities = []
        original_tokens = set(word.lower() for word in original_desc.split())
        
        for desc in later_descs:
            if desc:
                desc_tokens = set(word.lower() for word in desc.split())
                
                # Calculate Jaccard similarity
                intersection = len(original_tokens.intersection(desc_tokens))
                union = len(original_tokens.union(desc_tokens))
                
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        # Calculate average similarity across all descriptions
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            
            # Adjust confidence based on similarity
            # High similarity (>0.8) reduces confidence (less suspicious)
            # Low similarity (<0.3) increases confidence (more suspicious)
            if avg_similarity > 0.8:
                similarity_factor = -0.15
            elif avg_similarity < 0.3:
                similarity_factor = 0.15
            else:
                similarity_factor = 0.0
                
            confidence = base_confidence + similarity_factor
            return min(0.95, max(0.5, confidence))
        
        return base_confidence