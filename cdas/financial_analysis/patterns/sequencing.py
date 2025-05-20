"""
Sequencing pattern detector for the financial analysis engine.

This module detects sequential patterns in financial data, such as gradually
increasing amounts, splitting of larger amounts into multiple smaller ones,
and other patterns that may indicate manipulation of financial values.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class SequencingPatternDetector:
    """Detects sequential patterns in financial data."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the sequencing pattern detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_sequence_length = self.config.get('min_sequence_length', 3)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.max_time_between_sequential_items = self.config.get('max_time_between_sequential_items', 90)  # days
        
    def detect_sequential_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect amounts that follow a sequential pattern.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of sequential amount patterns
        """
        logger.info(f"Detecting sequential amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Detect different types of sequences
        increasing_sequences = self._detect_increasing_sequences(doc_id)
        decreasing_sequences = self._detect_decreasing_sequences(doc_id)
        arithmetic_sequences = self._detect_arithmetic_sequences(doc_id)
        
        # Combine all sequence types
        patterns = []
        patterns.extend(increasing_sequences)
        patterns.extend(decreasing_sequences)
        patterns.extend(arithmetic_sequences)
        
        logger.info(f"Found {len(patterns)} sequential amount patterns")
        return patterns
    
    def detect_split_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect large amounts that are later split into multiple smaller amounts.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of split amount patterns
        """
        logger.info(f"Detecting split amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Get large amounts
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
                li.amount > :min_amount
                :doc_filter
            ORDER BY
                li.amount DESC
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {
            "min_amount": 1000.0,  # Look for amounts above this threshold
            "doc_filter": doc_filter
        }
        if doc_id:
            params["doc_id"] = doc_id
            
        large_items = self.db_session.execute(query, params).fetchall()
        
        patterns = []
        for item in large_items:
            item_id, item_doc_id, description, amount, doc_type, party, date_created = item
            
            # Only proceed if we have a valid amount and date
            if amount is None or date_created is None:
                continue
                
            # Look for smaller amounts that sum close to this amount in later documents
            potential_splits = self._find_potential_splits(amount, date_created, item_id)
            
            if potential_splits:
                # Calculate confidence based on various factors
                confidence = self._calculate_split_confidence(amount, potential_splits, description)
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': 'split_amount',
                        'original_amount': float(amount) if amount is not None else None,
                        'original_item_id': item_id,
                        'original_doc_id': item_doc_id,
                        'original_doc_type': doc_type,
                        'original_date': date_created.isoformat() if date_created else None,
                        'split_items': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in item.items()} 
                            for item in potential_splits['items']
                        ],
                        'split_total': float(potential_splits['total']) if potential_splits['total'] is not None else None,
                        'difference': float(potential_splits['difference']) if potential_splits['difference'] is not None else None,
                        'confidence': confidence,
                        'explanation': f"Amount ${amount} appears to have been split into {len(potential_splits['items'])} smaller amounts"
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} split amount patterns")
        return patterns
    
    def detect_combined_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect multiple small amounts that are later combined into a single larger amount.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of combined amount patterns
        """
        logger.info(f"Detecting combined amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # This is essentially the reverse of detect_split_amounts
        # Find groups of smaller amounts that sum to a similar value
        small_amounts_query = text("""
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
                li.amount > 0
                AND li.amount < :max_amount
                :doc_filter
            ORDER BY
                d.date_created
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {
            "max_amount": 1000.0,  # Look for amounts below this threshold
            "doc_filter": doc_filter
        }
        if doc_id:
            params["doc_id"] = doc_id
            
        small_items = self.db_session.execute(small_amounts_query, params).fetchall()
        
        # Group by document and date
        grouped_items = {}
        for item in small_items:
            item_id, item_doc_id, description, amount, doc_type, party, date_created = item
            
            if not date_created:
                continue
                
            key = f"{item_doc_id}_{date_created.strftime('%Y-%m-%d')}"
            if key not in grouped_items:
                grouped_items[key] = []
                
            grouped_items[key].append({
                'item_id': item_id,
                'doc_id': item_doc_id,
                'description': description,
                'amount': amount,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            })
        
        patterns = []
        # Look for groups of small amounts that may have been combined later
        for key, items in grouped_items.items():
            if len(items) < 2:
                continue
                
            # Calculate sum of amounts in this group
            group_total = sum(item['amount'] for item in items if item['amount'])
            
            # If no meaningful sum, skip
            if group_total <= 0:
                continue
                
            # Find a larger amount close to this sum in later documents
            potential_combination = self._find_potential_combination(group_total, items[0]['date'], [item['item_id'] for item in items])
            
            if potential_combination:
                # Calculate confidence
                confidence = self._calculate_combination_confidence(group_total, potential_combination, items)
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': 'combined_amount',
                        'combined_amount': float(potential_combination['amount']) if potential_combination['amount'] is not None else None,
                        'combined_item_id': potential_combination['item_id'],
                        'combined_doc_id': potential_combination['doc_id'],
                        'combined_doc_type': potential_combination['doc_type'],
                        'combined_date': potential_combination['date'].isoformat() if potential_combination['date'] else None,
                        'source_items': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in item.items()} 
                            for item in items
                        ],
                        'source_total': float(group_total) if group_total is not None else None,
                        'difference': float(potential_combination['difference']) if potential_combination['difference'] is not None else None,
                        'confidence': confidence,
                        'explanation': f"{len(items)} smaller amounts totaling ${group_total} appear to have been combined into a single amount of ${potential_combination['amount']}"
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} combined amount patterns")
        return patterns
    
    def _detect_increasing_sequences(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect sequences of increasing amounts.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of increasing sequence patterns
        """
        # Get line items ordered by date
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
                li.amount > 0
                :doc_filter
            ORDER BY
                d.date_created, li.amount
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {"doc_filter": doc_filter}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Group by party and look for increasing sequences
        sequences = self._find_sequences(items, is_increasing=True)
        
        patterns = []
        for seq in sequences:
            if len(seq) >= self.min_sequence_length:
                # Calculate confidence based on sequence length and regularity
                confidence = self._calculate_sequence_confidence(seq)
                
                if confidence >= self.min_confidence:
                    first_amount = seq[0]['amount']
                    last_amount = seq[-1]['amount']
                    increase = last_amount - first_amount
                    percent_increase = (increase / first_amount) * 100 if first_amount else 0
                    
                    pattern = {
                        'type': 'increasing_sequence',
                        'items': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in item.items()} 
                            for item in seq
                        ],
                        'sequence_length': len(seq),
                        'start_amount': float(first_amount) if first_amount is not None else None,
                        'end_amount': float(last_amount) if last_amount is not None else None,
                        'total_increase': float(increase) if increase is not None else None,
                        'percent_increase': float(percent_increase) if percent_increase is not None else None,
                        'confidence': confidence,
                        'explanation': f"Sequence of {len(seq)} increasing amounts found, totaling a {percent_increase:.1f}% increase"
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_decreasing_sequences(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect sequences of decreasing amounts.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of decreasing sequence patterns
        """
        # Get line items ordered by date
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
                li.amount > 0
                :doc_filter
            ORDER BY
                d.date_created, li.amount DESC
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {"doc_filter": doc_filter}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Group by party and look for decreasing sequences
        sequences = self._find_sequences(items, is_increasing=False)
        
        patterns = []
        for seq in sequences:
            if len(seq) >= self.min_sequence_length:
                # Calculate confidence based on sequence length and regularity
                confidence = self._calculate_sequence_confidence(seq)
                
                if confidence >= self.min_confidence:
                    first_amount = seq[0]['amount']
                    last_amount = seq[-1]['amount']
                    decrease = first_amount - last_amount
                    percent_decrease = (decrease / first_amount) * 100 if first_amount else 0
                    
                    pattern = {
                        'type': 'decreasing_sequence',
                        'items': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in item.items()} 
                            for item in seq
                        ],
                        'sequence_length': len(seq),
                        'start_amount': float(first_amount) if first_amount is not None else None,
                        'end_amount': float(last_amount) if last_amount is not None else None,
                        'total_decrease': float(decrease) if decrease is not None else None,
                        'percent_decrease': float(percent_decrease) if percent_decrease is not None else None,
                        'confidence': confidence,
                        'explanation': f"Sequence of {len(seq)} decreasing amounts found, totaling a {percent_decrease:.1f}% decrease"
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_arithmetic_sequences(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect arithmetic sequences (constant differences) in amounts.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of arithmetic sequence patterns
        """
        # Get line items ordered by date
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
                li.amount > 0
                :doc_filter
            ORDER BY
                d.date_created
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {"doc_filter": doc_filter}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Group by party
        party_groups = {}
        for item in items:
            item_id, item_doc_id, description, amount, doc_type, party, date_created = item
            
            if not party or not date_created or amount is None:
                continue
                
            if party not in party_groups:
                party_groups[party] = []
                
            party_groups[party].append({
                'item_id': item_id,
                'doc_id': item_doc_id,
                'description': description,
                'amount': amount,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            })
        
        # Find arithmetic sequences in each party group
        patterns = []
        for party, group_items in party_groups.items():
            # Sort by date
            sorted_items = sorted(group_items, key=lambda x: x['date'])
            
            # Find potential arithmetic sequences
            arithmetic_seqs = self._find_arithmetic_sequences(sorted_items)
            
            for seq in arithmetic_seqs:
                if len(seq) >= self.min_sequence_length:
                    # Calculate confidence
                    confidence = self._calculate_arithmetic_confidence(seq)
                    
                    if confidence >= self.min_confidence:
                        # Calculate average difference
                        diffs = [seq[i+1]['amount'] - seq[i]['amount'] for i in range(len(seq)-1)]
                        avg_diff = sum(diffs) / len(diffs) if diffs else 0
                        
                        pattern = {
                            'type': 'arithmetic_sequence',
                            'items': [
                                {k: (v.isoformat() if k == 'date' and v else 
                                    float(v) if k == 'amount' and v is not None else v) 
                                 for k, v in item.items()} 
                                for item in seq
                            ],
                            'sequence_length': len(seq),
                            'average_difference': float(avg_diff) if avg_diff is not None else None,
                            'party': party,
                            'confidence': confidence,
                            'explanation': f"Arithmetic sequence of {len(seq)} amounts found with average difference of ${avg_diff:.2f}"
                        }
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _find_sequences(self, items: List[Tuple], is_increasing: bool = True) -> List[List[Dict[str, Any]]]:
        """Find sequences of consistently increasing or decreasing amounts.
        
        Args:
            items: List of line items
            is_increasing: True to find increasing sequences, False for decreasing
            
        Returns:
            List of sequences
        """
        # Group by party
        party_groups = {}
        for item in items:
            item_id, item_doc_id, description, amount, doc_type, party, date_created = item
            
            if not party or not date_created or amount is None:
                continue
                
            if party not in party_groups:
                party_groups[party] = []
                
            party_groups[party].append({
                'item_id': item_id,
                'doc_id': item_doc_id,
                'description': description,
                'amount': amount,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            })
        
        # Find sequences in each party group
        sequences = []
        for party, group_items in party_groups.items():
            # Sort by date
            sorted_items = sorted(group_items, key=lambda x: x['date'])
            
            current_seq = []
            for i, item in enumerate(sorted_items):
                if not current_seq:
                    # Start a new sequence
                    current_seq.append(item)
                else:
                    # Check if this item continues the sequence
                    prev_item = current_seq[-1]
                    
                    # Check date is within range
                    max_days = timedelta(days=self.max_time_between_sequential_items)
                    if item['date'] - prev_item['date'] > max_days:
                        # Too much time has passed, end sequence
                        if len(current_seq) >= self.min_sequence_length:
                            sequences.append(current_seq)
                        current_seq = [item]
                        continue
                    
                    # Check amount comparison based on sequence type
                    if is_increasing:
                        if item['amount'] > prev_item['amount']:
                            current_seq.append(item)
                        else:
                            # Sequence broken, check if long enough
                            if len(current_seq) >= self.min_sequence_length:
                                sequences.append(current_seq)
                            current_seq = [item]
                    else:  # decreasing
                        if item['amount'] < prev_item['amount']:
                            current_seq.append(item)
                        else:
                            # Sequence broken, check if long enough
                            if len(current_seq) >= self.min_sequence_length:
                                sequences.append(current_seq)
                            current_seq = [item]
            
            # Check the last sequence
            if len(current_seq) >= self.min_sequence_length:
                sequences.append(current_seq)
        
        return sequences
    
    def _find_arithmetic_sequences(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find arithmetic sequences (constant differences) in amounts.
        
        Args:
            items: List of line items
            
        Returns:
            List of arithmetic sequences
        """
        sequences = []
        
        # Need at least 3 items to form an arithmetic sequence
        if len(items) < 3:
            return sequences
            
        # Try different window sizes for consistency
        for window_size in range(3, min(len(items) + 1, 10)):  # Cap at 10 to avoid excessive computation
            for start_idx in range(len(items) - window_size + 1):
                window = items[start_idx:start_idx + window_size]
                
                # Calculate differences between consecutive amounts
                diffs = [window[i+1]['amount'] - window[i]['amount'] for i in range(len(window)-1)]
                
                # Check if differences are consistent (within a small tolerance)
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    is_consistent = all(abs(diff - avg_diff) < 0.1 * abs(avg_diff) for diff in diffs)
                    
                    if is_consistent:
                        # Check date sequence is reasonable
                        date_diffs = [(window[i+1]['date'] - window[i]['date']).days for i in range(len(window)-1)]
                        max_date_diff = max(date_diffs) if date_diffs else 0
                        
                        if max_date_diff <= self.max_time_between_sequential_items:
                            sequences.append(window)
        
        # Remove overlapping sequences, preferring longer ones
        sequences.sort(key=len, reverse=True)
        filtered_sequences = []
        covered_item_ids = set()
        
        for seq in sequences:
            seq_item_ids = {item['item_id'] for item in seq}
            if not seq_item_ids.intersection(covered_item_ids):
                filtered_sequences.append(seq)
                covered_item_ids.update(seq_item_ids)
        
        return filtered_sequences
    
    def _find_potential_splits(self, amount: Decimal, date: datetime, item_id: int) -> Optional[Dict[str, Any]]:
        """Find potential smaller amounts that sum close to a larger amount.
        
        Args:
            amount: The large amount
            date: Date of the large amount
            item_id: Item ID of the large amount
            
        Returns:
            Dict with potential split items and metrics, or None if none found
        """
        # Look for smaller amounts in a later timeframe
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
                li.amount > 0
                AND li.amount < :amount
                AND d.date_created > :date
                AND d.date_created <= :end_date
                AND li.item_id != :item_id
            ORDER BY
                d.date_created
        """)
        
        # Look for splits within the next X days
        end_date = date + timedelta(days=self.max_time_between_sequential_items)
        
        smaller_items = self.db_session.execute(query, {
            "amount": amount,
            "date": date,
            "end_date": end_date,
            "item_id": item_id
        }).fetchall()
        
        # Convert to dictionaries for easier handling
        smaller_items = [dict(zip(
            ['item_id', 'doc_id', 'description', 'amount', 'doc_type', 'party', 'date'],
            item
        )) for item in smaller_items]
        
        # Look for combinations that sum close to the original amount
        tolerance = Decimal('0.1') * amount  # 10% tolerance
        best_combo = None
        best_diff = float('inf')
        
        # Check all combinations (using a simple greedy approach for better performance)
        # Sort by amount descending
        sorted_items = sorted(smaller_items, key=lambda x: x['amount'], reverse=True)
        
        # Try to find combinations that sum close to the target amount
        combos = []
        
        # Start with each individual item
        for i, item in enumerate(sorted_items):
            combo = [item]
            combo_sum = item['amount']
            
            # Try adding other items to get closer to the target
            for j in range(i+1, len(sorted_items)):
                if combo_sum + sorted_items[j]['amount'] <= amount + tolerance:
                    combo.append(sorted_items[j])
                    combo_sum += sorted_items[j]['amount']
            
            # Check if this combination is a good match
            diff = abs(amount - combo_sum)
            
            if diff <= tolerance:
                combos.append({
                    'items': combo,
                    'total': combo_sum,
                    'difference': diff
                })
        
        # Find the best combination
        if combos:
            best_combo = min(combos, key=lambda x: x['difference'])
            
        return best_combo
    
    def _find_potential_combination(self, total_amount: Decimal, date: datetime, 
                                  item_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Find a potential larger amount that matches the sum of smaller amounts.
        
        Args:
            total_amount: Sum of the smaller amounts
            date: Date of the smaller amounts
            item_ids: List of item IDs for the smaller amounts
            
        Returns:
            Dict with potential combined item and metrics, or None if none found
        """
        # Look for a larger amount in a later timeframe
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
                li.amount BETWEEN :min_amount AND :max_amount
                AND d.date_created > :date
                AND d.date_created <= :end_date
                AND li.item_id NOT IN :item_ids
            ORDER BY
                ABS(li.amount - :total_amount)
        """)
        
        # Look for combinations within the next X days
        end_date = date + timedelta(days=self.max_time_between_sequential_items)
        tolerance = Decimal('0.1') * total_amount  # 10% tolerance
        
        try:
            larger_items = self.db_session.execute(query, {
                "min_amount": total_amount - tolerance,
                "max_amount": total_amount + tolerance,
                "date": date,
                "end_date": end_date,
                "item_ids": tuple(item_ids),
                "total_amount": total_amount
            }).fetchall()
        except:
            # If the tuple parameter fails (e.g., with SQLite), fall back to a simpler query
            larger_items = []
            for item_id in item_ids:
                fallback_query = text("""
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
                        li.amount BETWEEN :min_amount AND :max_amount
                        AND d.date_created > :date
                        AND d.date_created <= :end_date
                        AND li.item_id != :item_id
                    ORDER BY
                        ABS(li.amount - :total_amount)
                """)
                
                items = self.db_session.execute(fallback_query, {
                    "min_amount": total_amount - tolerance,
                    "max_amount": total_amount + tolerance,
                    "date": date,
                    "end_date": end_date,
                    "item_id": item_id,
                    "total_amount": total_amount
                }).fetchall()
                
                larger_items.extend(items)
        
        # Find the closest match
        if larger_items:
            best_match = larger_items[0]
            item_id, doc_id, description, amount, doc_type, party, match_date = best_match
            
            difference = abs(amount - total_amount) if amount is not None else None
            
            return {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'doc_type': doc_type,
                'party': party,
                'date': match_date,
                'difference': difference
            }
        
        return None
    
    def _calculate_sequence_confidence(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a sequence.
        
        Args:
            sequence: List of items in the sequence
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence depends on sequence length
        length_factor = min(1.0, (len(sequence) - 2) / 8.0)  # 10 items gives full score
        base_confidence = 0.5 + (0.3 * length_factor)
        
        # Check consistency of differences
        diffs = []
        for i in range(1, len(sequence)):
            diff = abs(sequence[i]['amount'] - sequence[i-1]['amount'])
            diffs.append(diff)
        
        consistency = 0.0
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0:
                # Lower variance increases confidence
                variance = sum((diff - avg_diff) ** 2 for diff in diffs) / len(diffs)
                normalized_variance = variance / (avg_diff ** 2)
                consistency = max(0.0, 0.15 * (1.0 - min(1.0, normalized_variance * 5)))
        
        # Check time regularity
        time_regularity = 0.0
        date_diffs = []
        for i in range(1, len(sequence)):
            days = (sequence[i]['date'] - sequence[i-1]['date']).days
            date_diffs.append(days)
        
        if date_diffs:
            avg_days = sum(date_diffs) / len(date_diffs)
            if avg_days > 0:
                # Lower date variance increases confidence
                date_variance = sum((days - avg_days) ** 2 for days in date_diffs) / len(date_diffs)
                normalized_date_variance = date_variance / (avg_days ** 2)
                time_regularity = max(0.0, 0.15 * (1.0 - min(1.0, normalized_date_variance * 5)))
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + consistency + time_regularity)
        
        return confidence
    
    def _calculate_arithmetic_confidence(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for an arithmetic sequence.
        
        Args:
            sequence: List of items in the arithmetic sequence
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Use the sequence confidence as a base
        base_confidence = self._calculate_sequence_confidence(sequence)
        
        # Add an extra boost for arithmetic sequences since they're more significant
        arithmetic_boost = 0.1
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + arithmetic_boost)
        
        return confidence
    
    def _calculate_split_confidence(self, original_amount: Decimal, 
                                 split_data: Dict[str, Any], 
                                 original_description: Optional[str] = None) -> float:
        """Calculate confidence score for a split amount pattern.
        
        Args:
            original_amount: Original large amount
            split_data: Dict with split items and metrics
            original_description: Description of the original item
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        base_confidence = 0.7
        
        # Higher confidence if the split is very close to the original
        difference_ratio = float(split_data['difference'] / original_amount) if original_amount else 1.0
        difference_factor = max(0.0, 0.15 * (1.0 - min(1.0, difference_ratio * 10)))
        
        # Higher confidence if more splits (up to a point)
        count_factor = min(0.1, (len(split_data['items']) - 1) * 0.02)
        
        # Higher confidence if descriptions are related
        description_factor = 0.0
        if original_description:
            # Simplified: Count items with similar descriptions
            similar_count = 0
            for item in split_data['items']:
                if item['description'] and self._are_descriptions_similar([original_description, item['description']]):
                    similar_count += 1
            
            description_factor = min(0.1, similar_count * 0.02)
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + difference_factor + count_factor + description_factor)
        
        return confidence
    
    def _calculate_combination_confidence(self, original_total: Decimal, 
                                       combined_item: Dict[str, Any], 
                                       original_items: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a combined amount pattern.
        
        Args:
            original_total: Total of the smaller amounts
            combined_item: Dict with combined item details
            original_items: List of smaller items
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        base_confidence = 0.7
        
        # Higher confidence if the combined amount is very close to the original total
        difference_ratio = float(combined_item['difference'] / original_total) if original_total else 1.0
        difference_factor = max(0.0, 0.15 * (1.0 - min(1.0, difference_ratio * 10)))
        
        # Higher confidence if more items were combined (up to a point)
        count_factor = min(0.1, (len(original_items) - 1) * 0.02)
        
        # Higher confidence if descriptions are related
        description_factor = 0.0
        if combined_item['description']:
            # Simplified: Count items with similar descriptions
            similar_count = 0
            for item in original_items:
                if item['description'] and self._are_descriptions_similar([combined_item['description'], item['description']]):
                    similar_count += 1
            
            description_factor = min(0.1, similar_count * 0.02)
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + difference_factor + count_factor + description_factor)
        
        return confidence
    
    def _are_descriptions_similar(self, descriptions: List[str]) -> bool:
        """Check if descriptions are similar.
        
        Args:
            descriptions: List of descriptions to compare
            
        Returns:
            True if descriptions are similar, False otherwise
        """
        # Similar to the implementation in the RecurringPatternDetector class
        # Simple implementation: check for common words
        
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
                    # If similarity is below threshold, descriptions are different
                    if similarity < 0.3:  # Lower threshold for this context
                        return False
        
        # All pairs were similar enough
        return True