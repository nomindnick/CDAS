"""
Fuzzy amount matcher for the financial analysis engine.

This module provides fuzzy matching of amounts across different documents,
accounting for variations in amounts that may be due to rounding, taxes,
fees, or other adjustments.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """Matches amounts with fuzzy tolerance across documents."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the fuzzy amount matcher.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.default_threshold = self.config.get('default_fuzzy_threshold', 0.1)  # 10% threshold
        self.match_across_parties = self.config.get('match_across_parties', True)
        self.match_across_doc_types = self.config.get('match_across_doc_types', True)
        
        # Configure standard markup percentages to check
        self.standard_markups = self.config.get('standard_markups', [
            0.05,   # 5% fee/tax
            0.0625, # 6.25% tax (common in some states)
            0.07,   # 7% tax (common in some states)
            0.08,   # 8% tax (common in some states)
            0.0875, # 8.75% tax (common in some states)
            0.10,   # 10% fee/markup
            0.15,   # 15% markup (common contractor markup)
            0.20,   # 20% markup (common contractor markup)
            0.25    # 25% markup (common contractor markup)
        ])
        
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
        
    def find_fuzzy_matches(self, amount: float, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find fuzzy matches for an amount.
        
        Args:
            amount: Amount to match
            threshold: Threshold percentage for fuzzy matching (defaults to self.default_threshold)
            
        Returns:
            List of fuzzy matches
        """
        logger.info(f"Finding fuzzy matches for amount: {amount}")
        
        threshold = threshold if threshold is not None else self.default_threshold
        
        # Calculate range for fuzzy matching
        min_amount = amount * (1 - threshold)
        max_amount = amount * (1 + threshold)
        
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                d.doc_type,
                d.party,
                d.date_created,
                ABS(li.amount - :amount) / :amount AS difference_percent
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount >= :min_amount AND li.amount <= :max_amount
                AND ABS(li.amount - :amount) / :amount <= :threshold
            ORDER BY
                difference_percent
        """)
        
        try:
            matches = self.db_session.execute(
                query, 
                {
                    "amount": amount, 
                    "min_amount": min_amount, 
                    "max_amount": max_amount, 
                    "threshold": threshold
                }
            ).fetchall()
        except Exception as e:
            logger.exception(f"Error in fuzzy match query: {e}")
            
            # Fallback to a more compatible query without the calculated column
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
                    li.amount >= :min_amount AND li.amount <= :max_amount
                ORDER BY
                    ABS(li.amount - :amount)
            """)
            
            matches = self.db_session.execute(
                fallback_query, 
                {
                    "amount": amount, 
                    "min_amount": min_amount, 
                    "max_amount": max_amount
                }
            ).fetchall()
            
            # Filter and calculate difference percent manually
            filtered_matches = []
            for match in matches:
                if len(match) == 7:  # Fallback query doesn't include difference_percent
                    item_id, doc_id, description, match_amount, doc_type, party, date_created = match
                    
                    # Calculate difference percent manually
                    if match_amount is not None and amount > 0:
                        difference_percent = abs(match_amount - amount) / amount
                        
                        # Apply threshold filter manually
                        if difference_percent <= threshold:
                            filtered_matches.append(match + (difference_percent,))
                
            matches = filtered_matches
        
        result = []
        for match in matches:
            if len(match) == 8:  # Regular query with difference_percent
                item_id, doc_id, description, match_amount, doc_type, party, date_created, difference_percent = match
            else:  # Shouldn't happen but handle just in case
                continue
                
            # Calculate confidence based on how close the match is
            confidence = 1.0 - (difference_percent / threshold) if threshold > 0 else 0.0
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created),
                'difference_percent': float(difference_percent) if difference_percent is not None else None,
                'confidence': float(confidence),
                'match_type': 'fuzzy'
            })
        
        logger.info(f"Found {len(result)} fuzzy matches for amount: {amount}")
        return result
    
    def find_markup_matches(self, amount: float, custom_markups: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Find matches that differ by standard or custom markup percentages.
        
        Args:
            amount: Amount to match
            custom_markups: Optional list of custom markup percentages to check
            
        Returns:
            List of markup matches
        """
        logger.info(f"Finding markup matches for amount: {amount}")
        
        # Combine standard and custom markups
        markups = list(self.standard_markups)
        if custom_markups:
            markups.extend(custom_markups)
            
        # Remove duplicates and sort
        markups = sorted(set(markups))
        
        # Look for both markup and markdown versions
        matches = []
        
        for markup in markups:
            # Calculate markup amount (amount + markup%)
            markup_amount = amount * (1 + markup)
            markup_matches = self._find_adjusted_amount_matches(amount, markup_amount, markup, 'markup')
            matches.extend(markup_matches)
            
            # Calculate markdown amount (amount - markup%)
            markdown_amount = amount / (1 + markup)
            markdown_matches = self._find_adjusted_amount_matches(amount, markdown_amount, markup, 'markdown')
            matches.extend(markdown_matches)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        logger.info(f"Found {len(matches)} markup/markdown matches for amount: {amount}")
        return matches
    
    def find_tax_matches(self, amount: float, tax_rates: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Find matches that differ by tax rates.
        
        Args:
            amount: Amount to match
            tax_rates: Optional list of tax rates to check (defaults to common rates)
            
        Returns:
            List of tax matches
        """
        logger.info(f"Finding tax matches for amount: {amount}")
        
        # Use provided tax rates or common defaults
        if tax_rates is None:
            tax_rates = [0.05, 0.0625, 0.07, 0.0775, 0.08, 0.0825, 0.0875, 0.09, 0.095, 0.10]
            
        # Remove duplicates and sort
        tax_rates = sorted(set(tax_rates))
        
        # Look for both with-tax and pre-tax versions
        matches = []
        
        for tax_rate in tax_rates:
            # Calculate with-tax amount (amount + tax)
            with_tax_amount = amount * (1 + tax_rate)
            with_tax_matches = self._find_adjusted_amount_matches(amount, with_tax_amount, tax_rate, 'with_tax')
            matches.extend(with_tax_matches)
            
            # Calculate pre-tax amount (amount - tax)
            pre_tax_amount = amount / (1 + tax_rate)
            pre_tax_matches = self._find_adjusted_amount_matches(amount, pre_tax_amount, tax_rate, 'pre_tax')
            matches.extend(pre_tax_matches)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        logger.info(f"Found {len(matches)} tax matches for amount: {amount}")
        return matches
    
    def find_rounded_matches(self, amount: float) -> List[Dict[str, Any]]:
        """Find matches that are rounded versions of the amount.
        
        Args:
            amount: Amount to match
            
        Returns:
            List of rounded matches
        """
        logger.info(f"Finding rounded matches for amount: {amount}")
        
        # Define rounding levels to check
        rounding_levels = [
            ('nearest_dollar', 1),
            ('nearest_10', 10),
            ('nearest_100', 100),
            ('nearest_1000', 1000)
        ]
        
        matches = []
        for name, level in rounding_levels:
            # Calculate rounded amount
            rounded_amount = round(amount / level) * level
            
            # Skip if rounding didn't change the amount
            if rounded_amount == amount:
                continue
                
            # Calculate difference percentage
            difference_percent = abs(rounded_amount - amount) / amount if amount > 0 else 0
            
            # Skip if difference is too large
            if difference_percent > 0.15:  # 15% max difference for rounding
                continue
                
            # Find matches for the rounded amount
            rounded_matches = self._find_exact_amount_matches(rounded_amount)
            
            # Add rounding metadata
            for match in rounded_matches:
                match['adjustment_type'] = 'rounding'
                match['rounding_level'] = name
                match['original_amount'] = float(amount)
                match['difference_percent'] = float(difference_percent)
                match['confidence'] = 1.0 - difference_percent
                
            matches.extend(rounded_matches)
        
        logger.info(f"Found {len(matches)} rounded matches for amount: {amount}")
        return matches
    
    def find_split_or_combined_matches(self, amount: float, max_parts: int = 4) -> List[Dict[str, Any]]:
        """Find matches that could be splits or combinations of the amount.
        
        Args:
            amount: Amount to match
            max_parts: Maximum number of parts to consider for splits/combinations
            
        Returns:
            List of split or combined matches
        """
        logger.info(f"Finding split or combined matches for amount: {amount}")
        
        # This could be computationally expensive with a large database
        # so we'll implement a simplified version that checks common patterns
        
        matches = []
        
        # Check for splits (amount = sum of smaller amounts)
        split_matches = self._find_potential_splits(amount, max_parts)
        matches.extend(split_matches)
        
        # Check for combinations (amount is part of a larger sum)
        combined_matches = self._find_potential_combinations(amount, max_parts)
        matches.extend(combined_matches)
        
        logger.info(f"Found {len(matches)} split or combined matches for amount: {amount}")
        return matches
    
    def _find_adjusted_amount_matches(self, original_amount: float, adjusted_amount: float, 
                                  adjustment_rate: float, adjustment_type: str) -> List[Dict[str, Any]]:
        """Find matches for an adjusted amount (with markup, tax, etc.)
        
        Args:
            original_amount: Original amount
            adjusted_amount: Adjusted amount to match
            adjustment_rate: Rate of adjustment
            adjustment_type: Type of adjustment ('markup', 'markdown', 'with_tax', 'pre_tax')
            
        Returns:
            List of adjusted amount matches
        """
        # Define a small tolerance for exact matching of adjusted amount
        tolerance = adjusted_amount * 0.01  # 1% tolerance
        
        # Find exact matches for the adjusted amount
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
            ORDER BY
                ABS(li.amount - :adjusted_amount)
        """)
        
        amount_min = adjusted_amount - tolerance
        amount_max = adjusted_amount + tolerance
        
        matches = self.db_session.execute(
            query, 
            {
                "adjusted_amount": adjusted_amount, 
                "amount_min": amount_min, 
                "amount_max": amount_max
            }
        ).fetchall()
        
        result = []
        for match in matches:
            item_id, doc_id, description, match_amount, doc_type, party, date_created = match
            
            # Calculate difference from expected adjusted amount
            difference = abs(match_amount - adjusted_amount) if match_amount is not None else 0
            difference_percent = difference / adjusted_amount if adjusted_amount > 0 else 0
            
            # Calculate confidence (higher for closer matches)
            confidence = 1.0 - (difference_percent / 0.01) if difference_percent <= 0.01 else 0.0
            
            # Format adjustment rate for display
            formatted_rate = f"{adjustment_rate * 100:.2f}%"
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created),
                'original_amount': float(original_amount),
                'adjustment_type': adjustment_type,
                'adjustment_rate': float(adjustment_rate),
                'adjustment_rate_formatted': formatted_rate,
                'difference': float(difference),
                'difference_percent': float(difference_percent),
                'confidence': float(confidence),
                'match_type': 'adjusted'
            })
        
        return result
    
    def _find_exact_amount_matches(self, amount: float) -> List[Dict[str, Any]]:
        """Find exact matches for a specific amount.
        
        Args:
            amount: Amount to match
            
        Returns:
            List of exact matches
        """
        # Small tolerance for floating point comparison
        tolerance = amount * 0.001  # 0.1% tolerance
        
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
            ORDER BY
                ABS(li.amount - :amount)
        """)
        
        amount_min = amount - tolerance
        amount_max = amount + tolerance
        
        matches = self.db_session.execute(
            query, 
            {
                "amount": amount, 
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
                'date': self._format_date(date_created),
                'match_type': 'exact'
            })
        
        return result
    
    def _find_potential_splits(self, amount: float, max_parts: int) -> List[Dict[str, Any]]:
        """Find potential splits of an amount into smaller parts.
        
        Args:
            amount: Amount to find splits for
            max_parts: Maximum number of parts to consider
            
        Returns:
            List of potential split matches
        """
        # This is a complex problem to solve perfectly, so we'll use heuristics
        # to find common split patterns
        
        # Check for common split patterns
        splits = []
        
        # 50/50 split
        if max_parts >= 2:
            half_amount = amount / 2
            half_matches = self._find_similar_amounts(half_amount, count=2)
            
            if len(half_matches) >= 2:
                splits.append({
                    'type': 'equal_split',
                    'parts': 2,
                    'original_amount': float(amount),
                    'part_amount': float(half_amount),
                    'matches': half_matches,
                    'confidence': 0.9,
                    'match_type': 'split'
                })
        
        # 33/33/33 split
        if max_parts >= 3:
            third_amount = amount / 3
            third_matches = self._find_similar_amounts(third_amount, count=3)
            
            if len(third_matches) >= 3:
                splits.append({
                    'type': 'equal_split',
                    'parts': 3,
                    'original_amount': float(amount),
                    'part_amount': float(third_amount),
                    'matches': third_matches,
                    'confidence': 0.85,
                    'match_type': 'split'
                })
        
        # 25/25/25/25 split
        if max_parts >= 4:
            quarter_amount = amount / 4
            quarter_matches = self._find_similar_amounts(quarter_amount, count=4)
            
            if len(quarter_matches) >= 4:
                splits.append({
                    'type': 'equal_split',
                    'parts': 4,
                    'original_amount': float(amount),
                    'part_amount': float(quarter_amount),
                    'matches': quarter_matches,
                    'confidence': 0.8,
                    'match_type': 'split'
                })
        
        return splits
    
    def _find_potential_combinations(self, amount: float, max_parts: int) -> List[Dict[str, Any]]:
        """Find potential combinations where this amount is part of a larger sum.
        
        Args:
            amount: Amount to find combinations for
            max_parts: Maximum number of parts to consider
            
        Returns:
            List of potential combination matches
        """
        # Check for common combination patterns
        combinations = []
        
        # Part of a 2-part sum
        if max_parts >= 2:
            # Find another part that might be combined with this
            other_matches = self._find_potential_combination_partners(amount, 1)
            
            for other_match in other_matches:
                total_amount = amount + other_match['amount']
                
                # Check if the total exists
                total_matches = self._find_similar_amounts(total_amount, count=1, threshold=0.01)
                
                if total_matches:
                    combinations.append({
                        'type': 'combination',
                        'parts': 2,
                        'part_amount': float(amount),
                        'partner_amounts': [float(other_match['amount'])],
                        'total_amount': float(total_amount),
                        'partner_matches': [other_match],
                        'total_matches': total_matches,
                        'confidence': 0.85,
                        'match_type': 'combination'
                    })
        
        # Only attempt more complex combinations if specifically configured
        # as these can be computationally expensive and have higher false positive rates
        
        return combinations
    
    def _find_similar_amounts(self, amount: float, count: int = 1, 
                          threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Find similar amounts within a threshold.
        
        Args:
            amount: Target amount
            count: Number of matches to find
            threshold: Similarity threshold (percentage)
            
        Returns:
            List of similar amount matches
        """
        # Calculate range for similarity
        min_amount = amount * (1 - threshold)
        max_amount = amount * (1 + threshold)
        
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
            ORDER BY
                ABS(li.amount - :amount)
            LIMIT :count
        """)
        
        matches = self.db_session.execute(
            query, 
            {
                "amount": amount, 
                "min_amount": min_amount, 
                "max_amount": max_amount,
                "count": count
            }
        ).fetchall()
        
        result = []
        for match in matches:
            item_id, doc_id, description, match_amount, doc_type, party, date_created = match
            
            # Calculate difference percentage
            difference_percent = abs(match_amount - amount) / amount if amount > 0 else 0
            
            # Calculate confidence (higher for closer matches)
            confidence = 1.0 - (difference_percent / threshold) if threshold > 0 else 0.0
            
            result.append({
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(match_amount) if match_amount is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': self._format_date(date_created),
                'difference_percent': float(difference_percent),
                'confidence': float(confidence),
                'match_type': 'similar'
            })
        
        return result
    
    def _find_potential_combination_partners(self, amount: float, 
                                         num_partners: int = 1) -> List[Dict[str, Any]]:
        """Find potential amounts that might combine with this one.
        
        Args:
            amount: Base amount
            num_partners: Number of potential partners to find
            
        Returns:
            List of potential partner amounts
        """
        # For simplicity, we'll find amounts that are similar to the base amount
        # (Many combinations tend to have similar-sized components)
        
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
                AND li.amount != :amount
            ORDER BY
                ABS(li.amount - :amount)
            LIMIT :limit
        """)
        
        # Look for amounts within 20% of the base amount
        min_amount = amount * 0.8
        max_amount = amount * 1.2
        
        matches = self.db_session.execute(
            query, 
            {
                "amount": amount, 
                "min_amount": min_amount, 
                "max_amount": max_amount,
                "limit": num_partners
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
                'date': date_created.isoformat() if date_created else None
            })
        
        return result