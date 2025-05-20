"""
Similarity pattern detector for the financial analysis engine.

This module detects similarity-based patterns in financial data, such as
inconsistent markups, similar amounts across different contexts, and other
patterns that may indicate financial irregularities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class SimilarityPatternDetector:
    """Detects similarity-based patterns in financial data."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the similarity pattern detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_item_count = self.config.get('min_item_count', 2)
        self.description_similarity_threshold = self.config.get('description_similarity_threshold', 0.3)
        
    def detect_inconsistent_markups(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect inconsistent markup percentages.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of inconsistent markup patterns
        """
        logger.info(f"Detecting inconsistent markups{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on whether doc_id is provided
        where_clause = "li.unit_price IS NOT NULL AND li.quantity IS NOT NULL"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Find line items with both unit price and quantity
        query = text(f"""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.total,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
            ORDER BY
                li.cost_code, li.description
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Group by cost code and similar description
        grouped_items = self._group_by_similarity(items)
        
        patterns = []
        for group_key, group_items in grouped_items.items():
            # We need at least a few items to analyze markup consistency
            if len(group_items) < self.min_item_count:
                continue
                
            # Calculate markup percentages for each item
            markup_data = self._calculate_markup_percentages(group_items)
            
            # Check for inconsistent markups
            if self._has_inconsistent_markups(markup_data):
                # Calculate confidence based on inconsistency level
                confidence = self._calculate_markup_confidence(markup_data)
                
                if confidence >= self.min_confidence:
                    # Format items for output
                    formatted_items = [
                        {k: (v.isoformat() if k == 'date' and v else 
                            float(v) if k in ['amount', 'quantity', 'unit_price', 'total', 'markup_percent'] and v is not None else v) 
                         for k, v in item.items()} 
                        for item in markup_data
                    ]
                    
                    # Get summary statistics
                    markup_stats = self._calculate_markup_stats(markup_data)
                    
                    pattern = {
                        'type': 'inconsistent_markup',
                        'items': formatted_items,
                        'min_markup': float(markup_stats['min']) if markup_stats['min'] is not None else None,
                        'max_markup': float(markup_stats['max']) if markup_stats['max'] is not None else None,
                        'avg_markup': float(markup_stats['avg']) if markup_stats['avg'] is not None else None,
                        'markup_variance': float(markup_stats['variance']) if markup_stats['variance'] is not None else None,
                        'cost_code': group_key.split('|')[0] if '|' in group_key else None,
                        'confidence': confidence,
                        'explanation': f"Inconsistent markup percentages found for similar items (variance: {markup_stats['variance']:.2f}%, range: {markup_stats['min']:.2f}% to {markup_stats['max']:.2f}%)"
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} inconsistent markup patterns")
        return patterns
    
    def detect_similar_items_different_prices(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect similar items with significantly different prices.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of similar items with different price patterns
        """
        logger.info(f"Detecting similar items with different prices{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on whether doc_id is provided
        where_clause = "li.description IS NOT NULL AND li.amount > 0"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Find items with descriptions and prices
        query = text(f"""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
            ORDER BY
                li.description
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Group by similar description
        grouped_items = self._group_by_description_similarity(items)
        
        patterns = []
        for desc_key, group_items in grouped_items.items():
            # We need at least a few items to compare prices
            if len(group_items) < self.min_item_count:
                continue
                
            # If quantities are available, normalize by quantity
            has_quantities = all(item['quantity'] is not None and item['quantity'] > 0 for item in group_items)
            
            if has_quantities:
                # Normalize to unit prices
                for item in group_items:
                    if 'unit_price' not in item or item['unit_price'] is None:
                        item['unit_price'] = item['amount'] / item['quantity']
                
                price_variance = self._calculate_price_variance([item['unit_price'] for item in group_items])
            else:
                # Use amount directly
                price_variance = self._calculate_price_variance([item['amount'] for item in group_items])
            
            # Check if variance is significant
            if price_variance['variation_coefficient'] > 0.2:  # 20% coefficient of variation threshold
                # Calculate confidence based on price variance
                confidence = self._calculate_price_variance_confidence(price_variance, group_items)
                
                if confidence >= self.min_confidence:
                    # Format items for output
                    formatted_items = [
                        {k: (v.isoformat() if k == 'date' and v else 
                            float(v) if k in ['amount', 'quantity', 'unit_price'] and v is not None else v) 
                         for k, v in item.items()} 
                        for item in group_items
                    ]
                    
                    pattern = {
                        'type': 'similar_items_different_prices',
                        'items': formatted_items,
                        'min_price': float(price_variance['min']) if price_variance['min'] is not None else None,
                        'max_price': float(price_variance['max']) if price_variance['max'] is not None else None,
                        'avg_price': float(price_variance['avg']) if price_variance['avg'] is not None else None,
                        'price_variance': float(price_variance['variance']) if price_variance['variance'] is not None else None,
                        'price_variation_coefficient': float(price_variance['variation_coefficient']) if price_variance['variation_coefficient'] is not None else None,
                        'is_unit_price': has_quantities,
                        'confidence': confidence,
                        'explanation': f"Similar items with significantly different prices found (coefficient of variation: {price_variance['variation_coefficient']:.2f}, range: {price_variance['min']:.2f} to {price_variance['max']:.2f})"
                    }
                    
                    patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} similar items with different prices patterns")
        return patterns
    
    def detect_outlier_pricing(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect pricing that is significantly higher or lower than average for similar items.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of outlier pricing patterns
        """
        logger.info(f"Detecting outlier pricing{f' for document {doc_id}' if doc_id else ''}")
        
        # This builds on the similar items detection, but focuses on significant outliers
        similar_item_patterns = self.detect_similar_items_different_prices(doc_id)
        
        patterns = []
        for pattern in similar_item_patterns:
            # Only consider high confidence patterns
            if pattern['confidence'] < 0.8:
                continue
                
            # Find outliers (>2x or <0.5x the average)
            items = pattern['items']
            avg_price = pattern['avg_price']
            is_unit_price = pattern['is_unit_price']
            
            outliers = []
            for item in items:
                price = item['unit_price'] if is_unit_price else item['amount']
                
                # Check if price is an outlier
                if price > 2 * avg_price or price < 0.5 * avg_price:
                    ratio = price / avg_price
                    outlier_info = {
                        **item,
                        'price_ratio': ratio,
                        'is_high_outlier': ratio > 2
                    }
                    outliers.append(outlier_info)
            
            # Only create pattern if we found outliers
            if outliers:
                pattern = {
                    'type': 'outlier_pricing',
                    'items': items,
                    'outliers': outliers,
                    'avg_price': avg_price,
                    'is_unit_price': is_unit_price,
                    'confidence': pattern['confidence'] + 0.05,  # Slightly higher confidence for outliers
                    'explanation': f"Found {len(outliers)} items with outlier pricing compared to similar items"
                }
                
                patterns.append(pattern)
        
        logger.info(f"Found {len(patterns)} outlier pricing patterns")
        return patterns
    
    def _group_by_similarity(self, items: List[Tuple]) -> Dict[str, List[Dict[str, Any]]]:
        """Group items by cost code and similar description.
        
        Args:
            items: List of items
            
        Returns:
            Dict mapping group key to list of items
        """
        # Convert tuple items to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, quantity, unit_price, total, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price,
                'total': total,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        # Group by cost code first
        cost_code_groups = {}
        for item in item_dicts:
            cost_code = item['cost_code'] or 'unknown'
            
            if cost_code not in cost_code_groups:
                cost_code_groups[cost_code] = []
                
            cost_code_groups[cost_code].append(item)
        
        # Further group by similar description within cost code groups
        grouped_items = {}
        for cost_code, code_items in cost_code_groups.items():
            description_groups = self._group_descriptions(code_items)
            
            for i, group in enumerate(description_groups):
                group_key = f"{cost_code}|group{i}"
                grouped_items[group_key] = group
        
        return grouped_items
    
    def _group_by_description_similarity(self, items: List[Tuple]) -> Dict[str, List[Dict[str, Any]]]:
        """Group items by similar description.
        
        Args:
            items: List of items
            
        Returns:
            Dict mapping description key to list of items
        """
        # Convert tuple items to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, quantity, unit_price, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        # Group by similar description
        description_groups = self._group_descriptions(item_dicts)
        
        grouped_items = {}
        for i, group in enumerate(description_groups):
            # Use the most common description as the key
            descriptions = {}
            for item in group:
                desc = item['description'] or ''
                if desc in descriptions:
                    descriptions[desc] += 1
                else:
                    descriptions[desc] = 1
            
            most_common_desc = max(descriptions.items(), key=lambda x: x[1])[0]
            group_key = most_common_desc[:50]  # Truncate for reasonable key length
            
            grouped_items[group_key] = group
        
        return grouped_items
    
    def _group_descriptions(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group items by similar description using text similarity.
        
        Args:
            items: List of items
            
        Returns:
            List of groups, where each group is a list of items
        """
        # Extract non-empty descriptions
        item_descriptions = [(i, item['description']) for i, item in enumerate(items) if item['description']]
        
        # If too few items with descriptions, return all as one group
        if len(item_descriptions) < 2:
            return [items]
        
        # Calculate similarity matrix
        similarity_matrix = {}
        for i in range(len(item_descriptions)):
            idx1, desc1 = item_descriptions[i]
            for j in range(i+1, len(item_descriptions)):
                idx2, desc2 = item_descriptions[j]
                
                similarity = self._calculate_text_similarity(desc1, desc2)
                similarity_matrix[(idx1, idx2)] = similarity
        
        # Group items based on similarity threshold
        groups = []
        remaining_indices = set(range(len(items)))
        
        while remaining_indices:
            # Start a new group with the first remaining item
            current_group = [next(iter(remaining_indices))]
            remaining_indices.remove(current_group[0])
            
            # Keep adding similar items to the group
            changed = True
            while changed:
                changed = False
                for idx in list(remaining_indices):
                    # Check if similar to any item in the current group
                    is_similar = False
                    for group_idx in current_group:
                        key = (min(idx, group_idx), max(idx, group_idx))
                        if key in similarity_matrix and similarity_matrix[key] >= self.description_similarity_threshold:
                            is_similar = True
                            break
                    
                    if is_similar:
                        current_group.append(idx)
                        remaining_indices.remove(idx)
                        changed = True
            
            # Add group to results
            group_items = [items[idx] for idx in current_group]
            groups.append(group_items)
        
        return groups
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple implementation using Jaccard similarity on word sets
        if not text1 or not text2:
            return 0.0
            
        # Tokenize and normalize text
        tokens1 = set(word.lower() for word in text1.split())
        tokens2 = set(word.lower() for word in text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union > 0:
            return intersection / union
        else:
            return 0.0
    
    def _calculate_markup_percentages(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate markup percentages for items.
        
        Args:
            items: List of items with unit_price and quantity
            
        Returns:
            List of items with markup_percent added
        """
        result_items = []
        
        for item in items:
            # Deep copy to avoid modifying original
            result_item = {**item}
            
            # Calculate expected amount based on unit_price * quantity
            if result_item['unit_price'] is not None and result_item['quantity'] is not None:
                expected_amount = result_item['unit_price'] * result_item['quantity']
                
                # Calculate markup percentage if amounts are available
                if result_item['amount'] is not None and expected_amount > 0:
                    markup_percent = ((result_item['amount'] - expected_amount) / expected_amount) * 100
                    result_item['markup_percent'] = markup_percent
                else:
                    result_item['markup_percent'] = None
            else:
                result_item['markup_percent'] = None
            
            result_items.append(result_item)
        
        return result_items
    
    def _has_inconsistent_markups(self, items: List[Dict[str, Any]]) -> bool:
        """Check if items have inconsistent markups.
        
        Args:
            items: List of items with markup_percent
            
        Returns:
            True if markups are inconsistent, False otherwise
        """
        # Extract valid markup percentages
        markups = [item['markup_percent'] for item in items if item['markup_percent'] is not None]
        
        # Need at least a few items to determine inconsistency
        if len(markups) < self.min_item_count:
            return False
            
        # Calculate markup variance
        markup_stats = self._calculate_markup_stats(items)
        
        # Consider markups inconsistent if variance is high or range is significant
        is_inconsistent = (
            markup_stats['variance'] > 5.0 or  # Variance > 5%
            (markup_stats['max'] - markup_stats['min']) > 10.0  # Range > 10%
        )
        
        return is_inconsistent
    
    def _calculate_markup_stats(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistics for markup percentages.
        
        Args:
            items: List of items with markup_percent
            
        Returns:
            Dict with min, max, avg, variance
        """
        # Extract valid markup percentages
        markups = [item['markup_percent'] for item in items if item['markup_percent'] is not None]
        
        # Calculate statistics
        if not markups:
            return {
                'min': None,
                'max': None,
                'avg': None,
                'variance': None
            }
            
        min_markup = min(markups)
        max_markup = max(markups)
        avg_markup = sum(markups) / len(markups)
        
        # Calculate variance
        variance = sum((m - avg_markup) ** 2 for m in markups) / len(markups)
        
        return {
            'min': min_markup,
            'max': max_markup,
            'avg': avg_markup,
            'variance': variance
        }
    
    def _calculate_price_variance(self, prices: List[Decimal]) -> Dict[str, float]:
        """Calculate variance statistics for prices.
        
        Args:
            prices: List of prices
            
        Returns:
            Dict with min, max, avg, variance, variation_coefficient
        """
        # Filter out None values
        valid_prices = [p for p in prices if p is not None]
        
        # Calculate statistics
        if not valid_prices:
            return {
                'min': None,
                'max': None,
                'avg': None,
                'variance': None,
                'variation_coefficient': None
            }
            
        min_price = min(valid_prices)
        max_price = max(valid_prices)
        avg_price = sum(valid_prices) / len(valid_prices)
        
        # Calculate variance
        variance = sum((p - avg_price) ** 2 for p in valid_prices) / len(valid_prices)
        
        # Calculate coefficient of variation (standardized measure of dispersion)
        std_dev = variance ** 0.5
        variation_coefficient = std_dev / avg_price if avg_price else float('inf')
        
        return {
            'min': min_price,
            'max': max_price,
            'avg': avg_price,
            'variance': variance,
            'variation_coefficient': variation_coefficient
        }
    
    def _calculate_markup_confidence(self, items: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for inconsistent markup pattern.
        
        Args:
            items: List of items with markup_percent
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Calculate markup statistics
        markup_stats = self._calculate_markup_stats(items)
        
        # Base confidence
        base_confidence = 0.7
        
        # Higher confidence with higher variance
        variance_factor = min(0.15, markup_stats['variance'] / 100.0)
        
        # Higher confidence with larger range
        range_factor = min(0.1, (markup_stats['max'] - markup_stats['min']) / 100.0)
        
        # Higher confidence with more items
        items_factor = min(0.1, (len(items) - self.min_item_count) * 0.02)
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + variance_factor + range_factor + items_factor)
        
        return confidence
    
    def _calculate_price_variance_confidence(self, variance_data: Dict[str, float], 
                                          items: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for price variance pattern.
        
        Args:
            variance_data: Dict with price variance statistics
            items: List of items with prices
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        base_confidence = 0.7
        
        # Higher confidence with higher coefficient of variation
        coef_factor = min(0.15, variance_data['variation_coefficient'] * 0.3)
        
        # Higher confidence with larger relative range
        range_ratio = (variance_data['max'] - variance_data['min']) / variance_data['avg'] if variance_data['avg'] else 0
        range_factor = min(0.1, range_ratio * 0.1)
        
        # Higher confidence with more items
        items_factor = min(0.1, (len(items) - self.min_item_count) * 0.02)
        
        # Calculate final confidence (cap at 0.95)
        confidence = min(0.95, base_confidence + coef_factor + range_factor + items_factor)
        
        return confidence