"""
Rule-based anomaly detector for the financial analysis engine.

This module detects anomalies in financial data based on predefined business rules,
heuristics, and domain-specific knowledge about construction finance.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class RuleBasedAnomalyDetector:
    """Detects rule-based anomalies in financial data."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the rule-based anomaly detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.large_amount_threshold = self.config.get('large_amount_threshold', 100000.0)
        self.round_amount_threshold = self.config.get('round_amount_threshold', 1000.0)
        
        # Rule parameters
        self.suspicious_keywords = self.config.get('suspicious_keywords', [
            'extra', 'additional', 'adjustment', 'misc', 'miscellaneous', 'other',
            'contingency', 'allowance', 'unforeseen', 'unknown', 'tbd'
        ])
        self.excluded_cost_codes = self.config.get('excluded_cost_codes', [
            'overhead', 'profit', 'insurance', 'bond', 'fee'
        ])
        
    def _parse_date(self, date_value):
        """Parse date value into a datetime object.
        
        Args:
            date_value: Date value (string, datetime, or None)
            
        Returns:
            datetime object or None
        """
        if date_value is None:
            return None
            
        # If it's already a datetime object, return it
        from datetime import datetime
        if isinstance(date_value, datetime):
            return date_value
            
        # If it's a string, try to parse it
        if isinstance(date_value, str):
            try:
                # Try different date formats
                formats = [
                    '%Y-%m-%d',       # 2023-03-15
                    '%Y-%m-%dT%H:%M:%S',  # ISO format
                    '%Y-%m-%dT%H:%M:%S.%f',  # ISO format with microseconds
                    '%B %d, %Y',      # March 15, 2023
                    '%b %d, %Y',      # Mar 15, 2023
                    '%m/%d/%Y',       # 03/15/2023
                    '%d-%m-%Y',       # 15-03-2023
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
                
        return None
        
    def detect_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect rule-based anomalies in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected anomalies
        """
        logger.info(f"Detecting rule-based anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        anomalies = []
        
        # Enhanced Mathematical Verification System
        # ----------------------------------------
        
        # Cross-document running total verification
        running_total_anomalies = self.detect_running_total_inconsistencies(doc_id)
        if running_total_anomalies:
            anomalies.extend(running_total_anomalies)
            
        # Enhanced rounding pattern detection
        systematic_rounding = self.detect_systematic_rounding_patterns(doc_id)
        if systematic_rounding:
            anomalies.extend(systematic_rounding)
            
        # Multi-level calculation validation
        math_errors = self.detect_math_errors(doc_id)
        if math_errors:
            anomalies.extend(math_errors)
        
        # Original Anomaly Detection
        # -------------------------
        
        # Detect unusually large amounts
        large_amounts = self.detect_unusually_large_amounts(doc_id)
        anomalies.extend(large_amounts)
        
        # Detect suspicious descriptions
        suspicious_descriptions = self.detect_suspicious_descriptions(doc_id)
        anomalies.extend(suspicious_descriptions)
        
        # Detect suspiciously round amounts (original version, less sophisticated)
        round_amounts = self.detect_round_amounts(doc_id)
        anomalies.extend(round_amounts)
        
        # Detect end-of-month anomalies
        end_of_month = self.detect_end_of_month_anomalies(doc_id)
        anomalies.extend(end_of_month)
        
        # Detect mismatched totals
        mismatched_totals = self.detect_mismatched_totals(doc_id)
        anomalies.extend(mismatched_totals)
        
        # Detect missing change order documentation
        missing_co = self.detect_missing_change_order_documentation(doc_id)
        anomalies.extend(missing_co)
        
        logger.info(f"Found {len(anomalies)} rule-based anomalies")
        return anomalies
    
    def detect_unusually_large_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect unusually large amounts that may indicate errors or fraud.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of large amount anomalies
        """
        logger.info(f"Detecting unusually large amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount > :threshold"
        if doc_id:
            where_clause += " AND li.doc_id = :doc_id"
        
        # Get line items with large amounts
        query = text(f"""
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
                {where_clause}
            ORDER BY
                li.amount DESC
        """)
        
        params = {"threshold": self.large_amount_threshold}
        if doc_id:
            params["doc_id"] = doc_id
            
        items = self.db_session.execute(query, params).fetchall()
        
        # Convert to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        # Find category averages for comparison
        category_averages = self._get_category_averages()
        
        # Check each large amount against category averages
        anomalies = []
        for item in item_dicts:
            # Get the category (cost code or a default)
            category = item['cost_code'] or 'unknown'
            
            # Skip certain categories
            if category.lower() in [code.lower() for code in self.excluded_cost_codes]:
                continue
                
            # Get average for this category
            category_avg = category_averages.get(category, None)
            
            # If we have an average, compare the amount
            if category_avg and category_avg > 0:
                # Calculate ratio of amount to category average
                ratio = item['amount'] / category_avg
                
                # If significantly larger than category average, flag as anomaly
                if ratio > 3.0:  # Threshold for "significantly larger"
                    # Calculate confidence based on ratio
                    confidence = min(0.95, 0.7 + (ratio - 3.0) * 0.05)
                    
                    if confidence >= self.min_confidence:
                        anomaly = {
                            'type': 'large_amount',
                            'item_id': item['item_id'],
                            'doc_id': item['doc_id'],
                            'amount': float(item['amount']) if item['amount'] is not None else None,
                            'description': item['description'],
                            'cost_code': item['cost_code'],
                            'doc_type': item['doc_type'],
                            'party': item['party'],
                            'date': item['date'].isoformat() if item['date'] is not None else None,
                            'category_average': float(category_avg) if category_avg is not None else None,
                            'ratio': float(ratio) if ratio is not None else None,
                            'confidence': confidence,
                            'explanation': f"Amount ${item['amount']} is {ratio:.1f}x larger than the average for {category} (${category_avg:.2f})"
                        }
                        
                        anomalies.append(anomaly)
            else:
                # If no category average, use a simpler rule
                if item['amount'] > self.large_amount_threshold * 2:
                    confidence = min(0.95, 0.7 + (item['amount'] / self.large_amount_threshold - 2.0) * 0.05)
                    
                    if confidence >= self.min_confidence:
                        anomaly = {
                            'type': 'large_amount',
                            'item_id': item['item_id'],
                            'doc_id': item['doc_id'],
                            'amount': float(item['amount']) if item['amount'] is not None else None,
                            'description': item['description'],
                            'cost_code': item['cost_code'],
                            'doc_type': item['doc_type'],
                            'party': item['party'],
                            'date': item['date'].isoformat() if item['date'] is not None else None,
                            'threshold': float(self.large_amount_threshold),
                            'confidence': confidence,
                            'explanation': f"Amount ${item['amount']} exceeds large amount threshold (${self.large_amount_threshold:.2f})"
                        }
                        
                        anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} unusually large amount anomalies")
        return anomalies
    
    def detect_suspicious_descriptions(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect items with suspicious or vague descriptions.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of suspicious description anomalies
        """
        logger.info(f"Detecting suspicious descriptions{f' for document {doc_id}' if doc_id else ''}")
        
        # Prepare list of suspicious keywords and phrases
        suspicious_patterns = [
            f"LOWER(li.description) LIKE '% {keyword} %'" 
            for keyword in self.suspicious_keywords
        ]
        suspicious_patterns.extend([
            f"LOWER(li.description) LIKE '{keyword} %'" 
            for keyword in self.suspicious_keywords
        ])
        suspicious_patterns.extend([
            f"LOWER(li.description) LIKE '% {keyword}'" 
            for keyword in self.suspicious_keywords
        ])
        
        # Define filter and params
        where_clause = "(LENGTH(li.description) < 10 OR " + ' OR '.join(suspicious_patterns) + ") AND li.amount > 0"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Get line items with suspicious descriptions
        query = text(f"""
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
                {where_clause}
            ORDER BY
                li.amount DESC
        """)
        
        params = {"doc_id": doc_id} if doc_id else {}
        
        try:
            items = self.db_session.execute(query, params).fetchall()
        except:
            # If the complex query fails (e.g., with SQLite), fall back to a simpler approach
            fallback_query = text(f"""
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
                    li.amount > 0
                    {' AND li.doc_id = :doc_id' if doc_id else ''}
                ORDER BY
                    li.amount DESC
            """)
            
            all_items = self.db_session.execute(fallback_query, params).fetchall()
            
            # Manually filter for suspicious descriptions
            items = []
            for item in all_items:
                description = item[2]  # Description is at index 2
                if description and (
                    len(description) < 10 or 
                    any(keyword.lower() in description.lower() for keyword in self.suspicious_keywords)
                ):
                    items.append(item)
        
        # Convert to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        # Analyze suspicious descriptions
        anomalies = []
        for item in item_dicts:
            description = item['description'] or ''
            amount = item['amount']
            
            # Skip non-positive amounts
            if not amount or amount <= 0:
                continue
                
            # Calculate suspiciousness score
            suspiciousness = self._calculate_description_suspiciousness(description, amount)
            
            # If suspicious enough, flag as anomaly
            if suspiciousness['score'] >= 0.7:
                confidence = min(0.95, 0.6 + suspiciousness['score'] * 0.3)
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'suspicious_description',
                        'item_id': item['item_id'],
                        'doc_id': item['doc_id'],
                        'amount': float(item['amount']) if item['amount'] is not None else None,
                        'description': item['description'],
                        'cost_code': item['cost_code'],
                        'doc_type': item['doc_type'],
                        'party': item['party'],
                        'date': item['date'].isoformat() if item['date'] is not None else None,
                        'suspiciousness_score': suspiciousness['score'],
                        'reasons': suspiciousness['reasons'],
                        'confidence': confidence,
                        'explanation': f"Item has suspicious description '{item['description']}' with amount ${item['amount']}"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} suspicious description anomalies")
        return anomalies
    
    def detect_round_amounts(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect suspiciously round amounts that may indicate estimation rather than actual costs.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of round amount anomalies
        """
        logger.info(f"Detecting round amounts{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount >= :threshold AND (li.amount % 1000 = 0 OR li.amount % 5000 = 0 OR li.amount % 10000 = 0 OR li.amount % 25000 = 0 OR li.amount % 50000 = 0 OR li.amount % 100000 = 0)"
        if doc_id:
            where_clause += " AND li.doc_id = :doc_id"
        
        # Get line items with potentially round amounts
        query = text(f"""
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
                {where_clause}
            ORDER BY
                li.amount DESC
        """)
        
        params = {"threshold": self.round_amount_threshold}
        if doc_id:
            params["doc_id"] = doc_id
            
        try:
            items = self.db_session.execute(query, params).fetchall()
        except:
            # If the modulo operations fail (e.g., with SQLite), fall back to a simpler approach
            fallback_query = text("""
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
                    li.amount >= :threshold
                    :doc_filter
                ORDER BY
                    li.amount DESC
            """)
            
            all_items = self.db_session.execute(fallback_query, params).fetchall()
            
            # Manually filter for round amounts
            items = []
            for item in all_items:
                amount = item[3]  # Amount is at index 3
                if amount and (
                    amount % 1000 == 0 or
                    amount % 5000 == 0 or
                    amount % 10000 == 0 or
                    amount % 25000 == 0 or
                    amount % 50000 == 0 or
                    amount % 100000 == 0
                ):
                    items.append(item)
        
        # Convert to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        # Analyze round amounts
        anomalies = []
        for item in item_dicts:
            amount = item['amount']
            
            # Skip non-positive amounts
            if not amount or amount <= 0:
                continue
                
            # Calculate roundness (higher values are "rounder")
            roundness = self._calculate_amount_roundness(amount)
            
            # If round enough, flag as anomaly
            if roundness['score'] >= 0.7:
                confidence = min(0.95, 0.6 + roundness['score'] * 0.3)
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'round_amount',
                        'item_id': item['item_id'],
                        'doc_id': item['doc_id'],
                        'amount': float(item['amount']) if item['amount'] is not None else None,
                        'description': item['description'],
                        'cost_code': item['cost_code'],
                        'doc_type': item['doc_type'],
                        'party': item['party'],
                        'date': item['date'].isoformat() if item['date'] is not None else None,
                        'roundness_score': roundness['score'],
                        'roundness_level': roundness['level'],
                        'confidence': confidence,
                        'explanation': f"Amount ${item['amount']} is suspiciously round ({roundness['level']})"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} round amount anomalies")
        return anomalies
    
    def detect_end_of_month_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect financial activity clustered at the end of the month that may indicate manipulation.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of end-of-month anomalies
        """
        logger.info(f"Detecting end-of-month anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        # This rule looks for patterns of large or unusual amounts
        # clustered in the last few days of the month
        
        # Construct WHERE clause based on doc_id
        where_clause = "d.date_created IS NOT NULL"
        if doc_id:
            where_clause += " AND d.doc_id = :doc_id"
        
        # Get documents with their dates
        query = text(f"""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.party,
                d.date_created,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount
            FROM 
                documents d
            JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                {where_clause}
            GROUP BY
                d.doc_id, d.doc_type, d.party, d.date_created
            ORDER BY
                d.date_created
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        results = self.db_session.execute(query, params).fetchall()
        
        # Convert to dictionaries for easier handling
        result_dicts = []
        for result in results:
            doc_id, doc_type, party, date_created, item_count, total_amount = result
            
            if not date_created:
                continue
                
            result_dict = {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'item_count': item_count,
                'total_amount': total_amount
            }
            
            result_dicts.append(result_dict)
        
        # Group by month and party
        month_party_groups = {}
        for result in result_dicts:
            if not result['date']:
                continue
                
            month_key = result['date'].strftime('%Y-%m')
            party = result['party'] or 'unknown'
            
            group_key = f"{month_key}|{party}"
            
            if group_key not in month_party_groups:
                month_party_groups[group_key] = []
                
            month_party_groups[group_key].append(result)
        
        # Analyze each month-party group
        anomalies = []
        for group_key, group_results in month_party_groups.items():
            # Skip if too few data points
            if len(group_results) < 3:
                continue
                
            # Calculate stats for the month
            month_stats = self._calculate_month_stats(group_results)
            
            # Check for end-of-month clustering
            if month_stats['end_of_month_ratio'] > 2.0:
                # Get end-of-month documents
                end_of_month_docs = [
                    r for r in group_results 
                    if r['date'].day >= month_stats['days_in_month'] - 3
                ]
                
                for doc in end_of_month_docs:
                    # Calculate confidence based on ratio and amount
                    amount_factor = min(1.0, doc['total_amount'] / 10000.0)
                    confidence = min(0.95, 0.6 + (month_stats['end_of_month_ratio'] - 2.0) * 0.1 + amount_factor * 0.1)
                    
                    if confidence >= self.min_confidence:
                        # Get line items for this document
                        doc_items = self._get_document_items(doc['doc_id'])
                        
                        anomaly = {
                            'type': 'end_of_month_anomaly',
                            'doc_id': doc['doc_id'],
                            'doc_type': doc['doc_type'],
                            'party': doc['party'],
                            'date': doc['date'].isoformat() if doc['date'] is not None else None,
                            'total_amount': float(doc['total_amount']) if doc['total_amount'] is not None else None,
                            'end_of_month_ratio': float(month_stats['end_of_month_ratio']) if month_stats['end_of_month_ratio'] is not None else None,
                            'items': [
                                {k: (v.isoformat() if k == 'date' and v is not None else 
                                    float(v) if k == 'amount' and v is not None else v) 
                                 for k, v in item.items()} 
                                for item in doc_items
                            ],
                            'confidence': confidence,
                            'explanation': f"Document dated {doc['date'].strftime('%Y-%m-%d') if doc['date'] is not None else 'unknown'} is part of a cluster of activity at the end of the month (ratio: {month_stats['end_of_month_ratio']:.1f}x)"
                        }
                        
                        anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} end-of-month anomalies")
        return anomalies
    
    def detect_mismatched_totals(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect documents where subtotals don't match totals, indicating potential manipulation.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of mismatched total anomalies
        """
        logger.info(f"Detecting mismatched totals{f' for document {doc_id}' if doc_id else ''}")
        
        # This rule looks for documents where there are line items marked as subtotals
        # and a total, but they don't match up correctly
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount IS NOT NULL"
        if doc_id:
            where_clause += " AND d.doc_id = :doc_id"
        
        # Get documents with multiple line items
        query = text(f"""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.party,
                d.date_created,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount
            FROM 
                documents d
            JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                {where_clause}
            GROUP BY
                d.doc_id, d.doc_type, d.party, d.date_created
            HAVING
                COUNT(li.item_id) > 5
            ORDER BY
                d.date_created
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        results = self.db_session.execute(query, params).fetchall()
        
        # Convert to dictionaries for easier handling
        result_dicts = []
        for result in results:
            doc_id, doc_type, party, date_created, item_count, total_amount = result
            
            result_dict = {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'item_count': item_count,
                'total_amount': total_amount
            }
            
            result_dicts.append(result_dict)
        
        # Analyze each document for mismatched totals
        anomalies = []
        for doc in result_dicts:
            # Get line items for this document
            doc_items = self._get_document_items(doc['doc_id'])
            
            # Identify potential total line items (usually at the end, with keywords)
            potential_totals = []
            for item in doc_items:
                desc = item['description'] or ''
                desc_lower = desc.lower()
                
                # Check for total-like descriptions
                if ('total' in desc_lower or 
                    'sum' in desc_lower or 
                    'grand' in desc_lower or 
                    'contract' in desc_lower):
                    potential_totals.append(item)
            
            # If we found potential totals, check for mismatches
            if potential_totals:
                for total_item in potential_totals:
                    # Calculate expected total
                    expected_total = sum(
                        item['amount'] for item in doc_items 
                        if item['item_id'] != total_item['item_id'] and item['amount'] is not None
                    )
                    
                    # Compare with the potential total
                    if total_item['amount'] is not None and expected_total > 0:
                        difference = abs(total_item['amount'] - expected_total)
                        difference_pct = (difference / expected_total) * 100 if expected_total > 0 else 0
                        
                        # If difference is significant, flag as anomaly
                        if difference_pct > 1.0:  # More than 1% difference
                            # Calculate confidence based on difference percentage
                            confidence = min(0.95, 0.7 + min(difference_pct / 20.0, 0.25))
                            
                            if confidence >= self.min_confidence:
                                anomaly = {
                                    'type': 'mismatched_total',
                                    'doc_id': doc['doc_id'],
                                    'doc_type': doc['doc_type'],
                                    'party': doc['party'],
                                    'date': doc['date'].isoformat() if doc['date'] is not None else None,
                                    'total_item_id': total_item['item_id'],
                                    'total_description': total_item['description'],
                                    'total_amount': float(total_item['amount']) if total_item['amount'] is not None else None,
                                    'expected_total': float(expected_total) if expected_total is not None else None,
                                    'difference': float(difference) if difference is not None else None,
                                    'difference_pct': float(difference_pct) if difference_pct is not None else None,
                                    'items': [
                                        {k: (v.isoformat() if k == 'date' and v is not None else 
                                            float(v) if k == 'amount' and v is not None else v) 
                                         for k, v in item.items()} 
                                        for item in doc_items
                                    ],
                                    'confidence': confidence,
                                    'explanation': f"Document total (${total_item['amount']}) doesn't match the sum of line items (${expected_total}), difference: ${difference} ({difference_pct:.2f}%)"
                                }
                                
                                anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} mismatched total anomalies")
        return anomalies
    
    def detect_missing_change_order_documentation(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect payments without corresponding change order documentation.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of missing change order documentation anomalies
        """
        logger.info(f"Detecting missing change order documentation{f' for document {doc_id}' if doc_id else ''}")
        
        # This looks for payment applications with significant amounts that don't have
        # corresponding change order documentation
        
        # Lower the threshold to catch more issues
        threshold = 5000.0  # Amount threshold for significant payments
        
        # Construct our WHERE clause based on whether doc_id is provided
        where_clause = f"d.doc_type IN ('payment_app', 'invoice') AND li.amount > {threshold}"
        if doc_id:
            where_clause += " AND li.doc_id = :doc_id"
            
        # Find payment application line items above threshold
        query = text(f"""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                d.doc_type,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
            ORDER BY
                li.amount DESC
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
        
        payment_items = self.db_session.execute(query, params).fetchall()
        
        anomalies = []
        for item in payment_items:
            item_id, doc_id, description, amount, doc_type, date_created = item
            
            # For each significant payment item, check if there's a corresponding change order
            # Look for similar amounts, descriptions, and earlier dates
            co_query = text("""
                SELECT 
                    COUNT(li.item_id) as count,
                    MAX(d.doc_id) as latest_co_id  -- Get an example CO ID to check status
                FROM 
                    line_items li
                JOIN
                    documents d ON li.doc_id = d.doc_id
                WHERE
                    d.doc_type = 'change_order'
                    AND li.amount BETWEEN :amount_min AND :amount_max
                    AND d.date_created < :payment_date
            """)
            
            # Define a tolerance for matching amounts (1%)
            amount_precision = float(amount) * 0.01
            amount_min = float(amount - amount_precision)
            amount_max = float(amount + amount_precision)
            
            # Execute query
            co_result = self.db_session.execute(
                co_query, 
                {
                    "amount_min": amount_min, 
                    "amount_max": amount_max, 
                    "payment_date": date_created
                }
            ).fetchone()
            
            co_count = co_result[0] if co_result else 0
            co_id = co_result[1] if co_result and len(co_result) > 1 else None
            
            # Check if the change order that exists is approved (if any)
            co_status = None
            if co_id:
                status_query = text("""
                    SELECT status
                    FROM documents
                    WHERE doc_id = :co_id
                """)
                
                status_result = self.db_session.execute(status_query, {"co_id": co_id}).fetchone()
                co_status = status_result[0].lower() if status_result and status_result[0] else None
            
            # An anomaly exists if either:
            # 1. No change order exists at all for this amount
            # 2. Or the change order exists but was rejected
            missing_co = (co_count == 0)
            rejected_co = (co_status and ('reject' in co_status or 'denied' in co_status or 'decline' in co_status))
            
            if missing_co or rejected_co:
                # Calculate confidence based on amount and description
                confidence = 0.8  # Base confidence
                
                # Higher confidence for very large amounts
                if amount > threshold * 2:
                    confidence += 0.1
                
                # Higher confidence if description contains words suggesting added scope
                if description:
                    scope_terms = ['add', 'extra', 'additional', 'new', 'change', 'modify']
                    if any(term in description.lower() for term in scope_terms):
                        confidence += 0.05
                
                # Cap at 0.95
                confidence = min(0.95, confidence)
                
                if confidence >= self.min_confidence:
                    if missing_co:
                        anomaly_type = 'missing_change_order_documentation'
                        explanation = f"Payment of ${float(amount):.2f} lacks corresponding change order documentation"
                    else:  # rejected_co
                        anomaly_type = 'payment_for_rejected_change_order'
                        explanation = f"Payment of ${float(amount):.2f} made despite rejected change order"
                        confidence = min(0.95, confidence + 0.1)  # Higher confidence for this case
                    
                    anomaly = {
                        'type': anomaly_type,
                        'item_id': item_id,
                        'doc_id': doc_id,
                        'amount': float(amount) if amount is not None else None,
                        'description': description,
                        'doc_type': doc_type,
                        'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created,
                        'confidence': confidence,
                        'explanation': explanation
                    }
                    
                    # Add reference to the rejected change order if applicable
                    if rejected_co and co_id:
                        anomaly['rejected_co_id'] = co_id
                        anomaly['rejected_co_status'] = co_status
                    
                    anomalies.append(anomaly)
            
            # Also check for items that appear to be outside the original contract scope
            # based on description keywords, if they don't have change order documentation
            elif co_count == 0 and description:
                out_of_scope_terms = [
                    'extra', 'additional', 'added', 'new scope', 'added scope', 
                    'scope change', 'modification', 'extension', 'expand', 'expanded'
                ]
                
                if any(term in description.lower() for term in out_of_scope_terms):
                    confidence = 0.85  # High confidence for scope changes without documentation
                    
                    anomaly = {
                        'type': 'missing_change_order_documentation',
                        'item_id': item_id,
                        'doc_id': doc_id,
                        'amount': float(amount) if amount is not None else None,
                        'description': description,
                        'doc_type': doc_type,
                        'date': date_created.isoformat() if hasattr(date_created, 'isoformat') else date_created,
                        'confidence': confidence,
                        'explanation': f"Payment of ${float(amount):.2f} appears to be for added scope but lacks change order documentation"
                    }
                    
                    anomalies.append(anomaly)
        
        # Also detect clusters of items in payment apps that might collectively need change order documentation
        # Group payment items by document to find documents with multiple items without COs
        doc_items = {}
        for item in payment_items:
            item_id, doc_id, description, amount, doc_type, date_created = item
            
            if doc_id not in doc_items:
                doc_items[doc_id] = {
                    'doc_type': doc_type,
                    'date': date_created,
                    'items': [],
                    'total_amount': 0
                }
            
            doc_items[doc_id]['items'].append({
                'item_id': item_id,
                'description': description,
                'amount': amount
            })
            doc_items[doc_id]['total_amount'] += amount
        
        # Check each document with multiple items
        for doc_id, doc_data in doc_items.items():
            if len(doc_data['items']) >= 3:  # At least 3 items to consider it a group
                # Check if many of these items contain similar keywords suggesting they're related
                descriptions = [item['description'].lower() for item in doc_data['items'] if item['description']]
                
                # Check for common terms or item numbers in descriptions
                common_terms = self._find_common_terms(descriptions)
                
                if common_terms and len(common_terms) >= 2:
                    # This looks like a clustered group of related items
                    # Check if total amount exceeds a higher threshold
                    if doc_data['total_amount'] > threshold * 2:
                        confidence = 0.9
                        
                        anomaly = {
                            'type': 'grouped_items_missing_change_order',
                            'doc_id': doc_id,
                            'doc_type': doc_data['doc_type'],
                            'items_count': len(doc_data['items']),
                            'total_amount': float(doc_data['total_amount']),
                            'date': doc_data['date'].isoformat() if hasattr(doc_data['date'], 'isoformat') else doc_data['date'],
                            'common_terms': common_terms,
                            'confidence': confidence,
                            'explanation': f"Group of {len(doc_data['items'])} related items totaling ${float(doc_data['total_amount']):.2f} lacks change order documentation"
                        }
                        
                        anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} missing change order documentation anomalies")
        return anomalies
        
    def _find_common_terms(self, descriptions: List[str]) -> List[str]:
        """Find common terms that appear in multiple descriptions.
        
        Args:
            descriptions: List of item descriptions
            
        Returns:
            List of common terms
        """
        if not descriptions:
            return []
            
        # Extract all words
        all_words = []
        for desc in descriptions:
            # Remove common punctuation and split
            words = desc.replace(',', ' ').replace(':', ' ').replace(';', ' ').replace('.', ' ').split()
            all_words.extend([w.lower() for w in words if len(w) > 3])  # Only consider words longer than 3 chars
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # Find words that appear in at least half the descriptions
        threshold = max(2, len(descriptions) // 2)  # At least 2 occurrences or half of descriptions
        common_terms = [word for word, count in word_counts.items() if count >= threshold]
        
        return common_terms
        
    def detect_math_errors(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enhanced detection of mathematical errors in financial documents.
        
        This function performs multi-level mathematical verification:
        1. Line item level: Validates quantity × unit_price = total
        2. Category level: Validates sums within categories
        3. Document level: Validates document totals against line item sums
        4. Retainage level: Validates retainage calculations
        5. Multi-document level: Validates consistent calculation approaches
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of math error anomalies with detailed explanation
        """
        logger.info(f"Detecting math errors{f' for document {doc_id}' if doc_id else ''}")
        
        # Check all financial document types, not just payment applications and invoices
        where_clause = "(d.doc_type IN ('payment_app', 'invoice', 'change_order', 'estimate'))"
        if doc_id:
            where_clause += " AND d.doc_id = :doc_id"
            
        # Get financial documents
        doc_query = text(f"""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.party,
                d.meta_data
            FROM 
                documents d
            WHERE
                {where_clause}
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
        
        docs = self.db_session.execute(doc_query, params).fetchall()
        
        anomalies = []
        for doc in docs:
            if len(doc) == 4:
                doc_id, doc_type, party, meta_data = doc
            else:
                doc_id, doc_type = doc[0], doc[1]
                party = doc[2] if len(doc) > 2 else "unknown"
                meta_data = doc[3] if len(doc) > 3 else None
            
            # Get line items for this document with metadata
            items_query = text("""
                SELECT 
                    li.item_id,
                    li.item_number,
                    li.description,
                    li.amount,
                    li.quantity,
                    li.unit_price,
                    li.total,
                    li.cost_code,
                    li.category,
                    li.meta_data
                FROM 
                    line_items li
                WHERE
                    li.doc_id = :doc_id
                ORDER BY
                    li.item_number,
                    li.cost_code
            """)
            
            items = self.db_session.execute(items_query, {"doc_id": doc_id}).fetchall()
            item_dicts = []
            
            # Convert to dictionaries for easier handling
            for item in items:
                item_id, item_number, description, amount, quantity, unit_price, total, cost_code, category, item_meta = item
                
                item_dict = {
                    'item_id': item_id,
                    'item_number': item_number,
                    'description': description,
                    'amount': float(amount) if amount is not None else None,
                    'quantity': float(quantity) if quantity is not None else None,
                    'unit_price': float(unit_price) if unit_price is not None else None,
                    'total': float(total) if total is not None else None,
                    'cost_code': cost_code,
                    'category': category,
                    'meta_data': item_meta
                }
                
                # Extract percentage complete from metadata if available
                if item_meta and isinstance(item_meta, dict):
                    if 'percent_complete' in item_meta:
                        try:
                            percent_complete = item_meta['percent_complete']
                            if isinstance(percent_complete, str) and percent_complete.endswith('%'):
                                percent_complete = float(percent_complete.rstrip('%')) / 100
                            else:
                                percent_complete = float(percent_complete)
                            item_dict['percent_complete'] = percent_complete
                        except (ValueError, TypeError):
                            pass
                    
                    if 'scheduled_value' in item_meta:
                        try:
                            item_dict['scheduled_value'] = float(item_meta['scheduled_value'])
                        except (ValueError, TypeError):
                            pass
                    
                    if 'previous_completed' in item_meta:
                        try:
                            item_dict['previous_completed'] = float(item_meta['previous_completed'])
                        except (ValueError, TypeError):
                            pass
                            
                item_dicts.append(item_dict)
            
            # LEVEL 1: Check for line-item level math errors
            for item in item_dicts:
                # Check for unit price * quantity calculation errors
                if all(k in item and item[k] is not None for k in ['quantity', 'unit_price', 'total']):
                    expected_total = item['quantity'] * item['unit_price']
                    actual_total = item['total']
                    
                    # If there's a discrepancy, flag it
                    if abs(expected_total - actual_total) > 0.01:  # Allow for small rounding differences
                        confidence = 0.95  # Math errors are pretty clear
                        
                        # Check if the error benefits the document submitter
                        error_benefits_submitter = False
                        if doc_type in ['invoice', 'payment_app', 'change_order'] and party == 'contractor':
                            error_benefits_submitter = actual_total > expected_total
                            if error_benefits_submitter:
                                confidence = 0.98  # Higher confidence when error benefits submitter
                        
                        anomaly = {
                            'type': 'math_error',
                            'subtype': 'quantity_unit_price_mismatch',
                            'item_id': item['item_id'],
                            'doc_id': doc_id,
                            'doc_type': doc_type,
                            'party': party,
                            'description': item['description'],
                            'quantity': item['quantity'],
                            'unit_price': item['unit_price'],
                            'expected_total': expected_total,
                            'actual_total': actual_total,
                            'difference': actual_total - expected_total,
                            'difference_percent': (abs(expected_total - actual_total) / max(expected_total, actual_total)) * 100 if max(expected_total, actual_total) > 0 else 0,
                            'benefits_submitter': error_benefits_submitter,
                            'confidence': confidence,
                            'explanation': f"Math error in calculation: {item['quantity']} x ${item['unit_price']:.2f} = ${expected_total:.2f}, but total is ${actual_total:.2f}" + 
                                         (f" (benefits {party})" if error_benefits_submitter else "")
                        }
                        
                        anomalies.append(anomaly)
                
                # Check if line item total matches claimed amount
                if all(k in item and item[k] is not None for k in ['amount', 'total']) and abs(item['amount'] - item['total']) > 0.01:
                    confidence = 0.9
                    
                    # Check if the error benefits the document submitter
                    error_benefits_submitter = False
                    if doc_type in ['invoice', 'payment_app', 'change_order'] and party == 'contractor':
                        error_benefits_submitter = item['amount'] > item['total']
                        if error_benefits_submitter:
                            confidence = 0.95  # Higher confidence when error benefits submitter
                    
                    anomaly = {
                        'type': 'math_error',
                        'subtype': 'amount_total_mismatch',
                        'item_id': item['item_id'],
                        'doc_id': doc_id,
                        'doc_type': doc_type,
                        'party': party,
                        'description': item['description'],
                        'amount': item['amount'],
                        'total': item['total'],
                        'difference': item['amount'] - item['total'],
                        'difference_percent': (abs(item['amount'] - item['total']) / max(item['amount'], item['total'])) * 100 if max(item['amount'], item['total']) > 0 else 0,
                        'benefits_submitter': error_benefits_submitter,
                        'confidence': confidence,
                        'explanation': f"Discrepancy between amount (${item['amount']:.2f}) and total (${item['total']:.2f})" +
                                     (f" (benefits {party})" if error_benefits_submitter else "")
                    }
                    
                    anomalies.append(anomaly)
                
                # Check for percent complete * scheduled value = amount
                if 'percent_complete' in item and 'scheduled_value' in item and 'amount' in item and item['amount'] is not None:
                    expected_amount = item['percent_complete'] * item['scheduled_value']
                    actual_amount = item['amount']
                    
                    # If there's a discrepancy, flag it (allow for slightly larger rounding differences)
                    if abs(expected_amount - actual_amount) > 0.1 and (abs(expected_amount - actual_amount) / max(expected_amount, actual_amount) > 0.01):
                        confidence = 0.9
                        
                        # Check if the error benefits the document submitter
                        error_benefits_submitter = False
                        if doc_type in ['invoice', 'payment_app'] and party == 'contractor':
                            error_benefits_submitter = actual_amount > expected_amount
                            if error_benefits_submitter:
                                confidence = 0.95  # Higher confidence when error benefits submitter
                        
                        anomaly = {
                            'type': 'math_error',
                            'subtype': 'percent_complete_mismatch',
                            'item_id': item['item_id'],
                            'doc_id': doc_id,
                            'doc_type': doc_type,
                            'party': party,
                            'description': item['description'],
                            'percent_complete': item['percent_complete'] * 100,  # For readability
                            'scheduled_value': item['scheduled_value'],
                            'expected_amount': expected_amount,
                            'actual_amount': actual_amount,
                            'difference': actual_amount - expected_amount,
                            'difference_percent': (abs(expected_amount - actual_amount) / max(expected_amount, actual_amount)) * 100 if max(expected_amount, actual_amount) > 0 else 0,
                            'benefits_submitter': error_benefits_submitter,
                            'confidence': confidence,
                            'explanation': f"Math error in percentage calculation: {item['percent_complete']*100:.1f}% of ${item['scheduled_value']:.2f} = ${expected_amount:.2f}, but amount is ${actual_amount:.2f}" +
                                         (f" (benefits {party})" if error_benefits_submitter else "")
                        }
                        
                        anomalies.append(anomaly)
                
                # Check for current work = total - previous (for payment apps)
                if doc_type == 'payment_app' and 'previous_completed' in item and 'amount' in item and item['amount'] is not None and 'total' in item and item['total'] is not None:
                    expected_current = item['total'] - item['previous_completed']
                    actual_current = item['amount']
                    
                    # If there's a discrepancy, flag it
                    if abs(expected_current - actual_current) > 0.1 and (abs(expected_current - actual_current) / max(abs(expected_current), abs(actual_current)) > 0.01):
                        confidence = 0.9
                        
                        # Check if the error benefits the document submitter
                        error_benefits_submitter = False
                        if party == 'contractor':
                            error_benefits_submitter = actual_current > expected_current
                            if error_benefits_submitter:
                                confidence = 0.95  # Higher confidence when error benefits submitter
                        
                        anomaly = {
                            'type': 'math_error',
                            'subtype': 'current_work_mismatch',
                            'item_id': item['item_id'],
                            'doc_id': doc_id,
                            'doc_type': doc_type,
                            'party': party,
                            'description': item['description'],
                            'total_completed': item['total'],
                            'previous_completed': item['previous_completed'],
                            'expected_current': expected_current,
                            'actual_current': actual_current,
                            'difference': actual_current - expected_current,
                            'difference_percent': (abs(expected_current - actual_current) / max(abs(expected_current), abs(actual_current))) * 100 if max(abs(expected_current), abs(actual_current)) > 0 else 0,
                            'benefits_submitter': error_benefits_submitter,
                            'confidence': confidence,
                            'explanation': f"Math error in current work calculation: ${item['total']:.2f} - ${item['previous_completed']:.2f} = ${expected_current:.2f}, but current is ${actual_current:.2f}" +
                                         (f" (benefits {party})" if error_benefits_submitter else "")
                        }
                        
                        anomalies.append(anomaly)
            
            # LEVEL 2: Check for category-level math errors
            categories = {}
            for item in item_dicts:
                if 'category' in item and item['category'] and 'amount' in item and item['amount'] is not None:
                    if item['category'] not in categories:
                        categories[item['category']] = 0
                    categories[item['category']] += item['amount']
            
            # If we have category totals in metadata, compare them
            if meta_data and isinstance(meta_data, dict) and 'category_totals' in meta_data:
                for category, reported_total in meta_data['category_totals'].items():
                    if category in categories:
                        try:
                            reported_total = float(reported_total)
                            calculated_total = categories[category]
                            
                            # Check for discrepancy
                            if abs(reported_total - calculated_total) > 0.01 and (abs(reported_total - calculated_total) / max(reported_total, calculated_total) > 0.01):
                                confidence = 0.9
                                
                                # Check if the error benefits the document submitter
                                error_benefits_submitter = False
                                if doc_type in ['invoice', 'payment_app', 'change_order'] and party == 'contractor':
                                    error_benefits_submitter = reported_total > calculated_total
                                    if error_benefits_submitter:
                                        confidence = 0.95  # Higher confidence when error benefits submitter
                                
                                anomaly = {
                                    'type': 'math_error',
                                    'subtype': 'category_total_mismatch',
                                    'doc_id': doc_id,
                                    'doc_type': doc_type,
                                    'party': party,
                                    'category': category,
                                    'reported_total': reported_total,
                                    'calculated_total': calculated_total,
                                    'difference': reported_total - calculated_total,
                                    'difference_percent': (abs(reported_total - calculated_total) / max(reported_total, calculated_total)) * 100,
                                    'benefits_submitter': error_benefits_submitter,
                                    'confidence': confidence,
                                    'explanation': f"Category total mismatch for '{category}': reported ${reported_total:.2f}, calculated ${calculated_total:.2f}" +
                                                 (f" (benefits {party})" if error_benefits_submitter else "")
                                }
                                
                                anomalies.append(anomaly)
                        except (ValueError, TypeError):
                            continue
            
            # LEVEL 3: Check document-level totals vs. sum of items
            # Get sum of line items
            sum_query = text("""
                SELECT 
                    SUM(li.amount) as items_total
                FROM 
                    line_items li
                WHERE
                    li.doc_id = :doc_id
            """)
            
            try:
                sum_result = self.db_session.execute(sum_query, {"doc_id": doc_id}).fetchone()
                items_total = float(sum_result[0]) if sum_result and sum_result[0] is not None else 0
                
                # Extract all relevant document-level totals
                doc_totals = {}
                if meta_data and isinstance(meta_data, dict):
                    # Get all potential total fields
                    total_fields = [
                        'total_amount', 'total', 'doc_total', 'document_total', 'amount',
                        'amount_requested', 'amount_approved', 'completed_to_date',
                        'contract_sum', 'contract_sum_to_date', 'change_order_amount',
                        'total_completed_and_stored', 'total_completed'
                    ]
                    
                    for field in total_fields:
                        if field in meta_data and meta_data[field] is not None:
                            try:
                                doc_totals[field] = float(meta_data[field])
                            except (ValueError, TypeError):
                                pass
                
                # Check all relevant document totals
                for field, doc_total in doc_totals.items():
                    # Skip if the field is not a direct sum of line items
                    if field in ['contract_sum', 'contract_sum_to_date']:
                        continue
                        
                    # For completed_to_date or total_completed_and_stored, we expect this to match items_total
                    if field in ['completed_to_date', 'total_completed_and_stored', 'total_completed']:
                        # Check if there's a discrepancy (more than 1% difference)
                        diff = abs(doc_total - items_total)
                        if diff > 0.01 and (diff / max(doc_total, items_total)) > 0.01:
                            confidence = 0.95  # Document level math errors are very suspicious
                            
                            # Check if the error benefits the document submitter
                            error_benefits_submitter = False
                            if doc_type in ['invoice', 'payment_app', 'change_order'] and party == 'contractor':
                                error_benefits_submitter = doc_total > items_total
                                if error_benefits_submitter:
                                    confidence = 0.98  # Higher confidence when error benefits submitter
                            
                            anomaly = {
                                'type': 'math_error',
                                'subtype': 'document_total_mismatch',
                                'doc_id': doc_id,
                                'doc_type': doc_type,
                                'party': party,
                                'total_field': field,
                                'document_total': doc_total,
                                'line_items_sum': items_total,
                                'difference': doc_total - items_total,
                                'difference_percent': (diff / max(doc_total, items_total)) * 100,
                                'benefits_submitter': error_benefits_submitter,
                                'confidence': confidence,
                                'explanation': f"Document {field} (${doc_total:.2f}) doesn't match sum of line items (${items_total:.2f}), difference: ${diff:.2f}" +
                                             (f" (benefits {party})" if error_benefits_submitter else "")
                            }
                            
                            anomalies.append(anomaly)
                
                # LEVEL 4: Validate retainage calculations
                if 'completed_to_date' in doc_totals and 'retainage' in doc_totals and 'retainage_percent' in meta_data:
                    try:
                        retainage_percent = float(meta_data['retainage_percent'])
                        if retainage_percent > 1:  # Assume it's a percentage like 5.0
                            retainage_percent /= 100
                            
                        expected_retainage = doc_totals['completed_to_date'] * retainage_percent
                        actual_retainage = doc_totals['retainage']
                        
                        # Check for discrepancy
                        diff = abs(expected_retainage - actual_retainage)
                        if diff > 0.01 and (diff / max(expected_retainage, actual_retainage) > 0.01):
                            confidence = 0.95
                            
                            # Check if the error benefits the document submitter
                            error_benefits_submitter = False
                            if doc_type == 'payment_app' and party == 'contractor':
                                error_benefits_submitter = actual_retainage < expected_retainage  # Less retainage held back
                                if error_benefits_submitter:
                                    confidence = 0.98
                            
                            anomaly = {
                                'type': 'math_error',
                                'subtype': 'retainage_calculation_error',
                                'doc_id': doc_id,
                                'doc_type': doc_type,
                                'party': party,
                                'completed_to_date': doc_totals['completed_to_date'],
                                'retainage_percent': retainage_percent * 100,  # For readability
                                'expected_retainage': expected_retainage,
                                'actual_retainage': actual_retainage,
                                'difference': actual_retainage - expected_retainage,
                                'difference_percent': (diff / max(expected_retainage, actual_retainage)) * 100,
                                'benefits_submitter': error_benefits_submitter,
                                'confidence': confidence,
                                'explanation': f"Retainage calculation error: {retainage_percent*100:.1f}% of ${doc_totals['completed_to_date']:.2f} = ${expected_retainage:.2f}, but reported retainage is ${actual_retainage:.2f}" +
                                             (f" (benefits {party})" if error_benefits_submitter else "")
                            }
                            
                            anomalies.append(anomaly)
                    except (ValueError, TypeError, KeyError):
                        pass
                
                # Check if current payment due is calculated correctly
                if all(k in doc_totals for k in ['total_earned_less_retainage', 'previous_certificates', 'current_payment_due']):
                    expected_payment = doc_totals['total_earned_less_retainage'] - doc_totals['previous_certificates']
                    actual_payment = doc_totals['current_payment_due']
                    
                    # Check for discrepancy
                    diff = abs(expected_payment - actual_payment)
                    if diff > 0.01 and (diff / max(abs(expected_payment), abs(actual_payment)) > 0.01):
                        confidence = 0.95
                        
                        # Check if the error benefits the document submitter
                        error_benefits_submitter = False
                        if doc_type == 'payment_app' and party == 'contractor':
                            error_benefits_submitter = actual_payment > expected_payment
                            if error_benefits_submitter:
                                confidence = 0.98
                        
                        anomaly = {
                            'type': 'math_error',
                            'subtype': 'payment_calculation_error',
                            'doc_id': doc_id,
                            'doc_type': doc_type,
                            'party': party,
                            'total_earned': doc_totals['total_earned_less_retainage'],
                            'previous_certificates': doc_totals['previous_certificates'],
                            'expected_payment': expected_payment,
                            'actual_payment': actual_payment,
                            'difference': actual_payment - expected_payment,
                            'difference_percent': (diff / max(abs(expected_payment), abs(actual_payment))) * 100,
                            'benefits_submitter': error_benefits_submitter,
                            'confidence': confidence,
                            'explanation': f"Payment calculation error: ${doc_totals['total_earned_less_retainage']:.2f} - ${doc_totals['previous_certificates']:.2f} = ${expected_payment:.2f}, but current payment due is ${actual_payment:.2f}" +
                                         (f" (benefits {party})" if error_benefits_submitter else "")
                        }
                        
                        anomalies.append(anomaly)
                
            except Exception as e:
                logger.warning(f"Error calculating document totals: {e}")
        
        # Look for systematic mathematical errors across multiple documents
        # Count errors by party to identify systematic bias
        error_counts = {}
        for anomaly in anomalies:
            if 'party' in anomaly and 'benefits_submitter' in anomaly and anomaly['benefits_submitter']:
                party = anomaly['party']
                if party not in error_counts:
                    error_counts[party] = 0
                error_counts[party] += 1
        
        # If a party has 3 or more errors that benefit them, flag as systematic bias
        for party, count in error_counts.items():
            if count >= 3:
                systematic_anomaly = {
                    'type': 'systematic_math_errors',
                    'party': party,
                    'error_count': count,
                    'confidence': min(0.95, 0.75 + count * 0.05),
                    'explanation': f"Systematic pattern of mathematical errors detected: {count} errors benefit {party}"
                }
                
                anomalies.append(systematic_anomaly)
            
        logger.info(f"Found {len(anomalies)} math error anomalies")
        return anomalies
        
    def detect_amount_anomalies(self, amount: float, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect rule-based anomalies related to a specific amount.
        
        Args:
            amount: Amount to analyze
            matches: List of matches for this amount
            
        Returns:
            List of anomalies related to this amount
        """
        logger.info(f"Detecting rule-based anomalies for amount: ${amount}")
        
        anomalies = []
        
        # Check for round amounts
        if amount >= self.round_amount_threshold:
            roundness = self._calculate_amount_roundness(amount)
            
            if roundness['score'] >= 0.7:
                confidence = min(0.95, 0.6 + roundness['score'] * 0.3)
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'specific_amount_roundness',
                        'amount': amount,
                        'roundness_score': roundness['score'],
                        'roundness_level': roundness['level'],
                        'matches': matches,
                        'confidence': confidence,
                        'explanation': f"Amount ${amount} is suspiciously round ({roundness['level']})"
                    }
                    
                    anomalies.append(anomaly)
        
        # Check for suspicious descriptions in matches
        suspicious_matches = []
        for match in matches:
            description = match.get('description', '')
            if description:
                suspiciousness = self._calculate_description_suspiciousness(description, amount)
                
                if suspiciousness['score'] >= 0.7:
                    suspicious_matches.append({
                        'match': match,
                        'suspiciousness': suspiciousness
                    })
        
        if suspicious_matches:
            # Average confidence across suspicious matches
            avg_confidence = sum(sm['suspiciousness']['score'] for sm in suspicious_matches) / len(suspicious_matches)
            confidence = min(0.95, 0.6 + avg_confidence * 0.3)
            
            if confidence >= self.min_confidence:
                anomaly = {
                    'type': 'specific_amount_suspicious_descriptions',
                    'amount': amount,
                    'suspicious_matches': suspicious_matches,
                    'confidence': confidence,
                    'explanation': f"Amount ${amount} appears with suspicious descriptions in {len(suspicious_matches)} occurrences"
                }
                
                anomalies.append(anomaly)
        
        # Check for end-of-month clustering
        month_counts = {}
        end_of_month_counts = {}
        
        for match in matches:
            date = match.get('date')
            if date:
                # Convert string dates to datetime if needed
                if isinstance(date, str):
                    try:
                        from datetime import datetime
                        date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue
                
                # If we now have a valid datetime object, process it
                if hasattr(date, 'strftime'):
                    month_key = date.strftime('%Y-%m')
                    
                    # Count by month
                    if month_key not in month_counts:
                        month_counts[month_key] = 0
                    month_counts[month_key] += 1
                    
                    # Count end-of-month occurrences
                    if date.day >= 25:
                        if month_key not in end_of_month_counts:
                            end_of_month_counts[month_key] = 0
                        end_of_month_counts[month_key] += 1
        
        # Check for months with high end-of-month ratios
        for month_key, count in month_counts.items():
            if month_key in end_of_month_counts and count >= 3:
                end_of_month_ratio = end_of_month_counts[month_key] / (count * (6 / 30))  # Normalize by expected ratio
                
                if end_of_month_ratio > 2.0:
                    confidence = min(0.95, 0.7 + (end_of_month_ratio - 2.0) * 0.1)
                    
                    if confidence >= self.min_confidence:
                        # Filter matches for this month
                        month_matches = [
                            match for match in matches 
                            if match.get('date') and match['date'].strftime('%Y-%m') == month_key
                        ]
                        
                        anomaly = {
                            'type': 'specific_amount_end_of_month',
                            'amount': amount,
                            'month': month_key,
                            'end_of_month_ratio': float(end_of_month_ratio),
                            'month_matches': month_matches,
                            'confidence': confidence,
                            'explanation': f"Amount ${amount} shows clustering at the end of month {month_key} (ratio: {end_of_month_ratio:.1f}x)"
                        }
                        
                        anomalies.append(anomaly)
        
        # Check for missing change order documentation
        payment_matches = [
            match for match in matches 
            if match.get('document', {}).get('doc_type', '').lower() in ['payment_app', 'invoice']
        ]
        
        if payment_matches and amount >= 5000.0:
            # Check if there are any change order matches for this amount
            has_change_order = any(
                match.get('document', {}).get('doc_type', '').lower() == 'change_order'
                for match in matches
            )
            
            if not has_change_order:
                confidence = 0.8
                
                # Higher confidence for larger amounts
                if amount > 15000.0:
                    confidence += 0.1
                
                anomaly = {
                    'type': 'missing_change_order_for_amount',
                    'amount': amount,
                    'payment_matches': payment_matches,
                    'confidence': confidence,
                    'explanation': f"Amount ${amount:.2f} appears in payment documents without corresponding change order"
                }
                
                anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} rule-based anomalies for amount: ${amount}")
        return anomalies
    
    def _get_category_averages(self) -> Dict[str, float]:
        """Get average amounts by category (cost code).
        
        Returns:
            Dict mapping category to average amount
        """
        query = text("""
            SELECT 
                li.cost_code,
                AVG(li.amount) AS avg_amount
            FROM 
                line_items li
            WHERE
                li.cost_code IS NOT NULL
                AND li.amount > 0
            GROUP BY
                li.cost_code
        """)
        
        try:
            results = self.db_session.execute(query).fetchall()
        except:
            # If the query fails, return an empty dict
            return {}
        
        # Convert to dictionary
        return {cost_code: avg_amount for cost_code, avg_amount in results}
    
    def _calculate_description_suspiciousness(self, description: str, amount: Decimal) -> Dict[str, Any]:
        """Calculate suspiciousness score for a description.
        
        Args:
            description: Item description
            amount: Item amount
            
        Returns:
            Dict with score and reasons
        """
        if not description:
            return {'score': 0.9, 'reasons': ['Empty description']}
            
        description_lower = description.lower()
        
        # Initialize score and reasons
        score = 0.0
        reasons = []
        
        # Check length
        if len(description) < 5:
            score += 0.4
            reasons.append('Very short description')
        elif len(description) < 10:
            score += 0.3
            reasons.append('Short description')
        
        # Check for suspicious keywords
        for keyword in self.suspicious_keywords:
            if keyword.lower() in description_lower:
                score += 0.3
                reasons.append(f"Contains suspicious keyword '{keyword}'")
                break
        
        # Check for vagueness
        vague_indicators = ['etc', '...', 'misc', 'other', 'additional']
        for indicator in vague_indicators:
            if indicator in description_lower:
                score += 0.2
                reasons.append('Vague description')
                break
        
        # Check for large amounts with minimal description
        if amount > self.large_amount_threshold and len(description) < 15:
            score += 0.3
            reasons.append('Large amount with minimal description')
        
        # Cap score at 1.0
        return {
            'score': min(1.0, score),
            'reasons': reasons
        }
    
    def _calculate_amount_roundness(self, amount: Decimal) -> Dict[str, Any]:
        """Calculate roundness score for an amount.
        
        Args:
            amount: Amount to analyze
            
        Returns:
            Dict with score and level
        """
        # Convert to string for digit analysis
        amount_str = str(amount)
        
        # Remove decimal point and trailing zeros
        if '.' in amount_str:
            integer_part, decimal_part = amount_str.split('.')
            # Remove trailing zeros from decimal part
            decimal_part = decimal_part.rstrip('0')
            
            if decimal_part:
                # Has non-zero decimal part
                digit_str = integer_part + decimal_part
                has_decimal = True
            else:
                # No significant decimal part
                digit_str = integer_part
                has_decimal = False
        else:
            digit_str = amount_str
            has_decimal = False
        
        # Count trailing zeros
        trailing_zeros = 0
        for char in reversed(digit_str):
            if char == '0':
                trailing_zeros += 1
            else:
                break
        
        # Determine level of roundness
        if amount % 100000 == 0:
            level = 'Hundred thousand'
            score = 0.95
        elif amount % 50000 == 0:
            level = 'Fifty thousand'
            score = 0.90
        elif amount % 25000 == 0:
            level = 'Twenty-five thousand'
            score = 0.85
        elif amount % 10000 == 0:
            level = 'Ten thousand'
            score = 0.80
        elif amount % 5000 == 0:
            level = 'Five thousand'
            score = 0.75
        elif amount % 1000 == 0:
            level = 'Thousand'
            score = 0.70
        elif amount % 500 == 0:
            level = 'Five hundred'
            score = 0.65
        elif amount % 100 == 0:
            level = 'Hundred'
            score = 0.60
        elif amount % 50 == 0:
            level = 'Fifty'
            score = 0.55
        elif amount % 10 == 0:
            level = 'Ten'
            score = 0.50
        elif amount % 5 == 0:
            level = 'Five'
            score = 0.45
        else:
            # Check for repeating digits (e.g. 11111, 22222)
            repeating = False
            prev_digit = None
            repeat_count = 0
            for digit in digit_str:
                if digit == prev_digit:
                    repeat_count += 1
                    if repeat_count >= 3:  # At least 4 same digits in a row
                        repeating = True
                        break
                else:
                    repeat_count = 0
                prev_digit = digit
            
            if repeating:
                level = 'Repeating digits'
                score = 0.70
            else:
                # Check for common multipliers like 9,999 or 99.99
                common_patterns = ['9' * x for x in range(3, 6)]  # 999, 9999, 99999
                for pattern in common_patterns:
                    if pattern in digit_str:
                        level = 'Common price pattern'
                        score = 0.60
                        break
                else:
                    level = 'Not particularly round'
                    score = 0.0
        
        # Adjust score based on amount size (larger round amounts are more suspicious)
        if score > 0:
            size_factor = min(0.1, 0.1 * (len(digit_str) - trailing_zeros - 1) / 5)
            score += size_factor
        
        # If there are decimals, reduce the score
        if has_decimal:
            score *= 0.8
        
        return {
            'score': min(1.0, score),
            'level': level
        }
        
    def detect_systematic_rounding_patterns(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect systematic rounding patterns that may indicate manipulation.
        
        This function analyzes rounding patterns across documents to identify systematic
        biases that may favor specific parties.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of systematic rounding pattern anomalies
        """
        logger.info(f"Detecting systematic rounding patterns{f' for document {doc_id}' if doc_id else ''}")
        
        # Get line items with their rounding information
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                d.doc_type,
                d.party
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount IS NOT NULL
                AND li.amount > 0
                :doc_filter
            ORDER BY
                d.party,
                d.doc_type
        """)
        
        doc_filter = ""
        params = {}
        if doc_id:
            doc_filter = "AND li.doc_id = :doc_id"
            params["doc_id"] = doc_id
            
        query_text = query.text.replace(':doc_filter', doc_filter)
        items = self.db_session.execute(text(query_text), params).fetchall()
        
        # Group items by party
        party_items = {}
        for item in items:
            item_id, doc_id, description, amount, doc_type, party = item
            if party not in party_items:
                party_items[party] = []
            
            # Calculate roundness
            roundness = self._calculate_amount_roundness(amount)
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': float(amount),
                'doc_type': doc_type,
                'roundness_score': roundness['score'],
                'roundness_level': roundness['level']
            }
            
            party_items[party].append(item_dict)
        
        # Analyze rounding patterns by party
        anomalies = []
        for party, items in party_items.items():
            if len(items) < 5:  # Need enough items to detect a pattern
                continue
                
            # Count round amounts by rounding level
            rounding_counts = {}
            total_items = len(items)
            round_items = 0
            
            for item in items:
                level = item['roundness_level']
                score = item['roundness_score']
                
                if score >= 0.5:  # Significantly round
                    round_items += 1
                    if level not in rounding_counts:
                        rounding_counts[level] = 0
                    rounding_counts[level] += 1
            
            if round_items == 0:
                continue
                
            # Calculate percentage of round amounts
            round_percent = round_items / total_items
            
            # If more than 50% of amounts are round, flag as suspicious
            if round_percent >= 0.5:
                # Prepare rounding breakdown for explanation
                rounding_breakdown = ", ".join([f"{count} {level}" for level, count in rounding_counts.items()])
                confidence = min(0.95, 0.75 + (round_percent - 0.5) * 2)
                
                anomaly = {
                    'type': 'systematic_rounding_pattern',
                    'party': party,
                    'total_items': total_items,
                    'round_items': round_items,
                    'round_percent': round_percent * 100,
                    'rounding_counts': rounding_counts,
                    'confidence': confidence,
                    'explanation': f"Systematic rounding pattern detected for {party}: {round_percent:.1%} of amounts are suspiciously round ({rounding_breakdown})"
                }
                
                anomalies.append(anomaly)
                
            # Now check for up-rounding vs down-rounding patterns
            # This requires having the original documents to compare with subcontractor invoices
            # which may not be available in all cases
            
        # Now check for consistent rounding that benefits a specific party
        if len(party_items) >= 2:
            # Get owner and contractor
            owner_items = party_items.get('owner', [])
            contractor_items = party_items.get('contractor', [])
            
            if owner_items and contractor_items:
                # Calculate average roundness scores
                owner_scores = [item['roundness_score'] for item in owner_items]
                contractor_scores = [item['roundness_score'] for item in contractor_items]
                
                owner_avg = sum(owner_scores) / len(owner_scores)
                contractor_avg = sum(contractor_scores) / len(contractor_scores)
                
                # If there's a significant difference, flag it
                diff = abs(owner_avg - contractor_avg)
                if diff >= 0.2:
                    favored_party = 'contractor' if contractor_avg > owner_avg else 'owner'
                    confidence = min(0.95, 0.75 + diff)
                    
                    anomaly = {
                        'type': 'biased_rounding_pattern',
                        'owner_score': owner_avg,
                        'contractor_score': contractor_avg,
                        'difference': diff,
                        'favored_party': favored_party,
                        'confidence': confidence,
                        'explanation': f"Biased rounding pattern detected: {favored_party}'s amounts are significantly more rounded (difference: {diff:.2f})"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} systematic rounding pattern anomalies")
        return anomalies
    
    def detect_running_total_inconsistencies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect inconsistencies in running totals across sequential payment applications.
        
        This function analyzes sequential payment applications to verify:
        1. Previous payment certificate amounts match with reported values
        2. Running totals are mathematically consistent
        3. Line item progress is logically consistent
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of running total inconsistency anomalies
        """
        logger.info(f"Detecting running total inconsistencies{f' for document {doc_id}' if doc_id else ''}")
        
        # Get all payment applications, sorted by date/number
        where_clause = "d.doc_type = 'payment_app'"
        if doc_id:
            where_clause += " AND d.doc_id = :doc_id"
            
        # Get payment applications with their metadata
        query = text(f"""
            SELECT 
                d.doc_id,
                d.meta_data,
                pa.payment_app_number,
                pa.period_start,
                pa.period_end,
                pa.amount_requested,
                pa.amount_approved,
                pa.status
            FROM 
                documents d
            JOIN
                payment_applications pa ON d.doc_id = pa.doc_id
            WHERE
                {where_clause}
            ORDER BY
                pa.payment_app_number ASC,
                pa.period_start ASC
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        pay_apps = self.db_session.execute(query, params).fetchall()
        
        # Convert to list of dictionaries for easier handling
        pay_app_dicts = []
        for app in pay_apps:
            doc_id, meta_data, app_number, period_start, period_end, requested, approved, status = app
            
            # Parse dates if needed
            period_start_date = self._parse_date(period_start)
            period_end_date = self._parse_date(period_end)
            
            pay_app_dict = {
                'doc_id': doc_id,
                'meta_data': meta_data,
                'payment_app_number': app_number,
                'period_start': period_start_date,
                'period_end': period_end_date,
                'amount_requested': float(requested) if requested is not None else None,
                'amount_approved': float(approved) if approved is not None else None,
                'status': status
            }
            
            # Extract document-level totals from metadata
            if meta_data and isinstance(meta_data, dict):
                for key, value in meta_data.items():
                    if key.lower() in ('completed_to_date', 'previous_certificates', 'current_payment_due', 
                                      'retainage', 'total_earned_less_retainage', 'contract_sum_to_date'):
                        try:
                            pay_app_dict[key.lower()] = float(value)
                        except (ValueError, TypeError):
                            pass
            
            pay_app_dicts.append(pay_app_dict)
        
        # If we have fewer than 2 payment applications, we can't check for running total issues
        if len(pay_app_dicts) < 2:
            return []
            
        # Sort by payment_app_number to ensure correct sequence
        pay_app_dicts.sort(key=lambda x: x['payment_app_number'] if x['payment_app_number'] is not None else 0)
        
        # Check for inconsistencies between sequential payment applications
        anomalies = []
        for i in range(1, len(pay_app_dicts)):
            current_app = pay_app_dicts[i]
            previous_app = pay_app_dicts[i-1]
            
            # Check 1: Does the 'previous_certificates' in current app match the total earned in previous app?
            if ('previous_certificates' in current_app and 
                'total_earned_less_retainage' in previous_app and
                current_app['previous_certificates'] is not None and 
                previous_app['total_earned_less_retainage'] is not None):
                
                current_prev_certs = current_app['previous_certificates']
                previous_total_earned = previous_app['total_earned_less_retainage']
                
                # Allow for small rounding differences (0.01 or 0.1% difference)
                diff = abs(current_prev_certs - previous_total_earned)
                if diff > 0.01 and (diff / max(current_prev_certs, previous_total_earned) > 0.001):
                    confidence = min(0.95, 0.80 + (diff / max(current_prev_certs, previous_total_earned)))
                    
                    anomaly = {
                        'type': 'previous_certificate_mismatch',
                        'doc_id': current_app['doc_id'],
                        'previous_doc_id': previous_app['doc_id'],
                        'payment_app_number': current_app['payment_app_number'],
                        'previous_payment_app_number': previous_app['payment_app_number'],
                        'reported_previous_certificates': current_prev_certs,
                        'actual_previous_total': previous_total_earned,
                        'difference': current_prev_certs - previous_total_earned,
                        'difference_percent': (diff / max(current_prev_certs, previous_total_earned)) * 100,
                        'confidence': confidence,
                        'explanation': f"Payment Application #{current_app['payment_app_number']} reports previous certificates of ${current_prev_certs:.2f}, but Payment Application #{previous_app['payment_app_number']} shows total earned of ${previous_total_earned:.2f}"
                    }
                    
                    anomalies.append(anomaly)
            
            # Check 2: Is the current payment due calculation correct?
            if ('total_earned_less_retainage' in current_app and 
                'previous_certificates' in current_app and
                'current_payment_due' in current_app and
                current_app['total_earned_less_retainage'] is not None and
                current_app['previous_certificates'] is not None and
                current_app['current_payment_due'] is not None):
                
                earned = current_app['total_earned_less_retainage']
                prev_certs = current_app['previous_certificates']
                payment_due = current_app['current_payment_due']
                
                expected_payment = earned - prev_certs
                
                # Check for discrepancy in payment calculation
                diff = abs(expected_payment - payment_due)
                if diff > 0.01 and (diff / max(expected_payment, payment_due) > 0.001):
                    confidence = min(0.95, 0.85 + (diff / max(expected_payment, payment_due)))
                    
                    anomaly = {
                        'type': 'payment_calculation_error',
                        'doc_id': current_app['doc_id'],
                        'payment_app_number': current_app['payment_app_number'],
                        'total_earned': earned,
                        'previous_certificates': prev_certs,
                        'expected_payment': expected_payment,
                        'reported_payment': payment_due,
                        'difference': payment_due - expected_payment,
                        'difference_percent': (diff / max(expected_payment, payment_due)) * 100,
                        'confidence': confidence,
                        'explanation': f"Payment Application #{current_app['payment_app_number']} reports current payment due of ${payment_due:.2f}, but calculation should be: ${earned:.2f} - ${prev_certs:.2f} = ${expected_payment:.2f}"
                    }
                    
                    anomalies.append(anomaly)
        
        # Now check line item progression for logical inconsistencies
        for app in pay_app_dicts:
            # Get line items for this payment application
            items_query = text("""
                SELECT 
                    li.item_id,
                    li.item_number,
                    li.description,
                    li.amount,
                    li.meta_data
                FROM 
                    line_items li
                WHERE
                    li.doc_id = :doc_id
                ORDER BY
                    li.item_number
            """)
            
            items = self.db_session.execute(items_query, {"doc_id": app['doc_id']}).fetchall()
            
            # Track unusual item progressions
            for item in items:
                item_id, item_number, description, amount, meta_data = item
                
                # Skip if missing key data
                if not item_number or not description:
                    continue
                    
                # Extract percentage complete from metadata if available
                percent_complete = None
                if meta_data and isinstance(meta_data, dict):
                    if 'percent_complete' in meta_data:
                        try:
                            percent_complete = float(meta_data['percent_complete'].rstrip('%')) / 100
                        except (ValueError, AttributeError):
                            try:
                                percent_complete = float(meta_data['percent_complete'])
                            except (ValueError, TypeError):
                                pass
                            
                # Find this same line item in previous payment applications
                if percent_complete is not None and app['payment_app_number'] > 1:
                    for prev_app in pay_app_dicts:
                        if prev_app['payment_app_number'] < app['payment_app_number']:
                            prev_items_query = text("""
                                SELECT 
                                    li.item_id,
                                    li.item_number,
                                    li.meta_data
                                FROM 
                                    line_items li
                                WHERE
                                    li.doc_id = :doc_id
                                    AND li.item_number = :item_number
                            """)
                            
                            prev_items = self.db_session.execute(prev_items_query, {
                                "doc_id": prev_app['doc_id'],
                                "item_number": item_number
                            }).fetchall()
                            
                            for prev_item in prev_items:
                                prev_item_id, prev_item_number, prev_meta_data = prev_item
                                
                                # Check for decrease in percent complete
                                prev_percent_complete = None
                                if prev_meta_data and isinstance(prev_meta_data, dict):
                                    if 'percent_complete' in prev_meta_data:
                                        try:
                                            prev_percent_complete = float(prev_meta_data['percent_complete'].rstrip('%')) / 100
                                        except (ValueError, AttributeError):
                                            try:
                                                prev_percent_complete = float(prev_meta_data['percent_complete'])
                                            except (ValueError, TypeError):
                                                pass
                                
                                if prev_percent_complete is not None and percent_complete < prev_percent_complete - 0.001:
                                    # Regression in percent complete
                                    decrease = prev_percent_complete - percent_complete
                                    confidence = min(0.95, 0.75 + decrease)
                                    
                                    anomaly = {
                                        'type': 'decreasing_completion_percentage',
                                        'doc_id': app['doc_id'],
                                        'previous_doc_id': prev_app['doc_id'],
                                        'item_id': item_id,
                                        'item_number': item_number,
                                        'description': description,
                                        'current_percent_complete': percent_complete * 100,
                                        'previous_percent_complete': prev_percent_complete * 100,
                                        'decrease': decrease * 100,
                                        'confidence': confidence,
                                        'explanation': f"Item #{item_number} ({description}) shows a decrease in completion percentage from {prev_percent_complete*100:.1f}% to {percent_complete*100:.1f}% between Payment Applications #{prev_app['payment_app_number']} and #{app['payment_app_number']}"
                                    }
                                    
                                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} running total inconsistencies")
        return anomalies
    
    def _calculate_month_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for documents in a month.
        
        Args:
            documents: List of documents in the month
            
        Returns:
            Dict with month statistics
        """
        if not documents:
            return {
                'total_amount': 0,
                'total_count': 0,
                'end_of_month_amount': 0,
                'end_of_month_count': 0,
                'end_of_month_ratio': 0,
                'days_in_month': 30
            }
            
        # Get the first date to determine month
        first_date = None
        for doc in documents:
            if doc['date']:
                first_date = doc['date']
                break
                
        if not first_date:
            return {
                'total_amount': 0,
                'total_count': 0,
                'end_of_month_amount': 0,
                'end_of_month_count': 0,
                'end_of_month_ratio': 0,
                'days_in_month': 30
            }
            
        # Determine days in the month
        year = first_date.year
        month = first_date.month
        
        if month == 2:
            # February
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                days_in_month = 29  # Leap year
            else:
                days_in_month = 28
        elif month in [4, 6, 9, 11]:
            # April, June, September, November
            days_in_month = 30
        else:
            # January, March, May, July, August, October, December
            days_in_month = 31
        
        # Calculate totals
        total_amount = sum(doc['total_amount'] for doc in documents if doc['total_amount'] is not None)
        total_count = len(documents)
        
        # Calculate end-of-month totals (last 3 days)
        end_of_month_docs = [doc for doc in documents if doc['date'] and doc['date'].day >= days_in_month - 2]
        end_of_month_amount = sum(doc['total_amount'] for doc in end_of_month_docs if doc['total_amount'] is not None)
        end_of_month_count = len(end_of_month_docs)
        
        # Calculate ratio (normalize by expected proportion)
        expected_ratio = 3 / days_in_month  # Expected proportion for last 3 days
        actual_ratio_count = end_of_month_count / total_count if total_count > 0 else 0
        actual_ratio_amount = end_of_month_amount / total_amount if total_amount > 0 else 0
        
        # Use the higher of the two ratios
        end_of_month_ratio = max(actual_ratio_count, actual_ratio_amount) / expected_ratio if expected_ratio > 0 else 0
        
        return {
            'total_amount': total_amount,
            'total_count': total_count,
            'end_of_month_amount': end_of_month_amount,
            'end_of_month_count': end_of_month_count,
            'end_of_month_ratio': end_of_month_ratio,
            'days_in_month': days_in_month
        }
    
    def _get_document_items(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get line items for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of line items
        """
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
                li.doc_id = :doc_id
            ORDER BY
                li.item_id
        """)
        
        items = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        # Convert to dictionaries
        return [dict(zip(
            ['item_id', 'doc_id', 'description', 'amount', 'cost_code', 'doc_type', 'party', 'date'],
            item
        )) for item in items]