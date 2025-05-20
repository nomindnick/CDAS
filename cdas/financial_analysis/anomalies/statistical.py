"""
Statistical anomaly detector for the financial analysis engine.

This module detects statistical anomalies in financial data, such as outliers,
unusual distributions, and other statistically significant irregularities.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal
import math

import logging
logger = logging.getLogger(__name__)


class StatisticalAnomalyDetector:
    """Detects statistical anomalies in financial data."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the statistical anomaly detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.outlier_z_score_threshold = self.config.get('outlier_z_score_threshold', 2.5)
        self.outlier_iqr_factor = self.config.get('outlier_iqr_factor', 1.5)
        self.min_data_points = self.config.get('min_data_points', 5)
        
    def detect_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected anomalies
        """
        logger.info(f"Detecting statistical anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        anomalies = []
        
        # Detect amount outliers
        amount_outliers = self.detect_amount_outliers(doc_id)
        anomalies.extend(amount_outliers)
        
        # Detect unusual distributions
        distribution_anomalies = self.detect_distribution_anomalies(doc_id)
        anomalies.extend(distribution_anomalies)
        
        # Detect frequency anomalies
        frequency_anomalies = self.detect_frequency_anomalies(doc_id)
        anomalies.extend(frequency_anomalies)
        
        logger.info(f"Found {len(anomalies)} statistical anomalies")
        return anomalies
    
    def detect_amount_outliers(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect outlier amounts in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of amount outlier anomalies
        """
        logger.info(f"Detecting amount outliers{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount IS NOT NULL"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Get line items
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
        """)
        
        params = {}
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
        
        # Group by cost code, doc type, and party for meaningful outlier detection
        grouped_items = {}
        for item in item_dicts:
            cost_code = item['cost_code'] or 'unknown'
            doc_type = item['doc_type'] or 'unknown'
            party = item['party'] or 'unknown'
            
            group_key = f"{cost_code}|{doc_type}|{party}"
            
            if group_key not in grouped_items:
                grouped_items[group_key] = []
                
            grouped_items[group_key].append(item)
        
        # Detect outliers in each group
        anomalies = []
        for group_key, group_items in grouped_items.items():
            # Need enough data points for statistical significance
            if len(group_items) < self.min_data_points:
                continue
                
            # Get amounts
            amounts = [item['amount'] for item in group_items if item['amount'] is not None]
            
            # Skip if no valid amounts
            if not amounts:
                continue
                
            # Detect outliers using Z-score method
            z_score_outliers = self._detect_z_score_outliers(amounts, group_items)
            
            # Detect outliers using IQR method
            iqr_outliers = self._detect_iqr_outliers(amounts, group_items)
            
            # Combine results (items that are outliers in both methods have higher confidence)
            combined_outliers = []
            outlier_ids = set()
            
            # Add Z-score outliers
            for outlier in z_score_outliers:
                outlier_ids.add(outlier['item_id'])
                outlier['detection_methods'] = ['z_score']
                combined_outliers.append(outlier)
            
            # Add IQR outliers or update existing ones
            for outlier in iqr_outliers:
                if outlier['item_id'] in outlier_ids:
                    # Update existing outlier
                    for existing in combined_outliers:
                        if existing['item_id'] == outlier['item_id']:
                            existing['detection_methods'].append('iqr')
                            # Increase confidence if detected by multiple methods
                            existing['confidence'] = min(0.95, existing['confidence'] + 0.1)
                            break
                else:
                    # Add new outlier
                    outlier_ids.add(outlier['item_id'])
                    outlier['detection_methods'] = ['iqr']
                    combined_outliers.append(outlier)
            
            # Add outliers to results
            anomalies.extend(combined_outliers)
        
        logger.info(f"Found {len(anomalies)} amount outlier anomalies")
        return anomalies
    
    def detect_distribution_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect unusual distributions in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of distribution anomalies
        """
        logger.info(f"Detecting distribution anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount IS NOT NULL"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Get line items
        query = text(f"""
            SELECT 
                li.doc_id,
                d.doc_type,
                d.party,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount,
                AVG(li.amount) AS avg_amount,
                MIN(li.amount) AS min_amount,
                MAX(li.amount) AS max_amount
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
            GROUP BY
                li.doc_id, d.doc_type, d.party
            HAVING
                COUNT(li.item_id) >= :min_items
        """)
        
        params = {"min_items": self.min_data_points}
        if doc_id:
            params["doc_id"] = doc_id
            
        # Execute query and handle potential database compatibility issues
        try:
            results = self.db_session.execute(query, params).fetchall()
        except:
            # If the query fails (e.g., with SQLite), fall back to a simpler approach
            return []
        
        # Convert to dictionaries for easier handling
        result_dicts = []
        for result in results:
            doc_id, doc_type, party, item_count, total_amount, avg_amount, min_amount, max_amount = result
            
            result_dict = {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'item_count': item_count,
                'total_amount': total_amount,
                'avg_amount': avg_amount,
                'min_amount': min_amount,
                'max_amount': max_amount
            }
            
            result_dicts.append(result_dict)
        
        # Group by doc type and party
        grouped_results = {}
        for result in result_dicts:
            doc_type = result['doc_type'] or 'unknown'
            party = result['party'] or 'unknown'
            
            group_key = f"{doc_type}|{party}"
            
            if group_key not in grouped_results:
                grouped_results[group_key] = []
                
            grouped_results[group_key].append(result)
        
        # Detect anomalies in each group
        anomalies = []
        for group_key, group_results in grouped_results.items():
            # Need enough data points for statistical significance
            if len(group_results) < self.min_data_points:
                continue
                
            # Get average amounts for each document
            avg_amounts = [result['avg_amount'] for result in group_results if result['avg_amount'] is not None]
            
            # Skip if no valid amounts
            if not avg_amounts:
                continue
                
            # Detect distribution anomalies
            distribution_stats = self._calculate_distribution_stats(avg_amounts)
            
            for result in group_results:
                if result['avg_amount'] is None:
                    continue
                    
                # Calculate Z-score for this document's average amount
                z_score = (result['avg_amount'] - distribution_stats['mean']) / distribution_stats['std_dev'] if distribution_stats['std_dev'] > 0 else 0
                
                # Check if this document is an outlier
                if abs(z_score) > self.outlier_z_score_threshold:
                    # Get line items for this document to include in the anomaly
                    doc_items = self._get_document_items(result['doc_id'])
                    
                    # Calculate confidence based on deviation
                    confidence = min(0.95, 0.7 + (abs(z_score) - self.outlier_z_score_threshold) * 0.1)
                    
                    anomaly = {
                        'type': 'distribution_anomaly',
                        'subtype': 'unusual_average',
                        'doc_id': result['doc_id'],
                        'doc_type': result['doc_type'],
                        'party': result['party'],
                        'item_count': result['item_count'],
                        'avg_amount': float(result['avg_amount']) if result['avg_amount'] is not None else None,
                        'total_amount': float(result['total_amount']) if result['total_amount'] is not None else None,
                        'group_avg': float(distribution_stats['mean']) if distribution_stats['mean'] is not None else None,
                        'z_score': float(z_score) if z_score is not None else None,
                        'items': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in item.items()} 
                            for item in doc_items
                        ],
                        'confidence': confidence,
                        'explanation': f"Document has an unusual average amount (${result['avg_amount']:.2f}) compared to similar documents (${distribution_stats['mean']:.2f}), z-score: {z_score:.2f}"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} distribution anomalies")
        return anomalies
    
    def detect_frequency_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect unusual frequencies in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of frequency anomalies
        """
        logger.info(f"Detecting frequency anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        # This analysis looks for unusual patterns in the frequency of specific amounts
        # For example, if a certain amount appears much more frequently than expected
        
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount IS NOT NULL AND li.amount > 0"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
        # Get frequency of amounts
        query = text(f"""
            SELECT 
                li.amount,
                COUNT(li.item_id) AS frequency
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                {where_clause}
            GROUP BY
                li.amount
            HAVING
                COUNT(li.item_id) > 1
            ORDER BY
                frequency DESC
        """)
        
        params = {}
        if doc_id:
            params["doc_id"] = doc_id
            
        results = self.db_session.execute(query, params).fetchall()
        
        # Convert to dictionaries for easier handling
        frequency_data = []
        for result in results:
            amount, frequency = result
            
            frequency_data.append({
                'amount': amount,
                'frequency': frequency
            })
        
        # Need enough data points for statistical significance
        if len(frequency_data) < self.min_data_points:
            return []
            
        # Get frequencies
        frequencies = [item['frequency'] for item in frequency_data]
        
        # Calculate statistics
        frequency_stats = self._calculate_distribution_stats(frequencies)
        
        # Detect outliers in frequency
        anomalies = []
        for item in frequency_data:
            if frequency_stats['std_dev'] > 0:
                z_score = (item['frequency'] - frequency_stats['mean']) / frequency_stats['std_dev']
                
                # Check if this frequency is an outlier
                if z_score > self.outlier_z_score_threshold:
                    # Get occurrences of this amount
                    occurrences = self._get_amount_occurrences(item['amount'], doc_id)
                    
                    # Calculate confidence based on deviation
                    confidence = min(0.95, 0.7 + (z_score - self.outlier_z_score_threshold) * 0.05)
                    
                    anomaly = {
                        'type': 'frequency_anomaly',
                        'amount': float(item['amount']) if item['amount'] is not None else None,
                        'frequency': item['frequency'],
                        'expected_frequency': float(frequency_stats['mean']) if frequency_stats['mean'] is not None else None,
                        'z_score': float(z_score) if z_score is not None else None,
                        'occurrences': [
                            {k: (v.isoformat() if k == 'date' and v else 
                                float(v) if k == 'amount' and v is not None else v) 
                             for k, v in occ.items()} 
                            for occ in occurrences
                        ],
                        'confidence': confidence,
                        'explanation': f"Amount ${item['amount']} appears with unusual frequency ({item['frequency']} occurrences) compared to other amounts (average: {frequency_stats['mean']:.1f}), z-score: {z_score:.2f}"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} frequency anomalies")
        return anomalies
    
    def detect_amount_anomalies(self, amount: float, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies related to a specific amount.
        
        Args:
            amount: Amount to analyze
            matches: List of matches for this amount
            
        Returns:
            List of anomalies related to this amount
        """
        logger.info(f"Detecting anomalies for amount: ${amount}")
        
        # If we have many matches, this might be a frequency anomaly
        if len(matches) > 5:  # Threshold for considering it potentially anomalous
            # Compare to overall frequency distribution
            query = text("""
                SELECT 
                    COUNT(DISTINCT amount) AS distinct_count,
                    AVG(amount_count) AS avg_frequency,
                    STDDEV(amount_count) AS stddev_frequency
                FROM (
                    SELECT 
                        amount, 
                        COUNT(*) AS amount_count
                    FROM 
                        line_items
                    WHERE 
                        amount > 0
                    GROUP BY 
                        amount
                ) AS amount_counts
            """)
            
            try:
                result = self.db_session.execute(query).fetchone()
                distinct_count, avg_frequency, stddev_frequency = result
            except:
                # If the query fails (e.g., with SQLite), use simplified approach
                distinct_count = 0
                avg_frequency = 0
                stddev_frequency = 0
                
                # Calculate manually
                amount_counts = {}
                query = text("""
                    SELECT amount, COUNT(*) AS count
                    FROM line_items
                    WHERE amount > 0
                    GROUP BY amount
                """)
                
                counts = self.db_session.execute(query).fetchall()
                
                for amt, count in counts:
                    amount_counts[amt] = count
                    
                if amount_counts:
                    distinct_count = len(amount_counts)
                    frequencies = list(amount_counts.values())
                    avg_frequency = sum(frequencies) / len(frequencies)
                    
                    # Calculate standard deviation
                    variance = sum((f - avg_frequency) ** 2 for f in frequencies) / len(frequencies)
                    stddev_frequency = math.sqrt(variance)
            
            if stddev_frequency and stddev_frequency > 0:
                # Calculate z-score for this amount's frequency
                z_score = (len(matches) - avg_frequency) / stddev_frequency
                
                # If this is significantly more frequent than average
                if z_score > self.outlier_z_score_threshold:
                    return [{
                        'type': 'specific_amount_frequency_anomaly',
                        'amount': amount,
                        'frequency': len(matches),
                        'average_frequency': float(avg_frequency) if avg_frequency is not None else None,
                        'z_score': float(z_score) if z_score is not None else None,
                        'matches': matches,
                        'confidence': min(0.95, 0.7 + (z_score - self.outlier_z_score_threshold) * 0.05),
                        'explanation': f"Amount ${amount} appears unusually frequently ({len(matches)} occurrences) compared to other amounts (average: {avg_frequency:.1f}), z-score: {z_score:.2f}"
                    }]
        
        return []
    
    def _detect_z_score_outliers(self, amounts: List[Decimal], 
                              items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect outliers using Z-score method.
        
        Args:
            amounts: List of amounts
            items: List of items with these amounts
            
        Returns:
            List of outlier anomalies
        """
        # Calculate mean and standard deviation
        if not amounts:
            return []
            
        n = len(amounts)
        mean = sum(amounts) / n
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in amounts) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0
        
        # If standard deviation is too small, can't reliably detect outliers
        if std_dev < 0.01 * mean:
            return []
            
        # Detect outliers
        outliers = []
        for item in items:
            if item['amount'] is None:
                continue
                
            # Calculate Z-score
            z_score = (item['amount'] - mean) / std_dev if std_dev > 0 else 0
            
            # Check if outlier
            if abs(z_score) > self.outlier_z_score_threshold:
                # Format for output
                formatted_item = {
                    k: (v.isoformat() if k == 'date' and v else 
                        float(v) if k == 'amount' and v is not None else v) 
                    for k, v in item.items()
                }
                
                # Add Z-score and other statistics
                formatted_item['z_score'] = float(z_score)
                formatted_item['group_mean'] = float(mean)
                formatted_item['group_std_dev'] = float(std_dev)
                
                # Set confidence based on how extreme the outlier is
                confidence = min(0.95, 0.7 + (abs(z_score) - self.outlier_z_score_threshold) * 0.1)
                
                outlier = {
                    'type': 'amount_outlier',
                    'subtype': 'z_score_outlier',
                    'item_id': item['item_id'],
                    'doc_id': item['doc_id'],
                    'amount': float(item['amount']),
                    'z_score': float(z_score),
                    'details': formatted_item,
                    'confidence': confidence,
                    'explanation': f"Amount ${item['amount']} is a statistical outlier (z-score: {z_score:.2f}, group mean: ${mean:.2f})"
                }
                
                outliers.append(outlier)
        
        return outliers
    
    def _detect_iqr_outliers(self, amounts: List[Decimal], 
                          items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            amounts: List of amounts
            items: List of items with these amounts
            
        Returns:
            List of outlier anomalies
        """
        # Need enough data points for quartile calculation
        if len(amounts) < 4:
            return []
            
        # Sort amounts
        sorted_amounts = sorted(amounts)
        n = len(sorted_amounts)
        
        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_amounts[q1_idx]
        q3 = sorted_amounts[q3_idx]
        
        # Calculate IQR and bounds
        iqr = q3 - q1
        lower_bound = q1 - (self.outlier_iqr_factor * iqr)
        upper_bound = q3 + (self.outlier_iqr_factor * iqr)
        
        # Detect outliers
        outliers = []
        for item in items:
            if item['amount'] is None:
                continue
                
            # Check if outlier
            is_outlier = item['amount'] < lower_bound or item['amount'] > upper_bound
            
            if is_outlier:
                # Format for output
                formatted_item = {
                    k: (v.isoformat() if k == 'date' and v else 
                        float(v) if k == 'amount' and v is not None else v) 
                    for k, v in item.items()
                }
                
                # Add IQR statistics
                formatted_item['q1'] = float(q1)
                formatted_item['q3'] = float(q3)
                formatted_item['iqr'] = float(iqr)
                formatted_item['lower_bound'] = float(lower_bound)
                formatted_item['upper_bound'] = float(upper_bound)
                
                # Calculate how extreme the outlier is
                if item['amount'] < lower_bound:
                    extremity = (lower_bound - item['amount']) / iqr if iqr > 0 else 0
                else:
                    extremity = (item['amount'] - upper_bound) / iqr if iqr > 0 else 0
                
                # Set confidence based on how extreme the outlier is
                confidence = min(0.95, 0.7 + extremity * 0.2)
                
                outlier = {
                    'type': 'amount_outlier',
                    'subtype': 'iqr_outlier',
                    'item_id': item['item_id'],
                    'doc_id': item['doc_id'],
                    'amount': float(item['amount']),
                    'details': formatted_item,
                    'confidence': confidence,
                    'explanation': f"Amount ${item['amount']} is a statistical outlier (outside IQR bounds: ${lower_bound:.2f} - ${upper_bound:.2f})"
                }
                
                outliers.append(outlier)
        
        return outliers
    
    def _calculate_distribution_stats(self, values: List[Union[Decimal, int, float]]) -> Dict[str, float]:
        """Calculate distribution statistics for a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Dict with mean, std_dev, min, max
        """
        if not values:
            return {
                'mean': None,
                'std_dev': None,
                'min': None,
                'max': None
            }
            
        n = len(values)
        mean = sum(values) / n
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0
        
        return {
            'mean': mean,
            'std_dev': std_dev,
            'min': min(values),
            'max': max(values)
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
        """)
        
        items = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        # Convert to dictionaries
        return [dict(zip(
            ['item_id', 'doc_id', 'description', 'amount', 'cost_code', 'doc_type', 'party', 'date'],
            item
        )) for item in items]
    
    def _get_amount_occurrences(self, amount: Decimal, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get occurrences of a specific amount.
        
        Args:
            amount: Amount to find
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of occurrences
        """
        # Construct WHERE clause based on doc_id
        where_clause = "li.amount BETWEEN :amount_min AND :amount_max"
        if doc_id:
            where_clause += f" AND li.doc_id = :doc_id"
        
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
        """)
        
        # Use small tolerance for exact amount matching
        amount_min = amount - Decimal('0.01')
        amount_max = amount + Decimal('0.01')
        
        params = {
            "amount_min": amount_min,
            "amount_max": amount_max
        }
        if doc_id:
            params["doc_id"] = doc_id
            
        occurrences = self.db_session.execute(query, params).fetchall()
        
        # Convert to dictionaries
        return [dict(zip(
            ['item_id', 'doc_id', 'description', 'amount', 'cost_code', 'doc_type', 'party', 'date'],
            item
        )) for item in occurrences]