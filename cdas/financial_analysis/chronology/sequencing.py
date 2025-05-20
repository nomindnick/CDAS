"""
Sequencing analyzer for the financial analysis engine.

This module analyzes financial event sequencing, detecting patterns in the
order of financial events and identifying suspicious or unusual timing patterns.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class SequencingAnalyzer:
    """Analyzes financial event sequencing and timing patterns."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the sequencing analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Timing pattern configuration
        self.end_of_month_threshold = self.config.get('end_of_month_threshold', 3)  # Last X days of month
        self.clustered_events_threshold = self.config.get('clustered_events_threshold', 3)  # Min events to detect clustering
        self.short_interval_days = self.config.get('short_interval_days', 2)  # Max days for short interval detection
        self.long_interval_days = self.config.get('long_interval_days', 60)  # Min days for long interval detection
        
    def detect_timing_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the timing of financial events.
        
        Returns:
            List of timing anomalies
        """
        logger.info("Detecting timing anomalies")
        
        anomalies = []
        
        # Detect end-of-month clustering
        eom_anomalies = self.detect_end_of_month_clustering()
        anomalies.extend(eom_anomalies)
        
        # Detect unusual gaps
        gap_anomalies = self.detect_unusual_gaps()
        anomalies.extend(gap_anomalies)
        
        # Detect event clustering
        clustering_anomalies = self.detect_event_clustering()
        anomalies.extend(clustering_anomalies)
        
        # Detect holiday/weekend timing
        holiday_anomalies = self.detect_holiday_weekend_timing()
        anomalies.extend(holiday_anomalies)
        
        # Detect amounts appearing before formal approval
        approval_anomalies = self.detect_amounts_before_approval()
        anomalies.extend(approval_anomalies)
        
        logger.info(f"Found {len(anomalies)} timing anomalies")
        return anomalies
    
    def detect_end_of_month_clustering(self) -> List[Dict[str, Any]]:
        """Detect clustering of financial events at the end of months.
        
        Returns:
            List of end-of-month clustering anomalies
        """
        logger.info("Detecting end-of-month clustering")
        
        # Get document dates
        try:
            # Use SQLite-compatible date functions
            query = text("""
                SELECT 
                    d.doc_id,
                    d.doc_type,
                    d.party,
                    d.date_created,
                    strftime('%m', d.date_created) AS month,
                    strftime('%Y', d.date_created) AS year,
                    strftime('%d', d.date_created) AS day,
                    COUNT(li.item_id) AS item_count,
                    SUM(li.amount) AS total_amount
                FROM 
                    documents d
                JOIN
                    line_items li ON d.doc_id = li.doc_id
                WHERE
                    d.date_created IS NOT NULL
                GROUP BY
                    d.doc_id, d.doc_type, d.party, d.date_created
                ORDER BY
                    d.date_created
            """)
            
            results = self.db_session.execute(query).fetchall()
        except Exception as e:
            logger.exception(f"Error executing end-of-month query: {e}")
            return []
        
        # Group by month and year
        month_groups = {}
        for result in results:
            doc_id, doc_type, party, date_created, month, year, day, item_count, total_amount = result
            
            if not date_created:
                continue
                
            # Determine the last day of the month
            last_day = self._get_last_day_of_month(date_created)
            
            # Make sure day is an integer
            try:
                day_int = int(day) if day else 0
            except (ValueError, TypeError):
                day_int = 0
                
            # Check if this is end of the month
            is_end_of_month = (last_day - day_int) < self.end_of_month_threshold
            
            # Create month key
            month_key = f"{int(year)}-{int(month)}"
            
            if month_key not in month_groups:
                month_groups[month_key] = {
                    'all_docs': [],
                    'end_of_month_docs': []
                }
                
            # Format date_created correctly
            if isinstance(date_created, str):
                date_formatted = date_created
            else:
                try:
                    date_formatted = date_created.isoformat()
                except AttributeError:
                    date_formatted = str(date_created)
            
            doc_info = {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_formatted,
                'day': day_int,
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None,
                'is_end_of_month': is_end_of_month
            }
            
            month_groups[month_key]['all_docs'].append(doc_info)
            
            if is_end_of_month:
                month_groups[month_key]['end_of_month_docs'].append(doc_info)
        
        # Analyze each month for clustering
        anomalies = []
        for month_key, group in month_groups.items():
            all_count = len(group['all_docs'])
            eom_count = len(group['end_of_month_docs'])
            
            # Skip months with too few documents
            if all_count < 5:
                continue
                
            # Calculate the expected proportion
            expected_proportion = self.end_of_month_threshold / 30.0  # Simplified assumption of 30-day months
            expected_count = all_count * expected_proportion
            
            # Check if end-of-month count is significantly higher than expected
            if eom_count > expected_count * 2 and eom_count >= self.clustered_events_threshold:
                # Calculate anomaly confidence
                ratio = eom_count / expected_count
                confidence = min(0.95, 0.7 + (ratio - 2) * 0.1)
                
                if confidence >= self.min_confidence:
                    # Get total amount for end-of-month docs
                    eom_total = sum(doc['total_amount'] for doc in group['end_of_month_docs'] if doc['total_amount'] is not None)
                    
                    anomaly = {
                        'type': 'end_of_month_clustering',
                        'month': month_key,
                        'total_documents': all_count,
                        'end_of_month_documents': eom_count,
                        'expected_end_of_month': expected_count,
                        'ratio': ratio,
                        'end_of_month_total': float(eom_total),
                        'affected_documents': [doc['doc_id'] for doc in group['end_of_month_docs']],
                        'confidence': confidence,
                        'explanation': f"Unusual clustering of {eom_count} documents at the end of {month_key} (expected: {expected_count:.1f})"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} end-of-month clustering anomalies")
        return anomalies
    
    def detect_unusual_gaps(self) -> List[Dict[str, Any]]:
        """Detect unusual gaps between financial events.
        
        Returns:
            List of unusual gap anomalies
        """
        logger.info("Detecting unusual gaps between financial events")
        
        # Get document dates in order
        query = text("""
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
                d.date_created IS NOT NULL
            GROUP BY
                d.doc_id, d.doc_type, d.party, d.date_created
            ORDER BY
                d.date_created
        """)
        
        results = self.db_session.execute(query).fetchall()
        
        # Convert to list of documents
        documents = []
        for result in results:
            doc_id, doc_type, party, date_created, item_count, total_amount = result
            
            if not date_created:
                continue
                
            documents.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None
            })
        
        # Calculate gaps between consecutive documents
        gaps = []
        for i in range(1, len(documents)):
            prev_doc = documents[i-1]
            curr_doc = documents[i]
            
            # Parse date strings to datetime objects
            try:
                # Try different date formats for prev_doc and curr_doc
                prev_date = self._parse_date_string(prev_doc['date'])
                curr_date = self._parse_date_string(curr_doc['date'])
                
                if prev_date and curr_date:
                    gap_days = (curr_date - prev_date).days
                else:
                    # Default to 0 if we can't parse the dates
                    gap_days = 0
            except Exception:
                # Default to 0 in case of any issues
                gap_days = 0
            
            gaps.append({
                'start_doc': prev_doc,
                'end_doc': curr_doc,
                'gap_days': gap_days
            })
        
        # Calculate statistics on gaps
        if not gaps:
            return []
            
        gap_days = [gap['gap_days'] for gap in gaps]
        avg_gap = sum(gap_days) / len(gap_days)
        
        # Calculate standard deviation
        variance = sum((days - avg_gap) ** 2 for days in gap_days) / len(gap_days)
        std_dev = variance ** 0.5
        
        # Identify unusually long gaps
        anomalies = []
        for gap in gaps:
            # Skip short gaps
            if gap['gap_days'] <= self.long_interval_days:
                continue
                
            # Calculate z-score
            if std_dev > 0:
                z_score = (gap['gap_days'] - avg_gap) / std_dev
            else:
                z_score = 0
                
            # Only consider gaps that are statistically significant
            if z_score > 2 or gap['gap_days'] > avg_gap * 3:
                # Calculate confidence
                confidence = min(0.95, 0.7 + min(z_score / 5, 0.25))
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'unusual_gap',
                        'start_date': gap['start_doc']['date'].isoformat(),
                        'end_date': gap['end_doc']['date'].isoformat(),
                        'gap_days': gap['gap_days'],
                        'average_gap': avg_gap,
                        'z_score': z_score,
                        'start_doc_id': gap['start_doc']['doc_id'],
                        'end_doc_id': gap['end_doc']['doc_id'],
                        'confidence': confidence,
                        'explanation': f"Unusual gap of {gap['gap_days']} days between documents (average: {avg_gap:.1f} days)"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} unusual gap anomalies")
        return anomalies
    
    def detect_event_clustering(self) -> List[Dict[str, Any]]:
        """Detect clustering of financial events.
        
        Returns:
            List of event clustering anomalies
        """
        logger.info("Detecting event clustering")
        
        # Get document dates in order
        query = text("""
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
                d.date_created IS NOT NULL
            GROUP BY
                d.doc_id, d.doc_type, d.party, d.date_created
            ORDER BY
                d.date_created
        """)
        
        results = self.db_session.execute(query).fetchall()
        
        # Convert to list of documents
        documents = []
        for result in results:
            doc_id, doc_type, party, date_created, item_count, total_amount = result
            
            if not date_created:
                continue
                
            documents.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None
            })
        
        # Find clusters using a sliding window approach
        anomalies = []
        for window_size in range(3, min(10, len(documents) + 1)):  # Try different window sizes
            # Slide window over documents
            for i in range(len(documents) - window_size + 1):
                window = documents[i:i+window_size]
                
                # Calculate time span of window using date parsing
                start_date_str = window[0]['date']
                end_date_str = window[-1]['date']
                
                # Parse dates
                start_date = self._parse_date_string(start_date_str)
                end_date = self._parse_date_string(end_date_str)
                
                # Calculate span days or default to a reasonable value
                if start_date and end_date:
                    span_days = (end_date - start_date).days
                else:
                    # Default to a reasonable value that won't trigger a clustering anomaly
                    span_days = self.short_interval_days * window_size
                
                # Skip if span is too long
                if span_days > self.short_interval_days * (window_size - 1):
                    continue
                    
                # Calculate total amount in cluster
                total_amount = sum(doc['total_amount'] for doc in window if doc['total_amount'] is not None)
                
                # Calculate confidence based on cluster density
                density = window_size / (span_days + 1)  # Documents per day
                confidence = min(0.95, 0.7 + min((density - 1) * 0.2, 0.25))
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'event_clustering',
                        'window_size': window_size,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'span_days': span_days,
                        'density': density,
                        'total_amount': float(total_amount),
                        'document_ids': [doc['doc_id'] for doc in window],
                        'confidence': confidence,
                        'explanation': f"Unusual clustering of {window_size} documents over {span_days} days (density: {density:.1f} docs/day)"
                    }
                    
                    anomalies.append(anomaly)
        
        # Remove overlapping clusters (keep the highest confidence ones)
        filtered_anomalies = self._filter_overlapping_clusters(anomalies)
        
        logger.info(f"Found {len(filtered_anomalies)} event clustering anomalies")
        return filtered_anomalies
    
    def detect_holiday_weekend_timing(self) -> List[Dict[str, Any]]:
        """Detect financial events timed around holidays and weekends.
        
        Returns:
            List of holiday/weekend timing anomalies
        """
        logger.info("Detecting holiday/weekend timing")
        
        # Get document dates
        try:
            # Use SQLite-compatible date functions
            query = text("""
                SELECT 
                    d.doc_id,
                    d.doc_type,
                    d.party,
                    d.date_created,
                    strftime('%w', d.date_created) AS day_of_week,
                    COUNT(li.item_id) AS item_count,
                    SUM(li.amount) AS total_amount
                FROM 
                    documents d
                JOIN
                    line_items li ON d.doc_id = li.doc_id
                WHERE
                    d.date_created IS NOT NULL
                GROUP BY
                    d.doc_id, d.doc_type, d.party, d.date_created
                ORDER BY
                    d.date_created
            """)
            
            results = self.db_session.execute(query).fetchall()
        except Exception as e:
            logger.exception(f"Error executing holiday/weekend query: {e}")
            
            # Fall back to a simpler query without the date extraction functions
            fallback_query = text("""
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
                    d.date_created IS NOT NULL
                GROUP BY
                    d.doc_id, d.doc_type, d.party, d.date_created
                ORDER BY
                    d.date_created
            """)
            
            results = self.db_session.execute(fallback_query).fetchall()
        
        # Convert to list of documents
        documents = []
        for result in results:
            if len(result) == 7:  # Original query with day_of_week
                doc_id, doc_type, party, date_created, day_of_week, item_count, total_amount = result
            else:  # Fallback query without day_of_week
                doc_id, doc_type, party, date_created, item_count, total_amount = result
                # Calculate day of week manually (0=Sunday, 6=Saturday)
                day_of_week = date_created.weekday() + 1  # Convert to 1=Monday, 7=Sunday
                if day_of_week == 7:
                    day_of_week = 0  # Adjust to match extract(DOW) from PostgreSQL
            
            if not date_created:
                continue
                
            documents.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'day_of_week': int(day_of_week),
                'is_weekend': int(day_of_week) in [0, 6],  # 0=Sunday, 6=Saturday
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None
            })
        
        # Look for weekend patterns
        weekend_docs = [doc for doc in documents if doc['is_weekend']]
        workday_docs = [doc for doc in documents if not doc['is_weekend']]
        
        anomalies = []
        
        # Skip if we don't have enough documents
        if len(documents) < 10:
            return anomalies
        
        # Calculate weekend percentage
        weekend_percent = (len(weekend_docs) / len(documents)) * 100
        
        # Expected weekend percentage (2 out of 7 days)
        expected_weekend_percent = (2 / 7) * 100
        
        # Check if weekend percentage is significantly higher than expected
        if weekend_percent > expected_weekend_percent * 1.5 and len(weekend_docs) >= 3:
            # Calculate weekend total amount
            weekend_total = sum(doc['total_amount'] for doc in weekend_docs if doc['total_amount'] is not None)
            
            # Calculate confidence
            ratio = weekend_percent / expected_weekend_percent
            confidence = min(0.95, 0.7 + (ratio - 1.5) * 0.2)
            
            if confidence >= self.min_confidence:
                anomaly = {
                    'type': 'weekend_timing',
                    'weekend_document_count': len(weekend_docs),
                    'total_document_count': len(documents),
                    'weekend_percent': weekend_percent,
                    'expected_weekend_percent': expected_weekend_percent,
                    'ratio': ratio,
                    'weekend_total_amount': float(weekend_total),
                    'affected_documents': [doc['doc_id'] for doc in weekend_docs],
                    'confidence': confidence,
                    'explanation': f"Unusual percentage of documents dated on weekends ({weekend_percent:.1f}%, expected: {expected_weekend_percent:.1f}%)"
                }
                
                anomalies.append(anomaly)
        
        # Check for US holidays (simplified approach)
        holidays = self._get_major_us_holidays(documents[0]['date'].year, documents[-1]['date'].year)
        
        holiday_docs = []
        for doc in documents:
            doc_date = doc['date'].date()
            
            # Check if document is on a holiday
            is_holiday = doc_date in holidays
            
            # Check if document is within 1 day of a holiday
            is_near_holiday = any(abs((doc_date - holiday).days) <= 1 for holiday in holidays)
            
            if is_holiday or is_near_holiday:
                doc['is_holiday'] = is_holiday
                doc['is_near_holiday'] = is_near_holiday
                holiday_docs.append(doc)
        
        # Calculate holiday percentage
        total_holiday_days = len(holidays) * 3  # Holiday plus day before/after
        total_days = (documents[-1]['date'] - documents[0]['date']).days + 1
        expected_holiday_percent = (total_holiday_days / total_days) * 100
        
        if len(holiday_docs) > 0:
            holiday_percent = (len(holiday_docs) / len(documents)) * 100
            
            # Check if holiday percentage is significantly higher than expected
            if holiday_percent > expected_holiday_percent * 1.5 and len(holiday_docs) >= 3:
                # Calculate holiday total amount
                holiday_total = sum(doc['total_amount'] for doc in holiday_docs if doc['total_amount'] is not None)
                
                # Calculate confidence
                ratio = holiday_percent / expected_holiday_percent
                confidence = min(0.95, 0.7 + (ratio - 1.5) * 0.2)
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'holiday_timing',
                        'holiday_document_count': len(holiday_docs),
                        'total_document_count': len(documents),
                        'holiday_percent': holiday_percent,
                        'expected_holiday_percent': expected_holiday_percent,
                        'ratio': ratio,
                        'holiday_total_amount': float(holiday_total),
                        'affected_documents': [doc['doc_id'] for doc in holiday_docs],
                        'confidence': confidence,
                        'explanation': f"Unusual percentage of documents dated on or near holidays ({holiday_percent:.1f}%, expected: {expected_holiday_percent:.1f}%)"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} holiday/weekend timing anomalies")
        return anomalies
    
    def detect_sequence_patterns(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in the sequence of financial items.
        
        Args:
            items: List of financial items
            
        Returns:
            List of sequence patterns
        """
        logger.info(f"Detecting sequence patterns in {len(items)} items")
        
        # Sort items by date
        sorted_items = sorted(items, key=lambda x: x['date'] if isinstance(x['date'], datetime) else datetime.fromisoformat(x['date'].replace('Z', '+00:00')) if isinstance(x['date'], str) else datetime.min)
        
        patterns = []
        
        # Skip if we don't have enough items
        if len(sorted_items) < 3:
            return patterns
            
        # Look for increasing/decreasing amount patterns
        amount_patterns = self._detect_amount_progression(sorted_items)
        patterns.extend(amount_patterns)
        
        # Look for alternating patterns
        alternating_patterns = self._detect_alternating_patterns(sorted_items)
        patterns.extend(alternating_patterns)
        
        # Look for repeated date patterns (e.g., always on the 15th)
        date_patterns = self._detect_date_patterns(sorted_items)
        patterns.extend(date_patterns)
        
        logger.info(f"Found {len(patterns)} sequence patterns")
        return patterns
    
    def _get_last_day_of_month(self, date: Union[datetime, str, None]) -> int:
        """Get the last day of the month for a date.
        
        Args:
            date: Datetime object, date string, or None
            
        Returns:
            Last day of the month or 0 if date is invalid
        """
        # Handle None or invalid dates
        if date is None:
            return 0
            
        # Convert string dates to datetime
        if isinstance(date, str):
            try:
                # Try to parse the date string
                from datetime import datetime
                date_formats = [
                    '%B %d, %Y',      # March 15, 2023
                    '%b %d, %Y',      # Mar 15, 2023
                    '%m/%d/%Y',       # 03/15/2023
                    '%Y-%m-%d',       # 2023-03-15
                    '%d-%m-%Y',       # 15-03-2023
                ]
                
                for fmt in date_formats:
                    try:
                        date = datetime.strptime(date, fmt)
                        break
                    except ValueError:
                        continue
                        
                # If no format worked, return default
                if isinstance(date, str):
                    return 0
            except Exception:
                return 0
        
        # Get the next month
        try:
            if date.month == 12:
                next_month = datetime(date.year + 1, 1, 1)
            else:
                next_month = datetime(date.year, date.month + 1, 1)
                
            # Subtract one day to get the last day of the current month
            last_day = (next_month - timedelta(days=1)).day
            
            return last_day
        except (AttributeError, TypeError):
            return 0
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string into a datetime object.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        if not date_str:
            return None
            
        # If it's already a datetime object, return it
        if isinstance(date_str, datetime):
            return date_str
            
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
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # If all parsing attempts fail, try extracting with regex
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 
            'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        # Match patterns like "March 15, 2023" or "March 2023"
        import re
        pattern = r'([a-zA-Z]+)\s+(\d{1,2})?,?\s+(\d{4})'
        match = re.search(pattern, date_str)
        if match:
            month_str, day_str, year_str = match.groups()
            month = month_names.get(month_str.lower())
            day = int(day_str) if day_str else 1
            year = int(year_str)
            
            if month and 1 <= day <= 31 and year >= 1900:
                return datetime(year, month, day)
        
        # Fallback to None if all else fails
        return None
    
    def _filter_overlapping_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out overlapping clusters, keeping the highest confidence ones.
        
        Args:
            clusters: List of cluster anomalies
            
        Returns:
            Filtered list of cluster anomalies
        """
        # Sort clusters by confidence (highest first)
        sorted_clusters = sorted(clusters, key=lambda x: x['confidence'], reverse=True)
        
        # Keep track of which document IDs have been covered
        covered_docs = set()
        filtered_clusters = []
        
        for cluster in sorted_clusters:
            # Check if this cluster overlaps significantly with already covered documents
            cluster_docs = set(cluster['document_ids'])
            overlap = len(cluster_docs.intersection(covered_docs))
            
            # If less than 50% overlap, keep this cluster
            if overlap < len(cluster_docs) / 2:
                filtered_clusters.append(cluster)
                covered_docs.update(cluster_docs)
        
        return filtered_clusters
    
    def _get_major_us_holidays(self, start_year: int, end_year: int) -> List[datetime.date]:
        """Get a list of major US holidays for a range of years.
        
        Args:
            start_year: Start year
            end_year: End year
            
        Returns:
            List of holiday dates
        """
        holidays = []
        
        for year in range(start_year, end_year + 1):
            # New Year's Day
            holidays.append(datetime(year, 1, 1).date())
            
            # Independence Day
            holidays.append(datetime(year, 7, 4).date())
            
            # Christmas Eve and Christmas Day
            holidays.append(datetime(year, 12, 24).date())
            holidays.append(datetime(year, 12, 25).date())
            
            # New Year's Eve
            holidays.append(datetime(year, 12, 31).date())
            
            # Memorial Day (last Monday in May)
            memorial_day = datetime(year, 5, 31).date()
            while memorial_day.weekday() != 0:  # 0 = Monday
                memorial_day = memorial_day - timedelta(days=1)
            holidays.append(memorial_day)
            
            # Labor Day (first Monday in September)
            labor_day = datetime(year, 9, 1).date()
            while labor_day.weekday() != 0:  # 0 = Monday
                labor_day = labor_day + timedelta(days=1)
            holidays.append(labor_day)
            
            # Thanksgiving (fourth Thursday in November)
            thanksgiving = datetime(year, 11, 1).date()
            while thanksgiving.weekday() != 3:  # 3 = Thursday
                thanksgiving = thanksgiving + timedelta(days=1)
            thanksgiving = thanksgiving + timedelta(days=21)  # Move to 4th Thursday
            holidays.append(thanksgiving)
            
            # Day after Thanksgiving
            holidays.append(thanksgiving + timedelta(days=1))
            
            # Simplistic handling of MLK Day, Presidents Day, etc.
            # (would require more complex logic for exact dates)
        
        return holidays
    
    def _detect_amount_progression(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect amount progression patterns.
        
        Args:
            items: List of financial items sorted by date
            
        Returns:
            List of amount progression patterns
        """
        patterns = []
        
        # Need at least 3 items to detect a progression
        if len(items) < 3:
            return patterns
            
        # Check for increasing/decreasing sequences
        increasing_runs = []
        decreasing_runs = []
        
        current_increasing_run = []
        current_decreasing_run = []
        
        for i, item in enumerate(items):
            if i == 0:
                # Start both types of runs with the first item
                current_increasing_run.append(item)
                current_decreasing_run.append(item)
                continue
                
            prev_amount = items[i-1]['amount']
            curr_amount = item['amount']
            
            if curr_amount is None or prev_amount is None:
                continue
                
            # Check for increasing run
            if curr_amount > prev_amount:
                current_increasing_run.append(item)
                
                # Check if decreasing run is ending
                if len(current_decreasing_run) >= 3:
                    decreasing_runs.append(current_decreasing_run.copy())
                current_decreasing_run = [item]  # Start a new decreasing run
                
            # Check for decreasing run
            elif curr_amount < prev_amount:
                current_decreasing_run.append(item)
                
                # Check if increasing run is ending
                if len(current_increasing_run) >= 3:
                    increasing_runs.append(current_increasing_run.copy())
                current_increasing_run = [item]  # Start a new increasing run
                
            # Equal amounts - can be part of both types
            else:
                current_increasing_run.append(item)
                current_decreasing_run.append(item)
        
        # Add the final runs if long enough
        if len(current_increasing_run) >= 3:
            increasing_runs.append(current_increasing_run)
        if len(current_decreasing_run) >= 3:
            decreasing_runs.append(current_decreasing_run)
        
        # Create patterns for significant runs
        for run in increasing_runs:
            if len(run) >= 3:
                # Calculate key metrics
                start_amount = run[0]['amount']
                end_amount = run[-1]['amount']
                total_increase = end_amount - start_amount
                percent_increase = (total_increase / start_amount) * 100 if start_amount > 0 else 0
                
                # Calculate confidence based on run length and percentage increase
                confidence = min(0.95, 0.7 + min(len(run) / 10, 0.1) + min(percent_increase / 100, 0.15))
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': 'increasing_amount_sequence',
                        'length': len(run),
                        'start_date': run[0]['date'].isoformat() if isinstance(run[0]['date'], datetime) else run[0]['date'],
                        'end_date': run[-1]['date'].isoformat() if isinstance(run[-1]['date'], datetime) else run[-1]['date'],
                        'start_amount': float(start_amount),
                        'end_amount': float(end_amount),
                        'total_increase': float(total_increase),
                        'percent_increase': float(percent_increase),
                        'affected_items': [item['item_id'] for item in run if 'item_id' in item],
                        'confidence': confidence,
                        'explanation': f"Sequence of {len(run)} increasing amounts from ${start_amount:.2f} to ${end_amount:.2f} ({percent_increase:.1f}% increase)"
                    }
                    
                    patterns.append(pattern)
        
        for run in decreasing_runs:
            if len(run) >= 3:
                # Calculate key metrics
                start_amount = run[0]['amount']
                end_amount = run[-1]['amount']
                total_decrease = start_amount - end_amount
                percent_decrease = (total_decrease / start_amount) * 100 if start_amount > 0 else 0
                
                # Calculate confidence based on run length and percentage decrease
                confidence = min(0.95, 0.7 + min(len(run) / 10, 0.1) + min(percent_decrease / 100, 0.15))
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': 'decreasing_amount_sequence',
                        'length': len(run),
                        'start_date': run[0]['date'].isoformat() if isinstance(run[0]['date'], datetime) else run[0]['date'],
                        'end_date': run[-1]['date'].isoformat() if isinstance(run[-1]['date'], datetime) else run[-1]['date'],
                        'start_amount': float(start_amount),
                        'end_amount': float(end_amount),
                        'total_decrease': float(total_decrease),
                        'percent_decrease': float(percent_decrease),
                        'affected_items': [item['item_id'] for item in run if 'item_id' in item],
                        'confidence': confidence,
                        'explanation': f"Sequence of {len(run)} decreasing amounts from ${start_amount:.2f} to ${end_amount:.2f} ({percent_decrease:.1f}% decrease)"
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_alternating_patterns(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect alternating patterns in financial items.
        
        Args:
            items: List of financial items sorted by date
            
        Returns:
            List of alternating patterns
        """
        patterns = []
        
        # Need at least 4 items to detect an alternating pattern
        if len(items) < 4:
            return patterns
            
        # Check for alternating high/low amounts
        high_low_pattern = []
        
        for i in range(len(items) - 1):
            curr_amount = items[i]['amount']
            next_amount = items[i+1]['amount']
            
            if curr_amount is None or next_amount is None:
                continue
                
            if (i % 2 == 0 and curr_amount > next_amount) or (i % 2 == 1 and curr_amount < next_amount):
                high_low_pattern.append(True)
            else:
                high_low_pattern.append(False)
        
        # Check if we have a significant alternating pattern
        if len(high_low_pattern) >= 3 and sum(high_low_pattern) / len(high_low_pattern) >= 0.75:
            # Calculate confidence based on pattern consistency
            consistency = sum(high_low_pattern) / len(high_low_pattern)
            confidence = min(0.95, 0.7 + (consistency - 0.75) * 0.8)
            
            if confidence >= self.min_confidence:
                pattern = {
                    'type': 'alternating_amount_pattern',
                    'length': len(items),
                    'consistency': consistency,
                    'start_date': items[0]['date'].isoformat() if isinstance(items[0]['date'], datetime) else items[0]['date'],
                    'end_date': items[-1]['date'].isoformat() if isinstance(items[-1]['date'], datetime) else items[-1]['date'],
                    'affected_items': [item['item_id'] for item in items if 'item_id' in item],
                    'confidence': confidence,
                    'explanation': f"Alternating high/low amount pattern with {consistency*100:.1f}% consistency across {len(items)} items"
                }
                
                patterns.append(pattern)
        
        return patterns
    
    def detect_amounts_before_approval(self) -> List[Dict[str, Any]]:
        """Detect amounts appearing in invoices/payment applications before formal approval.
        
        This function looks for cases where amounts in change orders appear in payment
        applications or invoices before the change order is formally approved.
        
        Returns:
            List of anomalies for amounts appearing before formal approval
        """
        logger.info("Detecting amounts appearing before formal approval")
        
        # First, get all change orders with approval dates
        # Approval dates could be in various fields - here we use date_created as a proxy
        query = text("""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.date_created,
                d.status,
                li.item_id,
                li.description,
                li.amount
            FROM 
                documents d
            JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                d.doc_type = 'change_order'
                AND d.date_created IS NOT NULL
            ORDER BY
                d.date_created
        """)
        
        change_orders = self.db_session.execute(query).fetchall()
        
        # Get all payment applications/invoices
        query = text("""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.date_created,
                li.item_id,
                li.description,
                li.amount
            FROM 
                documents d
            JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                d.doc_type IN ('payment_app', 'invoice')
                AND d.date_created IS NOT NULL
            ORDER BY
                d.date_created
        """)
        
        payments = self.db_session.execute(query).fetchall()
        
        # Convert change orders and payments to dictionaries
        change_order_data = []
        for co in change_orders:
            doc_id, doc_type, date_created, status, item_id, description, amount = co
            
            if not date_created or amount is None:
                continue
                
            change_order_data.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'date': date_created,
                'status': status,
                'item_id': item_id,
                'description': description,
                'amount': float(amount)
            })
        
        payment_data = []
        for payment in payments:
            doc_id, doc_type, date_created, item_id, description, amount = payment
            
            if not date_created or amount is None:
                continue
                
            payment_data.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'date': date_created,
                'item_id': item_id,
                'description': description,
                'amount': float(amount)
            })
        
        # Look for amounts that appear in payments before they appear in approved change orders
        anomalies = []
        
        # Loop through payments
        for payment in payment_data:
            payment_amount = payment['amount']
            payment_date = payment['date']
            
            # Find matching change order amounts
            matching_cos = []
            for co in change_order_data:
                # Skip if not the same amount (within small tolerance)
                tolerance = payment_amount * 0.01  # 1% tolerance
                if abs(co['amount'] - payment_amount) > tolerance:
                    continue
                    
                # Found a matching amount - check date
                co_date = co['date']
                
                # Convert dates to datetime objects if needed
                if isinstance(payment_date, str):
                    payment_date = self._parse_date_string(payment_date)
                if isinstance(co_date, str):
                    co_date = self._parse_date_string(co_date)
                
                # Calculate days difference (positive if payment is before CO)
                if payment_date and co_date:
                    days_diff = (co_date - payment_date).days
                    
                    # If payment is before CO approval (with small buffer for paperwork)
                    if days_diff > 1:
                        # Check if the CO is actually approved
                        co_status = co.get('status', '').lower()
                        
                        # Only consider this an issue if the CO is approved
                        if not co_status or 'reject' not in co_status:
                            matching_cos.append({
                                'co': co,
                                'days_diff': days_diff
                            })
            
            # If we found matches, create an anomaly
            if matching_cos:
                # Calculate confidence based on time difference and number of matches
                max_days = max(match['days_diff'] for match in matching_cos)
                confidence = min(0.95, 0.7 + min(max_days / 30, 0.2) + min(len(matching_cos) / 2, 0.1))
                
                if confidence >= self.min_confidence:
                    anomaly = {
                        'type': 'amount_before_approval',
                        'payment_doc_id': payment['doc_id'],
                        'payment_item_id': payment['item_id'],
                        'payment_date': payment_date.isoformat() if hasattr(payment_date, 'isoformat') else str(payment_date),
                        'amount': payment_amount,
                        'matching_change_orders': [match['co']['doc_id'] for match in matching_cos],
                        'max_days_difference': max_days,
                        'confidence': confidence,
                        'explanation': f"Amount ${payment_amount:.2f} appears in {payment['doc_type']} {max_days} days before it appears in an approved change order"
                    }
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} cases of amounts appearing before formal approval")
        return anomalies
    
    def _detect_date_patterns(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in the dates of financial items.
        
        Args:
            items: List of financial items sorted by date
            
        Returns:
            List of date patterns
        """
        patterns = []
        
        # Need at least 3 items to detect a date pattern
        if len(items) < 3:
            return patterns
            
        # Extract days of month
        days_of_month = []
        for item in items:
            date = item['date']
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    continue
            
            if isinstance(date, datetime):
                days_of_month.append(date.day)
        
        # Count frequency of each day
        day_counts = {}
        for day in days_of_month:
            if day not in day_counts:
                day_counts[day] = 0
            day_counts[day] += 1
        
        # Check for significant day patterns
        for day, count in day_counts.items():
            if count >= 3 and count / len(days_of_month) >= 0.3:
                # Extract items on this day
                day_items = []
                for item in items:
                    date = item['date']
                    if isinstance(date, str):
                        try:
                            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            continue
                    
                    if isinstance(date, datetime) and date.day == day:
                        day_items.append(item)
                
                # Calculate confidence based on pattern consistency
                consistency = count / len(days_of_month)
                confidence = min(0.95, 0.7 + (consistency - 0.3) * 0.75)
                
                if confidence >= self.min_confidence:
                    pattern = {
                        'type': 'recurring_day_pattern',
                        'day_of_month': day,
                        'count': count,
                        'total_items': len(days_of_month),
                        'consistency': consistency,
                        'affected_items': [item['item_id'] for item in day_items if 'item_id' in item],
                        'confidence': confidence,
                        'explanation': f"Pattern of financial activity on day {day} of the month ({count} out of {len(days_of_month)} items, {consistency*100:.1f}% consistency)"
                    }
                    
                    patterns.append(pattern)
        
        return patterns